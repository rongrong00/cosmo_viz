#include <mpi.h>
#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <map>
#include <set>
#include <cstring>
#include <filesystem>
#include "common/Config.h"
#include "renderer/Camera.h"
#include "renderer/GridReader.h"
#include "renderer/RayTracer.h"
#include "renderer/ImageWriter.h"

namespace fs = std::filesystem;

static void printUsage(const char* prog) {
    std::cerr << "Usage: " << prog << "\n"
              << "  Single-frame:  --grid grid.hdf5 --camera cam.yaml --output outdir\n"
              << "  Batch:         --grid grid.hdf5 --camera-list cams.txt\n"
              << "                 (each line: <cam_yaml> <output_dir>)\n"
              << "  Optional:      --fields gas_density[,temperature,...]\n";
}

struct Job { std::string cam_path; std::string out_dir; };

static std::vector<Job> parseCameraList(const std::string& path) {
    std::vector<Job> jobs;
    std::ifstream fin(path);
    if (!fin) throw std::runtime_error("Cannot open camera list: " + path);
    std::string line;
    while (std::getline(fin, line)) {
        if (line.empty() || line[0] == '#') continue;
        std::istringstream ss(line);
        Job j;
        ss >> j.cam_path >> j.out_dir;
        if (!j.cam_path.empty() && !j.out_dir.empty()) jobs.push_back(j);
    }
    return jobs;
}

static std::vector<std::string> splitCSV(const std::string& s) {
    std::vector<std::string> out;
    std::string cur;
    for (char c : s) {
        if (c == ',') { if (!cur.empty()) out.push_back(cur); cur.clear(); }
        else cur += c;
    }
    if (!cur.empty()) out.push_back(cur);
    return out;
}

// Fields needed on the grid to service a projection config.
static void collectFields(const std::vector<ProjectionConfig>& projs,
                          std::set<std::string>& out) {
    for (const auto& p : projs) {
        out.insert(p.field);
        if (p.mode == "mass_weighted") out.insert("gas_density");
        if (p.mode == "los_velocity") {
            out.insert("gas_density");
            out.insert("gas_velocity_x");
            out.insert("gas_velocity_y");
            out.insert("gas_velocity_z");
        }
    }
}

static void runOne(const Camera& camera, const GridData& grid,
                   const std::vector<ProjectionConfig>& projs,
                   const CameraConfig& cam_cfg,
                   const std::string& grid_path,
                   const std::string& out_dir) {
    fs::create_directories(out_dir);
    for (const auto& proj : projs) {
        std::vector<float> image;
        if (proj.mode == "column") {
            image = RayTracer::traceColumnDensity(camera, grid, proj.field);
        } else if (proj.mode == "mass_weighted") {
            image = RayTracer::traceMassWeighted(camera, grid, proj.field, "gas_density");
        } else if (proj.mode == "los_velocity") {
            image = RayTracer::traceLOSVelocity(camera, grid, "gas_density");
        } else {
            std::cerr << "Warning: unsupported projection mode '" << proj.mode
                      << "', skipping." << std::endl;
            continue;
        }
        std::string grid_name = fs::path(grid_path).stem().string();
        std::string out_file = out_dir + "/projection_" + grid_name + "_" + proj.field + ".hdf5";
        ImageWriter::write(out_file, image, cam_cfg.image_width, cam_cfg.image_height,
                           proj.field, cam_cfg, grid_path);
    }
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int world_rank, world_size;
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    MPI_Comm node_comm;
    MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, 0,
                        MPI_INFO_NULL, &node_comm);
    int node_rank, node_size;
    MPI_Comm_rank(node_comm, &node_rank);
    MPI_Comm_size(node_comm, &node_size);

    std::string grid_path, camera_path, camera_list, output_dir, fields_arg;
    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];
        if (arg == "--grid" && i + 1 < argc) grid_path = argv[++i];
        else if (arg == "--camera" && i + 1 < argc) camera_path = argv[++i];
        else if (arg == "--camera-list" && i + 1 < argc) camera_list = argv[++i];
        else if (arg == "--output" && i + 1 < argc) output_dir = argv[++i];
        else if (arg == "--fields" && i + 1 < argc) fields_arg = argv[++i];
    }

    if (grid_path.empty() ||
        (camera_path.empty() && camera_list.empty())) {
        if (world_rank == 0) printUsage(argv[0]);
        MPI_Finalize();
        return 1;
    }

    // Build the job list.
    std::vector<Job> jobs;
    if (!camera_list.empty()) {
        jobs = parseCameraList(camera_list);
    } else {
        if (output_dir.empty()) {
            if (world_rank == 0) printUsage(argv[0]);
            MPI_Finalize();
            return 1;
        }
        jobs.push_back({camera_path, output_dir});
    }

    if (world_rank == 0) {
        std::cout << "Renderer: " << world_size << " ranks across "
                  << (world_size / node_size) << " nodes, "
                  << node_size << " ranks/node (shared-mem grid), "
                  << jobs.size() << " jobs, grid=" << grid_path << std::endl;
    }

    // Read grid header on all ranks (cheap).
    GridData grid = GridReader::readHeader(grid_path);
    size_t total = grid.totalCells();

    // Determine which fields to load. Prefer --fields; else derive from the
    // projection configs across all jobs.
    std::vector<std::string> needed;
    if (!fields_arg.empty()) {
        needed = splitCSV(fields_arg);
    } else {
        std::set<std::string> fset;
        for (const auto& j : jobs) {
            auto projs = parseProjectionConfigs(j.cam_path);
            collectFields(projs, fset);
        }
        needed.assign(fset.begin(), fset.end());
    }

    // Allocate one shared-memory float buffer per field, per node.
    std::vector<MPI_Win> wins;
    std::map<std::string, float*> bufs;
    for (const auto& name : needed) {
        MPI_Aint bytes = (node_rank == 0) ?
            static_cast<MPI_Aint>(total * sizeof(float)) : 0;
        float* ptr = nullptr;
        MPI_Win win;
        MPI_Win_allocate_shared(bytes, sizeof(float), MPI_INFO_NULL,
                                node_comm, &ptr, &win);
        if (node_rank != 0) {
            MPI_Aint qsz; int qdisp;
            MPI_Win_shared_query(win, 0, &qsz, &qdisp, &ptr);
        }
        bufs[name] = ptr;
        wins.push_back(win);
    }

    // Node leader fills the buffers.
    if (node_rank == 0) {
        GridReader::readFieldsInto(grid_path, needed, bufs);
    }
    MPI_Barrier(node_comm);

    // Attach pointers to GridData view used by the ray tracer.
    for (const auto& name : needed) grid.fields[name] = bufs[name];

    // Split jobs round-robin across world ranks.
    for (size_t j = world_rank; j < jobs.size(); j += world_size) {
        const auto& job = jobs[j];
        CameraConfig cam_cfg = parseCameraConfig(job.cam_path);
        auto projs = parseProjectionConfigs(job.cam_path);
        Camera camera(cam_cfg);

        std::cout << "[rank " << world_rank << "] "
                  << (j + 1) << "/" << jobs.size()
                  << " cam=" << job.cam_path << " -> " << job.out_dir << std::endl;

        runOne(camera, grid, projs, cam_cfg, grid_path, job.out_dir);
    }

    MPI_Barrier(MPI_COMM_WORLD);

    for (auto& w : wins) MPI_Win_free(&w);
    MPI_Comm_free(&node_comm);
    MPI_Finalize();
    return 0;
}
