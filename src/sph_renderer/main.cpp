// SPH direct ray tracer entry point.
//
// Usage:
//   ./sph_renderer --snapshot <path> --region <region.yaml>
//                  [--camera <camera.yaml> | --camera-list <list.txt>]
//                  [--output <dir>] [--fields <csv>]
//
// Parallelism model (Phase D2 + D3):
//   * MPI world is split into per-node shared-memory sub-communicators.
//   * Node leaders form a `leader_comm`. They call ParticleLoader::loadMPI
//     to parallelize subfile I/O across nodes (round-robin over leaders).
//   * Each node leader builds the BVHs into local std::vectors.
//   * Each particle field and BVH array is mirrored into an MPI shared-memory
//     window; all ranks on the node read the single copy.
//   * Camera frames are round-robined across world ranks.
//
// Within each rank, OpenMP still parallelizes the inner pixel/build loops.

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <set>
#include <sstream>
#include <string>
#include <vector>

#include <mpi.h>

#include "common/Config.h"
#include "common/ShmArray.h"
#include "renderer/Camera.h"
#include "sph_renderer/BVH.h"
#include "sph_renderer/ImageWriter.h"
#include "sph_renderer/ParticleLoader.h"
#include "sph_renderer/ParticleStore.h"
#include "sph_renderer/SphRayTracer.h"

#include <sys/stat.h>
#include <sys/types.h>

// ---------------------------------------------------------------------------
// MPI shared-memory replication of ShmArray buffers.
// ---------------------------------------------------------------------------

// Windows owned by this process. MPI_Win_free must be called on each before
// MPI_Finalize, so we collect them here and free in main() tail.
static std::vector<MPI_Win> g_shm_wins;

// Allocate an MPI shared-memory window sized to hold `arr` as seen by the
// node leader, copy the leader's data in, and switch `arr` on every rank to
// read from the window.
template <class T>
static void replicateToShm(ShmArray<T>& arr, MPI_Comm node_comm, int node_rank) {
    std::uint64_t n = (node_rank == 0) ? static_cast<std::uint64_t>(arr.size()) : 0;
    MPI_Bcast(&n, 1, MPI_UINT64_T, 0, node_comm);

    MPI_Aint bytes = (node_rank == 0)
                         ? static_cast<MPI_Aint>(n * sizeof(T))
                         : static_cast<MPI_Aint>(0);
    T*      ptr = nullptr;
    MPI_Win win = MPI_WIN_NULL;
    MPI_Win_allocate_shared(bytes, sizeof(T), MPI_INFO_NULL, node_comm,
                            &ptr, &win);
    if (node_rank != 0) {
        MPI_Aint qsz;
        int      qdisp;
        MPI_Win_shared_query(win, 0, &qsz, &qdisp, &ptr);
    }
    if (node_rank == 0 && n > 0) {
        std::memcpy(ptr, arr.data(), n * sizeof(T));
    }
    MPI_Barrier(node_comm);
    arr.adoptExternal(ptr, static_cast<std::size_t>(n));
    g_shm_wins.push_back(win);
}

// Mirror all ParticleStore arrays + POD metadata across the node.
static void replicateParticleStore(ParticleStore& ps, MPI_Comm node_comm,
                                   int node_rank) {
    // Scalar metadata: just bytes-broadcast the POD fields.
    MPI_Bcast(&ps.region_center, sizeof(ps.region_center), MPI_BYTE, 0, node_comm);
    MPI_Bcast(&ps.region_size,   sizeof(ps.region_size),   MPI_BYTE, 0, node_comm);
    MPI_Bcast(&ps.bbox_gas,      sizeof(ps.bbox_gas),      MPI_BYTE, 0, node_comm);
    MPI_Bcast(&ps.bbox_dm,       sizeof(ps.bbox_dm),       MPI_BYTE, 0, node_comm);
    MPI_Bcast(&ps.h_max_gas, 1, MPI_FLOAT, 0, node_comm);
    MPI_Bcast(&ps.h_max_dm,  1, MPI_FLOAT, 0, node_comm);

    ShmArray<float>* gas_arrs[] = {
        &ps.gas_x, &ps.gas_y, &ps.gas_z, &ps.gas_h,
        &ps.gas_mass, &ps.gas_density,
        &ps.gas_temperature, &ps.gas_metallicity,
        &ps.gas_vx, &ps.gas_vy, &ps.gas_vz, &ps.gas_hii,
    };
    for (auto* a : gas_arrs) replicateToShm(*a, node_comm, node_rank);

    ShmArray<float>* dm_arrs[] = {
        &ps.dm_x, &ps.dm_y, &ps.dm_z, &ps.dm_h, &ps.dm_mass,
    };
    for (auto* a : dm_arrs) replicateToShm(*a, node_comm, node_rank);
}

static void replicateBVH(BVH& bvh, MPI_Comm node_comm, int node_rank) {
    replicateToShm(bvh.nodes, node_comm, node_rank);
    replicateToShm(bvh.perm,  node_comm, node_rank);
}

// ---------------------------------------------------------------------------

static void mkdirRecursive(const std::string& path) {
    if (path.empty()) return;
    size_t pos = 0;
    while (pos != std::string::npos) {
        pos = path.find('/', pos + 1);
        std::string sub = (pos == std::string::npos) ? path : path.substr(0, pos);
        if (sub.empty()) continue;
        ::mkdir(sub.c_str(), 0755);
    }
}

static void printUsage(const char* prog) {
    std::cerr << "Usage: " << prog << "\n"
              << "  --snapshot <path>      snapshot prefix (snap_NNN) or single .hdf5\n"
              << "  --region   <yaml>      region selection config\n"
              << "  --camera   <yaml>      single camera config\n"
              << "  --camera-list <file>   one camera path per line (batch mode)\n"
              << "  --output   <dir>       output directory\n"
              << "  --fields   <csv>       requested fields (drives optional field load)\n"
              << "  --no-volume            skip emission/absorption volume renders\n"
              << std::endl;
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

static std::set<std::string> optionalGasFields(const std::vector<std::string>& requested) {
    std::set<std::string> out;
    for (const auto& f : requested) {
        if      (f == "temperature"    || f == "temperature_dw"
              || f == "temperature_vw" || f == "heated_gas"
              || f == "gas_temperature"
              || f == "cold_col"       || f == "warm_col"
              || f == "hot_col"
              || f == "gas_cold"       || f == "gas_warm"
              || f == "gas_hot")                                  out.insert("temperature");
        else if (f == "metallicity"    || f == "metallicity_dw"
              || f == "metallicity_vw" || f == "metal_column"
              || f == "heavy_elements" || f == "gas_metallicity") out.insert("metallicity");
        else if (f == "gas_velocity"  || f == "los_velocity"
              || f == "velocity_column")                        out.insert("velocity");
        else if (f == "HII_density"   || f == "hii_fraction"
              || f == "ionized_H"     || f == "neutral_H"
              || f == "hii_column")                             out.insert("hii");
    }
    return out;
}

int main(int argc, char** argv) {
    int provided = 0;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);

    int world_rank = 0, world_size = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    MPI_Comm node_comm;
    MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, 0,
                        MPI_INFO_NULL, &node_comm);
    int node_rank = 0, node_size = 1;
    MPI_Comm_rank(node_comm, &node_rank);
    MPI_Comm_size(node_comm, &node_size);

    // One-rank-per-node communicator used for parallel subfile I/O.
    MPI_Comm leader_comm = MPI_COMM_NULL;
    int leader_color = (node_rank == 0) ? 0 : MPI_UNDEFINED;
    MPI_Comm_split(MPI_COMM_WORLD, leader_color, world_rank, &leader_comm);

    std::string snapshot_path, region_path, camera_path, camera_list_path,
                output_dir, fields_arg;
    int samples_per_axis = 1;
    bool skip_volume = false;
    float gas_h_scale = 1.0f;
    double star_h_max = 0.0;    // 0 = no cap
    double star_h_min = 0.0;    // 0 = no floor
    int    star_knn_k  = 16;
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        if      (a == "--snapshot"    && i + 1 < argc) snapshot_path    = argv[++i];
        else if (a == "--region"      && i + 1 < argc) region_path      = argv[++i];
        else if (a == "--camera"      && i + 1 < argc) camera_path      = argv[++i];
        else if (a == "--camera-list" && i + 1 < argc) camera_list_path = argv[++i];
        else if (a == "--output"      && i + 1 < argc) output_dir       = argv[++i];
        else if (a == "--fields"      && i + 1 < argc) fields_arg       = argv[++i];
        else if (a == "--supersample" && i + 1 < argc) samples_per_axis = std::max(1, atoi(argv[++i]));
        else if (a == "--no-volume")                   skip_volume      = true;
        else if (a == "--gas-h-scale" && i + 1 < argc) gas_h_scale      = static_cast<float>(atof(argv[++i]));
        else if (a == "--star-nn-k"   && i + 1 < argc) star_knn_k       = atoi(argv[++i]);
        else if (a == "--star-h-max"  && i + 1 < argc) star_h_max       = atof(argv[++i]);
        else if (a == "--star-h-min"  && i + 1 < argc) star_h_min       = atof(argv[++i]);
    }

    if (snapshot_path.empty() || region_path.empty()) {
        if (world_rank == 0) printUsage(argv[0]);
        MPI_Finalize();
        return 1;
    }

    RegionConfig region = parseRegionConfig(region_path);

    bool load_gas = false, load_dm = false, load_stars = false;
    for (const auto& t : region.particle_types) {
        if (t == "gas")   load_gas   = true;
        if (t == "dm")    load_dm    = true;
        if (t == "stars" || t == "star") load_stars = true;
    }
    if (!load_gas && !load_dm && !load_stars) load_gas = true;
    if (load_dm && load_stars) {
        if (world_rank == 0) {
            std::cerr << "Error: particle_types cannot contain both 'dm' and "
                         "'stars' (they share the dm_* array slot)." << std::endl;
        }
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    std::vector<std::string> requested_fields;
    if (!fields_arg.empty()) requested_fields = splitCSV(fields_arg);
    else                     requested_fields = {"gas_density"};
    auto gas_optional = optionalGasFields(requested_fields);

    if (world_rank == 0) {
        std::cout << "SPH renderer: " << world_size << " ranks across "
                  << (world_size / node_size) << " nodes, "
                  << node_size << " ranks/node (shared-mem particles + BVH)\n"
                  << "  snapshot = " << snapshot_path << "\n"
                  << "  region   = " << region.name
                  << "  center=[" << region.center.x << ", " << region.center.y << ", " << region.center.z << "]"
                  << " size=["    << region.size.x   << ", " << region.size.y   << ", " << region.size.z   << "]\n"
                  << "  types    =";
        for (const auto& t : region.particle_types) std::cout << " " << t;
        std::cout << "\n  fields   =";
        for (const auto& f : requested_fields) std::cout << " " << f;
        std::cout << std::endl;
    }

    // --- Load particles (node leaders, parallel subfile I/O) ---
    ParticleStore ps;
    using clk = std::chrono::steady_clock;
    if (node_rank == 0) {
        auto t0 = clk::now();
        ps = ParticleLoader::loadMPI(snapshot_path, region, gas_optional,
                                     load_dm, leader_comm, -1.0,
                                     load_stars,
                                     load_stars ? star_knn_k : 32,
                                     load_stars ? star_h_max : 0.0,
                                     load_stars ? star_h_min : 0.0);
        if (gas_h_scale != 1.0f && ps.numGas() > 0) {
            for (size_t i = 0; i < ps.numGas(); ++i) ps.gas_h[i] *= gas_h_scale;
            ps.h_max_gas *= gas_h_scale;
            std::cout << "  Applied --gas-h-scale=" << gas_h_scale
                      << "  new h_max=" << ps.h_max_gas << std::endl;
        }
        auto dt = std::chrono::duration<double>(clk::now() - t0).count();
        std::cout << "[rank " << world_rank << "] loaded particles in "
                  << dt << "s: gas=" << ps.numGas() << " dm=" << ps.numDM()
                  << std::endl;
    }

    // --- Build BVHs (node leader only) ---
    BVH gas_bvh, dm_bvh;
    if (node_rank == 0) {
        if (ps.numGas() > 0) {
            auto t0 = clk::now();
            gas_bvh.buildFromSpheres(ps.gas_x.data(), ps.gas_y.data(), ps.gas_z.data(),
                                     ps.gas_h.data(), ps.numGas());
            auto dt = std::chrono::duration<double>(clk::now() - t0).count();
            std::cout << "  gas BVH: " << gas_bvh.nodes.size() << " nodes  build="
                      << dt << "s" << std::endl;
        }
        if (ps.numDM() > 0) {
            auto t0 = clk::now();
            dm_bvh.buildFromSpheres(ps.dm_x.data(), ps.dm_y.data(), ps.dm_z.data(),
                                    ps.dm_h.data(), ps.numDM());
            auto dt = std::chrono::duration<double>(clk::now() - t0).count();
            std::cout << "  dm  BVH: " << dm_bvh.nodes.size() << " nodes  build="
                      << dt << "s" << std::endl;
        }
    }

    // --- Replicate to MPI shared-memory windows (all ranks on each node) ---
    replicateParticleStore(ps, node_comm, node_rank);
    replicateBVH(gas_bvh, node_comm, node_rank);
    replicateBVH(dm_bvh,  node_comm, node_rank);
    MPI_Barrier(MPI_COMM_WORLD);

    if (world_rank == 0) {
        std::cout << "Summary:\n"
                  << "  gas N = " << ps.numGas() << ", h_max = " << ps.h_max_gas << "\n"
                  << "  dm  N = " << ps.numDM()  << ", h_max = " << ps.h_max_dm  << "\n";
        if (ps.numGas() > 0) {
            std::cout << "  gas bbox: ["
                      << ps.bbox_gas.lo.x << ", " << ps.bbox_gas.lo.y << ", " << ps.bbox_gas.lo.z << "] - ["
                      << ps.bbox_gas.hi.x << ", " << ps.bbox_gas.hi.y << ", " << ps.bbox_gas.hi.z << "]\n";
        }
        if (ps.numDM() > 0) {
            std::cout << "  dm  bbox: ["
                      << ps.bbox_dm.lo.x  << ", " << ps.bbox_dm.lo.y  << ", " << ps.bbox_dm.lo.z  << "] - ["
                      << ps.bbox_dm.hi.x  << ", " << ps.bbox_dm.hi.y  << ", " << ps.bbox_dm.hi.z  << "]\n";
        }
    }

    // --- Gather camera paths ---
    std::vector<std::string> camera_paths;
    if (!camera_list_path.empty()) {
        std::ifstream f(camera_list_path);
        std::string line;
        while (std::getline(f, line)) {
            auto b = line.find_first_not_of(" \t\r\n");
            if (b == std::string::npos || line[b] == '#') continue;
            auto e = line.find_last_not_of(" \t\r\n");
            camera_paths.push_back(line.substr(b, e - b + 1));
        }
        if (camera_paths.empty()) {
            if (world_rank == 0)
                std::cerr << "camera-list " << camera_list_path << " is empty\n";
            MPI_Finalize();
            return 1;
        }
    } else if (!camera_path.empty()) {
        camera_paths.push_back(camera_path);
    }

    if (!camera_paths.empty() && (ps.numGas() > 0 || ps.numDM() > 0)) {
      bool batch = !camera_list_path.empty();
      if (world_rank == 0 && !output_dir.empty()) mkdirRecursive(output_dir);
      MPI_Barrier(MPI_COMM_WORLD);

      for (size_t frame = world_rank; frame < camera_paths.size(); frame += world_size) {
        const std::string& this_cam_path = camera_paths[frame];
        CameraConfig cam_cfg = parseCameraConfig(this_cam_path);
        Camera camera(cam_cfg);
        std::string frame_dir = output_dir;
        if (batch && !output_dir.empty()) {
            std::ostringstream fs;
            fs << output_dir << "/frame_" << std::setw(4) << std::setfill('0') << frame;
            frame_dir = fs.str();
            mkdirRecursive(frame_dir);
        }
        std::cout << "[rank " << world_rank << " / frame " << frame
                  << "/" << camera_paths.size() << "] "
                  << this_cam_path << " → " << frame_dir << "\n"
                  << "  camera " << cam_cfg.type << " "
                  << camera.width() << "x" << camera.height()
                  << "  los_slab=" << cam_cfg.los_slab << std::endl;
        const std::string& output_dir = frame_dir;   // shadow for the block

        auto stat_report = [&](const std::vector<float>& img, int C, const char* label, double dt) {
            size_t npix = static_cast<size_t>(camera.width()) * camera.height();
            std::cout << "  " << label << ": " << dt << "s  "
                      << (npix / dt) / 1e6 << " Mpix/s" << std::endl;
            for (int c = 0; c < C; ++c) {
                double vmin = 1e300, vmax = -1e300, sum = 0.0;
                size_t nonzero = 0;
                for (size_t i = 0; i < npix; ++i) {
                    float v = img[i * C + c];
                    sum += v;
                    if (v != 0.0f) ++nonzero;
                    if (v < vmin) vmin = v;
                    if (v > vmax) vmax = v;
                }
                std::cout << "    ch" << c << ": min=" << vmin << " max=" << vmax
                          << " mean=" << (sum / static_cast<double>(npix))
                          << " nonzero=" << nonzero << "/" << npix << std::endl;
            }
        };

        bool want_column = false, want_dm = false, want_star = false;
        for (const auto& f : requested_fields) {
            if (f == "gas_density")  want_column = true;
            if (f == "dm_density")   want_dm = true;
            if (f == "star_density") want_star = true;
        }
        if (want_column && ps.numGas() > 0) {
            auto t0 = clk::now();
            std::vector<float> image = SphRayTracer::traceGasColumn(camera, ps, gas_bvh);
            double dt = std::chrono::duration<double>(clk::now() - t0).count();
            stat_report(image, 1, "trace_gas_column", dt);
            if (!output_dir.empty()) {
                SphImageWriter::write(output_dir + "/gas_column.h5", image,
                                      camera.width(), camera.height(),
                                      "gas_column_density", cam_cfg, snapshot_path, region);
            }
        }

        // DM and Star both reuse the dm_* particle arrays and BVH; only the
        // output path/dataset name differ.
        if ((want_dm || want_star) && ps.numDM() > 0) {
            auto t0 = clk::now();
            std::vector<float> image = SphRayTracer::traceDMColumn(camera, ps, dm_bvh);
            double dt = std::chrono::duration<double>(clk::now() - t0).count();
            const bool as_stars = load_stars && want_star;
            stat_report(image, 1, as_stars ? "trace_star_column" : "trace_dm_column", dt);
            if (!output_dir.empty()) {
                const std::string out_name = as_stars ? "/star_column.h5" : "/dm_column.h5";
                const std::string ds_name  = as_stars ? "star_column_density" : "dm_column_density";
                SphImageWriter::write(output_dir + out_name, image,
                                      camera.width(), camera.height(),
                                      ds_name, cam_cfg, snapshot_path, region);
            }
        }

        // weight_kind: 0 = mass-weighted (no extra_w), 1 = density-weighted
        // (extra_w = ρ → ρ²-biased, emphasizes dense gas), 2 = volume-weighted
        // (extra_w = 1/ρ → emphasizes diffuse gas, matches Illustris-style
        // T maps where hot WHIM/ICM dominates a projection).
        struct MWField {
            std::string req;
            std::string outname;
            const float* buf;
            int weight_kind;
        };
        std::vector<MWField> mw;
        for (const auto& f : requested_fields) {
            if (f == "temperature" && !ps.gas_temperature.empty())
                mw.push_back({f, "gas_temperature_mw", ps.gas_temperature.data(), 0});
            else if (f == "temperature_dw" && !ps.gas_temperature.empty())
                mw.push_back({f, "gas_temperature_dw", ps.gas_temperature.data(), 1});
            else if (f == "temperature_vw" && !ps.gas_temperature.empty())
                mw.push_back({f, "gas_temperature_vw", ps.gas_temperature.data(), 2});
            else if (f == "metallicity" && !ps.gas_metallicity.empty())
                mw.push_back({f, "gas_metallicity_mw", ps.gas_metallicity.data(), 0});
            else if (f == "metallicity_dw" && !ps.gas_metallicity.empty())
                mw.push_back({f, "gas_metallicity_dw", ps.gas_metallicity.data(), 1});
            else if (f == "metallicity_vw" && !ps.gas_metallicity.empty())
                mw.push_back({f, "gas_metallicity_vw", ps.gas_metallicity.data(), 2});
        }
        std::vector<float> inv_rho;
        auto ensure_inv_rho = [&]() {
            if (!inv_rho.empty() || ps.gas_density.empty()) return;
            inv_rho.resize(ps.numGas());
            #pragma omp parallel for schedule(static)
            for (size_t i = 0; i < ps.numGas(); ++i) {
                float r = ps.gas_density[i];
                inv_rho[i] = (r > 0.0f) ? (1.0f / r) : 0.0f;
            }
        };
        for (const auto& m : mw) {
            const float* extra_w = nullptr;
            if (m.weight_kind == 1) extra_w = ps.gas_density.data();
            else if (m.weight_kind == 2) { ensure_inv_rho(); extra_w = inv_rho.data(); }
            auto t0 = clk::now();
            std::vector<float> image = SphRayTracer::traceGasWeighted(
                camera, ps, gas_bvh, extra_w, {m.buf});
            double dt = std::chrono::duration<double>(clk::now() - t0).count();
            stat_report(image, 1, ("trace_weighted " + m.req).c_str(), dt);
            if (!output_dir.empty()) {
                SphImageWriter::write(output_dir + "/" + m.req + ".h5", image,
                                      camera.width(), camera.height(),
                                      m.outname, cam_cfg, snapshot_path, region);
            }
        }

        bool want_metal_col = false, want_vel_col = false, want_hii_col = false;
        bool want_cold = false, want_warm = false, want_hot = false;
        for (const auto& f : requested_fields) {
            if (f == "metal_column")    want_metal_col = true;
            if (f == "velocity_column") want_vel_col = true;
            if (f == "hii_column")      want_hii_col = true;
            if (f == "cold_col")        want_cold = true;
            if (f == "warm_col")        want_warm = true;
            if (f == "hot_col")         want_hot = true;
        }
        // Phase-masked mass columns: Σ m·𝟙[T in bin]·F(b/h)/h². Three bins
        // (cold <10^4.5, warm 10^4.5–10^6, hot >10^6 K) feed a tri-color
        // Python compositor to reproduce the Illustris-style temperature look.
        auto render_phase = [&](const char* name, const char* out_tag,
                                float T_lo, float T_hi) {
            if (ps.gas_temperature.empty()) return;
            std::vector<float> w(ps.numGas());
            const float* T = ps.gas_temperature.data();
            #pragma omp parallel for schedule(static)
            for (size_t i = 0; i < ps.numGas(); ++i) {
                w[i] = (T[i] >= T_lo && T[i] < T_hi) ? 1.0f : 0.0f;
            }
            auto t0 = clk::now();
            std::vector<float> image = SphRayTracer::traceGasWeightedColumn(
                camera, ps, gas_bvh, w.data());
            double dt = std::chrono::duration<double>(clk::now() - t0).count();
            stat_report(image, 1, name, dt);
            if (!output_dir.empty()) {
                SphImageWriter::write(std::string(output_dir) + "/" + name + ".h5",
                                      image, camera.width(), camera.height(),
                                      out_tag, cam_cfg, snapshot_path, region);
            }
        };
        // Phase bounds tuned for z~5: no virialized clusters yet, so max T
        // is ~few·10^5 K. Split at 10^4 and 10^5 so cold=photo-ionized IGM,
        // warm=filament shocks, hot=halo virial shocks.
        // Physically-motivated z~5 phase split:
        //   cold <10^4 K   neutral / pre-reionization pockets
        //   warm 10^4-10^5 photo-ionized IGM + filament shocks
        //   hot  >10^5 K   shock-heated halo atmospheres
        if (want_cold) render_phase("cold_col", "gas_cold_column", 0.0f,    1.0e4f);
        if (want_warm) render_phase("warm_col", "gas_warm_column", 1.0e4f,  1.0e5f);
        if (want_hot)  render_phase("hot_col",  "gas_hot_column",  1.0e5f,  1.0e12f);
        if (want_hii_col && !ps.gas_hii.empty()) {
            // HII mass column = Σ m_i · X_H · xe_i · F(b/h)/h². This is the
            // Set-1 analogue of metal_column — a straight column density map
            // of ionized hydrogen.
            std::vector<float> w(ps.numGas());
            const float X_H = 0.76f;
            const float* xe = ps.gas_hii.data();
            #pragma omp parallel for schedule(static)
            for (size_t i = 0; i < ps.numGas(); ++i) {
                float x = std::max(0.0f, xe[i]);
                w[i] = X_H * x;
            }
            auto t0 = clk::now();
            std::vector<float> image = SphRayTracer::traceGasWeightedColumn(
                camera, ps, gas_bvh, w.data());
            double dt = std::chrono::duration<double>(clk::now() - t0).count();
            stat_report(image, 1, "trace_hii_column", dt);
            if (!output_dir.empty()) {
                SphImageWriter::write(output_dir + "/hii_column.h5", image,
                                      camera.width(), camera.height(),
                                      "hii_column_density", cam_cfg, snapshot_path, region);
            }
        }
        if (want_metal_col && !ps.gas_metallicity.empty()) {
            auto t0 = clk::now();
            std::vector<float> image = SphRayTracer::traceGasWeightedColumn(
                camera, ps, gas_bvh, ps.gas_metallicity.data());
            double dt = std::chrono::duration<double>(clk::now() - t0).count();
            stat_report(image, 1, "trace_metal_column", dt);
            if (!output_dir.empty()) {
                SphImageWriter::write(output_dir + "/metal_column.h5", image,
                                      camera.width(), camera.height(),
                                      "metal_column_density", cam_cfg, snapshot_path, region);
            }
        }
        if (want_vel_col && !ps.gas_vx.empty()) {
            std::vector<float> vmag(ps.numGas());
            const float* vx = ps.gas_vx.data();
            const float* vy = ps.gas_vy.data();
            const float* vz = ps.gas_vz.data();
            #pragma omp parallel for schedule(static)
            for (size_t i = 0; i < ps.numGas(); ++i) {
                double s = vx[i]*vx[i] + vy[i]*vy[i] + vz[i]*vz[i];
                vmag[i] = static_cast<float>(std::sqrt(s));
            }
            auto t0 = clk::now();
            std::vector<float> image = SphRayTracer::traceGasWeightedColumn(
                camera, ps, gas_bvh, vmag.data());
            double dt = std::chrono::duration<double>(clk::now() - t0).count();
            stat_report(image, 1, "trace_velocity_column", dt);
            if (!output_dir.empty()) {
                SphImageWriter::write(output_dir + "/velocity_column.h5", image,
                                      camera.width(), camera.height(),
                                      "gas_speed_column", cam_cfg, snapshot_path, region);
            }
        }

        // ------------------------------------------------------------------
        // Set-2 Renaissance-style volume renders.
        // ------------------------------------------------------------------
        struct VolField {
            std::string tag;
            std::string req;
            std::vector<float> (*build)(const ParticleStore&);
            double kappa_e, kappa_a;
        };

        // ρ·X_H·xe weighting. The extra ρ factor acts as a natural smoother:
        // dense particles dominate the line integral, so individual low-density
        // particle kernels at bubble edges no longer show as discrete weave.
        // Bubble interiors then reveal the ionized cosmic web inside Strömgren
        // spheres rather than uniform columns.
        auto build_neutral_H = +[](const ParticleStore& s) {
            std::vector<float> w(s.numGas());
            const float X_H = 0.76f;
            #pragma omp parallel for schedule(static)
            for (size_t i = 0; i < s.numGas(); ++i) {
                float xe = s.gas_hii.empty() ? 0.0f : s.gas_hii[i];
                w[i] = s.gas_density[i] * X_H * std::max(0.0f, 1.0f - xe);
            }
            return w;
        };
        auto build_ionized_H = +[](const ParticleStore& s) {
            std::vector<float> w(s.numGas());
            const float X_H = 0.76f;
            #pragma omp parallel for schedule(static)
            for (size_t i = 0; i < s.numGas(); ++i) {
                float xe = s.gas_hii.empty() ? 0.0f : std::max(0.0f, s.gas_hii[i]);
                w[i] = s.gas_density[i] * X_H * xe;
            }
            return w;
        };
        auto build_heated = +[](const ParticleStore& s) {
            std::vector<float> w(s.numGas());
            if (s.gas_temperature.empty()) return w;
            const float T_floor = 1.0e4f;
            #pragma omp parallel for schedule(static)
            for (size_t i = 0; i < s.numGas(); ++i) {
                float Texc = std::max(0.0f, s.gas_temperature[i] - T_floor);
                w[i] = s.gas_density[i] * Texc;
            }
            return w;
        };
        auto build_heavy = +[](const ParticleStore& s) {
            std::vector<float> w(s.numGas());
            if (s.gas_metallicity.empty()) return w;
            #pragma omp parallel for schedule(static)
            for (size_t i = 0; i < s.numGas(); ++i) {
                float rho = s.gas_density[i];
                w[i] = rho * rho * s.gas_metallicity[i];
            }
            return w;
        };

        auto build_gas_density = +[](const ParticleStore& s) {
            std::vector<float> w(s.numGas());
            #pragma omp parallel for schedule(static)
            for (size_t i = 0; i < s.numGas(); ++i) w[i] = s.gas_density[i];
            return w;
        };
        auto build_gas_temperature = +[](const ParticleStore& s) {
            std::vector<float> w(s.numGas());
            if (s.gas_temperature.empty()) return w;
            #pragma omp parallel for schedule(static)
            for (size_t i = 0; i < s.numGas(); ++i)
                w[i] = s.gas_density[i] * s.gas_temperature[i];
            return w;
        };
        // Phase-masked density emissions for tri-color volume render.
        // Weight = ρ · 𝟙[T in bin]. Same units as gas_density, so the same
        // κ_e/κ_a scales apply. Three bins: cold <10^4.5 K, warm 10^4.5–10^6,
        // hot >10^6.
        auto make_phase_build = [](float T_lo, float T_hi) {
            return [T_lo, T_hi](const ParticleStore& s) {
                std::vector<float> w(s.numGas());
                if (s.gas_temperature.empty()) return w;
                const float* T = s.gas_temperature.data();
                #pragma omp parallel for schedule(static)
                for (size_t i = 0; i < s.numGas(); ++i) {
                    w[i] = (T[i] >= T_lo && T[i] < T_hi) ? s.gas_density[i] : 0.0f;
                }
                return w;
            };
        };
        static auto build_gas_cold_store = make_phase_build(0.0f,    1.0e4f);
        static auto build_gas_warm_store = make_phase_build(1.0e4f,  1.0e5f);
        static auto build_gas_hot_store  = make_phase_build(1.0e5f,  1.0e12f);
        auto build_gas_cold = +[](const ParticleStore& s) { return build_gas_cold_store(s); };
        auto build_gas_warm = +[](const ParticleStore& s) { return build_gas_warm_store(s); };
        auto build_gas_hot  = +[](const ParticleStore& s) { return build_gas_hot_store(s);  };

        auto build_gas_metallicity = +[](const ParticleStore& s) {
            std::vector<float> w(s.numGas());
            if (s.gas_metallicity.empty()) return w;
            #pragma omp parallel for schedule(static)
            for (size_t i = 0; i < s.numGas(); ++i)
                w[i] = s.gas_density[i] * s.gas_metallicity[i];
            return w;
        };

        std::vector<VolField> vol_fields = {
            {"neutral_H",       "neutral_H",       build_neutral_H,       5.0e4, 1.0e4},
            {"ionized_H",       "ionized_H",       build_ionized_H,       5.0e4, 1.0e4},
            {"heated_gas",      "heated_gas",      build_heated,          1.0,   1.0e2},
            {"heavy_elements",  "heavy_elements",  build_heavy,           2.0e6, 2.0e5},
            {"gas_density",     "gas_density",     build_gas_density,     5.0e4, 1.0e4},
            {"gas_temperature", "gas_temperature", build_gas_temperature, 1.0,   1.0e2},
            {"gas_metallicity", "gas_metallicity", build_gas_metallicity, 5.0e6, 5.0e5},
            {"gas_cold",        "gas_cold",        build_gas_cold,        5.0e4, 1.0e4},
            {"gas_warm",        "gas_warm",        build_gas_warm,        5.0e4, 1.0e4},
            {"gas_hot",         "gas_hot",         build_gas_hot,         5.0e4, 1.0e4},
        };
        for (const auto& vf : vol_fields) {
            if (skip_volume) break;
            bool want = false;
            for (const auto& f : requested_fields) if (f == vf.req) { want = true; break; }
            if (!want) continue;
            std::vector<float> w = vf.build(ps);
            if (w.empty()) {
                std::cout << "  skipping " << vf.tag
                          << ": required particle field missing" << std::endl;
                continue;
            }
            auto t0 = clk::now();
            std::vector<float> image = SphRayTracer::traceGasEmission(
                camera, ps, gas_bvh, w.data(), w.data(),
                vf.kappa_e, vf.kappa_a, samples_per_axis);
            double dt = std::chrono::duration<double>(clk::now() - t0).count();
            stat_report(image, 2, ("trace_emission " + vf.tag).c_str(), dt);
            if (!output_dir.empty()) {
                SphImageWriter::writeMulti(
                    output_dir + "/" + vf.tag + "_vol.h5", image,
                    camera.width(), camera.height(),
                    {"emission", "transmittance"},
                    cam_cfg, snapshot_path, region);
            }
        }

        bool want_vlos = false;
        for (const auto& f : requested_fields) if (f == "los_velocity") want_vlos = true;
        if (want_vlos && !ps.gas_vx.empty()) {
            Vec3 n = camera.forward();
            std::vector<float> vlos(ps.numGas());
            const float* vx = ps.gas_vx.data();
            const float* vy = ps.gas_vy.data();
            const float* vz = ps.gas_vz.data();
            #pragma omp parallel for schedule(static)
            for (size_t i = 0; i < ps.numGas(); ++i) {
                vlos[i] = static_cast<float>(vx[i] * n.x + vy[i] * n.y + vz[i] * n.z);
            }
            auto t0 = clk::now();
            std::vector<float> image = SphRayTracer::traceGasMassWeighted(
                camera, ps, gas_bvh, {vlos.data()});
            double dt = std::chrono::duration<double>(clk::now() - t0).count();
            stat_report(image, 1, "trace_los_velocity", dt);
            if (!output_dir.empty()) {
                SphImageWriter::write(output_dir + "/los_velocity.h5", image,
                                      camera.width(), camera.height(),
                                      "gas_vlos_mw", cam_cfg, snapshot_path, region);
            }
        }

        bool want_vel = false;
        for (const auto& f : requested_fields) if (f == "gas_velocity") want_vel = true;
        if (want_vel && !ps.gas_vx.empty()) {
            auto t0 = clk::now();
            std::vector<float> image = SphRayTracer::traceGasMassWeighted(
                camera, ps, gas_bvh,
                {ps.gas_vx.data(), ps.gas_vy.data(), ps.gas_vz.data()});
            double dt = std::chrono::duration<double>(clk::now() - t0).count();
            stat_report(image, 3, "trace_gas_velocity", dt);
            if (!output_dir.empty()) {
                SphImageWriter::writeMulti(
                    output_dir + "/gas_velocity.h5", image,
                    camera.width(), camera.height(),
                    {"gas_vx_mw", "gas_vy_mw", "gas_vz_mw"},
                    cam_cfg, snapshot_path, region);
            }
        }
      }   // frame loop
    }

    MPI_Barrier(MPI_COMM_WORLD);

    // Release external pointers before freeing the windows they point into.
    ps.gas_x.clear();       ps.gas_y.clear();       ps.gas_z.clear();
    ps.gas_h.clear();       ps.gas_mass.clear();    ps.gas_density.clear();
    ps.gas_temperature.clear();
    ps.gas_metallicity.clear();
    ps.gas_vx.clear(); ps.gas_vy.clear(); ps.gas_vz.clear();
    ps.gas_hii.clear();
    ps.dm_x.clear(); ps.dm_y.clear(); ps.dm_z.clear();
    ps.dm_h.clear(); ps.dm_mass.clear();
    gas_bvh.nodes.clear(); gas_bvh.perm.clear();
    dm_bvh.nodes.clear();  dm_bvh.perm.clear();

    for (MPI_Win& w : g_shm_wins) MPI_Win_free(&w);
    g_shm_wins.clear();
    if (leader_comm != MPI_COMM_NULL) MPI_Comm_free(&leader_comm);
    MPI_Comm_free(&node_comm);
    MPI_Finalize();
    return 0;
}
