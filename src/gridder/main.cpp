#include <mpi.h>
#include <iostream>
#include <string>
#include <sstream>
#include <iomanip>
#include <filesystem>
#include <algorithm>
#include <cstring>
#include <vector>
#include <map>
#include "common/Config.h"
#include "common/SnapshotReader.h"
#include "common/SmoothingLength.h"
#include "gridder/Grid.h"
#include "gridder/Depositor.h"

namespace fs = std::filesystem;

static void printUsage(const char* prog) {
    std::cerr << "Usage: " << prog
              << " --snapshot /path/to/snapdir_NNN/snap_NNN"
              << " --config grids.yaml"
              << " --output /path/to/output/"
              << std::endl;
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int world_rank, world_size;
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    // One sub-communicator per shared-memory domain (= one per node on
    // typical systems). Ranks on the same node will share a single grid
    // buffer allocated with MPI_Win_allocate_shared, so N ranks per node
    // do not each hold a full copy of the grid.
    MPI_Comm node_comm;
    MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, 0,
                        MPI_INFO_NULL, &node_comm);
    int node_rank, node_size;
    MPI_Comm_rank(node_comm, &node_rank);
    MPI_Comm_size(node_comm, &node_size);

    // Parse command line
    std::string snapshot_path, config_path, output_dir;
    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];
        if (arg == "--snapshot" && i + 1 < argc) snapshot_path = argv[++i];
        else if (arg == "--config" && i + 1 < argc) config_path = argv[++i];
        else if (arg == "--output" && i + 1 < argc) output_dir = argv[++i];
    }

    if (snapshot_path.empty() || config_path.empty() || output_dir.empty()) {
        if (world_rank == 0) printUsage(argv[0]);
        MPI_Finalize();
        return 1;
    }

    if (world_rank == 0) fs::create_directories(output_dir);
    MPI_Barrier(MPI_COMM_WORLD);

    SnapshotHeader header = SnapshotReader::readHeader(snapshot_path);
    if (world_rank == 0) {
        std::cout << "Snapshot: BoxSize=" << header.boxsize
                  << " z=" << header.redshift
                  << " NumFiles=" << header.num_files
                  << " NumGas=" << header.num_part_total[0]
                  << " NumDM=" << header.num_part_total[1]
                  << std::endl;
        std::cout << "MPI: " << world_size << " ranks across "
                  << (world_size / node_size) << " nodes, "
                  << node_size << " ranks/node (shared-mem grid)"
                  << std::endl;
    }

    GridConfig gcfg = parseGridConfig(config_path);
    if (world_rank == 0) {
        std::cout << "Grid: name=" << gcfg.name
                  << " center=(" << gcfg.center.x << "," << gcfg.center.y << "," << gcfg.center.z << ")"
                  << " size=(" << gcfg.size.x << "," << gcfg.size.y << "," << gcfg.size.z << ")"
                  << " shape=(" << gcfg.shape[0] << "," << gcfg.shape[1] << "," << gcfg.shape[2] << ")"
                  << " fields:";
        for (const auto& f : gcfg.fields) std::cout << " " << f;
        std::cout << std::endl;
    }

    bool need_gas = false, need_dm = false;
    for (const auto& f : gcfg.fields) {
        if (f == "gas_density" || f == "temperature" || f == "metallicity" ||
            f == "HII_density" || f == "gas_velocity") need_gas = true;
        if (f == "dm_density") need_dm = true;
    }

    // --- Allocate shared-memory buffers, one per field, on each node. ---
    const size_t total_cells = static_cast<size_t>(gcfg.shape[0])
                             * static_cast<size_t>(gcfg.shape[1])
                             * static_cast<size_t>(gcfg.shape[2]);
    auto field_list = Grid::allocatedFieldNames(gcfg.fields);

    std::vector<MPI_Win> wins;
    std::map<std::string, double*> buffers;
    for (const auto& name : field_list) {
        MPI_Aint bytes = (node_rank == 0) ?
            static_cast<MPI_Aint>(total_cells * sizeof(double)) : 0;
        double* ptr = nullptr;
        MPI_Win win;
        MPI_Win_allocate_shared(bytes, sizeof(double), MPI_INFO_NULL,
                                node_comm, &ptr, &win);
        if (node_rank != 0) {
            MPI_Aint qsize; int qdisp;
            MPI_Win_shared_query(win, 0, &qsize, &qdisp, &ptr);
        } else {
            std::memset(ptr, 0, total_cells * sizeof(double));
        }
        buffers[name] = ptr;
        wins.push_back(win);
    }
    MPI_Barrier(node_comm);

    Grid grid(gcfg.center, gcfg.size,
              gcfg.shape[0], gcfg.shape[1], gcfg.shape[2],
              gcfg.fields, buffers);

    // Round-robin subfiles across all world ranks. Within a node, multiple
    // ranks concurrently deposit into the same shared buffer (atomic adds
    // in Depositor keep it correct).
    for (int fi = world_rank; fi < header.num_files; fi += world_size) {
        std::string subfile = SnapshotReader::subfilePath(snapshot_path, fi);
        std::cout << "Rank " << world_rank << " reading subfile " << fi << std::endl;

        if (need_gas) {
            auto gas = SnapshotReader::readGasParticles(subfile, header.boxsize);
            std::cout << "  Rank " << world_rank << " subfile " << fi
                      << ": " << gas.size() << " gas particles" << std::endl;
            Depositor::depositGas(grid, gas, header.boxsize);
        }

        if (need_dm) {
            auto dm = SnapshotReader::readDMParticles(subfile, header.boxsize);
            std::cout << "  Rank " << world_rank << " subfile " << fi
                      << ": " << dm.size() << " DM particles" << std::endl;
            SmoothingLength::computeKNN(dm, header.boxsize);
            Depositor::depositDM(grid, dm, header.boxsize);
        }
    }

    MPI_Barrier(MPI_COMM_WORLD);

    // Reduce across *node leaders* only. Each node already holds a single
    // fully-summed buffer, so this scales as (num_nodes - 1) * grid_bytes.
    MPI_Comm leader_comm = MPI_COMM_NULL;
    int leader_color = (node_rank == 0) ? 0 : MPI_UNDEFINED;
    MPI_Comm_split(MPI_COMM_WORLD, leader_color, world_rank, &leader_comm);

    if (node_rank == 0) {
        int n_leaders = 0;
        MPI_Comm_size(leader_comm, &n_leaders);
        if (n_leaders > 1) {
            std::vector<double> recv;
            if (world_rank == 0) recv.resize(total_cells);
            for (const auto& name : field_list) {
                double* send = buffers[name];
                if (world_rank == 0) {
                    MPI_Reduce(send, recv.data(), total_cells,
                               MPI_DOUBLE, MPI_SUM, 0, leader_comm);
                    std::memcpy(send, recv.data(), total_cells * sizeof(double));
                } else {
                    MPI_Reduce(send, nullptr, total_cells,
                               MPI_DOUBLE, MPI_SUM, 0, leader_comm);
                }
            }
        }
    }

    if (leader_comm != MPI_COMM_NULL) MPI_Comm_free(&leader_comm);

    // Write on global rank 0.
    if (world_rank == 0) {
        grid.normalizeIntensiveFields();

        int snap_num = 0;
        std::string base = fs::path(snapshot_path).filename().string();
        auto pos = base.find('_');
        if (pos != std::string::npos) {
            try { snap_num = std::stoi(base.substr(pos + 1)); } catch (...) {}
        }

        std::ostringstream fname;
        fname << output_dir << "/grid_" << gcfg.name << "_snap"
              << std::setfill('0') << std::setw(3) << snap_num << ".hdf5";

        grid.writeHDF5(fname.str(), header.redshift, header.time, snap_num,
                       header.boxsize, header.hubble_param,
                       header.omega0, header.omega_lambda);
    }

    MPI_Barrier(MPI_COMM_WORLD);

    for (auto& w : wins) MPI_Win_free(&w);
    MPI_Comm_free(&node_comm);

    MPI_Finalize();
    return 0;
}
