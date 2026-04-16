#include <mpi.h>
#include <iostream>
#include <string>
#include <sstream>
#include <iomanip>
#include <filesystem>
#include "common/Config.h"
#include "gridder/SnapshotReader.h"
#include "gridder/Grid.h"
#include "gridder/Depositor.h"
#include "gridder/SmoothingLength.h"

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

    int rank, nranks;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &nranks);

    // Parse command line
    std::string snapshot_path, config_path, output_dir;
    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];
        if (arg == "--snapshot" && i + 1 < argc) snapshot_path = argv[++i];
        else if (arg == "--config" && i + 1 < argc) config_path = argv[++i];
        else if (arg == "--output" && i + 1 < argc) output_dir = argv[++i];
    }

    if (snapshot_path.empty() || config_path.empty() || output_dir.empty()) {
        if (rank == 0) printUsage(argv[0]);
        MPI_Finalize();
        return 1;
    }

    if (rank == 0) fs::create_directories(output_dir);
    MPI_Barrier(MPI_COMM_WORLD);

    // Read header
    SnapshotHeader header = SnapshotReader::readHeader(snapshot_path);
    if (rank == 0) {
        std::cout << "Snapshot: BoxSize=" << header.boxsize
                  << " z=" << header.redshift
                  << " NumFiles=" << header.num_files
                  << " NumGas=" << header.num_part_total[0]
                  << " NumDM=" << header.num_part_total[1]
                  << std::endl;
    }

    // Parse grid config
    GridConfig gcfg = parseGridConfig(config_path);
    if (rank == 0) {
        std::cout << "Grid: name=" << gcfg.name
                  << " center=(" << gcfg.center.x << "," << gcfg.center.y << "," << gcfg.center.z << ")"
                  << " side=" << gcfg.side
                  << " res=" << gcfg.resolution
                  << " fields:";
        for (const auto& f : gcfg.fields) std::cout << " " << f;
        std::cout << std::endl;
    }

    // Check which particle types we need
    bool need_gas = false, need_dm = false;
    for (const auto& f : gcfg.fields) {
        if (f == "gas_density" || f == "temperature" || f == "metallicity" ||
            f == "HII_density" || f == "gas_velocity") {
            need_gas = true;
        }
        if (f == "dm_density") {
            need_dm = true;
        }
    }

    // Allocate grid
    Grid grid(gcfg.center, gcfg.side, gcfg.resolution, gcfg.fields);

    // Process subfiles round-robin
    for (int fi = rank; fi < header.num_files; fi += nranks) {
        std::string subfile = SnapshotReader::subfilePath(snapshot_path, fi);
        std::cout << "Rank " << rank << " reading subfile " << fi << std::endl;

        if (need_gas) {
            auto gas = SnapshotReader::readGasParticles(subfile, header.boxsize);
            std::cout << "  Rank " << rank << " subfile " << fi
                      << ": " << gas.size() << " gas particles" << std::endl;
            Depositor::depositGas(grid, gas, header.boxsize);
        }

        if (need_dm) {
            auto dm = SnapshotReader::readDMParticles(subfile, header.boxsize);
            std::cout << "  Rank " << rank << " subfile " << fi
                      << ": " << dm.size() << " DM particles" << std::endl;
            // Compute smoothing lengths
            SmoothingLength::computeKNN(dm, header.boxsize);
            Depositor::depositDM(grid, dm, header.boxsize);
        }
    }

    // MPI_Reduce all field buffers to rank 0
    size_t grid_size = grid.totalCells();
    if (nranks > 1) {
        // Gather all field names (including mass_weight and velocity components)
        std::vector<std::string> all_fields;
        for (const auto& f : gcfg.fields) {
            if (f == "gas_velocity") {
                all_fields.push_back("gas_velocity_x");
                all_fields.push_back("gas_velocity_y");
                all_fields.push_back("gas_velocity_z");
            } else {
                all_fields.push_back(f);
            }
        }
        all_fields.push_back("mass_weight");

        for (const auto& fname : all_fields) {
            if (!grid.hasField(fname)) continue;
            if (rank == 0) {
                std::vector<double> global(grid_size, 0.0);
                MPI_Reduce(grid.fieldData(fname), global.data(), grid_size,
                           MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
                grid.field(fname) = std::move(global);
            } else {
                MPI_Reduce(grid.fieldData(fname), nullptr, grid_size,
                           MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
            }
        }
    }

    // Normalize and write on rank 0
    if (rank == 0) {
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

    MPI_Finalize();
    return 0;
}
