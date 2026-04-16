#pragma once
#include <string>
#include <vector>
#include <cstdint>
#include "common/Vec3.h"

struct GasParticle {
    Vec3 pos;        // in code units (ckpc/h)
    float mass;
    float density;
    float hsml;      // smoothing length, computed from Volume = Mass/Density
};

struct SnapshotHeader {
    double boxsize;
    double redshift;
    double time;     // scale factor
    double hubble_param;
    double omega0;
    double omega_lambda;
    int num_files;
    std::vector<uint64_t> num_part_total;  // 6 elements
};

class SnapshotReader {
public:
    // Read header from first subfile
    static SnapshotHeader readHeader(const std::string& snapshot_path);

    // Read gas particles from a single subfile
    static std::vector<GasParticle> readGasParticles(const std::string& subfile_path,
                                                      double boxsize);

    // Get the number of subfiles
    static int getNumFiles(const std::string& snapshot_path);

    // Build subfile path: /path/to/snapdir_NNN/snap_NNN.K.hdf5
    static std::string subfilePath(const std::string& snapshot_path, int file_index);
};
