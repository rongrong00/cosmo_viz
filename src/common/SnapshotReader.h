#pragma once
#include <string>
#include <vector>
#include <cstdint>
#include "common/Vec3.h"

struct GasParticle {
    Vec3 pos;
    float mass;
    float density;
    float hsml;
    float internal_energy;
    float metallicity;
    float hii_fraction;
    Vec3 velocity;
};

struct DMParticle {
    Vec3 pos;
    float mass;
    float hsml;  // set later via kNN
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
    static SnapshotHeader readHeader(const std::string& snapshot_path);
    static std::vector<GasParticle> readGasParticles(const std::string& subfile_path,
                                                      double boxsize);
    static std::vector<DMParticle> readDMParticles(const std::string& subfile_path,
                                                    double boxsize);
    // Reads PartType4 (stars) and packs into the DMParticle struct
    // (same fields: pos, mass, hsml). Masses are per-particle from the
    // snapshot; hsml is set to 0 (compute via kNN later).
    static std::vector<DMParticle> readStarsAsDM(const std::string& subfile_path,
                                                  double boxsize);
    static int getNumFiles(const std::string& snapshot_path);
    static std::string subfilePath(const std::string& snapshot_path, int file_index);
};
