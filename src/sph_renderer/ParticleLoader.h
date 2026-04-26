#pragma once

#include "common/Config.h"
#include "sph_renderer/ParticleStore.h"
#include <mpi.h>
#include <set>
#include <string>

// Reads SPH particles from a multi-file snapshot into a ParticleStore. Does
// periodic unwrap relative to the region center and culls particles to
// [center ± size/2 + margin].
//
// This Phase-A implementation is single-rank (no MPI distribution). Phase D
// upgrades it with per-rank subfile distribution and shared-memory buffers.
class ParticleLoader {
public:
    // gas_optional_fields: any of "temperature", "metallicity", "velocity",
    // "hii". Empty set loads only the baseline (position, h, mass, density).
    //
    // load_dm: whether to also load PartType1 and assign kNN smoothing lengths.
    //
    // cull_margin_override: extra margin beyond 2*h_max (code units). If < 0,
    // uses region.margin.
    // If load_stars=true (and load_dm=false), PartType4 is loaded into the
    // dm_* arrays and kNN uses (dm_knn_k, dm_h_max). Setting both load_dm
    // and load_stars is not supported.
    static ParticleStore load(const std::string& snapshot_path,
                              const RegionConfig& region,
                              const std::set<std::string>& gas_optional_fields,
                              bool load_dm,
                              double cull_margin_override = -1.0,
                              bool load_stars = false,
                              int dm_knn_k = 32,
                              double dm_h_max = 0.0,
                              double dm_h_min = 0.0);

    // Distributed load: each rank reads subfiles where `f % size == rank`,
    // culls locally, then MPI_Allgatherv's the survivors so every rank in
    // `world_comm` ends up with the same consolidated ParticleStore. DM kNN
    // runs on world rank 0 only and is broadcast.
    //
    // Use this instead of `load()` when multiple MPI ranks are available to
    // parallelize the subfile I/O; HDF5 thread-safety prevents OpenMP from
    // doing the same job within a single rank.
    static ParticleStore loadMPI(const std::string& snapshot_path,
                                 const RegionConfig& region,
                                 const std::set<std::string>& gas_optional_fields,
                                 bool load_dm,
                                 MPI_Comm world_comm,
                                 double cull_margin_override = -1.0,
                                 bool load_stars = false,
                                 int dm_knn_k = 32,
                                 double dm_h_max = 0.0,
                                 double dm_h_min = 0.0);
};
