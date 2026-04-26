#pragma once

#include "common/Box3.h"
#include "common/ShmArray.h"
#include "common/Vec3.h"
#include <string>

// Struct-of-arrays storage for particles within a region, after periodic
// unwrapping relative to the region center. Positions are in code units
// (ckpc/h) and are guaranteed to sit in a contiguous span around the region
// center — no further periodic handling is needed downstream.
//
// A field array is empty() when that field was not requested at load time.
// Each ShmArray may be either locally owning (std::vector-backed) or backed
// by an MPI shared-memory window (set up after load by main.cpp).
struct ParticleStore {
    // --- Gas (PartType0) ---
    ShmArray<float> gas_x, gas_y, gas_z;      // always populated
    ShmArray<float> gas_h;                    // native SPH smoothing length
    ShmArray<float> gas_mass, gas_density;    // always populated
    ShmArray<float> gas_temperature;          // Kelvin, derived from u + xe
    ShmArray<float> gas_metallicity;          // mass fraction
    ShmArray<float> gas_vx, gas_vy, gas_vz;
    ShmArray<float> gas_hii;                  // HII fraction

    // --- DM (PartType1) ---
    ShmArray<float> dm_x, dm_y, dm_z;
    ShmArray<float> dm_h;                     // kNN-assigned
    ShmArray<float> dm_mass;

    // Region metadata (from RegionConfig, not re-derived).
    Vec3 region_center{0, 0, 0};
    Vec3 region_size{0, 0, 0};

    // Tight AABBs around the loaded centers (ignores kernel halo).
    Box3 bbox_gas, bbox_dm;

    // Max smoothing length across gas and DM. Used by ray-tracer AABB
    // margins.
    float h_max_gas = 0.0f;
    float h_max_dm  = 0.0f;

    size_t numGas() const { return gas_x.size(); }
    size_t numDM()  const { return dm_x.size(); }

    // Bounding box expanded by 2*h_max per type — this is the region a ray
    // must intersect to possibly touch a particle.
    Box3 bboxGasExpanded() const;
    Box3 bboxDMExpanded()  const;
};
