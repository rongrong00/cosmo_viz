#pragma once
#include <string>
#include <vector>
#include "common/Vec3.h"

struct GridConfig {
    std::string name;
    Vec3 center;       // in code units (ckpc/h)
    // Anisotropic size / resolution. For a cube, all three are equal.
    // If the yaml provides scalar `side`/`resolution`, they are replicated.
    Vec3 size;         // sx, sy, sz in code units (ckpc/h)
    int shape[3];      // Nx, Ny, Nz
    std::vector<std::string> fields;
};

struct CameraConfig {
    std::string type;  // "orthographic" or "perspective"
    Vec3 position;
    Vec3 look_at;
    Vec3 up;
    double fov;           // degrees, perspective only
    double ortho_width;   // code units, orthographic only
    int image_width;
    int image_height;
    // Optional LOS slab: if > 0, clip ray integration to a slab of this
    // thickness (in code units) centered on look_at along the camera forward
    // axis. 0 (default) disables and uses the full grid depth.
    double los_slab = 0.0;
};

struct ProjectionConfig {
    std::string field;
    std::string mode;  // "column", "mass_weighted", etc.
};

// Region selection for the direct SPH ray tracer. No resolution — the SPH
// renderer operates on particles directly.
struct RegionConfig {
    std::string name;
    Vec3 center;                            // code units (ckpc/h)
    Vec3 size;                              // full extents per axis
    double radius = 0.0;                    // if > 0, use spherical cull of this radius (code units); overrides size
    std::vector<std::string> particle_types; // subset of {"gas", "dm"}; default: ["gas"]
    double margin = 0.0;                    // extra halo (code units) beyond 2*h_max auto cull
};

GridConfig parseGridConfig(const std::string& filename);
CameraConfig parseCameraConfig(const std::string& filename);
std::vector<ProjectionConfig> parseProjectionConfigs(const std::string& filename);
RegionConfig parseRegionConfig(const std::string& filename);
