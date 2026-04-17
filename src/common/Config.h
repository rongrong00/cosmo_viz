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

GridConfig parseGridConfig(const std::string& filename);
CameraConfig parseCameraConfig(const std::string& filename);
std::vector<ProjectionConfig> parseProjectionConfigs(const std::string& filename);
