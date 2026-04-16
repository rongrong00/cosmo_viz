#pragma once
#include <string>
#include <vector>
#include "common/Vec3.h"

struct GridConfig {
    std::string name;
    Vec3 center;       // in code units (ckpc/h)
    double side;       // in code units (ckpc/h)
    int resolution;
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
};

struct ProjectionConfig {
    std::string field;
    std::string mode;  // "column", "mass_weighted", etc.
};

GridConfig parseGridConfig(const std::string& filename);
CameraConfig parseCameraConfig(const std::string& filename);
std::vector<ProjectionConfig> parseProjectionConfigs(const std::string& filename);
