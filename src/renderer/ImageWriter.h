#pragma once
#include <string>
#include <vector>
#include "common/Vec3.h"
#include "common/Config.h"

class ImageWriter {
public:
    static void write(const std::string& filename,
                      const std::vector<float>& image,
                      int width, int height,
                      const std::string& field_name,
                      const CameraConfig& cam_config,
                      const std::string& grid_file);
};
