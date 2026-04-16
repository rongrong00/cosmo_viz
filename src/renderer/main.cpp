#include <iostream>
#include <string>
#include <filesystem>
#include "common/Config.h"
#include "renderer/Camera.h"
#include "renderer/GridReader.h"
#include "renderer/RayTracer.h"
#include "renderer/ImageWriter.h"

namespace fs = std::filesystem;

static void printUsage(const char* prog) {
    std::cerr << "Usage: " << prog
              << " --grid grid_file.hdf5"
              << " --camera camera.yaml"
              << " --output /path/to/output/"
              << std::endl;
}

int main(int argc, char** argv) {
    std::string grid_path, camera_path, output_dir;
    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];
        if (arg == "--grid" && i + 1 < argc) grid_path = argv[++i];
        else if (arg == "--camera" && i + 1 < argc) camera_path = argv[++i];
        else if (arg == "--output" && i + 1 < argc) output_dir = argv[++i];
    }

    if (grid_path.empty() || camera_path.empty() || output_dir.empty()) {
        printUsage(argv[0]);
        return 1;
    }

    fs::create_directories(output_dir);

    // Read grid
    GridData grid = GridReader::read(grid_path);

    // Parse camera config
    CameraConfig cam_cfg = parseCameraConfig(camera_path);
    auto projections = parseProjectionConfigs(camera_path);

    std::cout << "Camera: type=" << cam_cfg.type
              << " pos=(" << cam_cfg.position.x << "," << cam_cfg.position.y << "," << cam_cfg.position.z << ")"
              << " look_at=(" << cam_cfg.look_at.x << "," << cam_cfg.look_at.y << "," << cam_cfg.look_at.z << ")"
              << " ortho_width=" << cam_cfg.ortho_width
              << " image=" << cam_cfg.image_width << "x" << cam_cfg.image_height
              << std::endl;

    Camera camera(cam_cfg);

    // For each projection
    for (const auto& proj : projections) {
        std::cout << "Projection: field=" << proj.field << " mode=" << proj.mode << std::endl;

        std::vector<float> image;
        if (proj.mode == "column") {
            image = RayTracer::traceColumnDensity(camera, grid);
        } else {
            std::cerr << "Warning: unsupported projection mode '" << proj.mode
                      << "', skipping." << std::endl;
            continue;
        }

        // Build output filename
        std::string grid_name = fs::path(grid_path).stem().string();
        std::string out_file = output_dir + "/projection_" + grid_name + "_" + proj.field + ".hdf5";
        ImageWriter::write(out_file, image, cam_cfg.image_width, cam_cfg.image_height,
                           proj.field, cam_cfg, grid_path);
    }

    return 0;
}
