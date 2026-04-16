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

    // Parse camera and projection configs
    CameraConfig cam_cfg = parseCameraConfig(camera_path);
    auto projections = parseProjectionConfigs(camera_path);

    // Collect all fields needed
    std::vector<std::string> needed_fields;
    for (const auto& proj : projections) {
        needed_fields.push_back(proj.field);
        // Mass-weighted projections need a weight field (gas_density by default)
        if (proj.mode == "mass_weighted") {
            needed_fields.push_back("gas_density");
        }
        if (proj.mode == "los_velocity") {
            needed_fields.push_back("gas_density");
            needed_fields.push_back("gas_velocity_x");
            needed_fields.push_back("gas_velocity_y");
            needed_fields.push_back("gas_velocity_z");
        }
    }

    // Read grid with needed fields
    GridData grid = GridReader::read(grid_path, needed_fields);

    std::cout << "Camera: type=" << cam_cfg.type
              << " pos=(" << cam_cfg.position.x << "," << cam_cfg.position.y << "," << cam_cfg.position.z << ")"
              << " image=" << cam_cfg.image_width << "x" << cam_cfg.image_height
              << std::endl;

    Camera camera(cam_cfg);

    for (const auto& proj : projections) {
        std::cout << "Projection: field=" << proj.field << " mode=" << proj.mode << std::endl;

        std::vector<float> image;

        if (proj.mode == "column") {
            image = RayTracer::traceColumnDensity(camera, grid, proj.field);
        } else if (proj.mode == "mass_weighted") {
            image = RayTracer::traceMassWeighted(camera, grid, proj.field, "gas_density");
        } else if (proj.mode == "los_velocity") {
            image = RayTracer::traceLOSVelocity(camera, grid, "gas_density");
        } else {
            std::cerr << "Warning: unsupported projection mode '" << proj.mode
                      << "', skipping." << std::endl;
            continue;
        }

        std::string grid_name = fs::path(grid_path).stem().string();
        std::string out_file = output_dir + "/projection_" + grid_name + "_" + proj.field + ".hdf5";
        ImageWriter::write(out_file, image, cam_cfg.image_width, cam_cfg.image_height,
                           proj.field, cam_cfg, grid_path);
    }

    return 0;
}
