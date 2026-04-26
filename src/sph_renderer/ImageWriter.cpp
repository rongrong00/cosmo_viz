#include "sph_renderer/ImageWriter.h"
#include "common/HDF5IO.h"
#include <iostream>

void SphImageWriter::write(const std::string& filename,
                           const std::vector<float>& image,
                           int width, int height,
                           const std::string& field_name,
                           const CameraConfig& cam_config,
                           const std::string& snapshot_path,
                           const RegionConfig& region) {
    HDF5Writer writer(filename);

    writer.createGroup("Header");
    writer.writeAttrString("Header", "pipeline", "sph_renderer");
    writer.writeAttrString("Header", "snapshot_path", snapshot_path);
    writer.writeAttrString("Header", "region_name", region.name);
    writer.writeAttrDoubleArray("Header", "region_center",
                                {region.center.x, region.center.y, region.center.z});
    writer.writeAttrDoubleArray("Header", "region_size",
                                {region.size.x, region.size.y, region.size.z});

    writer.writeAttrString("Header", "camera_type", cam_config.type);
    writer.writeAttrDoubleArray("Header", "camera_position",
                                {cam_config.position.x, cam_config.position.y, cam_config.position.z});
    writer.writeAttrDoubleArray("Header", "camera_look_at",
                                {cam_config.look_at.x, cam_config.look_at.y, cam_config.look_at.z});
    writer.writeAttrDoubleArray("Header", "camera_up",
                                {cam_config.up.x, cam_config.up.y, cam_config.up.z});
    writer.writeAttrDouble("Header", "ortho_width", cam_config.ortho_width);
    writer.writeAttrDouble("Header", "fov",         cam_config.fov);
    writer.writeAttrDouble("Header", "los_slab",    cam_config.los_slab);
    writer.writeAttrInt("Header",    "image_width",  width);
    writer.writeAttrInt("Header",    "image_height", height);

    writer.writeDataset2D(field_name, image, height, width);

    std::cout << "Wrote SPH projection to " << filename << std::endl;
}

void SphImageWriter::writeMulti(const std::string& filename,
                                const std::vector<float>& interleaved,
                                int width, int height,
                                const std::vector<std::string>& channel_names,
                                const CameraConfig& cam_config,
                                const std::string& snapshot_path,
                                const RegionConfig& region) {
    const int C = static_cast<int>(channel_names.size());
    HDF5Writer writer(filename);

    writer.createGroup("Header");
    writer.writeAttrString("Header", "pipeline", "sph_renderer");
    writer.writeAttrString("Header", "snapshot_path", snapshot_path);
    writer.writeAttrString("Header", "region_name", region.name);
    writer.writeAttrDoubleArray("Header", "region_center",
                                {region.center.x, region.center.y, region.center.z});
    writer.writeAttrDoubleArray("Header", "region_size",
                                {region.size.x, region.size.y, region.size.z});

    writer.writeAttrString("Header", "camera_type", cam_config.type);
    writer.writeAttrDoubleArray("Header", "camera_position",
                                {cam_config.position.x, cam_config.position.y, cam_config.position.z});
    writer.writeAttrDoubleArray("Header", "camera_look_at",
                                {cam_config.look_at.x, cam_config.look_at.y, cam_config.look_at.z});
    writer.writeAttrDoubleArray("Header", "camera_up",
                                {cam_config.up.x, cam_config.up.y, cam_config.up.z});
    writer.writeAttrDouble("Header", "ortho_width", cam_config.ortho_width);
    writer.writeAttrDouble("Header", "fov",         cam_config.fov);
    writer.writeAttrDouble("Header", "los_slab",    cam_config.los_slab);
    writer.writeAttrInt("Header",    "image_width",  width);
    writer.writeAttrInt("Header",    "image_height", height);

    const size_t npix = static_cast<size_t>(width) * height;
    std::vector<float> plane(npix);
    for (int c = 0; c < C; ++c) {
        for (size_t i = 0; i < npix; ++i) plane[i] = interleaved[i * C + c];
        writer.writeDataset2D(channel_names[c], plane, height, width);
    }

    std::cout << "Wrote SPH multi-channel projection to " << filename
              << " (" << C << " channels)" << std::endl;
}
