#include "renderer/ImageWriter.h"
#include "common/HDF5IO.h"
#include <iostream>

void ImageWriter::write(const std::string& filename,
                         const std::vector<float>& image,
                         int width, int height,
                         const std::string& field_name,
                         const CameraConfig& cam_config,
                         const std::string& grid_file) {
    HDF5Writer writer(filename);

    writer.createGroup("Header");
    writer.writeAttrString("Header", "grid_file", grid_file);
    writer.writeAttrString("Header", "camera_type", cam_config.type);
    writer.writeAttrDoubleArray("Header", "camera_position",
                                {cam_config.position.x, cam_config.position.y, cam_config.position.z});
    writer.writeAttrDoubleArray("Header", "camera_look_at",
                                {cam_config.look_at.x, cam_config.look_at.y, cam_config.look_at.z});
    writer.writeAttrDoubleArray("Header", "camera_up",
                                {cam_config.up.x, cam_config.up.y, cam_config.up.z});
    writer.writeAttrDouble("Header", "ortho_width", cam_config.ortho_width);
    writer.writeAttrInt("Header", "image_width", width);
    writer.writeAttrInt("Header", "image_height", height);

    writer.writeDataset2D(field_name, image, height, width);

    std::cout << "Wrote projection to " << filename << std::endl;
}
