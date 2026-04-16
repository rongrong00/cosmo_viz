#include "renderer/Camera.h"
#include "common/Constants.h"
#include <cmath>

Camera::Camera(const CameraConfig& config)
    : position_(config.position), type_(config.type),
      fov_rad_(config.fov * Constants::PI / 180.0),
      ortho_width_(config.ortho_width),
      width_(config.image_width), height_(config.image_height) {

    forward_ = (config.look_at - config.position).normalized();
    right_ = forward_.cross(config.up).normalized();
    up_ = right_.cross(forward_).normalized();
}

Ray Camera::generateRay(int px, int py) const {
    Ray ray;

    if (type_ == "orthographic") {
        double aspect = static_cast<double>(width_) / height_;
        double offset_x = (2.0 * (px + 0.5) / width_ - 1.0) * ortho_width_ / 2.0 * aspect;
        double offset_y = (1.0 - 2.0 * (py + 0.5) / height_) * ortho_width_ / 2.0;

        ray.origin = position_ + right_ * offset_x + up_ * offset_y;
        ray.dir = forward_;
    } else {
        // Perspective
        double aspect = static_cast<double>(width_) / height_;
        double tan_half_fov = std::tan(fov_rad_ / 2.0);
        double ndc_x = (2.0 * (px + 0.5) / width_ - 1.0) * aspect * tan_half_fov;
        double ndc_y = (1.0 - 2.0 * (py + 0.5) / height_) * tan_half_fov;

        Vec3 dir_cam(ndc_x, ndc_y, 1.0); // forward is +z in camera space
        ray.origin = position_;
        ray.dir = (right_ * dir_cam.x + up_ * dir_cam.y + forward_ * dir_cam.z).normalized();
    }

    return ray;
}
