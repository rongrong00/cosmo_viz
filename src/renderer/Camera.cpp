#include "renderer/Camera.h"
#include "common/Constants.h"
#include <cmath>
#include <limits>

Camera::Camera(const CameraConfig& config)
    : position_(config.position), look_at_(config.look_at), type_(config.type),
      fov_rad_(config.fov * Constants::PI / 180.0),
      ortho_width_(config.ortho_width),
      los_slab_(config.los_slab),
      width_(config.image_width), height_(config.image_height) {

    forward_ = (config.look_at - config.position).normalized();
    right_ = forward_.cross(config.up).normalized();
    up_ = right_.cross(forward_).normalized();
}

void Camera::slabTRange(const Ray& ray, double& t0, double& t1) const {
    if (los_slab_ <= 0.0) {
        t0 = -std::numeric_limits<double>::infinity();
        t1 =  std::numeric_limits<double>::infinity();
        return;
    }
    // Camera-space z of (ray.origin + t*ray.dir) relative to look_at:
    //   z(t) = (ray.origin - look_at) . forward + t * (ray.dir . forward)
    // Slab clips |z(t)| <= los_slab/2.
    double a = (look_at_ - ray.origin).dot(forward_);
    double b = ray.dir.dot(forward_);
    double half = los_slab_ * 0.5;
    if (std::fabs(b) < 1e-15) {
        // Ray parallel to slab plane; either entirely inside or outside
        if (std::fabs(a) <= half) {
            t0 = -std::numeric_limits<double>::infinity();
            t1 =  std::numeric_limits<double>::infinity();
        } else {
            t0 = 1.0; t1 = 0.0;  // empty
        }
        return;
    }
    double ta = (a - half) / b;
    double tb = (a + half) / b;
    if (ta > tb) std::swap(ta, tb);
    t0 = ta; t1 = tb;
}

Ray Camera::generateRay(int px, int py) const {
    return generateRay(px, py, 0.5, 0.5);
}

Ray Camera::generateRay(int px, int py, double dx, double dy) const {
    Ray ray;

    if (type_ == "orthographic") {
        double aspect = static_cast<double>(width_) / height_;
        double offset_x = (2.0 * (px + dx) / width_ - 1.0) * ortho_width_ / 2.0 * aspect;
        double offset_y = (1.0 - 2.0 * (py + dy) / height_) * ortho_width_ / 2.0;

        ray.origin = position_ + right_ * offset_x + up_ * offset_y;
        ray.dir = forward_;
    } else {
        // Perspective
        double aspect = static_cast<double>(width_) / height_;
        double tan_half_fov = std::tan(fov_rad_ / 2.0);
        double ndc_x = (2.0 * (px + dx) / width_ - 1.0) * aspect * tan_half_fov;
        double ndc_y = (1.0 - 2.0 * (py + dy) / height_) * tan_half_fov;

        Vec3 dir_cam(ndc_x, ndc_y, 1.0); // forward is +z in camera space
        ray.origin = position_;
        ray.dir = (right_ * dir_cam.x + up_ * dir_cam.y + forward_ * dir_cam.z).normalized();
    }

    return ray;
}
