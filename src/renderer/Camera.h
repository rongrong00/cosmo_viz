#pragma once
#include "common/Vec3.h"
#include "common/Config.h"

struct Ray {
    Vec3 origin;
    Vec3 dir;  // normalized
};

class Camera {
public:
    Camera(const CameraConfig& config);

    Ray generateRay(int px, int py) const;
    // Subpixel-offset variant: dx,dy ∈ [0,1], with (0.5,0.5) reproducing the
    // pixel-center ray. Used by supersampling anti-aliasing.
    Ray generateRay(int px, int py, double dx, double dy) const;

    int width() const { return width_; }
    int height() const { return height_; }
    const Vec3& forward() const { return forward_; }

    double losSlab() const { return los_slab_; }
    // If los_slab > 0, returns [t0, t1] along the ray for the slab of thickness
    // los_slab_ centered on look_at along the camera forward axis. Returns
    // {-inf, +inf} if slab is disabled.
    void slabTRange(const Ray& ray, double& t0, double& t1) const;

private:
    Vec3 position_;
    Vec3 look_at_;
    Vec3 forward_;
    Vec3 right_;
    Vec3 up_;
    std::string type_;
    double fov_rad_;
    double ortho_width_;
    double los_slab_;
    int width_, height_;
};
