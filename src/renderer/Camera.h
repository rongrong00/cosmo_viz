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

    int width() const { return width_; }
    int height() const { return height_; }
    const Vec3& forward() const { return forward_; }

private:
    Vec3 position_;
    Vec3 forward_;
    Vec3 right_;
    Vec3 up_;
    std::string type_;
    double fov_rad_;
    double ortho_width_;
    int width_, height_;
};
