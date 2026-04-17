#include "renderer/RayTracer.h"
#include <cmath>
#include <algorithm>
#include <iostream>
#include <limits>

void RayTracer::rayAABBIntersect(const Ray& ray, const Box3& bbox,
                                  double& tmin, double& tmax) {
    tmin = -std::numeric_limits<double>::infinity();
    tmax = std::numeric_limits<double>::infinity();

    for (int d = 0; d < 3; d++) {
        double dir_d = ray.dir[d];
        double orig_d = ray.origin[d];

        if (std::fabs(dir_d) < 1e-15) {
            if (orig_d < bbox.lo[d] || orig_d > bbox.hi[d]) {
                tmin = 1.0; tmax = 0.0; return;
            }
        } else {
            double inv_d = 1.0 / dir_d;
            double t1 = (bbox.lo[d] - orig_d) * inv_d;
            double t2 = (bbox.hi[d] - orig_d) * inv_d;
            if (t1 > t2) std::swap(t1, t2);
            tmin = std::max(tmin, t1);
            tmax = std::min(tmax, t2);
            if (tmin > tmax) return;
        }
    }
}

template <typename CellFunc>
void RayTracer::traceRayDDA(const Ray& ray, const Camera& camera,
                             const GridData& grid, CellFunc&& func) {
    double tmin, tmax;
    rayAABBIntersect(ray, grid.bbox, tmin, tmax);
    if (tmin >= tmax) return;

    // Optional LOS slab clipping
    double t_slab0, t_slab1;
    camera.slabTRange(ray, t_slab0, t_slab1);
    tmin = std::max(tmin, t_slab0);
    tmax = std::min(tmax, t_slab1);
    if (tmin >= tmax) return;

    if (tmin < 0.0) tmin = 0.0;

    int Nx = grid.nx, Ny = grid.ny, Nz = grid.nz;
    double csx = grid.cell_size.x, csy = grid.cell_size.y, csz = grid.cell_size.z;
    double cs_min = std::min({csx, csy, csz});
    const Vec3& lo = grid.bbox.lo;

    Vec3 entry = ray.origin + ray.dir * (tmin + 1e-10 * cs_min);

    int ix = std::clamp((int)std::floor((entry.x - lo.x) / csx), 0, Nx - 1);
    int iy = std::clamp((int)std::floor((entry.y - lo.y) / csy), 0, Ny - 1);
    int iz = std::clamp((int)std::floor((entry.z - lo.z) / csz), 0, Nz - 1);

    int step_x = (ray.dir.x >= 0) ? 1 : -1;
    int step_y = (ray.dir.y >= 0) ? 1 : -1;
    int step_z = (ray.dir.z >= 0) ? 1 : -1;

    double tDelta_x = (std::fabs(ray.dir.x) > 1e-15) ? csx / std::fabs(ray.dir.x) : 1e30;
    double tDelta_y = (std::fabs(ray.dir.y) > 1e-15) ? csy / std::fabs(ray.dir.y) : 1e30;
    double tDelta_z = (std::fabs(ray.dir.z) > 1e-15) ? csz / std::fabs(ray.dir.z) : 1e30;

    double next_x = lo.x + ((ray.dir.x >= 0) ? (ix + 1) : ix) * csx;
    double next_y = lo.y + ((ray.dir.y >= 0) ? (iy + 1) : iy) * csy;
    double next_z = lo.z + ((ray.dir.z >= 0) ? (iz + 1) : iz) * csz;

    double tMax_x = (std::fabs(ray.dir.x) > 1e-15) ? (next_x - ray.origin.x) / ray.dir.x : 1e30;
    double tMax_y = (std::fabs(ray.dir.y) > 1e-15) ? (next_y - ray.origin.y) / ray.dir.y : 1e30;
    double tMax_z = (std::fabs(ray.dir.z) > 1e-15) ? (next_z - ray.origin.z) / ray.dir.z : 1e30;

    double t_current = tmin;

    while (ix >= 0 && ix < Nx && iy >= 0 && iy < Ny && iz >= 0 && iz < Nz && t_current < tmax) {
        double t_next = std::min({tMax_x, tMax_y, tMax_z, tmax});
        double ds = t_next - t_current;
        if (ds < 0) ds = 0;

        size_t idx = (static_cast<size_t>(iz) * Ny + iy) * Nx + ix;
        func(idx, ds);

        t_current = t_next;
        if (tMax_x <= tMax_y && tMax_x <= tMax_z) {
            ix += step_x; tMax_x += tDelta_x;
        } else if (tMax_y <= tMax_z) {
            iy += step_y; tMax_y += tDelta_y;
        } else {
            iz += step_z; tMax_z += tDelta_z;
        }
    }
}

std::vector<float> RayTracer::traceColumnDensity(const Camera& camera,
                                                   const GridData& grid,
                                                   const std::string& field) {
    int W = camera.width();
    int H = camera.height();
    std::vector<float> image(H * W, 0.0f);
    const auto& fdata = grid.getField(field);

    std::cout << "Ray tracing column density: " << field << " (" << W << "x" << H << ")"
              << " los_slab=" << camera.losSlab() << std::endl;

    #pragma omp parallel for schedule(dynamic, 4)
    for (int py = 0; py < H; py++) {
        for (int px = 0; px < W; px++) {
            Ray ray = camera.generateRay(px, py);
            double accum = 0.0;
            traceRayDDA(ray, camera, grid, [&](size_t idx, double ds) {
                accum += fdata[idx] * ds;
            });
            image[py * W + px] = static_cast<float>(accum);
        }
        if (py % 200 == 0) {
            #pragma omp critical
            std::cout << "  Row " << py << " / " << H << std::endl;
        }
    }
    return image;
}

std::vector<float> RayTracer::traceMassWeighted(const Camera& camera,
                                                  const GridData& grid,
                                                  const std::string& field,
                                                  const std::string& weight_field) {
    int W = camera.width();
    int H = camera.height();
    std::vector<float> image(H * W, 0.0f);
    const auto& fdata = grid.getField(field);
    const auto& wdata = grid.getField(weight_field);

    std::cout << "Ray tracing mass-weighted: " << field << " weighted by " << weight_field
              << " (" << W << "x" << H << ")"
              << " los_slab=" << camera.losSlab() << std::endl;

    #pragma omp parallel for schedule(dynamic, 4)
    for (int py = 0; py < H; py++) {
        for (int px = 0; px < W; px++) {
            Ray ray = camera.generateRay(px, py);
            double num = 0.0, den = 0.0;
            traceRayDDA(ray, camera, grid, [&](size_t idx, double ds) {
                double w = wdata[idx] * ds;
                num += fdata[idx] * w;
                den += w;
            });
            image[py * W + px] = (den > 0.0) ? static_cast<float>(num / den) : 0.0f;
        }
        if (py % 200 == 0) {
            #pragma omp critical
            std::cout << "  Row " << py << " / " << H << std::endl;
        }
    }
    return image;
}

std::vector<float> RayTracer::traceLOSVelocity(const Camera& camera,
                                                 const GridData& grid,
                                                 const std::string& weight_field) {
    int W = camera.width();
    int H = camera.height();
    std::vector<float> image(H * W, 0.0f);
    const auto& vx = grid.getField("gas_velocity_x");
    const auto& vy = grid.getField("gas_velocity_y");
    const auto& vz = grid.getField("gas_velocity_z");
    const auto& wdata = grid.getField(weight_field);
    Vec3 n = camera.forward(); // LOS direction

    std::cout << "Ray tracing LOS velocity (" << W << "x" << H << ")"
              << " los_slab=" << camera.losSlab() << std::endl;

    #pragma omp parallel for schedule(dynamic, 4)
    for (int py = 0; py < H; py++) {
        for (int px = 0; px < W; px++) {
            Ray ray = camera.generateRay(px, py);
            // For perspective, use the actual ray direction; for ortho, use camera forward
            Vec3 los = ray.dir;
            double num = 0.0, den = 0.0;
            traceRayDDA(ray, camera, grid, [&](size_t idx, double ds) {
                double v_los = vx[idx] * los.x + vy[idx] * los.y + vz[idx] * los.z;
                double w = wdata[idx] * ds;
                num += v_los * w;
                den += w;
            });
            image[py * W + px] = (den > 0.0) ? static_cast<float>(num / den) : 0.0f;
        }
        if (py % 200 == 0) {
            #pragma omp critical
            std::cout << "  Row " << py << " / " << H << std::endl;
        }
    }
    return image;
}
