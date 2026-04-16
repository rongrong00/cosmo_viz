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
        double lo_d = bbox.lo[d];
        double hi_d = bbox.hi[d];

        if (std::fabs(dir_d) < 1e-15) {
            // Ray parallel to slab
            if (orig_d < lo_d || orig_d > hi_d) {
                tmin = 1.0;
                tmax = 0.0; // no intersection
                return;
            }
        } else {
            double inv_d = 1.0 / dir_d;
            double t1 = (lo_d - orig_d) * inv_d;
            double t2 = (hi_d - orig_d) * inv_d;
            if (t1 > t2) std::swap(t1, t2);
            tmin = std::max(tmin, t1);
            tmax = std::min(tmax, t2);
            if (tmin > tmax) return;
        }
    }
}

float RayTracer::traceRay(const Ray& ray, const GridData& grid) {
    double tmin, tmax;
    rayAABBIntersect(ray, grid.bbox, tmin, tmax);
    if (tmin >= tmax) return 0.0f;

    // Clamp tmin to 0 (don't go behind camera)
    if (tmin < 0.0) tmin = 0.0;

    int N = grid.resolution;
    double cs = grid.cell_size;
    const Vec3& lo = grid.bbox.lo;

    // DDA traversal (Amanatides-Woo)
    // Entry point
    Vec3 entry = ray.origin + ray.dir * (tmin + 1e-10 * cs);

    // Current voxel
    int ix = std::clamp((int)std::floor((entry.x - lo.x) / cs), 0, N - 1);
    int iy = std::clamp((int)std::floor((entry.y - lo.y) / cs), 0, N - 1);
    int iz = std::clamp((int)std::floor((entry.z - lo.z) / cs), 0, N - 1);

    // Step direction
    int step_x = (ray.dir.x >= 0) ? 1 : -1;
    int step_y = (ray.dir.y >= 0) ? 1 : -1;
    int step_z = (ray.dir.z >= 0) ? 1 : -1;

    // tDelta: distance along ray to cross one voxel in each axis
    double tDelta_x = (std::fabs(ray.dir.x) > 1e-15) ? cs / std::fabs(ray.dir.x) : 1e30;
    double tDelta_y = (std::fabs(ray.dir.y) > 1e-15) ? cs / std::fabs(ray.dir.y) : 1e30;
    double tDelta_z = (std::fabs(ray.dir.z) > 1e-15) ? cs / std::fabs(ray.dir.z) : 1e30;

    // tMax: distance to next voxel boundary
    double next_x = lo.x + ((ray.dir.x >= 0) ? (ix + 1) : ix) * cs;
    double next_y = lo.y + ((ray.dir.y >= 0) ? (iy + 1) : iy) * cs;
    double next_z = lo.z + ((ray.dir.z >= 0) ? (iz + 1) : iz) * cs;

    double tMax_x = (std::fabs(ray.dir.x) > 1e-15) ? (next_x - ray.origin.x) / ray.dir.x : 1e30;
    double tMax_y = (std::fabs(ray.dir.y) > 1e-15) ? (next_y - ray.origin.y) / ray.dir.y : 1e30;
    double tMax_z = (std::fabs(ray.dir.z) > 1e-15) ? (next_z - ray.origin.z) / ray.dir.z : 1e30;

    double t_current = tmin;
    double accum = 0.0;

    // Traverse
    while (ix >= 0 && ix < N && iy >= 0 && iy < N && iz >= 0 && iz < N && t_current < tmax) {
        // Distance through this cell
        double t_next = std::min({tMax_x, tMax_y, tMax_z, tmax});
        double ds = t_next - t_current;
        if (ds < 0) ds = 0;

        // Accumulate column density: integral of rho * ds
        size_t idx = static_cast<size_t>(iz) * N * N + static_cast<size_t>(iy) * N + ix;
        accum += grid.gas_density[idx] * ds;

        // Advance
        t_current = t_next;
        if (tMax_x <= tMax_y && tMax_x <= tMax_z) {
            ix += step_x;
            tMax_x += tDelta_x;
        } else if (tMax_y <= tMax_z) {
            iy += step_y;
            tMax_y += tDelta_y;
        } else {
            iz += step_z;
            tMax_z += tDelta_z;
        }
    }

    return static_cast<float>(accum);
}

std::vector<float> RayTracer::traceColumnDensity(const Camera& camera,
                                                   const GridData& grid) {
    int W = camera.width();
    int H = camera.height();
    std::vector<float> image(H * W, 0.0f);

    std::cout << "Ray tracing " << W << "x" << H << " image..." << std::endl;

    #pragma omp parallel for schedule(dynamic, 4)
    for (int py = 0; py < H; py++) {
        for (int px = 0; px < W; px++) {
            Ray ray = camera.generateRay(px, py);
            image[py * W + px] = traceRay(ray, grid);
        }
        if (py % 100 == 0) {
            #pragma omp critical
            std::cout << "  Row " << py << " / " << H << std::endl;
        }
    }

    std::cout << "Ray tracing complete." << std::endl;
    return image;
}
