#pragma once
#include "renderer/Camera.h"
#include "renderer/GridReader.h"
#include <vector>

class RayTracer {
public:
    // Trace all rays for column density projection.
    // Returns a 2D image (height x width) as flat vector.
    static std::vector<float> traceColumnDensity(const Camera& camera,
                                                  const GridData& grid);

private:
    // DDA ray-AABB intersection: returns (tmin, tmax), or tmin > tmax if no hit
    static void rayAABBIntersect(const Ray& ray, const Box3& bbox,
                                  double& tmin, double& tmax);

    // Trace a single ray through the grid, accumulating column density
    static float traceRay(const Ray& ray, const GridData& grid);
};
