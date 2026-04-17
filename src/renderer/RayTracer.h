#pragma once
#include "renderer/Camera.h"
#include "renderer/GridReader.h"
#include <vector>
#include <string>

class RayTracer {
public:
    // Column density: integral of field * ds
    static std::vector<float> traceColumnDensity(const Camera& camera,
                                                  const GridData& grid,
                                                  const std::string& field);

    // Mass-weighted projection: integral(rho * field * ds) / integral(rho * ds)
    static std::vector<float> traceMassWeighted(const Camera& camera,
                                                 const GridData& grid,
                                                 const std::string& field,
                                                 const std::string& weight_field);

    // LOS velocity: integral(rho * v_los * ds) / integral(rho * ds)
    static std::vector<float> traceLOSVelocity(const Camera& camera,
                                                const GridData& grid,
                                                const std::string& weight_field);

private:
    static void rayAABBIntersect(const Ray& ray, const Box3& bbox,
                                  double& tmin, double& tmax);

    // Generic single-ray DDA traversal that calls a per-cell callback.
    // If camera has a LOS slab, integration is clipped to the slab range.
    template <typename CellFunc>
    static void traceRayDDA(const Ray& ray, const Camera& camera,
                            const GridData& grid, CellFunc&& func);
};
