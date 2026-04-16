#include "gridder/Depositor.h"
#include "common/Kernel.h"
#include <cmath>
#include <algorithm>
#include <iostream>

void Depositor::depositGas(Grid& grid, const std::vector<GasParticle>& particles,
                            double boxsize) {
    int N = grid.resolution();
    double cell_size = grid.cellSize();
    const Box3& bbox = grid.bbox();
    double half_box = boxsize * 0.5;
    auto& density = grid.densityBuf();

    int deposited = 0;

    for (const auto& p : particles) {
        double h = p.hsml;
        double support = 2.0 * h;

        // Periodic-wrap particle position relative to grid center
        Vec3 rel;
        for (int d = 0; d < 3; d++) {
            double dx = p.pos[d] - grid.center()[d];
            if (dx > half_box)  dx -= boxsize;
            if (dx < -half_box) dx += boxsize;
            rel[d] = dx;
        }

        // Check if particle (with kernel support) overlaps the grid
        double half_side = grid.side() * 0.5;
        bool skip = false;
        for (int d = 0; d < 3; d++) {
            if (std::fabs(rel[d]) > half_side + support) { skip = true; break; }
        }
        if (skip) continue;

        // Particle position in grid-local coordinates (origin at bbox.lo)
        Vec3 pos_local;
        pos_local.x = rel.x + half_side;
        pos_local.y = rel.y + half_side;
        pos_local.z = rel.z + half_side;

        // Range of grid cells to touch
        int ix_lo = std::max(0, (int)std::floor((pos_local.x - support) / cell_size));
        int ix_hi = std::min(N - 1, (int)std::floor((pos_local.x + support) / cell_size));
        int iy_lo = std::max(0, (int)std::floor((pos_local.y - support) / cell_size));
        int iy_hi = std::min(N - 1, (int)std::floor((pos_local.y + support) / cell_size));
        int iz_lo = std::max(0, (int)std::floor((pos_local.z - support) / cell_size));
        int iz_hi = std::min(N - 1, (int)std::floor((pos_local.z + support) / cell_size));

        if (ix_lo > ix_hi || iy_lo > iy_hi || iz_lo > iz_hi) continue;

        deposited++;

        for (int iz = iz_lo; iz <= iz_hi; iz++) {
            double cz = (iz + 0.5) * cell_size;
            double dz = cz - pos_local.z;
            for (int iy = iy_lo; iy <= iy_hi; iy++) {
                double cy = (iy + 0.5) * cell_size;
                double dy = cy - pos_local.y;
                for (int ix = ix_lo; ix <= ix_hi; ix++) {
                    double cx = (ix + 0.5) * cell_size;
                    double dx = cx - pos_local.x;

                    double r = std::sqrt(dx * dx + dy * dy + dz * dz);
                    if (r >= support) continue;

                    double w = Kernel::W(r, h);
                    density[grid.index(ix, iy, iz)] += p.mass * w;
                }
            }
        }
    }

    std::cout << "  Deposited " << deposited << " / " << particles.size()
              << " particles onto grid" << std::endl;
}
