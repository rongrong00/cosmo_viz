#include "gridder/Depositor.h"
#include "common/Kernel.h"
#include "common/Constants.h"
#include <cmath>
#include <algorithm>
#include <iostream>

static double internalEnergyToTemperature(float u, float xe) {
    constexpr double gamma = 5.0 / 3.0;
    constexpr double m_p = 1.6726e-24;
    constexpr double k_B = 1.3807e-16;
    constexpr double X_H = 0.76;
    double mu = 4.0 / (1.0 + 3.0 * X_H + 4.0 * X_H * xe);
    return (gamma - 1.0) * u * 1e10 * mu * m_p / k_B;
}

void Depositor::depositGas(Grid& grid, const std::vector<GasParticle>& particles,
                            double boxsize) {
    const int Nx = grid.nx(), Ny = grid.ny(), Nz = grid.nz();
    const Vec3 cs = grid.cellSize();
    const Vec3 half_size(grid.size().x * 0.5, grid.size().y * 0.5, grid.size().z * 0.5);
    const Vec3 gc = grid.center();
    const double half_box = boxsize * 0.5;

    const bool do_density = grid.hasField("gas_density");
    const bool do_temp    = grid.hasField("temperature");
    const bool do_metal   = grid.hasField("metallicity");
    const bool do_hii     = grid.hasField("HII_density");
    const bool do_vel     = grid.hasField("gas_velocity_x");

    double* density_buf = do_density ? grid.fieldData("gas_density") : nullptr;
    double* temp_buf    = do_temp    ? grid.fieldData("temperature") : nullptr;
    double* metal_buf   = do_metal   ? grid.fieldData("metallicity") : nullptr;
    double* hii_buf     = do_hii     ? grid.fieldData("HII_density") : nullptr;
    double* vx_buf      = do_vel     ? grid.fieldData("gas_velocity_x") : nullptr;
    double* vy_buf      = do_vel     ? grid.fieldData("gas_velocity_y") : nullptr;
    double* vz_buf      = do_vel     ? grid.fieldData("gas_velocity_z") : nullptr;
    double* mw_buf      = grid.fieldData("mass_weight");

    const double min_cs = std::min({cs.x, cs.y, cs.z});
    const double h_min = 0.5 * min_cs;

    long long deposited = 0;
    const long long Np = static_cast<long long>(particles.size());

    #pragma omp parallel for schedule(dynamic, 256) reduction(+:deposited)
    for (long long pi = 0; pi < Np; pi++) {
        const auto& p = particles[pi];
        double h = std::max(static_cast<double>(p.hsml), h_min);
        double support = 2.0 * h;

        // Periodic wrap relative to grid center
        double rx = p.pos[0] - gc.x;
        double ry = p.pos[1] - gc.y;
        double rz = p.pos[2] - gc.z;
        if (rx >  half_box) rx -= boxsize;
        if (rx < -half_box) rx += boxsize;
        if (ry >  half_box) ry -= boxsize;
        if (ry < -half_box) ry += boxsize;
        if (rz >  half_box) rz -= boxsize;
        if (rz < -half_box) rz += boxsize;

        if (std::fabs(rx) > half_size.x + support ||
            std::fabs(ry) > half_size.y + support ||
            std::fabs(rz) > half_size.z + support) continue;

        const double lx = rx + half_size.x;
        const double ly = ry + half_size.y;
        const double lz = rz + half_size.z;

        int ix_lo = std::max(0,      (int)std::floor((lx - support) / cs.x));
        int ix_hi = std::min(Nx - 1, (int)std::floor((lx + support) / cs.x));
        int iy_lo = std::max(0,      (int)std::floor((ly - support) / cs.y));
        int iy_hi = std::min(Ny - 1, (int)std::floor((ly + support) / cs.y));
        int iz_lo = std::max(0,      (int)std::floor((lz - support) / cs.z));
        int iz_hi = std::min(Nz - 1, (int)std::floor((lz + support) / cs.z));

        if (ix_lo > ix_hi || iy_lo > iy_hi || iz_lo > iz_hi) continue;

        deposited++;

        const double T = do_temp ? internalEnergyToTemperature(p.internal_energy, p.hii_fraction) : 0.0;

        for (int iz = iz_lo; iz <= iz_hi; iz++) {
            double dz = (iz + 0.5) * cs.z - lz;
            for (int iy = iy_lo; iy <= iy_hi; iy++) {
                double dy = (iy + 0.5) * cs.y - ly;
                for (int ix = ix_lo; ix <= ix_hi; ix++) {
                    double dx = (ix + 0.5) * cs.x - lx;
                    double r = std::sqrt(dx * dx + dy * dy + dz * dz);
                    if (r >= support) continue;

                    double w = Kernel::W(r, h);
                    double mw = p.mass * w;
                    size_t idx = grid.index(ix, iy, iz);

                    // Atomic updates so threads (and ranks sharing the
                    // buffer via MPI_Win_allocate_shared) don't race.
                    if (density_buf) {
                        #pragma omp atomic
                        density_buf[idx] += mw;
                    }
                    if (temp_buf) {
                        #pragma omp atomic
                        temp_buf[idx] += mw * T;
                    }
                    if (metal_buf) {
                        #pragma omp atomic
                        metal_buf[idx] += mw * p.metallicity;
                    }
                    if (hii_buf) {
                        #pragma omp atomic
                        hii_buf[idx] += mw * p.hii_fraction;
                    }
                    if (vx_buf) {
                        #pragma omp atomic
                        vx_buf[idx] += mw * p.velocity.x;
                        #pragma omp atomic
                        vy_buf[idx] += mw * p.velocity.y;
                        #pragma omp atomic
                        vz_buf[idx] += mw * p.velocity.z;
                    }
                    #pragma omp atomic
                    mw_buf[idx] += mw;
                }
            }
        }
    }

    std::cout << "  Gas: deposited " << deposited << " / " << particles.size() << std::endl;
}

void Depositor::depositDM(Grid& grid, const std::vector<DMParticle>& particles,
                           double boxsize) {
    if (!grid.hasField("dm_density")) return;

    const int Nx = grid.nx(), Ny = grid.ny(), Nz = grid.nz();
    const Vec3 cs = grid.cellSize();
    const Vec3 half_size(grid.size().x * 0.5, grid.size().y * 0.5, grid.size().z * 0.5);
    const Vec3 gc = grid.center();
    const double half_box = boxsize * 0.5;
    double* dm_buf = grid.fieldData("dm_density");

    const double min_cs = std::min({cs.x, cs.y, cs.z});
    const double h_min = 0.5 * min_cs;

    long long deposited = 0;
    const long long Np = static_cast<long long>(particles.size());

    #pragma omp parallel for schedule(dynamic, 256) reduction(+:deposited)
    for (long long pi = 0; pi < Np; pi++) {
        const auto& p = particles[pi];
        if (p.hsml <= 0.0f) continue;
        double h = std::max(static_cast<double>(p.hsml), h_min);
        double support = 2.0 * h;

        double rx = p.pos[0] - gc.x;
        double ry = p.pos[1] - gc.y;
        double rz = p.pos[2] - gc.z;
        if (rx >  half_box) rx -= boxsize;
        if (rx < -half_box) rx += boxsize;
        if (ry >  half_box) ry -= boxsize;
        if (ry < -half_box) ry += boxsize;
        if (rz >  half_box) rz -= boxsize;
        if (rz < -half_box) rz += boxsize;

        if (std::fabs(rx) > half_size.x + support ||
            std::fabs(ry) > half_size.y + support ||
            std::fabs(rz) > half_size.z + support) continue;

        const double lx = rx + half_size.x;
        const double ly = ry + half_size.y;
        const double lz = rz + half_size.z;

        int ix_lo = std::max(0,      (int)std::floor((lx - support) / cs.x));
        int ix_hi = std::min(Nx - 1, (int)std::floor((lx + support) / cs.x));
        int iy_lo = std::max(0,      (int)std::floor((ly - support) / cs.y));
        int iy_hi = std::min(Ny - 1, (int)std::floor((ly + support) / cs.y));
        int iz_lo = std::max(0,      (int)std::floor((lz - support) / cs.z));
        int iz_hi = std::min(Nz - 1, (int)std::floor((lz + support) / cs.z));

        if (ix_lo > ix_hi || iy_lo > iy_hi || iz_lo > iz_hi) continue;

        deposited++;

        for (int iz = iz_lo; iz <= iz_hi; iz++) {
            double dz = (iz + 0.5) * cs.z - lz;
            for (int iy = iy_lo; iy <= iy_hi; iy++) {
                double dy = (iy + 0.5) * cs.y - ly;
                for (int ix = ix_lo; ix <= ix_hi; ix++) {
                    double dx = (ix + 0.5) * cs.x - lx;
                    double r = std::sqrt(dx * dx + dy * dy + dz * dz);
                    if (r >= support) continue;

                    size_t idx = grid.index(ix, iy, iz);
                    double add = p.mass * Kernel::W(r, h);
                    #pragma omp atomic
                    dm_buf[idx] += add;
                }
            }
        }
    }

    std::cout << "  DM: deposited " << deposited << " / " << particles.size() << std::endl;
}
