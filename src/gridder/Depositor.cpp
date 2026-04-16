#include "gridder/Depositor.h"
#include "common/Kernel.h"
#include "common/Constants.h"
#include <cmath>
#include <algorithm>
#include <iostream>

// Convert InternalEnergy to Temperature (K)
// T = (gamma - 1) * u * mu * m_p / k_B
// For fully ionized primordial gas: mu ≈ 0.588
// gamma = 5/3
static double internalEnergyToTemperature(float u, float xe) {
    constexpr double gamma = 5.0 / 3.0;
    constexpr double m_p = 1.6726e-24;     // proton mass in g
    constexpr double k_B = 1.3807e-16;     // Boltzmann in erg/K
    // Mean molecular weight: mu = 4 / (1 + 3*X_H + 4*X_H*xe)
    // For primordial X_H = 0.76
    constexpr double X_H = 0.76;
    double mu = 4.0 / (1.0 + 3.0 * X_H + 4.0 * X_H * xe);
    // u is in (km/s)^2 = 1e10 cm^2/s^2
    return (gamma - 1.0) * u * 1e10 * mu * m_p / k_B;
}

void Depositor::depositGas(Grid& grid, const std::vector<GasParticle>& particles,
                            double boxsize) {
    int N = grid.resolution();
    double cell_size = grid.cellSize();
    double half_box = boxsize * 0.5;
    double half_side = grid.side() * 0.5;

    bool do_density    = grid.hasField("gas_density");
    bool do_temp       = grid.hasField("temperature");
    bool do_metal      = grid.hasField("metallicity");
    bool do_hii        = grid.hasField("HII_density");
    bool do_vel_x      = grid.hasField("gas_velocity_x");

    auto* density_buf  = do_density ? grid.fieldData("gas_density") : nullptr;
    auto* temp_buf     = do_temp    ? grid.fieldData("temperature") : nullptr;
    auto* metal_buf    = do_metal   ? grid.fieldData("metallicity") : nullptr;
    auto* hii_buf      = do_hii    ? grid.fieldData("HII_density") : nullptr;
    auto* vel_x_buf    = do_vel_x  ? grid.fieldData("gas_velocity_x") : nullptr;
    auto* vel_y_buf    = do_vel_x  ? grid.fieldData("gas_velocity_y") : nullptr;
    auto* vel_z_buf    = do_vel_x  ? grid.fieldData("gas_velocity_z") : nullptr;
    auto* mw_buf       = grid.fieldData("mass_weight");

    int deposited = 0;

    for (const auto& p : particles) {
        double h = p.hsml;
        double support = 2.0 * h;

        // Periodic-wrap relative to grid center
        Vec3 rel;
        for (int d = 0; d < 3; d++) {
            double dx = p.pos[d] - grid.center()[d];
            if (dx > half_box)  dx -= boxsize;
            if (dx < -half_box) dx += boxsize;
            rel[d] = dx;
        }

        bool skip = false;
        for (int d = 0; d < 3; d++) {
            if (std::fabs(rel[d]) > half_side + support) { skip = true; break; }
        }
        if (skip) continue;

        // Position in grid-local coordinates
        Vec3 pos_local;
        pos_local.x = rel.x + half_side;
        pos_local.y = rel.y + half_side;
        pos_local.z = rel.z + half_side;

        // Cell range
        int ix_lo = std::max(0, (int)std::floor((pos_local.x - support) / cell_size));
        int ix_hi = std::min(N - 1, (int)std::floor((pos_local.x + support) / cell_size));
        int iy_lo = std::max(0, (int)std::floor((pos_local.y - support) / cell_size));
        int iy_hi = std::min(N - 1, (int)std::floor((pos_local.y + support) / cell_size));
        int iz_lo = std::max(0, (int)std::floor((pos_local.z - support) / cell_size));
        int iz_hi = std::min(N - 1, (int)std::floor((pos_local.z + support) / cell_size));

        if (ix_lo > ix_hi || iy_lo > iy_hi || iz_lo > iz_hi) continue;

        deposited++;

        // Precompute temperature for this particle
        // Use HII_Fraction as electron abundance proxy
        double T = do_temp ? internalEnergyToTemperature(p.internal_energy, p.hii_fraction) : 0.0;

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
                    double mw = p.mass * w;
                    size_t idx = grid.index(ix, iy, iz);

                    if (density_buf)  density_buf[idx] += mw;
                    if (temp_buf)     temp_buf[idx]    += mw * T;
                    if (metal_buf)    metal_buf[idx]   += mw * p.metallicity;
                    if (hii_buf)      hii_buf[idx]     += mw * p.hii_fraction;
                    if (vel_x_buf) {
                        vel_x_buf[idx] += mw * p.velocity.x;
                        vel_y_buf[idx] += mw * p.velocity.y;
                        vel_z_buf[idx] += mw * p.velocity.z;
                    }
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

    int N = grid.resolution();
    double cell_size = grid.cellSize();
    double half_box = boxsize * 0.5;
    double half_side = grid.side() * 0.5;
    auto* dm_buf = grid.fieldData("dm_density");

    int deposited = 0;

    for (const auto& p : particles) {
        if (p.hsml <= 0.0f) continue;
        double h = p.hsml;
        double support = 2.0 * h;

        Vec3 rel;
        for (int d = 0; d < 3; d++) {
            double dx = p.pos[d] - grid.center()[d];
            if (dx > half_box)  dx -= boxsize;
            if (dx < -half_box) dx += boxsize;
            rel[d] = dx;
        }

        bool skip = false;
        for (int d = 0; d < 3; d++) {
            if (std::fabs(rel[d]) > half_side + support) { skip = true; break; }
        }
        if (skip) continue;

        Vec3 pos_local;
        pos_local.x = rel.x + half_side;
        pos_local.y = rel.y + half_side;
        pos_local.z = rel.z + half_side;

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

                    dm_buf[grid.index(ix, iy, iz)] += p.mass * Kernel::W(r, h);
                }
            }
        }
    }

    std::cout << "  DM: deposited " << deposited << " / " << particles.size() << std::endl;
}
