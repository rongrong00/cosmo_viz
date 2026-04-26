#pragma once
#include "gridder/Grid.h"
#include "common/SnapshotReader.h"
#include <vector>

class Depositor {
public:
    // Deposit gas particles: density, temperature, metallicity, HII, velocity, mass_weight
    static void depositGas(Grid& grid, const std::vector<GasParticle>& particles,
                           double boxsize);

    // Deposit DM particles: dm_density only
    static void depositDM(Grid& grid, const std::vector<DMParticle>& particles,
                          double boxsize);
};
