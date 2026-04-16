#pragma once
#include "gridder/Grid.h"
#include "gridder/SnapshotReader.h"
#include <vector>

class Depositor {
public:
    // Deposit gas particles onto the grid using SPH kernel.
    // Handles periodic boundary wrapping.
    static void depositGas(Grid& grid, const std::vector<GasParticle>& particles,
                           double boxsize);
};
