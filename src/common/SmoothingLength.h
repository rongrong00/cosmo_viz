#pragma once
#include "common/SnapshotReader.h"
#include <vector>

// Compute adaptive smoothing lengths for DM particles using k-nearest neighbors.
// Uses a kd-tree (median-split, no periodic wrap) for the neighbor search.
class SmoothingLength {
public:
    static constexpr int K_NEIGHBORS = 32;

    // Compute hsml for all particles in the vector (modifies in place).
    // Default signature uses K_NEIGHBORS and no cap on hsml.
    static void computeKNN(std::vector<DMParticle>& particles, double boxsize);

    // Configurable variant: caller picks k and an optional hard cap on hsml
    // (ckpc/h). h_max<=0 disables the cap. h_min floors the hsml (used for
    // stars at the simulation's gravitational softening so 1/h^2 peaks don't
    // diverge below the resolved scale).
    static void computeKNN(std::vector<DMParticle>& particles,
                           double boxsize, int k, double h_max,
                           double h_min = 0.0);
};
