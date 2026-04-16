#pragma once
#include "gridder/SnapshotReader.h"
#include <vector>

// Compute adaptive smoothing lengths for DM particles using k-nearest neighbors.
// Uses a cell-linked list for efficient neighbor search.
class SmoothingLength {
public:
    static constexpr int K_NEIGHBORS = 32;

    // Compute hsml for all particles in the vector (modifies in place).
    // Only considers particles within the given bounding box + margin.
    static void computeKNN(std::vector<DMParticle>& particles, double boxsize);
};
