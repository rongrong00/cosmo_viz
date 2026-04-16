#include "gridder/SmoothingLength.h"
#include <cmath>
#include <algorithm>
#include <iostream>
#include <vector>
#include <numeric>

void SmoothingLength::computeKNN(std::vector<DMParticle>& particles, double boxsize) {
    if (particles.empty()) return;

    size_t N = particles.size();
    double half_box = boxsize * 0.5;

    // Estimate mean inter-particle spacing
    // For a uniform distribution: l = (V / N)^(1/3)
    double mean_spacing = std::cbrt(boxsize * boxsize * boxsize / static_cast<double>(N));

    // Cell size for linked list: ~2 * mean_spacing so each cell has ~8 particles on average
    double cell_size = 2.0 * mean_spacing;
    int ncells = std::max(1, static_cast<int>(std::ceil(boxsize / cell_size)));
    ncells = std::min(ncells, 512); // cap to avoid huge memory
    cell_size = boxsize / ncells;

    std::cout << "  kNN: " << N << " particles, ncells=" << ncells
              << " cell_size=" << cell_size << std::endl;

    // Build cell linked list
    // cell_start[c] = index of first particle in cell c, or -1
    // next[i] = index of next particle in same cell, or -1
    size_t total_cells = static_cast<size_t>(ncells) * ncells * ncells;
    std::vector<int> cell_start(total_cells, -1);
    std::vector<int> next(N, -1);

    auto cellIndex = [&](const Vec3& pos) -> size_t {
        int cx = std::clamp(static_cast<int>(pos.x / cell_size), 0, ncells - 1);
        int cy = std::clamp(static_cast<int>(pos.y / cell_size), 0, ncells - 1);
        int cz = std::clamp(static_cast<int>(pos.z / cell_size), 0, ncells - 1);
        return static_cast<size_t>(cz) * ncells * ncells + cy * ncells + cx;
    };

    for (size_t i = 0; i < N; i++) {
        size_t c = cellIndex(particles[i].pos);
        next[i] = cell_start[c];
        cell_start[c] = static_cast<int>(i);
    }

    // For each particle, find K nearest neighbors
    // Search expanding rings of cells until we have K neighbors
    #pragma omp parallel for schedule(dynamic, 256)
    for (size_t i = 0; i < N; i++) {
        const Vec3& pi = particles[i].pos;
        int cx = std::clamp(static_cast<int>(pi.x / cell_size), 0, ncells - 1);
        int cy = std::clamp(static_cast<int>(pi.y / cell_size), 0, ncells - 1);
        int cz = std::clamp(static_cast<int>(pi.z / cell_size), 0, ncells - 1);

        // Keep a sorted list of K nearest distances squared
        std::vector<double> dist2_heap(K_NEIGHBORS, 1e30);

        // Search expanding shell of cells
        for (int ring = 0; ring <= ncells; ring++) {
            // Check if the nearest point of this ring is farther than our K-th neighbor
            if (ring > 1) {
                double ring_dist = (ring - 1) * cell_size;
                if (ring_dist * ring_dist > dist2_heap[0]) break; // heap[0] is the max
            }

            for (int dz = -ring; dz <= ring; dz++) {
                for (int dy = -ring; dy <= ring; dy++) {
                    for (int dx = -ring; dx <= ring; dx++) {
                        // Only process the shell (not interior, already done)
                        if (ring > 0 && std::abs(dx) < ring && std::abs(dy) < ring && std::abs(dz) < ring)
                            continue;

                        int nx = (cx + dx % ncells + ncells) % ncells;
                        int ny = (cy + dy % ncells + ncells) % ncells;
                        int nz = (cz + dz % ncells + ncells) % ncells;
                        size_t nc = static_cast<size_t>(nz) * ncells * ncells + ny * ncells + nx;

                        int j = cell_start[nc];
                        while (j >= 0) {
                            if (static_cast<size_t>(j) != i) {
                                // Periodic distance
                                double ddx = particles[j].pos.x - pi.x;
                                double ddy = particles[j].pos.y - pi.y;
                                double ddz = particles[j].pos.z - pi.z;
                                if (ddx > half_box) ddx -= boxsize;
                                if (ddx < -half_box) ddx += boxsize;
                                if (ddy > half_box) ddy -= boxsize;
                                if (ddy < -half_box) ddy += boxsize;
                                if (ddz > half_box) ddz -= boxsize;
                                if (ddz < -half_box) ddz += boxsize;
                                double d2 = ddx * ddx + ddy * ddy + ddz * ddz;

                                if (d2 < dist2_heap[0]) {
                                    // Replace the largest element (max-heap root)
                                    std::pop_heap(dist2_heap.begin(), dist2_heap.end());
                                    dist2_heap.back() = d2;
                                    std::push_heap(dist2_heap.begin(), dist2_heap.end());
                                }
                            }
                            j = next[j];
                        }
                    }
                }
            }
        }

        // hsml = distance to K-th nearest neighbor
        particles[i].hsml = static_cast<float>(std::sqrt(dist2_heap[0]));
    }

    // Print stats
    float min_h = 1e30, max_h = 0, sum_h = 0;
    for (const auto& p : particles) {
        min_h = std::min(min_h, p.hsml);
        max_h = std::max(max_h, p.hsml);
        sum_h += p.hsml;
    }
    std::cout << "  kNN hsml: min=" << min_h << " max=" << max_h
              << " mean=" << sum_h / N << std::endl;
}
