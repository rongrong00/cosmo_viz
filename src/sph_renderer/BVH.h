#pragma once

#include "common/Box3.h"
#include "common/ShmArray.h"
#include "common/Vec3.h"
#include <cstdint>

// Binary BVH over per-particle AABBs built from positions and smoothing
// lengths (AABB = [pos ± 2h]). Top-down binned SAH, flat DFS storage.
//
// Node layout:
//   - Leaf  (count > 0): left_or_first = first index into `perm`;
//                        perm[first .. first+count) gives the original
//                        particle indices into the parallel position
//                        arrays.
//   - Internal (count == 0): left child is implicit at node_idx + 1;
//                            right child index is stored in left_or_first.
//
// Traversal is stack-based; see SphRayTracer.

struct alignas(32) BVHNode {
    float    bmin[3];
    uint32_t left_or_first;
    float    bmax[3];
    uint32_t count;
};
static_assert(sizeof(BVHNode) == 32, "BVHNode must be 32 bytes");

class BVH {
public:
    static constexpr int LEAF_SIZE = 16;
    static constexpr int N_BINS    = 16;
    static constexpr int MAX_DEPTH = 64;

    ShmArray<BVHNode>  nodes;
    ShmArray<uint32_t> perm;

    // Build from position + smoothing-length SoA arrays. AABB of primitive i
    // is [pos - 2*h[i], pos + 2*h[i]]. All arrays must have length N.
    void buildFromSpheres(const float* x, const float* y, const float* z,
                          const float* h, size_t N);

    // Root AABB (for outer ray-AABB test). Undefined if nodes empty.
    Box3 rootBbox() const;

private:
    void buildRecursive(uint32_t node_idx,
                        uint32_t first, uint32_t count, int depth,
                        const float* x, const float* y, const float* z,
                        const float* h);
};
