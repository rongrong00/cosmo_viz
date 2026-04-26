#include "sph_renderer/BVH.h"

#include <algorithm>
#include <cfloat>
#include <cmath>
#include <numeric>
#include <vector>

#ifdef _OPENMP
#include <omp.h>
#endif

// Nodes with at least this many primitives parallelize their inner loops.
// Below the threshold serial loops win (OpenMP team-up overhead dominates).
static constexpr uint32_t PAR_THRESHOLD = 8192;

namespace {

inline float surfaceArea(const float bmin[3], const float bmax[3]) {
    float ex = bmax[0] - bmin[0];
    float ey = bmax[1] - bmin[1];
    float ez = bmax[2] - bmin[2];
    if (ex < 0 || ey < 0 || ez < 0) return 0.0f;
    return 2.0f * (ex * ey + ey * ez + ez * ex);
}

struct Bin {
    float    bmin[3] = { FLT_MAX,  FLT_MAX,  FLT_MAX};
    float    bmax[3] = {-FLT_MAX, -FLT_MAX, -FLT_MAX};
    uint32_t n = 0;
    void expand(const float lo[3], const float hi[3]) {
        for (int d = 0; d < 3; ++d) {
            if (lo[d] < bmin[d]) bmin[d] = lo[d];
            if (hi[d] > bmax[d]) bmax[d] = hi[d];
        }
        ++n;
    }
    void merge(const Bin& o) {
        for (int d = 0; d < 3; ++d) {
            if (o.bmin[d] < bmin[d]) bmin[d] = o.bmin[d];
            if (o.bmax[d] > bmax[d]) bmax[d] = o.bmax[d];
        }
        n += o.n;
    }
};

} // namespace

Box3 BVH::rootBbox() const {
    if (nodes.empty()) return Box3();
    const auto& r = nodes[0];
    return {Vec3(r.bmin[0], r.bmin[1], r.bmin[2]),
            Vec3(r.bmax[0], r.bmax[1], r.bmax[2])};
}

void BVH::buildFromSpheres(const float* x, const float* y, const float* z,
                           const float* h, size_t N) {
    nodes.clear();
    perm.resize(N);
    std::iota(perm.begin(), perm.end(), 0u);

    if (N == 0) return;

    // Reserve a safe upper bound: perfect binary tree with ceil(N/LEAF_SIZE)
    // leaves has ~2*ceil(N/LEAF_SIZE)-1 nodes; round up generously.
    nodes.reserve(4 + 2 * ((N + LEAF_SIZE - 1) / LEAF_SIZE));
    nodes.emplace_back();   // root
    buildRecursive(0, 0, static_cast<uint32_t>(N), 0, x, y, z, h);
}

void BVH::buildRecursive(uint32_t node_idx,
                         uint32_t first, uint32_t count, int depth,
                         const float* x, const float* y, const float* z,
                         const float* h) {
    // --- Compute full AABB + centroid AABB over [first, first+count) ---
    float bmin[3] = { FLT_MAX,  FLT_MAX,  FLT_MAX};
    float bmax[3] = {-FLT_MAX, -FLT_MAX, -FLT_MAX};
    float cmin[3] = { FLT_MAX,  FLT_MAX,  FLT_MAX};
    float cmax[3] = {-FLT_MAX, -FLT_MAX, -FLT_MAX};
    if (count >= PAR_THRESHOLD) {
        #pragma omp parallel for schedule(static) \
            reduction(min:bmin[:3], cmin[:3]) reduction(max:bmax[:3], cmax[:3])
        for (int64_t i = (int64_t)first; i < (int64_t)(first + count); ++i) {
            uint32_t p = perm[i];
            float hp = h[p] * 2.0f;
            float cx = x[p], cy = y[p], cz = z[p];
            if (cx - hp < bmin[0]) bmin[0] = cx - hp;
            if (cy - hp < bmin[1]) bmin[1] = cy - hp;
            if (cz - hp < bmin[2]) bmin[2] = cz - hp;
            if (cx + hp > bmax[0]) bmax[0] = cx + hp;
            if (cy + hp > bmax[1]) bmax[1] = cy + hp;
            if (cz + hp > bmax[2]) bmax[2] = cz + hp;
            if (cx < cmin[0]) cmin[0] = cx;
            if (cy < cmin[1]) cmin[1] = cy;
            if (cz < cmin[2]) cmin[2] = cz;
            if (cx > cmax[0]) cmax[0] = cx;
            if (cy > cmax[1]) cmax[1] = cy;
            if (cz > cmax[2]) cmax[2] = cz;
        }
    } else {
        for (uint32_t i = first; i < first + count; ++i) {
            uint32_t p = perm[i];
            float hp = h[p] * 2.0f;
            float cx = x[p], cy = y[p], cz = z[p];
            if (cx - hp < bmin[0]) bmin[0] = cx - hp;
            if (cy - hp < bmin[1]) bmin[1] = cy - hp;
            if (cz - hp < bmin[2]) bmin[2] = cz - hp;
            if (cx + hp > bmax[0]) bmax[0] = cx + hp;
            if (cy + hp > bmax[1]) bmax[1] = cy + hp;
            if (cz + hp > bmax[2]) bmax[2] = cz + hp;
            if (cx < cmin[0]) cmin[0] = cx;
            if (cy < cmin[1]) cmin[1] = cy;
            if (cz < cmin[2]) cmin[2] = cz;
            if (cx > cmax[0]) cmax[0] = cx;
            if (cy > cmax[1]) cmax[1] = cy;
            if (cz > cmax[2]) cmax[2] = cz;
        }
    }
    for (int d = 0; d < 3; ++d) {
        nodes[node_idx].bmin[d] = bmin[d];
        nodes[node_idx].bmax[d] = bmax[d];
    }

    auto makeLeaf = [&]() {
        nodes[node_idx].count = count;
        nodes[node_idx].left_or_first = first;
    };

    if (count <= static_cast<uint32_t>(LEAF_SIZE) || depth >= MAX_DEPTH) {
        makeLeaf();
        return;
    }

    // --- Pick split axis = longest centroid extent ---
    float cext[3] = {cmax[0] - cmin[0], cmax[1] - cmin[1], cmax[2] - cmin[2]};
    int axis = 0;
    if (cext[1] > cext[axis]) axis = 1;
    if (cext[2] > cext[axis]) axis = 2;
    if (cext[axis] < 1e-12f) { makeLeaf(); return; }

    float axis_lo = cmin[axis];
    float inv_extent = static_cast<float>(N_BINS) / cext[axis];

    // --- Bin primitives (thread-local bins, merge at end) ---
    Bin bins[N_BINS];
    if (count >= PAR_THRESHOLD) {
        #pragma omp parallel
        {
            Bin local[N_BINS];
            #pragma omp for schedule(static) nowait
            for (int64_t i = (int64_t)first; i < (int64_t)(first + count); ++i) {
                uint32_t p = perm[i];
                float cv  = (axis == 0) ? x[p] : (axis == 1) ? y[p] : z[p];
                int   bi  = static_cast<int>((cv - axis_lo) * inv_extent);
                if (bi < 0) bi = 0;
                if (bi >= N_BINS) bi = N_BINS - 1;
                float hp = h[p] * 2.0f;
                float lo[3] = {x[p] - hp, y[p] - hp, z[p] - hp};
                float hi[3] = {x[p] + hp, y[p] + hp, z[p] + hp};
                local[bi].expand(lo, hi);
            }
            #pragma omp critical
            for (int b = 0; b < N_BINS; ++b) bins[b].merge(local[b]);
        }
    } else {
        for (uint32_t i = first; i < first + count; ++i) {
            uint32_t p = perm[i];
            float cv  = (axis == 0) ? x[p] : (axis == 1) ? y[p] : z[p];
            int   bi  = static_cast<int>((cv - axis_lo) * inv_extent);
            if (bi < 0) bi = 0;
            if (bi >= N_BINS) bi = N_BINS - 1;
            float hp = h[p] * 2.0f;
            float lo[3] = {x[p] - hp, y[p] - hp, z[p] - hp};
            float hi[3] = {x[p] + hp, y[p] + hp, z[p] + hp};
            bins[bi].expand(lo, hi);
        }
    }

    // --- Sweep SAH from both sides ---
    Bin   left_acc[N_BINS - 1];
    Bin   right_acc[N_BINS - 1];
    Bin   acc;
    for (int i = 0; i < N_BINS - 1; ++i) {
        acc.merge(bins[i]);
        left_acc[i] = acc;
    }
    acc = Bin();
    for (int i = N_BINS - 1; i >= 1; --i) {
        acc.merge(bins[i]);
        right_acc[i - 1] = acc;
    }

    float best_cost  = FLT_MAX;
    int   best_split = -1;
    for (int i = 0; i < N_BINS - 1; ++i) {
        if (left_acc[i].n == 0 || right_acc[i].n == 0) continue;
        float c = static_cast<float>(left_acc[i].n)  * surfaceArea(left_acc[i].bmin,  left_acc[i].bmax)
                + static_cast<float>(right_acc[i].n) * surfaceArea(right_acc[i].bmin, right_acc[i].bmax);
        if (c < best_cost) { best_cost = c; best_split = i; }
    }

    float parent_sa  = surfaceArea(bmin, bmax);
    float leaf_cost  = static_cast<float>(count) * parent_sa;
    if (best_split < 0 || (best_cost >= leaf_cost && count <= 2 * LEAF_SIZE)) {
        makeLeaf();
        return;
    }
    if (best_split < 0) {
        // Large count but no useful SAH split found: force a median split.
        best_split = N_BINS / 2 - 1;
    }

    // --- In-place partition around split threshold ---
    float threshold = axis_lo + (static_cast<float>(best_split + 1) / inv_extent);
    uint32_t mid = first;
    for (uint32_t i = first; i < first + count; ++i) {
        uint32_t p  = perm[i];
        float    cv = (axis == 0) ? x[p] : (axis == 1) ? y[p] : z[p];
        if (cv < threshold) {
            std::swap(perm[i], perm[mid]);
            ++mid;
        }
    }
    uint32_t left_count = mid - first;
    if (left_count == 0 || left_count == count) {
        // Degenerate partition — force equal median split.
        uint32_t median = first + count / 2;
        std::nth_element(perm.begin() + first, perm.begin() + median, perm.begin() + first + count,
                         [axis, x, y, z](uint32_t a, uint32_t b) {
                             float va = (axis == 0) ? x[a] : (axis == 1) ? y[a] : z[a];
                             float vb = (axis == 0) ? x[b] : (axis == 1) ? y[b] : z[b];
                             return va < vb;
                         });
        mid = median;
        left_count = count / 2;
        if (left_count == 0) { makeLeaf(); return; }
    }

    // --- Allocate children; left is implicit at node_idx + 1 ---
    uint32_t left_idx = static_cast<uint32_t>(nodes.size());
    nodes.emplace_back();
    buildRecursive(left_idx, first, left_count, depth + 1, x, y, z, h);

    uint32_t right_idx = static_cast<uint32_t>(nodes.size());
    nodes.emplace_back();
    buildRecursive(right_idx, mid, count - left_count, depth + 1, x, y, z, h);

    nodes[node_idx].count          = 0;
    nodes[node_idx].left_or_first  = right_idx;
}
