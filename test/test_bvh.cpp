// Brute-force vs. BVH: ray-sphere intersection count must match exactly.
#include "sph_renderer/BVH.h"
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <vector>

static int s_fail = 0;
#define CHECK(cond, msg) do { \
    if (!(cond)) { std::fprintf(stderr, "FAIL %s:%d  %s\n", __FILE__, __LINE__, msg); ++s_fail; } \
} while (0)

struct Ray { float ox, oy, oz, dx, dy, dz; };

// Infinite-line slab test for line vs. BVH AABB. The unit test checks
// agreement on infinite-line-sphere counts (no t >= 0 filter) so that the
// brute-force and BVH traversals use identical geometry semantics.
static inline bool rayAABB(const Ray& r, const float bmin[3], const float bmax[3]) {
    float t0 = -1e30f, t1 = 1e30f;
    for (int d = 0; d < 3; ++d) {
        float origin = (d == 0 ? r.ox : d == 1 ? r.oy : r.oz);
        float dir    = (d == 0 ? r.dx : d == 1 ? r.dy : r.dz);
        if (std::fabs(dir) < 1e-20f) {
            if (origin < bmin[d] || origin > bmax[d]) return false;
        } else {
            float inv = 1.0f / dir;
            float a = (bmin[d] - origin) * inv;
            float b = (bmax[d] - origin) * inv;
            if (a > b) std::swap(a, b);
            if (a > t0) t0 = a;
            if (b < t1) t1 = b;
            if (t0 > t1) return false;
        }
    }
    return true;
}

// Returns impact parameter squared and along-ray t.
static inline void rayPoint(const Ray& r, float px, float py, float pz,
                            float& b2, float& t) {
    float dx = px - r.ox, dy = py - r.oy, dz = pz - r.oz;
    t = dx * r.dx + dy * r.dy + dz * r.dz;
    float cx = r.ox + r.dx * t;
    float cy = r.oy + r.dy * t;
    float cz = r.oz + r.dz * t;
    float ex = px - cx, ey = py - cy, ez = pz - cz;
    b2 = ex * ex + ey * ey + ez * ez;
}

// Brute-force: every particle whose sphere of radius 2h the ray pierces.
static int bruteHits(const Ray& r,
                     const std::vector<float>& x, const std::vector<float>& y,
                     const std::vector<float>& z, const std::vector<float>& h) {
    int hits = 0;
    for (size_t i = 0; i < x.size(); ++i) {
        float b2, t;
        rayPoint(r, x[i], y[i], z[i], b2, t);
        float sup = 2.0f * h[i];
        if (b2 < sup * sup) ++hits;
    }
    return hits;
}

// BVH traversal with the same precise per-particle test.
static int bvhHits(const BVH& bvh, const Ray& r,
                   const std::vector<float>& x, const std::vector<float>& y,
                   const std::vector<float>& z, const std::vector<float>& h) {
    int hits = 0;
    uint32_t stack[128];
    int top = 0;
    stack[top++] = 0;
    while (top > 0) {
        uint32_t ni = stack[--top];
        const BVHNode& node = bvh.nodes[ni];
        if (!rayAABB(r, node.bmin, node.bmax)) continue;
        if (node.count > 0) {
            for (uint32_t i = 0; i < node.count; ++i) {
                uint32_t p = bvh.perm[node.left_or_first + i];
                float b2, t;
                rayPoint(r, x[p], y[p], z[p], b2, t);
                float sup = 2.0f * h[p];
                if (b2 < sup * sup) ++hits;
            }
        } else {
            stack[top++] = node.left_or_first;      // right child
            stack[top++] = ni + 1;                   // left child (implicit)
        }
    }
    return hits;
}

int main() {
    std::printf("test_bvh:\n");
    std::mt19937 rng(42);
    std::uniform_real_distribution<float> u01(0.0f, 1.0f);

    constexpr size_t N = 10000;
    std::vector<float> x(N), y(N), z(N), h(N);
    for (size_t i = 0; i < N; ++i) {
        x[i] = u01(rng) * 100.0f;
        y[i] = u01(rng) * 100.0f;
        z[i] = u01(rng) * 100.0f;
        h[i] = 0.5f + 1.5f * u01(rng);
    }

    BVH bvh;
    bvh.buildFromSpheres(x.data(), y.data(), z.data(), h.data(), N);
    std::printf("  N=%zu, nodes=%zu\n", N, bvh.nodes.size());
    CHECK(bvh.perm.size() == N, "perm size mismatch");
    CHECK(bvh.nodes.size() > 0, "empty tree");

    // Run 200 random rays; counts must agree.
    constexpr int R = 200;
    int total_disagree = 0;
    for (int k = 0; k < R; ++k) {
        Ray r;
        r.ox = u01(rng) * 100.0f;
        r.oy = u01(rng) * 100.0f;
        r.oz = u01(rng) * 100.0f;
        float dx = u01(rng) * 2.0f - 1.0f;
        float dy = u01(rng) * 2.0f - 1.0f;
        float dz = u01(rng) * 2.0f - 1.0f;
        float n = std::sqrt(dx * dx + dy * dy + dz * dz);
        r.dx = dx / n; r.dy = dy / n; r.dz = dz / n;

        int bh = bruteHits(r, x, y, z, h);
        int bv = bvhHits(bvh, r, x, y, z, h);
        if (bh != bv) {
            std::fprintf(stderr, "  ray %d: brute=%d bvh=%d\n", k, bh, bv);
            ++total_disagree;
        }
    }
    CHECK(total_disagree == 0, "BVH and brute-force disagree on some rays");
    std::printf("  %d rays, %d disagreements\n", R, total_disagree);

    if (s_fail) {
        std::fprintf(stderr, "\n%d check(s) failed\n", s_fail);
        return 1;
    }
    std::printf("  OK\n");
    return 0;
}
