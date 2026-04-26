#include "common/SmoothingLength.h"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <numeric>
#include <vector>

// kNN-based adaptive smoothing length. Replaces the older uniform-cell
// search; for large (>1e8) DM arrays the cell grid would hit its 512^3
// cap and each cell held millions of particles, making the expanding-ring
// scan very slow. A kd-tree keeps the query cost ~O(log N * K) regardless
// of clustering. Periodic wrap is deliberately omitted: the cube is
// extracted + pre-unwrapped, so particles are contiguous.

namespace {

inline double axis_val(const Vec3& p, int axis) {
    return axis == 0 ? p.x : (axis == 1 ? p.y : p.z);
}

struct KDTree {
    const std::vector<DMParticle>& particles;
    std::vector<uint32_t> idx;

    explicit KDTree(const std::vector<DMParticle>& p)
        : particles(p), idx(p.size()) {
        std::iota(idx.begin(), idx.end(), 0u);
        #pragma omp parallel
        {
            #pragma omp single nowait
            build(0, idx.size(), 0);
        }
    }

    void build(size_t lo, size_t hi, int depth) {
        if (hi - lo <= 1) return;
        int axis = depth % 3;
        size_t mid = lo + (hi - lo) / 2;
        auto cmp = [&, axis](uint32_t a, uint32_t b) {
            return axis_val(particles[a].pos, axis)
                 < axis_val(particles[b].pos, axis);
        };
        std::nth_element(idx.begin() + lo, idx.begin() + mid,
                         idx.begin() + hi, cmp);
        const size_t parallel_threshold = 200000;
        if (hi - lo > parallel_threshold) {
            #pragma omp task default(none) firstprivate(lo, mid, depth)
            build(lo, mid, depth + 1);
            #pragma omp task default(none) firstprivate(mid, hi, depth)
            build(mid + 1, hi, depth + 1);
            #pragma omp taskwait
        } else {
            build(lo, mid, depth + 1);
            build(mid + 1, hi, depth + 1);
        }
    }

    // Max-heap of squared distances (largest at heap[0]); caller pre-fills
    // with sentinels. `self` is the index to skip (the query particle itself).
    void knn(const Vec3& q, uint32_t self, std::vector<double>& heap) const {
        query(0, idx.size(), 0, q, self, heap);
    }

    void query(size_t lo, size_t hi, int depth, const Vec3& q,
               uint32_t self, std::vector<double>& heap) const {
        if (lo >= hi) return;
        int axis = depth % 3;
        size_t mid = lo + (hi - lo) / 2;
        uint32_t mj = idx[mid];
        const Vec3& pm = particles[mj].pos;

        if (mj != self) {
            double dx = pm.x - q.x;
            double dy = pm.y - q.y;
            double dz = pm.z - q.z;
            double d2 = dx * dx + dy * dy + dz * dz;
            if (d2 < heap[0]) {
                std::pop_heap(heap.begin(), heap.end());
                heap.back() = d2;
                std::push_heap(heap.begin(), heap.end());
            }
        }

        double split = axis_val(pm, axis);
        double diff = axis_val(q, axis) - split;
        if (diff < 0.0) {
            query(lo, mid, depth + 1, q, self, heap);
            if (diff * diff < heap[0])
                query(mid + 1, hi, depth + 1, q, self, heap);
        } else {
            query(mid + 1, hi, depth + 1, q, self, heap);
            if (diff * diff < heap[0])
                query(lo, mid, depth + 1, q, self, heap);
        }
    }
};

}  // namespace

void SmoothingLength::computeKNN(std::vector<DMParticle>& particles,
                                 double boxsize) {
    computeKNN(particles, boxsize, K_NEIGHBORS, /*h_max=*/0.0, /*h_min=*/0.0);
}

void SmoothingLength::computeKNN(std::vector<DMParticle>& particles,
                                 double /*boxsize*/, int k, double h_max,
                                 double h_min) {
    if (particles.empty()) return;
    const size_t N = particles.size();
    if (k < 1) k = 1;

    std::cout << "  kNN kd-tree: N=" << N << " K=" << k;
    if (h_max > 0.0) std::cout << " h_max=" << h_max;
    if (h_min > 0.0) std::cout << " h_min=" << h_min;
    std::cout << std::endl;
    auto t0 = std::chrono::steady_clock::now();

    KDTree tree(particles);

    auto t1 = std::chrono::steady_clock::now();
    std::cout << "  kd-tree built in "
              << std::chrono::duration<double>(t1 - t0).count() << "s"
              << std::endl;

    const float h_cap   = (h_max > 0.0) ? static_cast<float>(h_max) : 0.0f;
    const float h_floor = (h_min > 0.0) ? static_cast<float>(h_min) : 0.0f;
    #pragma omp parallel for schedule(dynamic, 4096)
    for (size_t i = 0; i < N; i++) {
        std::vector<double> heap(k, 1e30);
        tree.knn(particles[i].pos, static_cast<uint32_t>(i), heap);
        float h = static_cast<float>(std::sqrt(heap[0]));
        if (h_cap   > 0.0f && h > h_cap)   h = h_cap;
        if (h_floor > 0.0f && h < h_floor) h = h_floor;
        particles[i].hsml = h;
    }

    auto t2 = std::chrono::steady_clock::now();
    std::cout << "  kNN query in "
              << std::chrono::duration<double>(t2 - t1).count() << "s"
              << std::endl;

    float min_h = 1e30f, max_h = 0.0f;
    double sum_h = 0.0;
    for (const auto& p : particles) {
        min_h = std::min(min_h, p.hsml);
        max_h = std::max(max_h, p.hsml);
        sum_h += p.hsml;
    }
    std::cout << "  kNN hsml: min=" << min_h << " max=" << max_h
              << " mean=" << sum_h / N << std::endl;
}
