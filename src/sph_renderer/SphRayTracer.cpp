#include "sph_renderer/SphRayTracer.h"
#include "common/KernelLUT.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <limits>

namespace {

// Deterministic per-pixel/per-sample hash → float in [0,1). Used to jitter
// stratified subpixel offsets so the regular supersample grid doesn't itself
// beat against the particle layout and re-create moiré.
inline float hash01(uint32_t px, uint32_t py, uint32_t sx, uint32_t sy,
                    uint32_t axis) {
    uint32_t h = px * 0x9E3779B1u;
    h ^= py * 0x85EBCA77u;
    h ^= sx * 0xC2B2AE3Du;
    h ^= sy * 0x27D4EB2Fu;
    h ^= axis * 0x165667B1u;
    h ^= h >> 16; h *= 0x7FEB352Du;
    h ^= h >> 15; h *= 0x846CA68Bu;
    h ^= h >> 16;
    return (h & 0x00FFFFFFu) * (1.0f / 16777216.0f);
}

// Ray vs. axis-aligned slab test, clipped against an incoming [tmin,tmax]
// range (so callers can fold in slab/camera clipping). Returns false when the
// ray misses the box within the incoming range.
inline bool rayAABB(const Vec3& origin, const Vec3& inv_dir,
                    const float bmin[3], const float bmax[3],
                    double tmin_in, double tmax_in,
                    double& tmin_out, double& tmax_out) {
    double tmin = tmin_in, tmax = tmax_in;
    for (int d = 0; d < 3; ++d) {
        double t1 = (static_cast<double>(bmin[d]) - origin[d]) * inv_dir[d];
        double t2 = (static_cast<double>(bmax[d]) - origin[d]) * inv_dir[d];
        if (t1 > t2) std::swap(t1, t2);
        if (t1 > tmin) tmin = t1;
        if (t2 < tmax) tmax = t2;
        if (tmin > tmax) return false;
    }
    tmin_out = tmin;
    tmax_out = tmax;
    return true;
}

} // namespace

static std::vector<float> traceColumnSoA(
    const Camera& camera, const BVH& bvh,
    const float* X, const float* Y, const float* Z,
    const float* Hh, const float* Mass,
    size_t N, const char* label,
    const float* extra_weight = nullptr) {

    KernelLUT::init();

    const int W = camera.width();
    const int H = camera.height();
    std::vector<float> image(static_cast<size_t>(H) * W, 0.0f);
    if (N == 0 || bvh.nodes.empty()) return image;

    std::cout << "SPH " << label << ": " << W << "x" << H
              << "  N=" << N
              << "  los_slab=" << camera.losSlab() << std::endl;

    #pragma omp parallel for schedule(dynamic, 4)
    for (int py = 0; py < H; ++py) {
        for (int px = 0; px < W; ++px) {
            Ray ray = camera.generateRay(px, py);

            Vec3 inv_dir;
            inv_dir.x = (std::fabs(ray.dir.x) > 1e-30) ? 1.0 / ray.dir.x : 1e30;
            inv_dir.y = (std::fabs(ray.dir.y) > 1e-30) ? 1.0 / ray.dir.y : 1e30;
            inv_dir.z = (std::fabs(ray.dir.z) > 1e-30) ? 1.0 / ray.dir.z : 1e30;

            double slab0, slab1;
            camera.slabTRange(ray, slab0, slab1);
            // Reject particles behind the camera eye. For orthographic rays
            // the origin sits on the image plane so all physical hits have
            // t >= 0 anyway, and slab0 is already positive. For perspective
            // rays this clamp drops particles lying behind the camera that
            // would otherwise leak into the integral.
            if (slab0 < 0.0) slab0 = 0.0;
            if (slab0 >= slab1) { image[py * W + px] = 0.0f; continue; }

            double accum = 0.0;
            // Origin stays double so the per-particle subtraction does not
            // lose precision against large box coordinates; direction and
            // intermediate dx/dy/dz/t/b² run in float for SIMD throughput.
            const double Ox = ray.origin.x;
            const double Oy = ray.origin.y;
            const double Oz = ray.origin.z;
            const float  Dx = static_cast<float>(ray.dir.x);
            const float  Dy = static_cast<float>(ray.dir.y);
            const float  Dz = static_cast<float>(ray.dir.z);
            const float  s0 = static_cast<float>(slab0);
            const float  s1 = static_cast<float>(slab1);

            uint32_t stack[128];
            int top = 0;
            stack[top++] = 0;

            while (top > 0) {
                uint32_t ni = stack[--top];
                const BVHNode& node = bvh.nodes[ni];

                double t0, t1;
                if (!rayAABB(ray.origin, inv_dir, node.bmin, node.bmax,
                             slab0, slab1, t0, t1)) continue;

                if (node.count > 0) {
                    // Leaf: precise per-particle kernel line integral.
                    const uint32_t first = node.left_or_first;
                    const uint32_t nprim = node.count;
                    for (uint32_t i = 0; i < nprim; ++i) {
                        uint32_t p = bvh.perm[first + i];

                        float dx = static_cast<float>(static_cast<double>(X[p]) - Ox);
                        float dy = static_cast<float>(static_cast<double>(Y[p]) - Oy);
                        float dz = static_cast<float>(static_cast<double>(Z[p]) - Oz);
                        float t  = dx * Dx + dy * Dy + dz * Dz;
                        if (t < s0 || t > s1) continue;

                        float ex = dx - t * Dx;
                        float ey = dy - t * Dy;
                        float ez = dz - t * Dz;
                        float b2 = ex * ex + ey * ey + ez * ez;

                        float h  = Hh[p];
                        float h2 = h * h;
                        if (b2 < 4.0f * h2) {
                            float contrib = KernelLUT::evalU2(b2 / h2) / h2;
                            double mw = static_cast<double>(Mass[p]);
                            if (extra_weight) mw *= static_cast<double>(extra_weight[p]);
                            accum += mw * static_cast<double>(contrib);
                        }
                    }
                } else {
                    // Internal: push right then left (left visited first).
                    stack[top++] = node.left_or_first;  // right child
                    stack[top++] = ni + 1;              // left child (implicit)
                }
            }
            image[py * W + px] = static_cast<float>(accum);
        }
        if ((py & 0xFF) == 0) {
            #pragma omp critical
            std::cout << "  Row " << py << " / " << H << std::endl;
        }
    }
    return image;
}

std::vector<float> SphRayTracer::traceGasColumn(
    const Camera& camera, const ParticleStore& ps, const BVH& bvh) {
    return traceColumnSoA(camera, bvh,
                          ps.gas_x.data(), ps.gas_y.data(), ps.gas_z.data(),
                          ps.gas_h.data(), ps.gas_mass.data(),
                          ps.numGas(), "traceGasColumn");
}

std::vector<float> SphRayTracer::traceDMColumn(
    const Camera& camera, const ParticleStore& ps, const BVH& bvh) {
    return traceColumnSoA(camera, bvh,
                          ps.dm_x.data(), ps.dm_y.data(), ps.dm_z.data(),
                          ps.dm_h.data(), ps.dm_mass.data(),
                          ps.numDM(), "traceDMColumn");
}

std::vector<float> SphRayTracer::traceGasWeightedColumn(
    const Camera& camera, const ParticleStore& ps, const BVH& bvh,
    const float* extra_weight) {
    return traceColumnSoA(camera, bvh,
                          ps.gas_x.data(), ps.gas_y.data(), ps.gas_z.data(),
                          ps.gas_h.data(), ps.gas_mass.data(),
                          ps.numGas(), "traceGasWeightedColumn", extra_weight);
}

std::vector<float> SphRayTracer::traceGasWeighted(
    const Camera& camera, const ParticleStore& ps, const BVH& bvh,
    const float* extra_weight,
    const std::vector<const float*>& fields) {

    KernelLUT::init();

    const int W = camera.width();
    const int H = camera.height();
    const int C = static_cast<int>(fields.size());
    std::vector<float> image(static_cast<size_t>(H) * W * C, 0.0f);
    if (ps.numGas() == 0 || bvh.nodes.empty() || C == 0) return image;

    const float* X    = ps.gas_x.data();
    const float* Y    = ps.gas_y.data();
    const float* Z    = ps.gas_z.data();
    const float* Hh   = ps.gas_h.data();
    const float* Mass = ps.gas_mass.data();

    std::cout << "SPH traceGasMassWeighted: " << W << "x" << H
              << "  N_gas=" << ps.numGas()
              << "  channels=" << C
              << "  los_slab=" << camera.losSlab() << std::endl;

    #pragma omp parallel
    {
        std::vector<double> num(C, 0.0);

        #pragma omp for schedule(dynamic, 4)
        for (int py = 0; py < H; ++py) {
            for (int px = 0; px < W; ++px) {
                Ray ray = camera.generateRay(px, py);

                Vec3 inv_dir;
                inv_dir.x = (std::fabs(ray.dir.x) > 1e-30) ? 1.0 / ray.dir.x : 1e30;
                inv_dir.y = (std::fabs(ray.dir.y) > 1e-30) ? 1.0 / ray.dir.y : 1e30;
                inv_dir.z = (std::fabs(ray.dir.z) > 1e-30) ? 1.0 / ray.dir.z : 1e30;

                double slab0, slab1;
                camera.slabTRange(ray, slab0, slab1);
                if (slab0 < 0.0) slab0 = 0.0;
                if (slab0 >= slab1) continue;

                for (int c = 0; c < C; ++c) num[c] = 0.0;
                double den = 0.0;

                const double Ox = ray.origin.x;
                const double Oy = ray.origin.y;
                const double Oz = ray.origin.z;
                const float  Dx = static_cast<float>(ray.dir.x);
                const float  Dy = static_cast<float>(ray.dir.y);
                const float  Dz = static_cast<float>(ray.dir.z);
                const float  s0 = static_cast<float>(slab0);
                const float  s1 = static_cast<float>(slab1);

                uint32_t stack[128];
                int top = 0;
                stack[top++] = 0;

                while (top > 0) {
                    uint32_t ni = stack[--top];
                    const BVHNode& node = bvh.nodes[ni];

                    double t0, t1;
                    if (!rayAABB(ray.origin, inv_dir, node.bmin, node.bmax,
                                 slab0, slab1, t0, t1)) continue;

                    if (node.count > 0) {
                        const uint32_t first = node.left_or_first;
                        const uint32_t nprim = node.count;
                        for (uint32_t i = 0; i < nprim; ++i) {
                            uint32_t p = bvh.perm[first + i];

                            float dx = static_cast<float>(static_cast<double>(X[p]) - Ox);
                            float dy = static_cast<float>(static_cast<double>(Y[p]) - Oy);
                            float dz = static_cast<float>(static_cast<double>(Z[p]) - Oz);
                            float t  = dx * Dx + dy * Dy + dz * Dz;
                            if (t < s0 || t > s1) continue;

                            float ex = dx - t * Dx;
                            float ey = dy - t * Dy;
                            float ez = dz - t * Dz;
                            float b2 = ex * ex + ey * ey + ez * ez;

                            float h  = Hh[p];
                            float h2 = h * h;
                            if (b2 < 4.0f * h2) {
                                float k = KernelLUT::evalU2(b2 / h2) / h2;
                                double w = static_cast<double>(Mass[p])
                                         * static_cast<double>(k);
                                if (extra_weight) w *= static_cast<double>(extra_weight[p]);
                                den += w;
                                for (int c = 0; c < C; ++c) {
                                    num[c] += w * static_cast<double>(fields[c][p]);
                                }
                            }
                        }
                    } else {
                        stack[top++] = node.left_or_first;
                        stack[top++] = ni + 1;
                    }
                }

                if (den > 0.0) {
                    size_t base = (static_cast<size_t>(py) * W + px) * C;
                    double inv_den = 1.0 / den;
                    for (int c = 0; c < C; ++c) {
                        image[base + c] = static_cast<float>(num[c] * inv_den);
                    }
                }
            }
            if ((py & 0xFF) == 0) {
                #pragma omp critical
                std::cout << "  Row " << py << " / " << H << std::endl;
            }
        }
    }
    return image;
}

std::vector<float> SphRayTracer::traceGasEmission(
    const Camera& camera, const ParticleStore& ps, const BVH& bvh,
    const float* emission, const float* absorption,
    double kappa_e, double kappa_a, int samples_per_axis) {
    if (samples_per_axis < 1) samples_per_axis = 1;
    const int NS = samples_per_axis;
    const double inv_nsamples = 1.0 / (static_cast<double>(NS) * NS);

    KernelLUT::init();

    const int W = camera.width();
    const int H = camera.height();
    std::vector<float> image(static_cast<size_t>(H) * W * 2, 0.0f);
    if (ps.numGas() == 0 || bvh.nodes.empty()) return image;

    const float* X    = ps.gas_x.data();
    const float* Y    = ps.gas_y.data();
    const float* Zc   = ps.gas_z.data();
    const float* Hh   = ps.gas_h.data();
    const float* Mass = ps.gas_mass.data();

    // Initialize transmittance = 1 (channel 1) so pixels that see no gas stay
    // fully transparent on the Python side.
    for (size_t i = 0; i < static_cast<size_t>(W) * H; ++i) image[i * 2 + 1] = 1.0f;

    std::cout << "SPH traceGasEmission: " << W << "x" << H
              << "  N_gas=" << ps.numGas()
              << "  kappa_e=" << kappa_e << " kappa_a=" << kappa_a
              << "  los_slab=" << camera.losSlab()
              << "  samples/axis=" << NS << std::endl;

    // Front-to-back BVH traversal — near child popped first.
    // Transmittance cutoff: once T < T_EPS the ray is effectively opaque, so
    // we skip further traversal for that pixel.
    constexpr double T_EPS = 1e-3;

    #pragma omp parallel
    {
        // thread-local scratch for per-leaf hit sorting
        std::vector<std::pair<double, uint32_t>> hits;
        hits.reserve(64);

        #pragma omp for schedule(dynamic, 4)
        for (int py = 0; py < H; ++py) {
            for (int px = 0; px < W; ++px) {
                double E_sum = 0.0, T_sum = 0.0;
                for (int sy = 0; sy < NS; ++sy)
                for (int sx = 0; sx < NS; ++sx) {
                    double dx_sub = (sx + hash01(px, py, sx, sy, 0u)) / NS;
                    double dy_sub = (sy + hash01(px, py, sx, sy, 1u)) / NS;
                    Ray ray = camera.generateRay(px, py, dx_sub, dy_sub);

                    Vec3 inv_dir;
                    inv_dir.x = (std::fabs(ray.dir.x) > 1e-30) ? 1.0 / ray.dir.x : 1e30;
                    inv_dir.y = (std::fabs(ray.dir.y) > 1e-30) ? 1.0 / ray.dir.y : 1e30;
                    inv_dir.z = (std::fabs(ray.dir.z) > 1e-30) ? 1.0 / ray.dir.z : 1e30;

                    double slab0, slab1;
                    camera.slabTRange(ray, slab0, slab1);
                    if (slab0 < 0.0) slab0 = 0.0;
                    if (slab0 >= slab1) { T_sum += 1.0; continue; }

                    double T = 1.0;
                    double E = 0.0;

                // Stack holds (node_index, t_near) so we can prune when
                // cumulative transmittance already makes further integration
                // invisible.
                struct Entry { uint32_t ni; double tnear; };
                Entry stack[128];
                int top = 0;
                stack[top++] = {0u, slab0};

                while (top > 0 && T > T_EPS) {
                    Entry e = stack[--top];
                    if (e.tnear > slab1) continue;
                    const BVHNode& node = bvh.nodes[e.ni];

                    double t0, t1;
                    if (!rayAABB(ray.origin, inv_dir, node.bmin, node.bmax,
                                 slab0, slab1, t0, t1)) continue;

                    if (node.count > 0) {
                        // Leaf: collect hits and sort by t so alpha compositing
                        // proceeds in physical front-to-back order.
                        hits.clear();
                        const uint32_t first = node.left_or_first;
                        const uint32_t nprim = node.count;
                        for (uint32_t i = 0; i < nprim; ++i) {
                            uint32_t p = bvh.perm[first + i];
                            double dx = static_cast<double>(X[p])  - ray.origin.x;
                            double dy = static_cast<double>(Y[p])  - ray.origin.y;
                            double dz = static_cast<double>(Zc[p]) - ray.origin.z;
                            double t  = dx * ray.dir.x + dy * ray.dir.y + dz * ray.dir.z;
                            if (t < slab0 || t > slab1) continue;

                            double cx = ray.origin.x + ray.dir.x * t;
                            double cy = ray.origin.y + ray.dir.y * t;
                            double cz = ray.origin.z + ray.dir.z * t;
                            double ex = static_cast<double>(X[p])  - cx;
                            double ey = static_cast<double>(Y[p])  - cy;
                            double ez = static_cast<double>(Zc[p]) - cz;
                            double b2 = ex * ex + ey * ey + ez * ez;
                            double hsml = Hh[p];
                            if (b2 >= (2.0 * hsml) * (2.0 * hsml)) continue;

                            hits.push_back({t, p});
                        }
                        std::sort(hits.begin(), hits.end(),
                                  [](const std::pair<double,uint32_t>& a,
                                     const std::pair<double,uint32_t>& b) {
                                      return a.first < b.first;
                                  });

                        for (const auto& hp : hits) {
                            if (T <= T_EPS) break;
                            uint32_t p = hp.second;
                            double hsml = Hh[p];
                            // recompute b² (cheap vs. storing it)
                            double t = hp.first;
                            double cx = ray.origin.x + ray.dir.x * t;
                            double cy = ray.origin.y + ray.dir.y * t;
                            double cz = ray.origin.z + ray.dir.z * t;
                            double ex = static_cast<double>(X[p])  - cx;
                            double ey = static_cast<double>(Y[p])  - cy;
                            double ez = static_cast<double>(Zc[p]) - cz;
                            double b2 = ex * ex + ey * ey + ez * ez;
                            float u = static_cast<float>(std::sqrt(b2) / hsml);
                            double k = static_cast<double>(KernelLUT::eval(u))
                                     / (hsml * hsml);
                            double mk = static_cast<double>(Mass[p]) * k;
                            double e_i = emission   ? static_cast<double>(emission[p])   : 1.0;
                            double a_i = absorption ? static_cast<double>(absorption[p]) : 1.0;
                            double emit = kappa_e * e_i * mk;
                            double alph = kappa_a * a_i * mk;
                            if (alph > 1.0) alph = 1.0;
                            E += T * emit;
                            T *= (1.0 - alph);
                        }
                    } else {
                        // Internal node: push far child first, then near — so
                        // the near child is popped and traversed first.
                        uint32_t left  = e.ni + 1;
                        uint32_t right = node.left_or_first;
                        const BVHNode& lnode = bvh.nodes[left];
                        const BVHNode& rnode = bvh.nodes[right];
                        double lt0, lt1, rt0, rt1;
                        bool lhit = rayAABB(ray.origin, inv_dir, lnode.bmin, lnode.bmax,
                                            slab0, slab1, lt0, lt1);
                        bool rhit = rayAABB(ray.origin, inv_dir, rnode.bmin, rnode.bmax,
                                            slab0, slab1, rt0, rt1);
                        if (lhit && rhit) {
                            if (lt0 <= rt0) {
                                stack[top++] = {right, rt0};
                                stack[top++] = {left,  lt0};
                            } else {
                                stack[top++] = {left,  lt0};
                                stack[top++] = {right, rt0};
                            }
                        } else if (lhit) {
                            stack[top++] = {left, lt0};
                        } else if (rhit) {
                            stack[top++] = {right, rt0};
                        }
                    }
                }

                    E_sum += E;
                    T_sum += T;
                } // end sample loop
                size_t base = (static_cast<size_t>(py) * W + px) * 2;
                image[base + 0] = static_cast<float>(E_sum * inv_nsamples);
                image[base + 1] = static_cast<float>(T_sum * inv_nsamples);
            }
            if ((py & 0xFF) == 0) {
                #pragma omp critical
                std::cout << "  Row " << py << " / " << H << std::endl;
            }
        }
    }
    return image;
}
