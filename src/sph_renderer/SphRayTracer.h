#pragma once

#include "renderer/Camera.h"
#include "sph_renderer/BVH.h"
#include "sph_renderer/ParticleStore.h"
#include <vector>

// Direct SPH ray tracer: traverses a per-particle BVH and evaluates the exact
// cubic-spline line integral for every particle whose kernel support the ray
// pierces.
//
// Per-hit contribution (column-density mode):
//   contrib_i = m_i * F(b_i / h_i) / h_i^2
// where b_i is the impact parameter of the ray to particle i and F is the
// tabulated kernel line integral (KernelLUT::eval).
//
// Image layout: row-major [H * W], row 0 at the top (matches renderer/).
class SphRayTracer {
public:
    // Gas column density using native SPH smoothing lengths.
    static std::vector<float> traceGasColumn(
        const Camera& camera,
        const ParticleStore& ps,
        const BVH& bvh);

    // Dark matter column density using kNN smoothing lengths. Same math
    // as gas (mass * F(b/h) / h^2) — DM kernel support is 2*h from the kNN
    // scheme. ps.dm_* arrays must be populated and `bvh` built over them.
    static std::vector<float> traceDMColumn(
        const Camera& camera,
        const ParticleStore& ps,
        const BVH& bvh);

    // Weighted column on gas: Σ (m_i * extra_w[i]) * F(b_i/h_i)/h_i^2.
    // Examples: pass gas_metallicity to get a metal column (Σ m·Z·F/h²);
    // pass per-particle |v| to get a speed-weighted column. No denominator —
    // this is a column, not a mass-weighted ratio.
    static std::vector<float> traceGasWeightedColumn(
        const Camera& camera,
        const ParticleStore& ps,
        const BVH& bvh,
        const float* extra_weight);

    // Emission + absorption volume render (front-to-back, order-dependent).
    // For each ray, particles are visited in t-sorted order and a classical
    // emission/absorption integral is accumulated:
    //   E_out = Σ_i T_i · (kappa_e · emit_i · m_i · F/h²)
    //   T_{i+1} = T_i · (1 - alpha_i),  alpha_i = kappa_a · absorb_i · m_i · F/h²
    // Returns an interleaved [H*W*2] buffer: channel 0 = emission, channel 1
    // = transmittance remaining (T). Python side turns (E, T) + colormap
    // into the final translucent-layered RGB+A frame.
    static std::vector<float> traceGasEmission(
        const Camera& camera,
        const ParticleStore& ps,
        const BVH& bvh,
        const float* emission,
        const float* absorption,
        double kappa_e,
        double kappa_a,
        int samples_per_axis = 1);

    // Gas weighted mean of one or more per-particle fields along each ray.
    //   result_c = (Σ (m_i * extra_w_i) * F(b_i/h_i)/h_i^2 * q_c[i])
    //              / (Σ (m_i * extra_w_i) * F(b_i/h_i)/h_i^2)
    // If `extra_weight` is null, falls back to plain mass weighting:
    // `result = <q>_mass`. Passing `extra_weight = gas_density` gives the
    // density-weighted (ρ²-weighted) average, which highlights hot/diffuse
    // structure the way most published cosmo temperature maps do.
    // Per-channel output is interleaved: out[(py*W + px) * C + c].
    // Pixels with no gas coverage (zero denominator) get 0.
    static std::vector<float> traceGasWeighted(
        const Camera& camera,
        const ParticleStore& ps,
        const BVH& bvh,
        const float* extra_weight,
        const std::vector<const float*>& fields);

    // Convenience shim: mass-weighted (extra_weight = null).
    static std::vector<float> traceGasMassWeighted(
        const Camera& camera,
        const ParticleStore& ps,
        const BVH& bvh,
        const std::vector<const float*>& fields) {
        return traceGasWeighted(camera, ps, bvh, nullptr, fields);
    }
};
