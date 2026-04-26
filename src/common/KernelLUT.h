#pragma once

// Tabulated 2D line integral of the cubic-spline SPH kernel.
//
// F(u) = ∫_{-∞}^{+∞} W(sqrt(u² + s²), h=1) ds
//
// For an SPH particle of mass m, smoothing length h, at impact parameter b
// from a ray, the contribution to the ray's column density is
//   m · F(b/h) / h²                   (units: mass / length²)
// This is the exact line integral through the 3D cubic-spline kernel.
//
// F(0) = 3/(2π) ≈ 0.47746 (peak), F(u≥2) = 0, ∫₀² F(u) 2π u du = 1.

class KernelLUT {
public:
    static constexpr int   N_SAMPLES   = 1024;
    static constexpr float U_MAX       = 2.0f;
    // u² table is finer because the leaf loop indexes it directly without a
    // sqrt, and uniform spacing in u² is coarser in u near the origin.
    static constexpr int   N_U2_SAMPLES = 4096;
    static constexpr float U2_MAX       = 4.0f;

    static void init();

    static inline float eval(float u) {
        if (u >= U_MAX) return 0.0f;
        if (u < 0.0f)   u = -u;
        float x = u * INV_DU_;
        int   i = static_cast<int>(x);
        if (i >= N_SAMPLES - 1) return 0.0f;
        float frac = x - static_cast<float>(i);
        return table_[i] * (1.0f - frac) + table_[i + 1] * frac;
    }

    // Same kernel line integral, indexed by u² to avoid a sqrt per particle hit.
    static inline float evalU2(float u2) {
        if (u2 >= U2_MAX) return 0.0f;
        if (u2 < 0.0f)    u2 = 0.0f;
        float x = u2 * INV_DU2_;
        int   i = static_cast<int>(x);
        if (i >= N_U2_SAMPLES - 1) return 0.0f;
        float frac = x - static_cast<float>(i);
        return table_u2_[i] * (1.0f - frac) + table_u2_[i + 1] * frac;
    }

    // High-accuracy reference used by unit tests and for LUT population.
    static double F_reference(double u);

private:
    static float table_[N_SAMPLES];
    static float table_u2_[N_U2_SAMPLES];
    static bool  initialized_;
    static constexpr float DU_      = U_MAX / static_cast<float>(N_SAMPLES - 1);
    static constexpr float INV_DU_  = 1.0f / DU_;
    static constexpr float DU2_     = U2_MAX / static_cast<float>(N_U2_SAMPLES - 1);
    static constexpr float INV_DU2_ = 1.0f / DU2_;
};
