#include "common/KernelLUT.h"
#include "common/Kernel.h"
#include <cmath>
#include <cstdio>
#include <cstdlib>

static int s_fail = 0;

#define CHECK(cond, msg) do { \
    if (!(cond)) { std::fprintf(stderr, "FAIL %s:%d  %s\n", __FILE__, __LINE__, msg); ++s_fail; } \
} while (0)

// 1. RMS agreement between LUT and reference quadrature over u ∈ [0, 2].
static void test_lut_vs_reference() {
    KernelLUT::init();
    constexpr int N = 20000;
    double sum_sq = 0.0, max_abs = 0.0;
    for (int i = 0; i < N; ++i) {
        double u = 2.0 * i / (N - 1);
        double ref  = KernelLUT::F_reference(u);
        double lut  = KernelLUT::eval(static_cast<float>(u));
        double err  = lut - ref;
        sum_sq += err * err;
        if (std::fabs(err) > max_abs) max_abs = std::fabs(err);
    }
    double rms = std::sqrt(sum_sq / N);
    std::printf("  LUT vs ref: RMS=%.3e, max=%.3e\n", rms, max_abs);
    CHECK(rms < 1e-4, "RMS error too large");
    CHECK(max_abs < 1e-3, "max error too large");
}

// 2. Mass normalization: ∫₀² F(u) · 2π u du == 1 for unit-mass particle.
static void test_mass_normalization() {
    KernelLUT::init();
    const double PI = 3.14159265358979323846;
    constexpr int M = 200000;
    double du = 2.0 / M;
    double sum = 0.0;
    for (int k = 1; k < M; ++k) {
        double u = k * du;
        double w = (k % 2 == 0) ? 2.0 : 4.0;
        sum += w * KernelLUT::F_reference(u) * 2.0 * PI * u;
    }
    // endpoints: u=0 gives 0, u=2 gives 0.
    double integral = (du / 3.0) * sum;
    std::printf("  Mass norm: ∫₀² F(u)·2π u du = %.6f (want 1.0)\n", integral);
    CHECK(std::fabs(integral - 1.0) < 1e-4, "kernel line integral not mass-normalized");
}

// 3. Edge behavior: F(2) = 0, F(u >= 2) = 0 via eval.
static void test_edges() {
    KernelLUT::init();
    CHECK(KernelLUT::eval(2.0f) == 0.0f, "eval(2.0) should be zero");
    CHECK(KernelLUT::eval(3.5f) == 0.0f, "eval beyond support should be zero");
    // F(0) should equal 2 ∫₀² W(s, 1) ds. For cubic spline σ=1/π, this works
    // out to about 0.95493. Check LUT close to reference at u=0.
    double f0_ref = KernelLUT::F_reference(0.0);
    double f0_lut = KernelLUT::eval(0.0f);
    std::printf("  F(0) ref=%.6f lut=%.6f\n", f0_ref, f0_lut);
    CHECK(std::fabs(f0_lut - f0_ref) < 1e-5, "F(0) LUT disagrees with reference");
}

// 4. evalU2(u²) agrees with eval(u) and the reference quadrature.
static void test_lut_u2_vs_reference() {
    KernelLUT::init();
    constexpr int N = 20000;
    double sum_sq = 0.0, max_abs = 0.0;
    for (int i = 0; i < N; ++i) {
        double u  = 2.0 * i / (N - 1);
        double u2 = u * u;
        double ref = KernelLUT::F_reference(u);
        double lut = KernelLUT::evalU2(static_cast<float>(u2));
        double err = lut - ref;
        sum_sq += err * err;
        if (std::fabs(err) > max_abs) max_abs = std::fabs(err);
    }
    double rms = std::sqrt(sum_sq / N);
    std::printf("  LUT_u2 vs ref: RMS=%.3e, max=%.3e\n", rms, max_abs);
    CHECK(rms < 1e-4, "evalU2 RMS error too large");
    CHECK(max_abs < 1e-3, "evalU2 max error too large");
    CHECK(KernelLUT::evalU2(4.0f) == 0.0f, "evalU2(4) should be zero");
    CHECK(KernelLUT::evalU2(9.0f) == 0.0f, "evalU2 beyond support should be zero");
}

int main() {
    std::printf("test_kernel_lut:\n");
    test_lut_vs_reference();
    test_mass_normalization();
    test_edges();
    test_lut_u2_vs_reference();
    if (s_fail) {
        std::fprintf(stderr, "\n%d check(s) failed\n", s_fail);
        return 1;
    }
    std::printf("  OK\n");
    return 0;
}
