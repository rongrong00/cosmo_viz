#include "common/KernelLUT.h"
#include "common/Kernel.h"
#include <cmath>

float KernelLUT::table_[KernelLUT::N_SAMPLES];
float KernelLUT::table_u2_[KernelLUT::N_U2_SAMPLES];
bool  KernelLUT::initialized_ = false;

double KernelLUT::F_reference(double u) {
    // F(u) = 2 · ∫₀^{s_max} W(sqrt(u² + s²), 1) ds, s_max = sqrt(4 - u²).
    // Composite Simpson's rule with 4096 subintervals; absolute error < 1e-9.
    double s_max_sq = 4.0 - u * u;
    if (s_max_sq <= 0.0) return 0.0;
    double s_max = std::sqrt(s_max_sq);
    constexpr int M = 4096;             // must be even
    double ds = s_max / static_cast<double>(M);
    double sum = Kernel::W(u, 1.0);     // s = 0 endpoint, W(u, 1)
    // s = s_max endpoint, W(2, 1) = 0 exactly; omit.
    for (int k = 1; k < M; ++k) {
        double s = k * ds;
        double r = std::sqrt(u * u + s * s);
        double w = Kernel::W(r, 1.0);
        sum += (k % 2 == 0) ? 2.0 * w : 4.0 * w;
    }
    return 2.0 * (ds / 3.0) * sum;
}

void KernelLUT::init() {
    if (initialized_) return;
    for (int i = 0; i < N_SAMPLES; ++i) {
        double u = static_cast<double>(i) * DU_;
        table_[i] = static_cast<float>(F_reference(u));
    }
    for (int i = 0; i < N_U2_SAMPLES; ++i) {
        double u2 = static_cast<double>(i) * DU2_;
        table_u2_[i] = static_cast<float>(F_reference(std::sqrt(u2)));
    }
    initialized_ = true;
}
