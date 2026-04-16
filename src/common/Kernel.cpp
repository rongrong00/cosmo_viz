#include "common/Kernel.h"
#include "common/Constants.h"

double Kernel::norm3D(double h) {
    return 1.0 / (Constants::PI * h * h * h);
}

double Kernel::W(double r, double h) {
    double q = r / h;
    if (q >= 2.0) return 0.0;

    double sigma = norm3D(h);

    if (q <= 1.0) {
        // W = sigma * (1 - 3/2 q^2 + 3/4 q^3)
        return sigma * (1.0 - 1.5 * q * q + 0.75 * q * q * q);
    } else {
        // W = sigma * 1/4 * (2 - q)^3
        double t = 2.0 - q;
        return sigma * 0.25 * t * t * t;
    }
}
