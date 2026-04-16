#pragma once

// Cubic spline SPH kernel W(r, h) with compact support at r = 2h.
// Normalized in 3D: integral of W over all space = 1.
class Kernel {
public:
    // Evaluate W(r, h)
    static double W(double r, double h);

    // Kernel normalization constant in 3D: 1 / (pi * h^3)
    static double norm3D(double h);
};
