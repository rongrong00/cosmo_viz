#pragma once
#include <cmath>

namespace Constants {
    constexpr double PI = 3.14159265358979323846;
    // Arepo internal units: UnitLength = 1 kpc/h, UnitMass = 10^10 Msun/h
    constexpr double INTCOORD_SCALE = 1.0 / 4294967296.0; // 1 / 2^32
}
