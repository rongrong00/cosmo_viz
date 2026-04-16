#pragma once
#include <string>
#include <vector>
#include "common/Vec3.h"
#include "common/Box3.h"

struct GridData {
    Vec3 center;
    double side;
    int resolution;
    double cell_size;
    double redshift;
    Box3 bbox;
    std::vector<float> gas_density;  // N^3 flat array, C-order
};

class GridReader {
public:
    static GridData read(const std::string& filename);
};
