#pragma once
#include <string>
#include <vector>
#include <map>
#include "common/Vec3.h"
#include "common/Box3.h"

struct GridData {
    Vec3 center;
    double side;
    int resolution;
    double cell_size;
    double redshift;
    Box3 bbox;
    std::map<std::string, std::vector<float>> fields;  // name -> N^3 flat array

    const std::vector<float>& getField(const std::string& name) const;
    bool hasField(const std::string& name) const;
};

class GridReader {
public:
    // Read grid, loading only the specified fields (or all if empty)
    static GridData read(const std::string& filename,
                         const std::vector<std::string>& field_names = {});
};
