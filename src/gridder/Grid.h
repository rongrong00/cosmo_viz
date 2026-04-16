#pragma once
#include <vector>
#include <string>
#include <map>
#include "common/Vec3.h"
#include "common/Box3.h"

class Grid {
public:
    Grid(const Vec3& center, double side, int resolution,
         const std::vector<std::string>& field_names);

    int resolution() const { return N_; }
    double cellSize() const { return cell_size_; }
    const Vec3& center() const { return center_; }
    double side() const { return side_; }
    const Box3& bbox() const { return bbox_; }
    size_t totalCells() const { return static_cast<size_t>(N_) * N_ * N_; }

    size_t index(int ix, int iy, int iz) const {
        return static_cast<size_t>(iz) * N_ * N_ + static_cast<size_t>(iy) * N_ + ix;
    }

    // Field access
    bool hasField(const std::string& name) const;
    std::vector<double>& field(const std::string& name);
    const std::vector<double>& field(const std::string& name) const;
    const std::vector<std::string>& fieldNames() const { return field_names_; }

    // Get raw pointer to field data (for MPI_Reduce)
    double* fieldData(const std::string& name);

    // Normalize intensive fields by mass_weight after MPI_Reduce
    void normalizeIntensiveFields();

    // Write all fields to HDF5
    void writeHDF5(const std::string& filename, double redshift, double scale_factor,
                   int snapshot_num, double boxsize, double hubble_param,
                   double omega0, double omega_lambda) const;

private:
    Vec3 center_;
    double side_;
    int N_;
    double cell_size_;
    Box3 bbox_;
    std::vector<std::string> field_names_;
    std::map<std::string, std::vector<double>> fields_;
};
