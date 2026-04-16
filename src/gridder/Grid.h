#pragma once
#include <vector>
#include <string>
#include "common/Vec3.h"
#include "common/Box3.h"

class Grid {
public:
    Grid(const Vec3& center, double side, int resolution);

    int resolution() const { return N_; }
    double cellSize() const { return cell_size_; }
    const Vec3& center() const { return center_; }
    double side() const { return side_; }
    const Box3& bbox() const { return bbox_; }

    // Get flat index for (ix, iy, iz) — C-order: iz * N*N + iy * N + ix
    size_t index(int ix, int iy, int iz) const {
        return static_cast<size_t>(iz) * N_ * N_ + static_cast<size_t>(iy) * N_ + ix;
    }

    // Convert world position to grid cell indices (can be out of bounds)
    void worldToCell(const Vec3& pos, int& ix, int& iy, int& iz) const;

    // Access the density accumulation buffer (float64 for deposition accuracy)
    std::vector<double>& densityBuf() { return density_; }
    const std::vector<double>& densityBuf() const { return density_; }

    // Write to HDF5
    void writeHDF5(const std::string& filename, double redshift, double scale_factor,
                   int snapshot_num, double boxsize, double hubble_param,
                   double omega0, double omega_lambda) const;

private:
    Vec3 center_;
    double side_;
    int N_;
    double cell_size_;
    Box3 bbox_;
    std::vector<double> density_; // accumulated gas density (double for precision)
};
