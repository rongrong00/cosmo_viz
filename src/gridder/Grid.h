#pragma once
#include <vector>
#include <string>
#include <map>
#include <memory>
#include "common/Vec3.h"
#include "common/Box3.h"

// Anisotropic grid: Nx x Ny x Nz cells over a box of size (sx, sy, sz).
class Grid {
public:
    // Self-allocated grid (one buffer per allocated field).
    Grid(const Vec3& center, const Vec3& size, int nx, int ny, int nz,
         const std::vector<std::string>& field_names);

    // Grid using caller-provided buffers (e.g. MPI shared memory). The
    // `buffers` map must contain entries for every name returned by
    // allocatedFieldNames(field_names), each of length totalCells() doubles.
    Grid(const Vec3& center, const Vec3& size, int nx, int ny, int nz,
         const std::vector<std::string>& field_names,
         const std::map<std::string, double*>& buffers);

    // Names of all buffers that must exist for the given requested field
    // list (expands gas_velocity -> _x/_y/_z and always appends mass_weight).
    static std::vector<std::string> allocatedFieldNames(
        const std::vector<std::string>& field_names);

    int nx() const { return nx_; }
    int ny() const { return ny_; }
    int nz() const { return nz_; }
    int resolution() const { return nx_; }

    const Vec3& cellSize() const { return cell_size_; }
    const Vec3& center() const { return center_; }
    const Vec3& size() const { return size_; }
    double side() const { return size_.x; }
    const Box3& bbox() const { return bbox_; }
    size_t totalCells() const {
        return static_cast<size_t>(nx_) * ny_ * nz_;
    }

    size_t index(int ix, int iy, int iz) const {
        return (static_cast<size_t>(iz) * ny_ + iy) * nx_ + ix;
    }

    bool hasField(const std::string& name) const;
    const std::vector<std::string>& fieldNames() const { return field_names_; }

    double* fieldData(const std::string& name);
    const double* fieldData(const std::string& name) const;

    void normalizeIntensiveFields();

    void writeHDF5(const std::string& filename, double redshift, double scale_factor,
                   int snapshot_num, double boxsize, double hubble_param,
                   double omega0, double omega_lambda) const;

private:
    Vec3 center_;
    Vec3 size_;
    int nx_, ny_, nz_;
    Vec3 cell_size_;
    Box3 bbox_;
    std::vector<std::string> field_names_;
    std::map<std::string, double*> fields_;
    // Only populated for self-allocated grids; empty when external buffers used.
    std::vector<std::unique_ptr<double[]>> owned_;
};
