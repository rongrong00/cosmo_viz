#include "gridder/Grid.h"
#include "common/HDF5IO.h"
#include <iostream>
#include <cmath>

Grid::Grid(const Vec3& center, double side, int resolution)
    : center_(center), side_(side), N_(resolution),
      cell_size_(side / resolution),
      bbox_(Box3::fromCenterSide(center, side)) {
    size_t total = static_cast<size_t>(N_) * N_ * N_;
    density_.resize(total, 0.0);
    std::cout << "Grid allocated: " << N_ << "^3 = " << total << " cells ("
              << (total * sizeof(double)) / (1024.0 * 1024.0) << " MB)" << std::endl;
}

void Grid::worldToCell(const Vec3& pos, int& ix, int& iy, int& iz) const {
    ix = static_cast<int>(std::floor((pos.x - bbox_.lo.x) / cell_size_));
    iy = static_cast<int>(std::floor((pos.y - bbox_.lo.y) / cell_size_));
    iz = static_cast<int>(std::floor((pos.z - bbox_.lo.z) / cell_size_));
}

void Grid::writeHDF5(const std::string& filename, double redshift, double scale_factor,
                      int snapshot_num, double boxsize, double hubble_param,
                      double omega0, double omega_lambda) const {
    HDF5Writer writer(filename);

    // Write header
    writer.createGroup("Header");
    writer.writeAttrDoubleArray("Header", "center", {center_.x, center_.y, center_.z});
    writer.writeAttrDouble("Header", "side", side_);
    writer.writeAttrInt("Header", "resolution", N_);
    writer.writeAttrDouble("Header", "cell_size", cell_size_);
    writer.writeAttrDouble("Header", "redshift", redshift);
    writer.writeAttrDouble("Header", "scale_factor", scale_factor);
    writer.writeAttrInt("Header", "snapshot", snapshot_num);
    writer.writeAttrDouble("Header", "boxsize", boxsize);
    writer.writeAttrDouble("Header", "hubble_param", hubble_param);
    writer.writeAttrDouble("Header", "omega0", omega0);
    writer.writeAttrDouble("Header", "omega_lambda", omega_lambda);

    // Convert double density to float32 for output
    size_t total = static_cast<size_t>(N_) * N_ * N_;
    std::vector<float> data_f32(total);
    for (size_t i = 0; i < total; i++) {
        data_f32[i] = static_cast<float>(density_[i]);
    }

    writer.writeDataset3D("gas_density", data_f32, N_, N_, N_);
    std::cout << "Wrote grid to " << filename << std::endl;
}
