#include "gridder/Grid.h"
#include "common/HDF5IO.h"
#include <iostream>
#include <cmath>
#include <stdexcept>
#include <cstring>

std::vector<std::string> Grid::allocatedFieldNames(
    const std::vector<std::string>& field_names) {
    std::vector<std::string> out;
    for (const auto& name : field_names) {
        if (name == "gas_velocity") {
            out.push_back("gas_velocity_x");
            out.push_back("gas_velocity_y");
            out.push_back("gas_velocity_z");
        } else {
            out.push_back(name);
        }
    }
    out.push_back("mass_weight");
    return out;
}

Grid::Grid(const Vec3& center, const Vec3& size, int nx, int ny, int nz,
           const std::vector<std::string>& field_names)
    : center_(center), size_(size), nx_(nx), ny_(ny), nz_(nz),
      cell_size_(size.x / nx, size.y / ny, size.z / nz),
      bbox_(Box3::fromCenterSize(center, size)),
      field_names_(field_names) {
    size_t total = totalCells();

    // Self-allocate one zeroed buffer per needed field.
    auto needed = allocatedFieldNames(field_names);
    owned_.reserve(needed.size());
    for (const auto& name : needed) {
        auto buf = std::unique_ptr<double[]>(new double[total]());
        fields_[name] = buf.get();
        owned_.push_back(std::move(buf));
    }

    size_t num_buffers = fields_.size();
    double mem_mb = (total * sizeof(double) * num_buffers) / (1024.0 * 1024.0);
    std::cout << "Grid allocated: " << nx_ << "x" << ny_ << "x" << nz_
              << " = " << total << " cells, "
              << num_buffers << " fields (" << mem_mb << " MB), "
              << "cell=(" << cell_size_.x << "," << cell_size_.y << "," << cell_size_.z << ")"
              << std::endl;
}

Grid::Grid(const Vec3& center, const Vec3& size, int nx, int ny, int nz,
           const std::vector<std::string>& field_names,
           const std::map<std::string, double*>& buffers)
    : center_(center), size_(size), nx_(nx), ny_(ny), nz_(nz),
      cell_size_(size.x / nx, size.y / ny, size.z / nz),
      bbox_(Box3::fromCenterSize(center, size)),
      field_names_(field_names) {
    for (const auto& name : allocatedFieldNames(field_names)) {
        auto it = buffers.find(name);
        if (it == buffers.end())
            throw std::runtime_error("external buffer missing for field: " + name);
        fields_[name] = it->second;
    }
}

bool Grid::hasField(const std::string& name) const {
    return fields_.count(name) > 0;
}

double* Grid::fieldData(const std::string& name) {
    auto it = fields_.find(name);
    if (it == fields_.end())
        throw std::runtime_error("Grid field not found: " + name);
    return it->second;
}

const double* Grid::fieldData(const std::string& name) const {
    auto it = fields_.find(name);
    if (it == fields_.end())
        throw std::runtime_error("Grid field not found: " + name);
    return it->second;
}

void Grid::normalizeIntensiveFields() {
    size_t total = totalCells();
    const double* mw = fieldData("mass_weight");

    std::vector<std::string> intensive = {
        "temperature", "metallicity",
        "gas_velocity_x", "gas_velocity_y", "gas_velocity_z"
    };

    for (const auto& fname : intensive) {
        if (!hasField(fname)) continue;
        double* f = fieldData(fname);
        for (size_t i = 0; i < total; i++) {
            if (mw[i] > 0.0) f[i] /= mw[i];
        }
    }
    std::cout << "Normalized intensive fields by mass_weight." << std::endl;
}

void Grid::writeHDF5(const std::string& filename, double redshift, double scale_factor,
                      int snapshot_num, double boxsize, double hubble_param,
                      double omega0, double omega_lambda) const {
    HDF5Writer writer(filename);

    writer.createGroup("Header");
    writer.writeAttrDoubleArray("Header", "center", {center_.x, center_.y, center_.z});
    writer.writeAttrDoubleArray("Header", "size", {size_.x, size_.y, size_.z});
    writer.writeAttrDouble("Header", "side", size_.x);
    writer.writeAttrDoubleArray("Header", "shape",
                                {(double)nx_, (double)ny_, (double)nz_});
    writer.writeAttrInt("Header", "resolution", nx_);
    writer.writeAttrDoubleArray("Header", "cell_size",
                                {cell_size_.x, cell_size_.y, cell_size_.z});
    writer.writeAttrDouble("Header", "redshift", redshift);
    writer.writeAttrDouble("Header", "scale_factor", scale_factor);
    writer.writeAttrInt("Header", "snapshot", snapshot_num);
    writer.writeAttrDouble("Header", "boxsize", boxsize);
    writer.writeAttrDouble("Header", "hubble_param", hubble_param);
    writer.writeAttrDouble("Header", "omega0", omega0);
    writer.writeAttrDouble("Header", "omega_lambda", omega_lambda);

    size_t total = totalCells();

    for (const auto& [name, ptr] : fields_) {
        if (name == "gas_velocity") continue;

        std::vector<float> data_f32(total);
        for (size_t i = 0; i < total; i++) {
            data_f32[i] = static_cast<float>(ptr[i]);
        }
        writer.writeDataset3D(name, data_f32, nz_, ny_, nx_);
    }

    std::cout << "Wrote " << fields_.size() << " fields to " << filename << std::endl;
}
