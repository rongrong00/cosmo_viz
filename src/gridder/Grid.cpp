#include "gridder/Grid.h"
#include "common/HDF5IO.h"
#include <iostream>
#include <cmath>
#include <stdexcept>

Grid::Grid(const Vec3& center, double side, int resolution,
           const std::vector<std::string>& field_names)
    : center_(center), side_(side), N_(resolution),
      cell_size_(side / resolution),
      bbox_(Box3::fromCenterSide(center, side)),
      field_names_(field_names) {
    size_t total = totalCells();

    // Always allocate mass_weight for normalizing intensive fields
    fields_["mass_weight"].resize(total, 0.0);

    for (const auto& name : field_names) {
        fields_[name].resize(total, 0.0);
        // Vector fields: also allocate _x, _y, _z
        if (name == "gas_velocity") {
            fields_["gas_velocity_x"].resize(total, 0.0);
            fields_["gas_velocity_y"].resize(total, 0.0);
            fields_["gas_velocity_z"].resize(total, 0.0);
        }
    }

    size_t num_buffers = fields_.size();
    double mem_mb = (total * sizeof(double) * num_buffers) / (1024.0 * 1024.0);
    std::cout << "Grid allocated: " << N_ << "^3 = " << total << " cells, "
              << num_buffers << " fields (" << mem_mb << " MB)" << std::endl;
}

bool Grid::hasField(const std::string& name) const {
    return fields_.count(name) > 0;
}

std::vector<double>& Grid::field(const std::string& name) {
    auto it = fields_.find(name);
    if (it == fields_.end())
        throw std::runtime_error("Grid field not found: " + name);
    return it->second;
}

const std::vector<double>& Grid::field(const std::string& name) const {
    auto it = fields_.find(name);
    if (it == fields_.end())
        throw std::runtime_error("Grid field not found: " + name);
    return it->second;
}

double* Grid::fieldData(const std::string& name) {
    return field(name).data();
}

void Grid::normalizeIntensiveFields() {
    size_t total = totalCells();
    const auto& mw = fields_["mass_weight"];

    // Intensive (mass-weighted) fields that need normalization
    std::vector<std::string> intensive = {
        "temperature", "metallicity",
        "gas_velocity_x", "gas_velocity_y", "gas_velocity_z"
    };

    for (const auto& fname : intensive) {
        if (!hasField(fname)) continue;
        auto& f = fields_[fname];
        for (size_t i = 0; i < total; i++) {
            if (mw[i] > 0.0) {
                f[i] /= mw[i];
            }
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

    size_t total = totalCells();

    // Write each field
    for (const auto& [name, data] : fields_) {
        // Skip gas_velocity parent entry (we write _x, _y, _z instead)
        if (name == "gas_velocity") continue;

        std::vector<float> data_f32(total);
        for (size_t i = 0; i < total; i++) {
            data_f32[i] = static_cast<float>(data[i]);
        }
        writer.writeDataset3D(name, data_f32, N_, N_, N_);
    }

    std::cout << "Wrote " << fields_.size() << " fields to " << filename << std::endl;
}
