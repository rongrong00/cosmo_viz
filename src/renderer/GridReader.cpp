#include "renderer/GridReader.h"
#include "common/HDF5IO.h"
#include <iostream>
#include <stdexcept>

const std::vector<float>& GridData::getField(const std::string& name) const {
    auto it = fields.find(name);
    if (it == fields.end())
        throw std::runtime_error("Grid field not found: " + name);
    return it->second;
}

bool GridData::hasField(const std::string& name) const {
    return fields.count(name) > 0;
}

GridData GridReader::read(const std::string& filename,
                           const std::vector<std::string>& field_names) {
    HDF5Reader reader(filename);
    GridData g;

    // Read header attributes
    hid_t gid = H5Gopen2(reader.fileId(), "Header", H5P_DEFAULT);
    double vals[3];
    hid_t aid = H5Aopen(gid, "center", H5P_DEFAULT);
    H5Aread(aid, H5T_NATIVE_DOUBLE, vals);
    H5Aclose(aid);
    g.center = Vec3(vals[0], vals[1], vals[2]);
    H5Gclose(gid);

    g.side = reader.readAttrDouble("Header", "side");
    g.resolution = reader.readAttrInt("Header", "resolution");
    g.cell_size = reader.readAttrDouble("Header", "cell_size");
    g.redshift = reader.readAttrDouble("Header", "redshift");
    g.bbox = Box3::fromCenterSide(g.center, g.side);

    // Determine which fields to load
    std::vector<std::string> to_load = field_names;
    if (to_load.empty()) {
        // Load all datasets at root level
        hid_t fid = reader.fileId();
        hsize_t num_objs;
        H5Gget_num_objs(fid, &num_objs);
        for (hsize_t i = 0; i < num_objs; i++) {
            char name[256];
            H5Gget_objname_by_idx(fid, i, name, sizeof(name));
            if (H5Gget_objtype_by_idx(fid, i) == H5G_DATASET) {
                to_load.push_back(name);
            }
        }
    }

    for (const auto& name : to_load) {
        if (!reader.datasetExists(name)) {
            std::cerr << "Warning: field '" << name << "' not found in grid, skipping." << std::endl;
            continue;
        }
        g.fields[name] = reader.readDatasetFloat(name);
    }

    std::cout << "Read grid: res=" << g.resolution
              << " fields=" << g.fields.size() << std::endl;

    return g;
}
