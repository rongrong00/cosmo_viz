#include "renderer/GridReader.h"
#include "common/HDF5IO.h"
#include <iostream>
#include <stdexcept>

const float* GridData::getField(const std::string& name) const {
    auto it = fields.find(name);
    if (it == fields.end())
        throw std::runtime_error("Grid field not found: " + name);
    return it->second;
}

bool GridData::hasField(const std::string& name) const {
    return fields.count(name) > 0;
}

static bool readVec3Attr(hid_t gid, const char* name, double out[3]) {
    if (H5Aexists(gid, name) <= 0) return false;
    hid_t aid = H5Aopen(gid, name, H5P_DEFAULT);
    hid_t space = H5Aget_space(aid);
    hsize_t dim;
    H5Sget_simple_extent_dims(space, &dim, nullptr);
    if (dim != 3) {
        H5Sclose(space); H5Aclose(aid); return false;
    }
    H5Aread(aid, H5T_NATIVE_DOUBLE, out);
    H5Sclose(space);
    H5Aclose(aid);
    return true;
}

static void fillMetadata(HDF5Reader& reader, GridData& g) {
    hid_t gid = H5Gopen2(reader.fileId(), "Header", H5P_DEFAULT);
    double vals[3];
    hid_t aid = H5Aopen(gid, "center", H5P_DEFAULT);
    H5Aread(aid, H5T_NATIVE_DOUBLE, vals);
    H5Aclose(aid);
    g.center = Vec3(vals[0], vals[1], vals[2]);

    double sz[3];
    if (readVec3Attr(gid, "size", sz)) {
        g.size = Vec3(sz[0], sz[1], sz[2]);
    } else {
        double side = reader.readAttrDouble("Header", "side");
        g.size = Vec3(side, side, side);
    }

    double shape[3];
    if (readVec3Attr(gid, "shape", shape)) {
        g.nx = (int)shape[0]; g.ny = (int)shape[1]; g.nz = (int)shape[2];
    } else {
        int n = reader.readAttrInt("Header", "resolution");
        g.nx = g.ny = g.nz = n;
    }

    double cs[3];
    if (readVec3Attr(gid, "cell_size", cs)) {
        g.cell_size = Vec3(cs[0], cs[1], cs[2]);
    } else {
        g.cell_size = Vec3(g.size.x / g.nx, g.size.y / g.ny, g.size.z / g.nz);
    }

    H5Gclose(gid);
    g.redshift = reader.readAttrDouble("Header", "redshift");
    g.bbox = Box3::fromCenterSize(g.center, g.size);
}

GridData GridReader::readHeader(const std::string& filename) {
    HDF5Reader reader(filename);
    GridData g;
    fillMetadata(reader, g);
    return g;
}

GridData GridReader::read(const std::string& filename,
                           const std::vector<std::string>& field_names) {
    HDF5Reader reader(filename);
    GridData g;
    fillMetadata(reader, g);

    std::vector<std::string> to_load = field_names;
    if (to_load.empty()) {
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

    size_t n = g.totalCells();
    for (const auto& name : to_load) {
        if (!reader.datasetExists(name)) {
            std::cerr << "Warning: field '" << name << "' not found, skipping." << std::endl;
            continue;
        }
        auto buf = std::unique_ptr<float[]>(new float[n]);
        reader.readDatasetFloatInto(name, buf.get(), n);
        g.fields[name] = buf.get();
        g.owned_.push_back(std::move(buf));
    }

    std::cout << "Read grid: " << g.nx << "x" << g.ny << "x" << g.nz
              << " size=(" << g.size.x << "," << g.size.y << "," << g.size.z << ")"
              << " fields=" << g.fields.size() << std::endl;

    return g;
}

void GridReader::readFieldsInto(const std::string& filename,
                                 const std::vector<std::string>& field_names,
                                 const std::map<std::string, float*>& out) {
    HDF5Reader reader(filename);
    GridData meta;
    fillMetadata(reader, meta);
    size_t n = meta.totalCells();

    for (const auto& name : field_names) {
        if (!reader.datasetExists(name)) {
            std::cerr << "Warning: field '" << name << "' not found, skipping." << std::endl;
            continue;
        }
        auto it = out.find(name);
        if (it == out.end())
            throw std::runtime_error("readFieldsInto: no buffer for " + name);
        reader.readDatasetFloatInto(name, it->second, n);
    }
}
