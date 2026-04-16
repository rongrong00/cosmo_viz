#include "common/HDF5IO.h"
#include <iostream>

// --- HDF5Reader ---

HDF5Reader::HDF5Reader(const std::string& filename) {
    file_id_ = H5Fopen(filename.c_str(), H5F_ACC_RDONLY, H5P_DEFAULT);
    if (file_id_ < 0)
        throw std::runtime_error("Cannot open HDF5 file: " + filename);
}

HDF5Reader::~HDF5Reader() {
    if (file_id_ >= 0) H5Fclose(file_id_);
}

double HDF5Reader::readAttrDouble(const std::string& group, const std::string& name) {
    hid_t gid = H5Gopen2(file_id_, group.c_str(), H5P_DEFAULT);
    if (gid < 0) throw std::runtime_error("Cannot open group: " + group);
    hid_t aid = H5Aopen(gid, name.c_str(), H5P_DEFAULT);
    if (aid < 0) { H5Gclose(gid); throw std::runtime_error("Cannot open attr: " + name); }
    double val;
    H5Aread(aid, H5T_NATIVE_DOUBLE, &val);
    H5Aclose(aid);
    H5Gclose(gid);
    return val;
}

int HDF5Reader::readAttrInt(const std::string& group, const std::string& name) {
    hid_t gid = H5Gopen2(file_id_, group.c_str(), H5P_DEFAULT);
    if (gid < 0) throw std::runtime_error("Cannot open group: " + group);
    hid_t aid = H5Aopen(gid, name.c_str(), H5P_DEFAULT);
    if (aid < 0) { H5Gclose(gid); throw std::runtime_error("Cannot open attr: " + name); }
    int val;
    H5Aread(aid, H5T_NATIVE_INT, &val);
    H5Aclose(aid);
    H5Gclose(gid);
    return val;
}

std::vector<uint64_t> HDF5Reader::readAttrUint64Array(const std::string& group, const std::string& name) {
    hid_t gid = H5Gopen2(file_id_, group.c_str(), H5P_DEFAULT);
    if (gid < 0) throw std::runtime_error("Cannot open group: " + group);
    hid_t aid = H5Aopen(gid, name.c_str(), H5P_DEFAULT);
    if (aid < 0) { H5Gclose(gid); throw std::runtime_error("Cannot open attr: " + name); }

    hid_t space = H5Aget_space(aid);
    int ndims = H5Sget_simple_extent_ndims(space);
    std::vector<hsize_t> dims(ndims);
    H5Sget_simple_extent_dims(space, dims.data(), nullptr);
    hsize_t total = 1;
    for (auto d : dims) total *= d;

    std::vector<uint64_t> vals(total);
    H5Aread(aid, H5T_NATIVE_UINT64, vals.data());
    H5Sclose(space);
    H5Aclose(aid);
    H5Gclose(gid);
    return vals;
}

std::vector<float> HDF5Reader::readDatasetFloat(const std::string& name) {
    hid_t did = H5Dopen2(file_id_, name.c_str(), H5P_DEFAULT);
    if (did < 0) throw std::runtime_error("Cannot open dataset: " + name);
    hid_t space = H5Dget_space(did);
    int ndims = H5Sget_simple_extent_ndims(space);
    std::vector<hsize_t> dims(ndims);
    H5Sget_simple_extent_dims(space, dims.data(), nullptr);
    hsize_t total = 1;
    for (auto d : dims) total *= d;

    std::vector<float> data(total);
    H5Dread(did, H5T_NATIVE_FLOAT, H5S_ALL, H5S_ALL, H5P_DEFAULT, data.data());
    H5Sclose(space);
    H5Dclose(did);
    return data;
}

std::vector<double> HDF5Reader::readDatasetDouble(const std::string& name) {
    hid_t did = H5Dopen2(file_id_, name.c_str(), H5P_DEFAULT);
    if (did < 0) throw std::runtime_error("Cannot open dataset: " + name);
    hid_t space = H5Dget_space(did);
    int ndims = H5Sget_simple_extent_ndims(space);
    std::vector<hsize_t> dims(ndims);
    H5Sget_simple_extent_dims(space, dims.data(), nullptr);
    hsize_t total = 1;
    for (auto d : dims) total *= d;

    std::vector<double> data(total);
    H5Dread(did, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, data.data());
    H5Sclose(space);
    H5Dclose(did);
    return data;
}

std::vector<uint32_t> HDF5Reader::readDatasetUint32(const std::string& name) {
    hid_t did = H5Dopen2(file_id_, name.c_str(), H5P_DEFAULT);
    if (did < 0) throw std::runtime_error("Cannot open dataset: " + name);
    hid_t space = H5Dget_space(did);
    int ndims = H5Sget_simple_extent_ndims(space);
    std::vector<hsize_t> dims(ndims);
    H5Sget_simple_extent_dims(space, dims.data(), nullptr);
    hsize_t total = 1;
    for (auto d : dims) total *= d;

    std::vector<uint32_t> data(total);
    H5Dread(did, H5T_NATIVE_UINT32, H5S_ALL, H5S_ALL, H5P_DEFAULT, data.data());
    H5Sclose(space);
    H5Dclose(did);
    return data;
}

std::vector<hsize_t> HDF5Reader::getDatasetDims(const std::string& name) {
    hid_t did = H5Dopen2(file_id_, name.c_str(), H5P_DEFAULT);
    if (did < 0) throw std::runtime_error("Cannot open dataset: " + name);
    hid_t space = H5Dget_space(did);
    int ndims = H5Sget_simple_extent_ndims(space);
    std::vector<hsize_t> dims(ndims);
    H5Sget_simple_extent_dims(space, dims.data(), nullptr);
    H5Sclose(space);
    H5Dclose(did);
    return dims;
}

bool HDF5Reader::datasetExists(const std::string& name) {
    return H5Lexists(file_id_, name.c_str(), H5P_DEFAULT) > 0;
}

// --- HDF5Writer ---

HDF5Writer::HDF5Writer(const std::string& filename) {
    file_id_ = H5Fcreate(filename.c_str(), H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT);
    if (file_id_ < 0)
        throw std::runtime_error("Cannot create HDF5 file: " + filename);
}

HDF5Writer::~HDF5Writer() {
    if (file_id_ >= 0) H5Fclose(file_id_);
}

void HDF5Writer::createGroup(const std::string& name) {
    hid_t gid = H5Gcreate2(file_id_, name.c_str(), H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    if (gid < 0) throw std::runtime_error("Cannot create group: " + name);
    H5Gclose(gid);
}

void HDF5Writer::writeAttrDouble(const std::string& group, const std::string& name, double value) {
    hid_t gid = H5Gopen2(file_id_, group.c_str(), H5P_DEFAULT);
    hid_t space = H5Screate(H5S_SCALAR);
    hid_t aid = H5Acreate2(gid, name.c_str(), H5T_IEEE_F64LE, space, H5P_DEFAULT, H5P_DEFAULT);
    H5Awrite(aid, H5T_NATIVE_DOUBLE, &value);
    H5Aclose(aid);
    H5Sclose(space);
    H5Gclose(gid);
}

void HDF5Writer::writeAttrInt(const std::string& group, const std::string& name, int value) {
    hid_t gid = H5Gopen2(file_id_, group.c_str(), H5P_DEFAULT);
    hid_t space = H5Screate(H5S_SCALAR);
    hid_t aid = H5Acreate2(gid, name.c_str(), H5T_STD_I32LE, space, H5P_DEFAULT, H5P_DEFAULT);
    H5Awrite(aid, H5T_NATIVE_INT, &value);
    H5Aclose(aid);
    H5Sclose(space);
    H5Gclose(gid);
}

void HDF5Writer::writeAttrString(const std::string& group, const std::string& name, const std::string& value) {
    hid_t gid = H5Gopen2(file_id_, group.c_str(), H5P_DEFAULT);
    hid_t strtype = H5Tcopy(H5T_C_S1);
    H5Tset_size(strtype, value.size() + 1);
    hid_t space = H5Screate(H5S_SCALAR);
    hid_t aid = H5Acreate2(gid, name.c_str(), strtype, space, H5P_DEFAULT, H5P_DEFAULT);
    H5Awrite(aid, strtype, value.c_str());
    H5Aclose(aid);
    H5Sclose(space);
    H5Tclose(strtype);
    H5Gclose(gid);
}

void HDF5Writer::writeAttrDoubleArray(const std::string& group, const std::string& name,
                                       const std::vector<double>& values) {
    hid_t gid = H5Gopen2(file_id_, group.c_str(), H5P_DEFAULT);
    hsize_t dim = values.size();
    hid_t space = H5Screate_simple(1, &dim, nullptr);
    hid_t aid = H5Acreate2(gid, name.c_str(), H5T_IEEE_F64LE, space, H5P_DEFAULT, H5P_DEFAULT);
    H5Awrite(aid, H5T_NATIVE_DOUBLE, values.data());
    H5Aclose(aid);
    H5Sclose(space);
    H5Gclose(gid);
}

void HDF5Writer::writeDataset3D(const std::string& name, const std::vector<float>& data,
                                  int nz, int ny, int nx) {
    hsize_t dims[3] = {(hsize_t)nz, (hsize_t)ny, (hsize_t)nx};
    hid_t space = H5Screate_simple(3, dims, nullptr);
    hid_t did = H5Dcreate2(file_id_, name.c_str(), H5T_IEEE_F32LE, space,
                            H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    H5Dwrite(did, H5T_NATIVE_FLOAT, H5S_ALL, H5S_ALL, H5P_DEFAULT, data.data());
    H5Dclose(did);
    H5Sclose(space);
}

void HDF5Writer::writeDataset2D(const std::string& name, const std::vector<float>& data,
                                  int ny, int nx) {
    hsize_t dims[2] = {(hsize_t)ny, (hsize_t)nx};
    hid_t space = H5Screate_simple(2, dims, nullptr);
    hid_t did = H5Dcreate2(file_id_, name.c_str(), H5T_IEEE_F32LE, space,
                            H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    H5Dwrite(did, H5T_NATIVE_FLOAT, H5S_ALL, H5S_ALL, H5P_DEFAULT, data.data());
    H5Dclose(did);
    H5Sclose(space);
}
