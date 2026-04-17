#pragma once
#include <hdf5.h>
#include <string>
#include <vector>
#include <stdexcept>
#include <cstdint>

class HDF5Reader {
public:
    explicit HDF5Reader(const std::string& filename);
    ~HDF5Reader();

    // Read a scalar attribute from a group
    double readAttrDouble(const std::string& group, const std::string& name);
    int readAttrInt(const std::string& group, const std::string& name);
    std::vector<uint64_t> readAttrUint64Array(const std::string& group, const std::string& name);

    // Read a full dataset
    std::vector<float> readDatasetFloat(const std::string& name);
    std::vector<double> readDatasetDouble(const std::string& name);
    std::vector<uint32_t> readDatasetUint32(const std::string& name);

    // Read a float dataset into a pre-allocated buffer (for shared-memory use).
    // Throws if the dataset element count != expected_n.
    void readDatasetFloatInto(const std::string& name, float* out, size_t expected_n);

    // Get dataset dimensions
    std::vector<hsize_t> getDatasetDims(const std::string& name);

    // Check if dataset exists
    bool datasetExists(const std::string& name);

    hid_t fileId() const { return file_id_; }

private:
    hid_t file_id_;
};

class HDF5Writer {
public:
    explicit HDF5Writer(const std::string& filename);
    ~HDF5Writer();

    // Create a group
    void createGroup(const std::string& name);

    // Write attributes to a group
    void writeAttrDouble(const std::string& group, const std::string& name, double value);
    void writeAttrInt(const std::string& group, const std::string& name, int value);
    void writeAttrString(const std::string& group, const std::string& name, const std::string& value);
    void writeAttrDoubleArray(const std::string& group, const std::string& name,
                              const std::vector<double>& values);

    // Write datasets
    void writeDataset3D(const std::string& name, const std::vector<float>& data,
                        int nx, int ny, int nz);
    void writeDataset2D(const std::string& name, const std::vector<float>& data,
                        int ny, int nx);

    hid_t fileId() const { return file_id_; }

private:
    hid_t file_id_;
};
