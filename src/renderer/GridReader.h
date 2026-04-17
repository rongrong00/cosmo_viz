#pragma once
#include <string>
#include <vector>
#include <map>
#include <memory>
#include "common/Vec3.h"
#include "common/Box3.h"

struct GridData {
    Vec3 center;
    Vec3 size;
    int nx, ny, nz;
    Vec3 cell_size;
    double redshift;
    Box3 bbox;
    // Field pointers. Ownership lives in `owned_` (self-allocated reads)
    // or with the caller (batch / shared-memory path).
    std::map<std::string, const float*> fields;
    std::vector<std::unique_ptr<float[]>> owned_;

    double side() const { return size.x; }
    int resolution() const { return nx; }
    size_t totalCells() const { return (size_t)nx * ny * nz; }

    const float* getField(const std::string& name) const;
    bool hasField(const std::string& name) const;
};

class GridReader {
public:
    // Read everything: metadata + fields into self-allocated buffers.
    static GridData read(const std::string& filename,
                         const std::vector<std::string>& field_names = {});

    // Read only header/metadata (cheap; no dataset reads).
    static GridData readHeader(const std::string& filename);

    // Read the named fields into caller-provided float buffers
    // (one buffer per field, length = totalCells()). Used by the batch
    // renderer to load fields directly into MPI shared memory.
    static void readFieldsInto(const std::string& filename,
                               const std::vector<std::string>& field_names,
                               const std::map<std::string, float*>& out);
};
