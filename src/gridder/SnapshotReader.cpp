#include "gridder/SnapshotReader.h"
#include "common/HDF5IO.h"
#include "common/Constants.h"
#include <cmath>
#include <iostream>
#include <sstream>
#include <filesystem>

namespace fs = std::filesystem;

// Determine the snapshot base path and naming convention.
// Supports:
//   /path/to/snapdir_011/snap_011   (multi-file)
//   /path/to/snap_011               (single-file)
//   /path/to/snap_011.hdf5          (single-file, explicit)

std::string SnapshotReader::subfilePath(const std::string& snapshot_path, int file_index) {
    // If snapshot_path ends with .hdf5, it's a single file
    if (snapshot_path.size() > 5 &&
        snapshot_path.substr(snapshot_path.size() - 5) == ".hdf5") {
        return snapshot_path;
    }

    // Try: snapshot_path.{file_index}.hdf5
    std::string path = snapshot_path + "." + std::to_string(file_index) + ".hdf5";
    if (fs::exists(path)) return path;

    // Try: snapdir convention
    // If snapshot_path is like /base/output/snapdir_011/snap_011
    // Already correct form, just append .K.hdf5
    return path; // let caller handle non-existence
}

int SnapshotReader::getNumFiles(const std::string& snapshot_path) {
    // Open the first subfile and read NumFilesPerSnapshot
    std::string first;
    if (snapshot_path.size() > 5 &&
        snapshot_path.substr(snapshot_path.size() - 5) == ".hdf5") {
        first = snapshot_path;
    } else {
        first = snapshot_path + ".0.hdf5";
    }

    HDF5Reader reader(first);
    return reader.readAttrInt("Header", "NumFilesPerSnapshot");
}

SnapshotHeader SnapshotReader::readHeader(const std::string& snapshot_path) {
    std::string first;
    if (snapshot_path.size() > 5 &&
        snapshot_path.substr(snapshot_path.size() - 5) == ".hdf5") {
        first = snapshot_path;
    } else {
        first = snapshot_path + ".0.hdf5";
    }

    HDF5Reader reader(first);
    SnapshotHeader h;
    h.boxsize = reader.readAttrDouble("Header", "BoxSize");
    h.redshift = reader.readAttrDouble("Header", "Redshift");
    h.time = reader.readAttrDouble("Header", "Time");
    h.hubble_param = reader.readAttrDouble("Parameters", "HubbleParam");
    h.omega0 = reader.readAttrDouble("Parameters", "Omega0");
    h.omega_lambda = reader.readAttrDouble("Parameters", "OmegaLambda");
    h.num_files = reader.readAttrInt("Header", "NumFilesPerSnapshot");
    h.num_part_total = reader.readAttrUint64Array("Header", "NumPart_Total");
    return h;
}

std::vector<GasParticle> SnapshotReader::readGasParticles(const std::string& subfile_path,
                                                           double boxsize) {
    HDF5Reader reader(subfile_path);

    // Read NumPart_ThisFile to get gas count
    auto npart = reader.readAttrUint64Array("Header", "NumPart_ThisFile");
    uint64_t ngas = npart[0];
    if (ngas == 0) return {};

    std::vector<GasParticle> particles(ngas);

    // Read coordinates — could be IntCoordinates (uint32) or Coordinates (float/double)
    bool has_int_coords = reader.datasetExists("PartType0/IntCoordinates");

    if (has_int_coords) {
        auto icoords = reader.readDatasetUint32("PartType0/IntCoordinates");
        for (uint64_t i = 0; i < ngas; i++) {
            particles[i].pos.x = static_cast<double>(icoords[i * 3 + 0]) * Constants::INTCOORD_SCALE * boxsize;
            particles[i].pos.y = static_cast<double>(icoords[i * 3 + 1]) * Constants::INTCOORD_SCALE * boxsize;
            particles[i].pos.z = static_cast<double>(icoords[i * 3 + 2]) * Constants::INTCOORD_SCALE * boxsize;
        }
    } else {
        // Try float Coordinates
        auto coords = reader.readDatasetFloat("PartType0/Coordinates");
        for (uint64_t i = 0; i < ngas; i++) {
            particles[i].pos.x = coords[i * 3 + 0];
            particles[i].pos.y = coords[i * 3 + 1];
            particles[i].pos.z = coords[i * 3 + 2];
        }
    }

    // Read masses
    auto masses = reader.readDatasetFloat("PartType0/Masses");
    for (uint64_t i = 0; i < ngas; i++) {
        particles[i].mass = masses[i];
    }

    // Read density
    auto density = reader.readDatasetFloat("PartType0/Density");
    for (uint64_t i = 0; i < ngas; i++) {
        particles[i].density = density[i];
    }

    // Compute smoothing length from Volume = Mass/Density
    // h = (3 * V / (4 * pi))^(1/3)
    for (uint64_t i = 0; i < ngas; i++) {
        double vol = particles[i].mass / particles[i].density;
        particles[i].hsml = std::cbrt(3.0 * vol / (4.0 * Constants::PI));
    }

    return particles;
}
