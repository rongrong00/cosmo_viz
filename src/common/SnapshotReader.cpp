#include "common/SnapshotReader.h"
#include "common/HDF5IO.h"
#include "common/Constants.h"
#include <cmath>
#include <iostream>
#include <filesystem>

namespace fs = std::filesystem;

std::string SnapshotReader::subfilePath(const std::string& snapshot_path, int file_index) {
    if (snapshot_path.size() > 5 &&
        snapshot_path.substr(snapshot_path.size() - 5) == ".hdf5") {
        return snapshot_path;
    }
    return snapshot_path + "." + std::to_string(file_index) + ".hdf5";
}

int SnapshotReader::getNumFiles(const std::string& snapshot_path) {
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

    auto npart = reader.readAttrUint64Array("Header", "NumPart_ThisFile");
    uint64_t ngas = npart[0];
    if (ngas == 0) return {};

    std::vector<GasParticle> particles(ngas);

    // Read coordinates
    bool has_int_coords = reader.datasetExists("PartType0/IntCoordinates");
    if (has_int_coords) {
        auto icoords = reader.readDatasetUint32("PartType0/IntCoordinates");
        for (uint64_t i = 0; i < ngas; i++) {
            particles[i].pos.x = static_cast<double>(icoords[i * 3 + 0]) * Constants::INTCOORD_SCALE * boxsize;
            particles[i].pos.y = static_cast<double>(icoords[i * 3 + 1]) * Constants::INTCOORD_SCALE * boxsize;
            particles[i].pos.z = static_cast<double>(icoords[i * 3 + 2]) * Constants::INTCOORD_SCALE * boxsize;
        }
    } else {
        auto coords = reader.readDatasetFloat("PartType0/Coordinates");
        for (uint64_t i = 0; i < ngas; i++) {
            particles[i].pos.x = coords[i * 3 + 0];
            particles[i].pos.y = coords[i * 3 + 1];
            particles[i].pos.z = coords[i * 3 + 2];
        }
    }

    // Masses
    auto masses = reader.readDatasetFloat("PartType0/Masses");
    for (uint64_t i = 0; i < ngas; i++) particles[i].mass = masses[i];

    // Density
    auto density = reader.readDatasetFloat("PartType0/Density");
    for (uint64_t i = 0; i < ngas; i++) particles[i].density = density[i];

    // Smoothing length from Volume = Mass/Density
    for (uint64_t i = 0; i < ngas; i++) {
        double vol = particles[i].mass / particles[i].density;
        particles[i].hsml = std::cbrt(3.0 * vol / (4.0 * Constants::PI));
    }

    // InternalEnergy
    auto ie = reader.readDatasetFloat("PartType0/InternalEnergy");
    for (uint64_t i = 0; i < ngas; i++) particles[i].internal_energy = ie[i];

    // Metallicity
    if (reader.datasetExists("PartType0/GFM_Metallicity")) {
        auto met = reader.readDatasetFloat("PartType0/GFM_Metallicity");
        for (uint64_t i = 0; i < ngas; i++) particles[i].metallicity = met[i];
    } else {
        for (uint64_t i = 0; i < ngas; i++) particles[i].metallicity = 0.0f;
    }

    // HII fraction
    if (reader.datasetExists("PartType0/HII_Fraction")) {
        auto hii = reader.readDatasetFloat("PartType0/HII_Fraction");
        for (uint64_t i = 0; i < ngas; i++) particles[i].hii_fraction = hii[i];
    } else {
        for (uint64_t i = 0; i < ngas; i++) particles[i].hii_fraction = 0.0f;
    }

    // Velocities
    auto vel = reader.readDatasetFloat("PartType0/Velocities");
    for (uint64_t i = 0; i < ngas; i++) {
        particles[i].velocity.x = vel[i * 3 + 0];
        particles[i].velocity.y = vel[i * 3 + 1];
        particles[i].velocity.z = vel[i * 3 + 2];
    }

    return particles;
}

std::vector<DMParticle> SnapshotReader::readDMParticles(const std::string& subfile_path,
                                                         double boxsize) {
    HDF5Reader reader(subfile_path);

    auto npart = reader.readAttrUint64Array("Header", "NumPart_ThisFile");
    uint64_t ndm = npart[1];
    if (ndm == 0) return {};

    std::vector<DMParticle> particles(ndm);

    // Read coordinates
    bool has_int_coords = reader.datasetExists("PartType1/IntCoordinates");
    if (has_int_coords) {
        auto icoords = reader.readDatasetUint32("PartType1/IntCoordinates");
        for (uint64_t i = 0; i < ndm; i++) {
            particles[i].pos.x = static_cast<double>(icoords[i * 3 + 0]) * Constants::INTCOORD_SCALE * boxsize;
            particles[i].pos.y = static_cast<double>(icoords[i * 3 + 1]) * Constants::INTCOORD_SCALE * boxsize;
            particles[i].pos.z = static_cast<double>(icoords[i * 3 + 2]) * Constants::INTCOORD_SCALE * boxsize;
        }
    } else {
        auto coords = reader.readDatasetFloat("PartType1/Coordinates");
        for (uint64_t i = 0; i < ndm; i++) {
            particles[i].pos.x = coords[i * 3 + 0];
            particles[i].pos.y = coords[i * 3 + 1];
            particles[i].pos.z = coords[i * 3 + 2];
        }
    }

    // Masses — check MassTable first
    if (reader.datasetExists("PartType1/Masses")) {
        auto masses = reader.readDatasetFloat("PartType1/Masses");
        for (uint64_t i = 0; i < ndm; i++) particles[i].mass = masses[i];
    } else {
        // Use MassTable[1]
        // For now, set to 1.0 — will be overridden if MassTable is available
        for (uint64_t i = 0; i < ndm; i++) particles[i].mass = 1.0f;
    }

    // hsml will be set later via kNN
    for (uint64_t i = 0; i < ndm; i++) particles[i].hsml = 0.0f;

    return particles;
}

std::vector<DMParticle> SnapshotReader::readStarsAsDM(const std::string& subfile_path,
                                                      double boxsize) {
    HDF5Reader reader(subfile_path);

    auto npart = reader.readAttrUint64Array("Header", "NumPart_ThisFile");
    uint64_t nstar = npart.size() > 4 ? npart[4] : 0;
    if (nstar == 0) return {};

    std::vector<DMParticle> particles(nstar);

    bool has_int_coords = reader.datasetExists("PartType4/IntCoordinates");
    if (has_int_coords) {
        auto icoords = reader.readDatasetUint32("PartType4/IntCoordinates");
        for (uint64_t i = 0; i < nstar; i++) {
            particles[i].pos.x = static_cast<double>(icoords[i * 3 + 0]) *
                                 Constants::INTCOORD_SCALE * boxsize;
            particles[i].pos.y = static_cast<double>(icoords[i * 3 + 1]) *
                                 Constants::INTCOORD_SCALE * boxsize;
            particles[i].pos.z = static_cast<double>(icoords[i * 3 + 2]) *
                                 Constants::INTCOORD_SCALE * boxsize;
        }
    } else {
        auto coords = reader.readDatasetFloat("PartType4/Coordinates");
        for (uint64_t i = 0; i < nstar; i++) {
            particles[i].pos.x = coords[i * 3 + 0];
            particles[i].pos.y = coords[i * 3 + 1];
            particles[i].pos.z = coords[i * 3 + 2];
        }
    }

    // PartType4 Masses are always per-particle (variable mass).
    if (reader.datasetExists("PartType4/Masses")) {
        auto masses = reader.readDatasetFloat("PartType4/Masses");
        for (uint64_t i = 0; i < nstar; i++) particles[i].mass = masses[i];
    } else {
        for (uint64_t i = 0; i < nstar; i++) particles[i].mass = 0.0f;
    }

    for (uint64_t i = 0; i < nstar; i++) particles[i].hsml = 0.0f;
    return particles;
}
