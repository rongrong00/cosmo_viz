#include "renderer/GridReader.h"
#include "common/HDF5IO.h"
#include <iostream>

GridData GridReader::read(const std::string& filename) {
    HDF5Reader reader(filename);
    GridData g;

    // Read header attributes
    hid_t gid = H5Gopen2(reader.fileId(), "Header", H5P_DEFAULT);
    hid_t aid;
    double vals[3];

    aid = H5Aopen(gid, "center", H5P_DEFAULT);
    H5Aread(aid, H5T_NATIVE_DOUBLE, vals);
    H5Aclose(aid);
    g.center = Vec3(vals[0], vals[1], vals[2]);

    H5Gclose(gid);

    g.side = reader.readAttrDouble("Header", "side");
    g.resolution = reader.readAttrInt("Header", "resolution");
    g.cell_size = reader.readAttrDouble("Header", "cell_size");
    g.redshift = reader.readAttrDouble("Header", "redshift");

    g.bbox = Box3::fromCenterSide(g.center, g.side);

    // Read gas_density dataset
    g.gas_density = reader.readDatasetFloat("gas_density");

    std::cout << "Read grid: center=(" << g.center.x << "," << g.center.y << "," << g.center.z
              << ") side=" << g.side << " res=" << g.resolution
              << " cells=" << g.gas_density.size() << std::endl;

    return g;
}
