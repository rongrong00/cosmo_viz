#include "sph_renderer/ParticleLoader.h"
#include "common/SnapshotReader.h"
#include "common/SmoothingLength.h"
#include <algorithm>
#include <cmath>
#include <iostream>
#include <vector>

// u → T in Kelvin using same prescription as the gridder's Depositor:
// gamma=5/3, mu from primordial X_H=0.76 and electron abundance xe.
static float internalEnergyToTemperature(float u, float xe) {
    constexpr double gamma = 5.0 / 3.0;
    constexpr double m_p   = 1.6726e-24;
    constexpr double k_B   = 1.3807e-16;
    constexpr double X_H   = 0.76;
    double mu = 4.0 / (1.0 + 3.0 * X_H + 4.0 * X_H * xe);
    return static_cast<float>((gamma - 1.0) * u * 1e10 * mu * m_p / k_B);
}

// Returns the minimum-image delta (p - center) wrapped into [-L/2, L/2] per
// axis. Used to test region membership and to place the particle in the
// region-local frame.
static inline Vec3 minImageDelta(const Vec3& p, const Vec3& c, double boxsize) {
    double half = 0.5 * boxsize;
    double dx = p.x - c.x, dy = p.y - c.y, dz = p.z - c.z;
    if (dx >  half) dx -= boxsize;
    if (dx < -half) dx += boxsize;
    if (dy >  half) dy -= boxsize;
    if (dy < -half) dy += boxsize;
    if (dz >  half) dz -= boxsize;
    if (dz < -half) dz += boxsize;
    return {dx, dy, dz};
}

ParticleStore ParticleLoader::load(const std::string& snapshot_path,
                                   const RegionConfig& region,
                                   const std::set<std::string>& gas_optional_fields,
                                   bool load_dm,
                                   double cull_margin_override,
                                   bool load_stars,
                                   int dm_knn_k,
                                   double dm_h_max,
                                   double dm_h_min) {
    SnapshotHeader hdr = SnapshotReader::readHeader(snapshot_path);
    int nfiles = SnapshotReader::getNumFiles(snapshot_path);

    ParticleStore ps;
    ps.region_center = region.center;
    ps.region_size   = region.size;

    const bool want_temp  = gas_optional_fields.count("temperature") > 0;
    const bool want_metal = gas_optional_fields.count("metallicity") > 0;
    const bool want_vel   = gas_optional_fields.count("velocity")    > 0;
    const bool want_hii   = gas_optional_fields.count("hii")         > 0;

    // Cull half-extents along each axis. We cull against size/2; later we
    // widen the BVH leaf-test AABBs with per-particle h. For DM we do a
    // second pass after kNN assigns h because at load time h is unknown.
    double margin = cull_margin_override >= 0.0 ? cull_margin_override : region.margin;
    double hx = 0.5 * region.size.x + margin;
    double hy = 0.5 * region.size.y + margin;
    double hz = 0.5 * region.size.z + margin;
    const bool spherical = region.radius > 0.0;
    const double R_base  = region.radius + margin;

    double x_min = 1e30, x_max = -1e30;
    double y_min = 1e30, y_max = -1e30;
    double z_min = 1e30, z_max = -1e30;
    float  gas_h_max = 0.0f;

    for (int f = 0; f < nfiles; ++f) {
        std::string subfile = SnapshotReader::subfilePath(snapshot_path, f);

        auto gas = SnapshotReader::readGasParticles(subfile, hdr.boxsize);
        for (const auto& g : gas) {
            Vec3 d = minImageDelta(g.pos, region.center, hdr.boxsize);
            // Conservative cull: add 2*hsml so we keep particles whose kernel
            // support intersects the region even if their center doesn't.
            if (spherical) {
                double Re = R_base + 2.0 * g.hsml;
                if (d.x * d.x + d.y * d.y + d.z * d.z > Re * Re) continue;
            } else {
                double margin_x = hx + 2.0 * g.hsml;
                double margin_y = hy + 2.0 * g.hsml;
                double margin_z = hz + 2.0 * g.hsml;
                if (std::fabs(d.x) > margin_x ||
                    std::fabs(d.y) > margin_y ||
                    std::fabs(d.z) > margin_z) continue;
            }

            float px = static_cast<float>(region.center.x + d.x);
            float py = static_cast<float>(region.center.y + d.y);
            float pz = static_cast<float>(region.center.z + d.z);

            ps.gas_x.push_back(px);
            ps.gas_y.push_back(py);
            ps.gas_z.push_back(pz);
            ps.gas_h.push_back(g.hsml);
            ps.gas_mass.push_back(g.mass);
            ps.gas_density.push_back(g.density);

            if (want_temp)  ps.gas_temperature.push_back(
                internalEnergyToTemperature(g.internal_energy, g.hii_fraction));
            if (want_metal) ps.gas_metallicity.push_back(g.metallicity);
            if (want_vel) {
                ps.gas_vx.push_back(g.velocity.x);
                ps.gas_vy.push_back(g.velocity.y);
                ps.gas_vz.push_back(g.velocity.z);
            }
            if (want_hii)   ps.gas_hii.push_back(g.hii_fraction);

            x_min = std::min(x_min, (double)px); x_max = std::max(x_max, (double)px);
            y_min = std::min(y_min, (double)py); y_max = std::max(y_max, (double)py);
            z_min = std::min(z_min, (double)pz); z_max = std::max(z_max, (double)pz);
            if (g.hsml > gas_h_max) gas_h_max = g.hsml;
        }

        if (load_dm || load_stars) {
            auto dm = load_stars
                ? SnapshotReader::readStarsAsDM(subfile, hdr.boxsize)
                : SnapshotReader::readDMParticles(subfile, hdr.boxsize);
            for (const auto& d_part : dm) {
                Vec3 d = minImageDelta(d_part.pos, region.center, hdr.boxsize);
                if (spherical) {
                    if (d.x * d.x + d.y * d.y + d.z * d.z > R_base * R_base) continue;
                } else {
                    if (std::fabs(d.x) > hx || std::fabs(d.y) > hy || std::fabs(d.z) > hz) continue;
                }

                ps.dm_x.push_back(static_cast<float>(region.center.x + d.x));
                ps.dm_y.push_back(static_cast<float>(region.center.y + d.y));
                ps.dm_z.push_back(static_cast<float>(region.center.z + d.z));
                ps.dm_mass.push_back(d_part.mass);
            }
        }

        std::cout << "  loaded subfile " << f << "/" << nfiles
                  << ": gas=" << ps.gas_x.size() << " dm=" << ps.dm_x.size() << std::endl;
    }

    if (!ps.gas_x.empty()) {
        ps.bbox_gas = Box3(Vec3(x_min, y_min, z_min), Vec3(x_max, y_max, z_max));
        ps.h_max_gas = gas_h_max;
    }

    if ((load_dm || load_stars) && !ps.dm_x.empty()) {
        std::vector<DMParticle> dm_aos(ps.dm_x.size());
        for (size_t i = 0; i < dm_aos.size(); ++i) {
            dm_aos[i].pos  = Vec3(ps.dm_x[i], ps.dm_y[i], ps.dm_z[i]);
            dm_aos[i].mass = ps.dm_mass[i];
            dm_aos[i].hsml = 0.0f;
        }
        SmoothingLength::computeKNN(dm_aos, hdr.boxsize, dm_knn_k, dm_h_max, dm_h_min);

        ps.dm_h.resize(dm_aos.size());
        double dxmn = 1e30, dymn = 1e30, dzmn = 1e30;
        double dxmx = -1e30, dymx = -1e30, dzmx = -1e30;
        float  dm_h_max = 0.0f;
        for (size_t i = 0; i < dm_aos.size(); ++i) {
            ps.dm_h[i] = dm_aos[i].hsml;
            if (dm_aos[i].hsml > dm_h_max) dm_h_max = dm_aos[i].hsml;
            dxmn = std::min(dxmn, (double)ps.dm_x[i]); dxmx = std::max(dxmx, (double)ps.dm_x[i]);
            dymn = std::min(dymn, (double)ps.dm_y[i]); dymx = std::max(dymx, (double)ps.dm_y[i]);
            dzmn = std::min(dzmn, (double)ps.dm_z[i]); dzmx = std::max(dzmx, (double)ps.dm_z[i]);
        }
        ps.bbox_dm = Box3(Vec3(dxmn, dymn, dzmn), Vec3(dxmx, dymx, dzmx));
        ps.h_max_dm = dm_h_max;
    }

    std::cout << "  Loaded gas=" << ps.numGas()
              << " (h_max=" << ps.h_max_gas << ")"
              << ", dm=" << ps.numDM()
              << " (h_max=" << ps.h_max_dm << ")" << std::endl;

    return ps;
}

// ---------------------------------------------------------------------------
// MPI-parallel load: each rank reads its own share of subfiles, then all
// ranks Allgatherv the culled survivors so every rank has the same store.
// ---------------------------------------------------------------------------

namespace {

// Read + cull from a single subfile into `ps` (owning vectors).
static void readAndCullSubfile(const std::string& subfile_path,
                               double boxsize,
                               const RegionConfig& region,
                               double hx, double hy, double hz,
                               bool spherical, double R_base,
                               bool want_temp, bool want_metal,
                               bool want_vel, bool want_hii,
                               bool load_dm,
                               bool load_stars,
                               ParticleStore& ps,
                               double& x_min, double& x_max,
                               double& y_min, double& y_max,
                               double& z_min, double& z_max,
                               float& gas_h_max)
{
    auto gas = SnapshotReader::readGasParticles(subfile_path, boxsize);
    for (const auto& g : gas) {
        Vec3 d = minImageDelta(g.pos, region.center, boxsize);
        if (spherical) {
            double Re = R_base + 2.0 * g.hsml;
            if (d.x * d.x + d.y * d.y + d.z * d.z > Re * Re) continue;
        } else {
            double mx = hx + 2.0 * g.hsml;
            double my = hy + 2.0 * g.hsml;
            double mz = hz + 2.0 * g.hsml;
            if (std::fabs(d.x) > mx || std::fabs(d.y) > my || std::fabs(d.z) > mz) continue;
        }
        float px = static_cast<float>(region.center.x + d.x);
        float py = static_cast<float>(region.center.y + d.y);
        float pz = static_cast<float>(region.center.z + d.z);
        ps.gas_x.push_back(px);
        ps.gas_y.push_back(py);
        ps.gas_z.push_back(pz);
        ps.gas_h.push_back(g.hsml);
        ps.gas_mass.push_back(g.mass);
        ps.gas_density.push_back(g.density);
        if (want_temp)  ps.gas_temperature.push_back(
            internalEnergyToTemperature(g.internal_energy, g.hii_fraction));
        if (want_metal) ps.gas_metallicity.push_back(g.metallicity);
        if (want_vel) {
            ps.gas_vx.push_back(g.velocity.x);
            ps.gas_vy.push_back(g.velocity.y);
            ps.gas_vz.push_back(g.velocity.z);
        }
        if (want_hii)   ps.gas_hii.push_back(g.hii_fraction);
        x_min = std::min(x_min, (double)px); x_max = std::max(x_max, (double)px);
        y_min = std::min(y_min, (double)py); y_max = std::max(y_max, (double)py);
        z_min = std::min(z_min, (double)pz); z_max = std::max(z_max, (double)pz);
        if (g.hsml > gas_h_max) gas_h_max = g.hsml;
    }

    if (load_dm || load_stars) {
        auto dm = load_stars
            ? SnapshotReader::readStarsAsDM(subfile_path, boxsize)
            : SnapshotReader::readDMParticles(subfile_path, boxsize);
        for (const auto& d_part : dm) {
            Vec3 d = minImageDelta(d_part.pos, region.center, boxsize);
            if (spherical) {
                if (d.x * d.x + d.y * d.y + d.z * d.z > R_base * R_base) continue;
            } else {
                if (std::fabs(d.x) > hx || std::fabs(d.y) > hy || std::fabs(d.z) > hz) continue;
            }
            ps.dm_x.push_back(static_cast<float>(region.center.x + d.x));
            ps.dm_y.push_back(static_cast<float>(region.center.y + d.y));
            ps.dm_z.push_back(static_cast<float>(region.center.z + d.z));
            ps.dm_mass.push_back(d_part.mass);
        }
    }
}

// Replace `arr` with the concatenation of every rank's `arr` via Allgatherv.
// All ranks end up with identical data. Only called if the field is populated
// on all ranks (i.e. requested).
static void allgatherFloat(ShmArray<float>& arr, MPI_Comm comm) {
    int size;
    MPI_Comm_size(comm, &size);
    int local_n = static_cast<int>(arr.size());
    std::vector<int> counts(size), displs(size);
    MPI_Allgather(&local_n, 1, MPI_INT, counts.data(), 1, MPI_INT, comm);
    int total = 0;
    for (int i = 0; i < size; ++i) { displs[i] = total; total += counts[i]; }
    std::vector<float> merged(total);
    MPI_Allgatherv(arr.data(), local_n, MPI_FLOAT,
                   merged.data(), counts.data(), displs.data(), MPI_FLOAT,
                   comm);
    arr.ownedVec() = std::move(merged);
}

} // namespace

ParticleStore ParticleLoader::loadMPI(const std::string& snapshot_path,
                                      const RegionConfig& region,
                                      const std::set<std::string>& gas_optional_fields,
                                      bool load_dm,
                                      MPI_Comm world_comm,
                                      double cull_margin_override,
                                      bool load_stars,
                                      int dm_knn_k,
                                      double dm_h_max,
                                      double dm_h_min) {
    int rank = 0, size = 1;
    MPI_Comm_rank(world_comm, &rank);
    MPI_Comm_size(world_comm, &size);

    SnapshotHeader hdr = SnapshotReader::readHeader(snapshot_path);
    int nfiles = SnapshotReader::getNumFiles(snapshot_path);

    ParticleStore ps;
    ps.region_center = region.center;
    ps.region_size   = region.size;

    const bool want_temp  = gas_optional_fields.count("temperature") > 0;
    const bool want_metal = gas_optional_fields.count("metallicity") > 0;
    const bool want_vel   = gas_optional_fields.count("velocity")    > 0;
    const bool want_hii   = gas_optional_fields.count("hii")         > 0;

    double margin = cull_margin_override >= 0.0 ? cull_margin_override : region.margin;
    double hx = 0.5 * region.size.x + margin;
    double hy = 0.5 * region.size.y + margin;
    double hz = 0.5 * region.size.z + margin;
    const bool spherical = region.radius > 0.0;
    const double R_base  = region.radius + margin;

    double x_min = 1e30, x_max = -1e30;
    double y_min = 1e30, y_max = -1e30;
    double z_min = 1e30, z_max = -1e30;
    float  gas_h_max = 0.0f;

    // Each rank reads round-robin-assigned subfiles.
    int my_count = 0;
    for (int f = rank; f < nfiles; f += size) {
        std::string subfile = SnapshotReader::subfilePath(snapshot_path, f);
        readAndCullSubfile(subfile, hdr.boxsize, region,
                           hx, hy, hz, spherical, R_base,
                           want_temp, want_metal, want_vel, want_hii,
                           load_dm, load_stars, ps,
                           x_min, x_max, y_min, y_max, z_min, z_max,
                           gas_h_max);
        ++my_count;
    }
    std::cout << "[rank " << rank << "] read " << my_count << " subfiles: "
              << "gas=" << ps.numGas() << " dm=" << ps.numDM() << std::endl;

    MPI_Barrier(world_comm);

    // Merge across world. Fields that weren't requested are empty everywhere
    // so we can skip them cleanly.
    allgatherFloat(ps.gas_x, world_comm);
    allgatherFloat(ps.gas_y, world_comm);
    allgatherFloat(ps.gas_z, world_comm);
    allgatherFloat(ps.gas_h, world_comm);
    allgatherFloat(ps.gas_mass, world_comm);
    allgatherFloat(ps.gas_density, world_comm);
    if (want_temp)  allgatherFloat(ps.gas_temperature, world_comm);
    if (want_metal) allgatherFloat(ps.gas_metallicity, world_comm);
    if (want_vel) {
        allgatherFloat(ps.gas_vx, world_comm);
        allgatherFloat(ps.gas_vy, world_comm);
        allgatherFloat(ps.gas_vz, world_comm);
    }
    if (want_hii)   allgatherFloat(ps.gas_hii, world_comm);
    if (load_dm || load_stars) {
        allgatherFloat(ps.dm_x,    world_comm);
        allgatherFloat(ps.dm_y,    world_comm);
        allgatherFloat(ps.dm_z,    world_comm);
        allgatherFloat(ps.dm_mass, world_comm);
    }

    // Aggregate bbox + h_max via reductions on the per-rank local values.
    MPI_Allreduce(MPI_IN_PLACE, &x_min, 1, MPI_DOUBLE, MPI_MIN, world_comm);
    MPI_Allreduce(MPI_IN_PLACE, &y_min, 1, MPI_DOUBLE, MPI_MIN, world_comm);
    MPI_Allreduce(MPI_IN_PLACE, &z_min, 1, MPI_DOUBLE, MPI_MIN, world_comm);
    MPI_Allreduce(MPI_IN_PLACE, &x_max, 1, MPI_DOUBLE, MPI_MAX, world_comm);
    MPI_Allreduce(MPI_IN_PLACE, &y_max, 1, MPI_DOUBLE, MPI_MAX, world_comm);
    MPI_Allreduce(MPI_IN_PLACE, &z_max, 1, MPI_DOUBLE, MPI_MAX, world_comm);
    MPI_Allreduce(MPI_IN_PLACE, &gas_h_max, 1, MPI_FLOAT, MPI_MAX, world_comm);

    if (!ps.gas_x.empty()) {
        ps.bbox_gas = Box3(Vec3(x_min, y_min, z_min), Vec3(x_max, y_max, z_max));
        ps.h_max_gas = gas_h_max;
    }

    // DM kNN: rank 0 runs it (OpenMP-parallel) and Bcasts results.
    if ((load_dm || load_stars) && !ps.dm_x.empty()) {
        size_t N = ps.dm_x.size();
        ps.dm_h.resize(N);
        if (rank == 0) {
            std::vector<DMParticle> dm_aos(N);
            for (size_t i = 0; i < N; ++i) {
                dm_aos[i].pos  = Vec3(ps.dm_x[i], ps.dm_y[i], ps.dm_z[i]);
                dm_aos[i].mass = ps.dm_mass[i];
                dm_aos[i].hsml = 0.0f;
            }
            SmoothingLength::computeKNN(dm_aos, hdr.boxsize, dm_knn_k, dm_h_max, dm_h_min);
            for (size_t i = 0; i < N; ++i) ps.dm_h[i] = dm_aos[i].hsml;
        }
        MPI_Bcast(ps.dm_h.data(), static_cast<int>(N), MPI_FLOAT, 0, world_comm);

        // DM bbox + h_max can now be computed on everyone uniformly.
        double dxmn = 1e30, dymn = 1e30, dzmn = 1e30;
        double dxmx = -1e30, dymx = -1e30, dzmx = -1e30;
        float dm_h_max = 0.0f;
        for (size_t i = 0; i < N; ++i) {
            double xi = ps.dm_x[i], yi = ps.dm_y[i], zi = ps.dm_z[i];
            if (xi < dxmn) dxmn = xi; if (xi > dxmx) dxmx = xi;
            if (yi < dymn) dymn = yi; if (yi > dymx) dymx = yi;
            if (zi < dzmn) dzmn = zi; if (zi > dzmx) dzmx = zi;
            if (ps.dm_h[i] > dm_h_max) dm_h_max = ps.dm_h[i];
        }
        ps.bbox_dm = Box3(Vec3(dxmn, dymn, dzmn), Vec3(dxmx, dymx, dzmx));
        ps.h_max_dm = dm_h_max;
    }

    if (rank == 0) {
        std::cout << "  [MPI load] gas=" << ps.numGas()
                  << " (h_max=" << ps.h_max_gas << ")"
                  << ", dm=" << ps.numDM()
                  << " (h_max=" << ps.h_max_dm << ") across "
                  << size << " ranks, " << nfiles << " subfiles"
                  << std::endl;
    }
    return ps;
}
