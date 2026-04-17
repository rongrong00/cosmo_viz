# Cosmological Simulation Visualization Pipeline — Technical Specification

## Overview

Build a two-stage pipeline for rendering images and movies from large cosmological simulations (Arepo/GADGET format: Lumina, Thesan-XL, IllustrisTNG).

- **Stage 1 — `gridder` (C++/MPI):** Reads raw HDF5 snapshots, selects spatial regions, deposits particle/cell data onto Cartesian grids using an SPH cubic spline kernel, writes gridded HDF5 cubes.
- **Stage 2 — `renderer` (C++):** Reads gridded cubes, ray-traces through them for a given camera configuration, writes 2D projection maps as HDF5.
- **Stage 3 — `plotter` (Python):** Reads 2D maps, applies colormaps/compositing, outputs final PNG/PDF figures. A Python orchestrator script ties everything together.

The separation means you pay the expensive snapshot I/O cost once, then render many views cheaply.

---

## Recent Updates (April 16, 2026)

### Zoom Movie Workflow

- Added a deep 16:9 zoom workflow with more tiers and more frames.
- Added automatic lead-in trimming so output starts at the first frame where the 16:9 view is fully covered.
- Added full-bleed plotting mode so rendered PNG frames fill the canvas (no dark rim from figure margins).
- Added depth-mode control for LOS slab:
  - `fixed`: use a constant slab (`--los-slab`).
  - `same-as-y`: set z depth equal to per-frame orthographic y extent.

### Updated Files

- Main zoom generator: `python/zoom_frames.py`
  - New options: `--los-depth`, `--los-slab`, `--render-parallelism`, `--tier-set`, `--trim-start-frame`, `--full-bleed`.
  - Tier render calls can now run concurrently (`--render-parallelism > 1`).
- Deep run driver: `run_zoom16_deep.sh`
  - Environment-configurable depth mode and slab (`LOS_DEPTH_MODE`, `LOS_SLAB`).
  - Uses multi-node tier fanout by default with:
    - `RENDER_PARALLELISM=4`
    - `RENDER_LAUNCHER="srun --exclusive -N 1 -n 8 -c 4"`
- Convenience submission wrappers:
  - `run_zoom16_fixed10k.sh`
  - `run_zoom16_samey.sh`
- New deep-tier grid configs (10 Mpc z extent used for fixed-depth grids):
  - `config/zoom_grid_10mpc_wide.yaml`
  - `config/zoom_grid_10mpc_t2.yaml`
  - `config/zoom_grid_10mpc_mid.yaml`
  - `config/zoom_grid_10mpc_t4.yaml`
  - `config/zoom_grid_10mpc_close.yaml`
  - `config/zoom_grid_10mpc_t5.yaml`
  - `config/zoom_grid_10mpc_t6.yaml`

### How To Run

- Fixed depth run (10,000 ckpc/h slab):

```bash
sbatch run_zoom16_fixed10k.sh
```

- Same-as-y depth run:

```bash
sbatch run_zoom16_samey.sh
```

### Regrid / Reuse Controls

- Rebuild deep grids (default): `REGRID=1`.
- Reuse existing deep grids:

```bash
REGRID=0 sbatch run_zoom16_fixed10k.sh
REGRID=0 sbatch run_zoom16_samey.sh
```

### Tuning Notes

- If queue limits or node pressure are high, reduce node fanout by lowering either:
  - `RENDER_PARALLELISM`
  - `RENDER_LAUNCHER` node/rank request
- If walltime is dominated by plotting/encoding, adding render nodes yields diminishing returns.

### C++ Engine Changes

The gridder and renderer were reworked to support the deep-zoom workflow
and to scale beyond a single node. All changes are backwards compatible
with legacy scalar `side`/`resolution` configs and HDF5 headers.

#### Anisotropic grids

`GridConfig` (`src/common/Config.{h,cpp}`) now stores a `Vec3 size` and
`int shape[3]` instead of a scalar `side` + `resolution`. YAML accepts
either form:

```yaml
# legacy (still works; replicated to all axes)
side: 10000
resolution: 512

# anisotropic
size: [40000, 40000, 10000]
resolution: [1024, 1024, 256]
```

`Grid`, `Depositor`, `GridReader`, `RayTracer`, and the HDF5 writer/reader
were all updated to carry per-axis cell sizes. HDF5 headers now carry
`size`, `shape`, and vector `cell_size` attributes; scalar `side` and
`resolution` are also written for backward compatibility.

New helpers:
- `Box3::fromCenterSize(center, Vec3 size)` (`src/common/Box3.h`).
- `Grid::allocatedFieldNames(field_names)` — the canonical list of buffers
  a grid needs for a given requested field list (expands `gas_velocity` →
  `_x/_y/_z` and always appends `mass_weight`).

#### MPI shared-memory grid (gridder + renderer)

Both executables now split `MPI_COMM_WORLD` into a per-node communicator
(`MPI_Comm_split_type(..., MPI_COMM_TYPE_SHARED, ...)`) and allocate grid
buffers with `MPI_Win_allocate_shared`. N ranks on the same node share one
buffer per field instead of each holding a full copy. This is what made
deep grids (10 Mpc slab at high resolution) fit on Frontier nodes.

- Gridder (`src/gridder/main.cpp`, `src/gridder/Grid.{h,cpp}`):
  - `Grid` has a second constructor that accepts a `map<string, double*>`
    of externally-owned buffers (the shared-memory path). The
    self-allocating constructor still exists for single-rank use.
  - Reduction runs only across **node leaders** (a sub-communicator of
    `node_rank == 0` ranks), so the all-reduce cost is
    `(num_nodes - 1) * grid_bytes` rather than
    `(world_size - 1) * grid_bytes`.
  - OpenMP atomic adds in `Depositor` make concurrent per-node deposition
    from many ranks into the same shared buffer safe.
- Renderer (`src/renderer/main.cpp`, `src/renderer/GridReader.{h,cpp}`):
  - New `GridReader::readHeader` (metadata only) and
    `GridReader::readFieldsInto(filename, names, buffers)` read datasets
    directly into caller-provided buffers.
  - `HDF5Reader::readDatasetFloatInto(name, out, expected_n)` underpins
    the above (`src/common/HDF5IO.{h,cpp}`).
  - Node leader reads fields once into shared memory; all ranks on the
    node then render against the same buffer.
  - `renderer` now links `MPI::MPI_CXX` (`CMakeLists.txt`).

#### OpenMP-parallel deposition

`Depositor::depositGas` and `depositDM` (`src/gridder/Depositor.cpp`) are
now `#pragma omp parallel for schedule(dynamic, 256)` over particles, with
`#pragma omp atomic` on every field accumulation. This is thread-safe for
both intra-rank parallelism and inter-rank shared-memory deposition.

Smoothing length is additionally clamped to
`h_min = 0.5 * min(cell_size)` so particles smaller than a cell are not
dropped by the kernel-support test on anisotropic thin-slab grids.

#### Batch rendering (many cameras per grid read)

`renderer` now accepts `--camera-list <file>` where each line is
`<camera.yaml> <output_dir>` (blank lines and `#` comments ignored). The
grid is read once into shared memory; jobs are distributed round-robin
across world ranks. Example:

```
# cameras.txt
config/orbit_frames/cam_0000.yaml  frames/0000
config/orbit_frames/cam_0001.yaml  frames/0001
```

```bash
srun -N 4 -n 32 ./renderer --grid deep.hdf5 --camera-list cameras.txt
```

The single-frame `--camera cam.yaml --output dir` form still works.
Optional `--fields a,b,c` overrides the auto-derived field set.

#### LOS slab clipping

`CameraConfig` has a new optional `los_slab` (code units). When `> 0`,
ray integration is clipped to a slab of that thickness centered on
`look_at` along the camera forward axis. `Camera::slabTRange(ray, t0, t1)`
computes the per-ray clip range; `RayTracer::traceRayDDA` intersects it
with the AABB range before DDA traversal. This is what the Python
`--los-depth same-as-y` / `--los-slab` flags feed into at the C++ layer.

`los_slab = 0` (default) disables the clip and integrates the full grid
depth, matching pre-change behavior.

---

## Directory Structure

```
cosmo_viz/
├── CMakeLists.txt              # Top-level CMake (builds gridder + renderer)
├── config/
│   ├── example_grid.yaml       # Example grid config
│   └── example_camera.yaml     # Example camera config
├── src/
│   ├── common/
│   │   ├── Vec3.h              # 3D vector class (arithmetic, dot, cross, normalize)
│   │   ├── Box3.h              # Axis-aligned bounding box (contains, intersects, expand)
│   │   ├── Config.h/cpp        # YAML config parser
│   │   ├── HDF5IO.h/cpp        # HDF5 read/write helpers (wrappers around HDF5 C API)
│   │   ├── Kernel.h/cpp        # SPH cubic spline kernel W(r,h), dW/dr
│   │   └── Constants.h         # Physical constants, unit conversions
│   ├── gridder/
│   │   ├── main.cpp            # MPI init, config parse, orchestrate
│   │   ├── SnapshotReader.h/cpp # Read Arepo snapshot subfiles (gas, DM, stars)
│   │   ├── Grid.h/cpp          # 3D Cartesian grid: allocate, deposit, normalize, write
│   │   ├── Depositor.h/cpp     # SPH kernel deposition onto grid (scatter approach)
│   │   └── SmoothingLength.h/cpp # kNN-based hsml for DM/star particles
│   ├── renderer/
│   │   ├── main.cpp            # Config parse, orchestrate
│   │   ├── Camera.h/cpp        # Perspective + orthographic camera, ray generation
│   │   ├── GridReader.h/cpp    # Read gridded HDF5 cubes
│   │   ├── RayTracer.h/cpp     # DDA grid traversal, accumulate projections
│   │   └── ImageWriter.h/cpp   # Write 2D projection maps to HDF5
│   └── sps/
│       └── SPSTable.h/cpp      # Stellar population synthesis lookup table
├── python/
│   ├── orchestrate.py          # Master script: calls gridder, renderer, plotter
│   ├── plotter.py              # Read 2D maps, apply colormaps, save figures
│   ├── flythrough.py           # Generate camera path configs for movie frames
│   └── sps_table_generator.py  # Precompute L(age, Z, band) table from FSPS
├── test/
│   └── test_kernel.cpp         # Unit tests for kernel, deposition, ray tracing
└── README.md
```

---

## Stage 1: `gridder`

### 1.1 Input

**Command line:**
```bash
mpirun -np 64 ./gridder --snapshot /path/to/snap_099 --config grids.yaml --output /path/to/output/
```

**Config file (`grids.yaml`):**
```yaml
grids:
  - name: "cluster_zoom"
    center: [35.2, 18.7, 42.1]   # cMpc/h
    side: 2.0                     # cMpc/h, cubic box
    resolution: 1024              # cells per side
    fields:
      - gas_density
      - temperature
      - metallicity
      - HII_density
      - gas_velocity             # 3-component vector
      - dm_density
      - stellar_gband
      - stellar_rband
      - stellar_iband

  - name: "overview"
    center: [25.0, 25.0, 25.0]
    side: 50.0
    resolution: 512
    fields:
      - gas_density
      - dm_density
```

### 1.2 Snapshot format (Arepo/GADGET HDF5)

Each snapshot is split into `N` subfiles: `snap_099.0.hdf5` ... `snap_099.{N-1}.hdf5`.

Each subfile contains groups:
- `PartType0` — gas (Voronoi cells): `Coordinates`, `Masses`, `Density`, `InternalEnergy`, `Velocities`, `GFM_Metallicity`, `ElectronAbundance`, `Volume` (or derive from `Density` and `Masses`)
- `PartType1` — DM: `Coordinates`, `Masses` (or `Header/MassTable`)
- `PartType4` — stars: `Coordinates`, `Masses`, `GFM_Metallicity`, `GFM_StellarFormationTime` (scale factor at birth → age)

Header attributes: `BoxSize`, `Redshift`, `Time`, `NumFilesPerSnapshot`, `NumPart_Total`, `HubbleParam`, `Omega0`, `OmegaLambda`.

**Important:** The field names above are for IllustrisTNG/Thesan convention. Other simulations may differ slightly — keep field name mapping configurable.

### 1.3 MPI strategy

```
For each MPI rank:
    my_subfiles = distribute subfile indices round-robin across ranks
    allocate local_grids[num_grids]   // each grid: N^3 * num_fields, initialized to 0

    for each subfile in my_subfiles:
        read all particle types from subfile
        for each grid definition:
            bbox = grid bounding box (center ± side/2), handle periodic wrapping
            for each particle that falls within bbox (with kernel support margin = 2*h):
                deposit onto local_grids[grid_index] using SPH kernel

    MPI_Reduce(local_grids → global_grids, MPI_SUM, root=0)

    if rank == 0:
        for each grid:
            normalize intensive fields (divide by mass_weight)
            write to HDF5
```

**Memory consideration:** A 1024^3 grid with 12 float32 fields = ~48 GB. If running multiple grids, total memory per rank = sum of all grid sizes. For very large grids, consider:
- Processing one grid at a time (loop over grids in the outer loop, do full snapshot pass per grid).
- Using `MPI_Reduce` instead of `MPI_Allreduce` to save memory (only rank 0 needs the result).
- For grids that don't fit in memory even on one node: tile the grid spatially, deposit one tile at a time.

### 1.4 Deposition details

**Gas cells (PartType0):**
- Smoothing length: `h = (3 * Volume / (4 * pi))^(1/3)`. If `Volume` is not stored, compute `Volume = Mass / Density`.
- Kernel: cubic spline `W(r, h)` with compact support at `r = 2h`.
- For each gas cell, find all grid cells within `2h` of the particle position. For each overlapping grid cell, evaluate `W(|r_grid - r_particle|, h)` and accumulate:
  - `gas_density_grid += mass * W`
  - `temperature_grid += mass * T * W`   (T from InternalEnergy: `T = (gamma-1) * u * mu * m_p / k_B`)
  - `metallicity_grid += mass * Z * W`
  - `HII_density_grid += mass * x_HII * W`  (x_HII from ElectronAbundance or neutral fraction field)
  - `velocity_grid[3] += mass * v[3] * W`
  - `mass_weight_grid += mass * W`   (for normalizing intensive quantities)
- After MPI reduce, normalize: `temperature_grid /= mass_weight_grid`, same for metallicity, velocity.

**DM particles (PartType1):**
- No intrinsic smoothing length. Options (in order of preference):
  1. Use `SubfindHsml` if present in snapshot (SUBFIND output).
  2. Compute adaptive smoothing length from k-nearest neighbors (k=32 or 64). Build a k-d tree from DM particles that passed the bounding box test, query kNN distance for each.
- Deposit `dm_density_grid += mass * W(r, h_dm)`.

**Star particles (PartType4):**
- Smoothing length: same approach as DM (kNN or SubfindHsml).
- Stellar luminosity: look up `L_band(age, metallicity)` from a precomputed SPS table (see §1.5).
- `stellar_band_grid += L_band * W(r, h_star)`.

### 1.5 Stellar population synthesis table

Precompute a 2D lookup table: `L(age, Z)` per photometric band (e.g., SDSS g, r, i for RGB composites, or JWST bands).

**Generator (`python/sps_table_generator.py`):**
```python
import fsps
sp = fsps.StellarPopulation(...)
ages = np.logspace(6, 10.2, 200)    # 1 Myr to 15 Gyr
metallicities = np.logspace(-4, -1, 50)
# For each (age, Z), compute L_band per solar mass
# Save as HDF5: L_table[n_ages, n_Z, n_bands]
```

The C++ code loads this table at startup and does bilinear interpolation in (log age, log Z) space per star particle. This is fast — no FSPS dependency in C++.

### 1.6 Output format

One HDF5 file per grid per snapshot:

```
grid_{name}_snap{NNN}.hdf5
│
├── Header/                        (attributes)
│   ├── center: [3] float64        # grid center in cMpc/h
│   ├── side: float64              # box side length in cMpc/h
│   ├── resolution: int            # cells per side
│   ├── cell_size: float64         # = side / resolution
│   ├── redshift: float64
│   ├── scale_factor: float64
│   ├── snapshot: int
│   ├── boxsize: float64           # simulation box size
│   ├── hubble_param: float64
│   ├── omega0: float64
│   ├── omega_lambda: float64
│   └── fields: string[]           # list of field names present
│
├── gas_density:      [N, N, N] float32   # physical density (g/cm^3 or 10^10 Msun/h / (ckpc/h)^3)
├── temperature:      [N, N, N] float32   # mass-weighted T in Kelvin
├── metallicity:      [N, N, N] float32   # mass-weighted Z (mass fraction)
├── HII_density:      [N, N, N] float32   # ionized hydrogen density
├── gas_velocity_x:   [N, N, N] float32   # mass-weighted vx
├── gas_velocity_y:   [N, N, N] float32   # mass-weighted vy
├── gas_velocity_z:   [N, N, N] float32   # mass-weighted vz
├── dm_density:       [N, N, N] float32
├── stellar_gband:    [N, N, N] float32   # stellar luminosity density in g-band
├── stellar_rband:    [N, N, N] float32
├── stellar_iband:    [N, N, N] float32
└── mass_weight:      [N, N, N] float32   # for reference / re-weighting
```

Units should be documented in attributes on each dataset.

---

## Stage 2: `renderer`

### 2.1 Input

**Command line:**
```bash
./renderer --grid grid_cluster_zoom_snap099.hdf5 --camera camera.yaml --output projections/
```

**Camera config (`camera.yaml`):**
```yaml
camera:
  type: perspective           # or "orthographic"
  position: [36.0, 18.7, 40.0]   # cMpc/h (physical position in grid coordinates)
  look_at: [35.2, 18.7, 42.1]
  up: [0, 1, 0]
  fov: 60.0                  # degrees (perspective only)
  ortho_width: 2.0           # cMpc/h (orthographic only)
  image_width: 2048          # pixels
  image_height: 2048

projections:
  - field: gas_density
    mode: column              # ∫ ρ ds

  - field: temperature
    mode: mass_weighted       # ∫ ρ T ds / ∫ ρ ds

  - field: metallicity
    mode: mass_weighted

  - field: dm_density
    mode: column

  - field: stellar_gband
    mode: emission_absorption  # dI/ds = j - κI
    dust_kappa_field: gas_density  # κ ∝ ρ_gas * Z
    dust_kappa_scale: 1.0e3   # tunable normalization

  - field: gas_velocity
    mode: los_velocity        # mass-weighted v · n_ray
```

### 2.2 Camera

**Perspective camera:**
```
For each pixel (i, j):
    # Map pixel to normalized device coordinates [-1, 1]
    ndc_x = (2 * (i + 0.5) / width - 1) * aspect * tan(fov/2)
    ndc_y = (1 - 2 * (j + 0.5) / height) * tan(fov/2)

    # Ray in camera space
    ray_dir_cam = normalize(ndc_x, ndc_y, -1)

    # Transform to world space using camera basis vectors
    ray_origin = camera_position
    ray_dir = cam_right * ray_dir_cam.x + cam_up * ray_dir_cam.y + cam_forward * ray_dir_cam.z
```

**Orthographic camera:**
```
For each pixel (i, j):
    offset_x = (2 * (i + 0.5) / width - 1) * ortho_width / 2
    offset_y = (1 - 2 * (j + 0.5) / height) * ortho_width / 2

    ray_origin = camera_position + cam_right * offset_x + cam_up * offset_y
    ray_dir = cam_forward   # all rays parallel
```

### 2.3 Ray tracing (DDA / Amanatides-Woo)

For each ray:
1. Compute entry and exit points with the grid bounding box (ray-AABB intersection).
2. Initialize the DDA stepper: current voxel indices, tMax (distance to next voxel boundary in each axis), tDelta (distance between voxel boundaries along ray in each axis).
3. Step through voxels:
   ```
   while inside grid:
       ds = min(tMax_x, tMax_y, tMax_z) - t_current   # path length through this cell
       accumulate based on projection mode (see below)
       advance to next voxel (step in the axis with smallest tMax)
   ```

**Projection modes:**

| Mode | Accumulation | Finalization |
|------|-------------|--------------|
| `column` | `I += ρ[cell] * ds` | — |
| `mass_weighted` | `num += ρ[cell] * f[cell] * ds`; `den += ρ[cell] * ds` | `I = num / den` |
| `emission_absorption` | `I += j[cell] * ds * exp(-τ)`; `τ += κ[cell] * ds` | — |
| `los_velocity` | `num += ρ[cell] * (v · n̂) * ds`; `den += ρ[cell] * ds` | `I = num / den` |

For `emission_absorption`: `j` is the stellar emission density (from the stellar band field in the grid), and `κ` is the dust absorption coefficient, typically modeled as `κ = κ_scale * ρ_gas * Z` where Z is metallicity. The ray tracer needs to read both the emission field and the dust-related fields from the grid simultaneously.

### 2.4 Output format

One HDF5 file per camera setup per grid:

```
projection_{grid_name}_{camera_name}.hdf5
│
├── Header/                        (attributes)
│   ├── grid_file: string
│   ├── camera_type: string
│   ├── camera_position: [3] float64
│   ├── camera_look_at: [3] float64
│   ├── camera_up: [3] float64
│   ├── fov: float64
│   ├── image_width: int
│   ├── image_height: int
│   └── fields: string[]
│
├── gas_density:    [H, W] float32
├── temperature:    [H, W] float32
├── dm_density:     [H, W] float32
├── stellar_gband:  [H, W] float32
├── stellar_rband:  [H, W] float32
├── stellar_iband:  [H, W] float32
└── gas_velocity:   [H, W] float32    # line-of-sight velocity
```

---

## Stage 3: `plotter` (Python)

This is a standard matplotlib/astropy script. No special architecture needed. Typical operations:

- Read 2D maps from HDF5.
- Apply `np.log10` for density fields.
- Apply colormaps (e.g., `inferno` for gas density, `coolwarm` for temperature, `viridis` for metallicity).
- RGB composite for stellar light: map g→blue, r→green, i→red channels, apply `arcsinh` stretch.
- Add scale bar, colorbar, redshift label, annotations.
- Save as PNG/PDF.

---

## Python Orchestrator

`python/orchestrate.py` ties everything together:

```python
#!/usr/bin/env python3
"""
Master script: runs gridder, renderer, plotter in sequence.

Usage:
    python orchestrate.py --snapshot /path/to/snap_099 \
                          --grid-config grids.yaml \
                          --camera-config camera.yaml \
                          --output-dir /path/to/output/ \
                          --np 64
"""

import subprocess, sys, argparse

def run_gridder(args):
    cmd = f"mpirun -np {args.np} ./build/gridder --snapshot {args.snapshot} --config {args.grid_config} --output {args.output_dir}/grids/"
    subprocess.run(cmd, shell=True, check=True)

def run_renderer(args):
    # Find all grid files produced by gridder
    grid_files = glob(f"{args.output_dir}/grids/*.hdf5")
    for gf in grid_files:
        cmd = f"./build/renderer --grid {gf} --camera {args.camera_config} --output {args.output_dir}/projections/"
        subprocess.run(cmd, shell=True, check=True)

def run_plotter(args):
    cmd = f"python plotter.py --input {args.output_dir}/projections/ --output {args.output_dir}/figures/"
    subprocess.run(cmd, shell=True, check=True)

if __name__ == "__main__":
    # parse args, run stages in order
    ...
```

For flythroughs: `flythrough.py` generates a sequence of camera configs (interpolating position along a spline path), the orchestrator loops over them, and the final step is `ffmpeg` to stitch frames into a video.

---

## Build System

**CMakeLists.txt:**
```cmake
cmake_minimum_required(VERSION 3.16)
project(cosmo_viz LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(MPI REQUIRED)
find_package(HDF5 REQUIRED COMPONENTS C)
# Optional: yaml-cpp for config parsing
find_package(yaml-cpp QUIET)

# --- Common library ---
add_library(common STATIC
    src/common/Config.cpp
    src/common/HDF5IO.cpp
    src/common/Kernel.cpp
)
target_include_directories(common PUBLIC src/common)
target_link_libraries(common PUBLIC ${HDF5_C_LIBRARIES})
target_include_directories(common PUBLIC ${HDF5_INCLUDE_DIRS})
if(yaml-cpp_FOUND)
    target_link_libraries(common PUBLIC yaml-cpp)
endif()

# --- Gridder ---
add_executable(gridder
    src/gridder/main.cpp
    src/gridder/SnapshotReader.cpp
    src/gridder/Grid.cpp
    src/gridder/Depositor.cpp
    src/gridder/SmoothingLength.cpp
)
target_link_libraries(gridder PRIVATE common MPI::MPI_CXX)

# --- Renderer ---
add_executable(renderer
    src/renderer/main.cpp
    src/renderer/Camera.cpp
    src/renderer/GridReader.cpp
    src/renderer/RayTracer.cpp
    src/renderer/ImageWriter.cpp
)
target_link_libraries(renderer PRIVATE common)
```

**Dependencies:**
- C++17 compiler (GCC 9+ or Clang 10+)
- MPI (OpenMPI or MPICH)
- HDF5 (C library, with parallel I/O support for large snapshots)
- yaml-cpp (for config parsing; alternatively, roll a simple parser)
- Python 3 with numpy, h5py, matplotlib, astropy (for plotter)
- FSPS (Python, only for precomputing the SPS table)

---

## Periodic Boundary Handling

Simulations use periodic boxes. When a grid region straddles a box boundary, particles near the edge on the opposite side must be wrapped. During the bounding-box test in the gridder:

```cpp
// For each axis, check if particle is within bbox considering periodicity
for (int d = 0; d < 3; d++) {
    float dx = pos[d] - grid_center[d];
    if (dx > boxsize / 2)  dx -= boxsize;
    if (dx < -boxsize / 2) dx += boxsize;
    // Now test dx against grid half-side + kernel margin
    if (fabs(dx) > grid_halfside + 2 * hsml) skip;
}
```

---

## Key Design Decisions

1. **Cubic spline kernel** for all depositions. Compact support (2h), well-tested, standard in SPH.
2. **Grid is always a cube** (same resolution in x, y, z). Simplifies DDA and memory layout. Non-cubic regions can be handled by choosing the cube to contain the region of interest.
3. **Float32 for grid data.** Sufficient precision, halves memory vs float64. Deposition accumulators should use float64 to avoid catastrophic cancellation, then cast to float32 when writing.
4. **C-order (row-major) for 3D arrays.** Index as `grid[iz * N * N + iy * N + ix]`. This matches HDF5 default and numpy convention.
5. **Units:** store everything in simulation internal units (typically 10^10 Msun/h, ckpc/h, km/s). Document in HDF5 attributes. Convert to physical in the plotter.

---

## Performance Notes

- **I/O bound:** The gridder will be I/O dominated for large snapshots. Use HDF5 hyperslabs to read only the coordinate columns first for the bounding-box test, then read full particle data only for particles that pass.
- **Two-pass read per subfile:** (1) Read coordinates, test against all grid bboxes, build index list of passing particles. (2) Read remaining fields only for those particles. Avoids loading GB of velocity data for particles outside all grids.
- **OpenMP inside MPI ranks:** The deposition loop over particles within a single rank can be parallelized with OpenMP. Use `#pragma omp parallel for reduction(+:grid_data[:grid_size])` or atomic adds. Thread-local grid copies with a final reduction are cleaner but cost more memory.
- **Renderer is embarrassingly parallel over pixels.** Use OpenMP `parallel for` over rows. No MPI needed for the renderer unless image sizes are extreme.

---

## Build Order (Implementation Phases)

### Phase 1: Minimal end-to-end
- `gridder`: single rank, single grid, gas_density only, reads one subfile
- `renderer`: orthographic, column density only
- `plotter`: single colormap image
- **Validation:** compare output image against a known SPH rendering tool (e.g., `py-sphviewer`, Paicos)

### Phase 2: MPI + multi-field
- `gridder`: MPI-parallel subfile distribution, MPI_Reduce, multiple fields (T, Z, DM, HI, velocity)
- `renderer`: mass-weighted projections, LOS velocity
- Test on a 50 Mpc box snapshot

### Phase 3: Multi-grid + stars
- `gridder`: multiple grid definitions per run (Oliver's multi-grid-per-pass idea)
- Stellar light: SPS table, stellar luminosity deposition
- `renderer`: emission-absorption mode (stellar light + dust)
- `plotter`: RGB compositing from stellar bands

### Phase 4: Cameras + flythrough
- `renderer`: perspective camera
- `flythrough.py`: camera path interpolation (Catmull-Rom spline)
- Batch render frames, ffmpeg video assembly

### Phase 5: Optimization
- Two-pass I/O (coordinates first, then fields)
- OpenMP within MPI ranks
- HDF5 chunking + compression for grid output
- Memory-constrained mode (one grid at a time)
