# SPH Direct Ray Tracer — Implementation Plan

A new pipeline that ray-traces directly through SPH particles instead of first
depositing them onto a Cartesian grid. Runs **alongside** the existing
`gridder` + `renderer` pipeline; neither is modified.

---

## 0. Scope & non-goals

**In scope:**
- New executable `sph_renderer`. Reads SPH particles directly from snapshot,
  builds per-type BVHs, ray-traces through kernels.
- Modes: `column`, `mass_weighted`. Gas velocity is rendered as three
  mass-weighted channels (`vx`, `vy`, `vz`) — no dedicated LOS-velocity mode.
- Gas (PartType0, native `h`) and DM (PartType1, kNN `h`).
- Orthographic + perspective cameras. Optional LOS slab clipping.
- MPI shared-memory particle + BVH storage across ranks on a node.
- Batch-camera mode (`--camera-list`).

**Not in scope (deferred):**
- Stellar bands / emission–absorption. Only stellar light needs order-
  dependent ray integration; all in-scope modes are order-independent.
- Disk particle caching (noted as Phase F extension).
- Removal or modification of existing `gridder/` or `renderer/` targets.

---

## 1. File layout

```
src/
├── common/                    (existing + 2 promoted + 2 new)
│   ├── Vec3.h, Box3.h, Config.{h,cpp}, Kernel.{h,cpp},
│   │   HDF5IO.{h,cpp}, Constants.h            (unchanged)
│   ├── SnapshotReader.{h,cpp}  ← moved from gridder/
│   ├── SmoothingLength.{h,cpp} ← moved from gridder/
│   ├── KernelLUT.{h,cpp}       ← NEW: tabulated 2D line integral F(b/h)
│   └── RegionConfig.{h,cpp}    ← NEW: region YAML parser
│
├── gridder/                   (unchanged, re-links through common/)
├── renderer/                  (unchanged, grid-based ray tracer)
└── sph_renderer/              NEW
    ├── main.cpp               MPI init, particle load, BVH build, camera loop
    ├── ParticleStore.{h,cpp}  SoA particle buffers in MPI shared mem
    ├── ParticleLoader.{h,cpp} Subfile distribution, region cull, consolidate
    ├── BVH.{h,cpp}            Binary SAH BVH over per-particle AABBs
    ├── SphRayTracer.{h,cpp}   BVH traversal + per-particle line integral
    └── ImageWriter.{h,cpp}    Same HDF5 schema as renderer/ImageWriter
```

`SnapshotReader` + `SmoothingLength` move: purely mechanical. Two `#include`
path updates in gridder sources and a CMake source-path rename.

---

## 2. Data structures

### 2.1 Particle storage (SoA for cache efficiency)

```cpp
// ParticleStore.h
struct GasStore {
    size_t N;
    float *x, *y, *z;          // positions
    float *h;                  // smoothing length (code units)
    float *mass, *density;
    float *temperature;        // optional, null if not requested
    float *metallicity;        // optional
    float *vx, *vy, *vz;       // optional
    float *hii_fraction;       // optional
    std::map<std::string, float*> extra;  // future fields
};

struct DMStore {
    size_t N;
    float *x, *y, *z, *h, *mass;
};
```

Why SoA, not AoS: a ray evaluates `F(b/h) * mass` for every hit but touches
`temperature[i]` only for mass-weighted modes. SoA keeps cache lines packed
with the accumulator's hot fields.

Each array is its own MPI shared-memory window (same pattern as the current
grid/renderer). Node leader allocates + fills, all ranks read.

### 2.2 BVH node layout

```cpp
// BVH.h — 32 bytes, cache-line friendly
struct alignas(32) BVHNode {
    float bmin[3];
    uint32_t left_or_first;   // leaf: first particle index; internal: left child
    float bmax[3];
    uint32_t count;           // >0 → leaf with this many particles; 0 → internal
};

class BVH {
    std::vector<BVHNode> nodes;   // flat DFS array, root at 0
    std::vector<uint32_t> perm;   // permutation into ParticleStore
};
```

Build inputs: per-particle AABB = `[pos − 2h, pos + 2h]`. Built top-down with
16-bin SAH on the longest centroid-extent axis. Leaf size 4, depth bound 64.

### 2.3 Memory budget (40M-particle zoom region)

| Buffer | Size |
|---|---|
| BVH nodes (~2N) | 2.6 GB |
| Permutation (4 B × N) | 160 MB |
| Particle arrays (~9 × 4 B × N) | 1.4 GB |
| **Total per node** | **~4.2 GB** |

Fits Frontier comfortably, shared once per node via MPI shared-mem.

### 2.4 Kernel line-integral LUT

```cpp
// common/KernelLUT.h
class KernelLUT {
public:
    // F(u) = ∫_{-∞}^{+∞} W(√(u²+s²), 1) ds   for cubic spline, u = b/h
    // Per-particle contribution to a ray passing at impact parameter b:
    //   contrib = F(b/h) / h²   (units: 1/length², i.e. 2D column kernel)
    static void init(int nsamples = 1024);
    static inline float eval(float u);
};
```

1024-entry table + linear interpolation is < 1e-4 RMS error and ~1 ns per
lookup. Evaluated ~10⁹ times per image; worth the table.

### 2.5 Region config

```yaml
# config/zoom_region_10mpc.yaml
region:
  name: "zoom_10mpc"
  center: [35200, 18700, 42100]   # ckpc/h
  size:   [10000, 10000, 10000]   # ckpc/h (asymmetric OK)
  particle_types: [gas, dm]
  margin: 0.0                     # extra halo beyond auto 2*h_max cull
```

Camera YAMLs are **identical** to the grid renderer — no changes.

---

## 3. Algorithms

### 3.1 Region selection (two-pass I/O, reusing SnapshotReader pattern)

1. Read coordinates only. Test `|p_d − center_d| ≤ size_d/2 + 2 h_max_guess`
   per axis with periodic wrap.
2. For passing particles, HDF5 hyperslab-read remaining fields.

Halo cull uses `size/2 + 2 · h_max`. `h_max` from a first pass over gas `h`
and post-kNN for DM. Typical expansion ≈ +5–10 %.

### 3.2 BVH build (top-down binned SAH)

```
build(prim_indices[0..N]):
    aabb_union      = union of per-particle AABBs
    centroid_union  = union of centroids
    if N <= leaf_size: emit leaf; return

    axis = argmax(centroid_union.extent)
    for b in 0..15: accumulate count_b, aabb_b per centroid bin
    best = argmin over 15 split positions of
           SA(left)*n_left + SA(right)*n_right
    partition in place around best split
    recurse (OpenMP task at top K levels)
    emit internal node
```

Complexity O(N log N). Target: < 3 s for 40M particles, 64 threads.

### 3.3 Ray traversal

```cpp
for each pixel ray:
    stack[64]; top = 0; push(root)
    (tmin, tmax) = rayAABB(ray, root_bbox) ∩ camera.slabTRange(ray)
    while top > 0:
        node = nodes[pop()]
        if !rayAABB(ray, node.bmin, node.bmax, within [tmin,tmax]): continue
        if leaf:
            for i in [first, first+count):
                p = particles[perm[i]]
                t_hit = dot(p.pos - ray.origin, ray.dir)
                if t_hit < tmin or t_hit > tmax: continue
                b² = |p.pos - (ray.origin + t_hit * ray.dir)|²
                if b² < (2*p.h)²:
                    contrib = KernelLUT::eval(sqrt(b²)/p.h) / (p.h*p.h)
                    accumulator.add(p, contrib)
        else:
            push(node.right); push(node.left)
```

- `rayAABB`: slab test — 6 mul + 6 sub. Reused from `renderer/RayTracer`.
- `rayPointClosest`: 9 mul-add. Cheap.
- Pixel loop: `#pragma omp parallel for` over rows — same as grid renderer.

### 3.4 Projection modes

Per ray–particle hit let `w = contrib * p.mass`.

| Mode | Per-hit | Per-ray finalize |
|---|---|---|
| `column` on `gas_density` | `accum += w` | `accum` |
| `column` on `dm_density`  | `accum += w` (DM BVH) | `accum` |
| `mass_weighted` on `f` | `num += w * f[i]`; `den += w` | `num/den` if `den>0` |
| gas velocity (three channels) | run `mass_weighted` on `vx`, `vy`, `vz` independently | three image channels |

- `column gas_density` is **exact SPH**:  `Σᵢ mᵢ · F(bᵢ/hᵢ) / hᵢ²`.
- `mass_weighted` matches the grid result in the limit of small cells, and
  avoids grid discretization entirely.

---

## 4. MPI / OpenMP strategy

Mirrors the current renderer:

```
world → per-node comm (MPI_COMM_TYPE_SHARED)
node leader (node_rank == 0):
    ├─ loads subfiles (round-robin over world ranks)
    ├─ Allgatherv region-culled particles (all nodes see all particles)
    ├─ kNN for DM (OpenMP parallel)
    ├─ BVH build (OpenMP parallel, two BVHs: gas + DM)
all ranks:
    ├─ query shared BVH + particle arrays
    ├─ round-robin camera jobs
    ├─ ray-trace pixel tile (OpenMP parallel for)
    └─ write HDF5
```

**Consolidation rationale:** a ray may touch any particle in the region, so
each node needs the full set. For a 40M-particle region that is one 1.4 GB
all-gather per region load. Spatial partitioning would save memory but
requires per-ray cross-node accumulation — not worth the complexity while
the data fits per node.

---

## 5. CLI

```bash
# Single camera
./sph_renderer --snapshot /path/snap_099 \
               --region   config/zoom_region_10mpc.yaml \
               --camera   config/zoom_frames/cam_0000.yaml \
               --output   frames/0000

# Batch (particles + BVH loaded once, round-robin over cameras)
./sph_renderer --snapshot /path/snap_099 \
               --region   config/zoom_region_10mpc.yaml \
               --camera-list cams.txt

# Field override
./sph_renderer ... --fields gas_density,temperature,gas_velocity
```

Stellar field names produce an explicit error until Phase F ships.

---

## 6. Build system changes (CMakeLists.txt)

```cmake
# Promote shared sources into common library
target_sources(common PRIVATE
    src/common/SnapshotReader.cpp
    src/common/SmoothingLength.cpp
    src/common/KernelLUT.cpp
    src/common/RegionConfig.cpp
)

# SPH Renderer target
add_executable(sph_renderer
    src/sph_renderer/main.cpp
    src/sph_renderer/ParticleStore.cpp
    src/sph_renderer/ParticleLoader.cpp
    src/sph_renderer/BVH.cpp
    src/sph_renderer/SphRayTracer.cpp
    src/sph_renderer/ImageWriter.cpp
)
target_link_libraries(sph_renderer PRIVATE common MPI::MPI_CXX)
if(OpenMP_CXX_FOUND)
    target_link_libraries(sph_renderer PRIVATE OpenMP::OpenMP_CXX)
endif()
```

Gridder source list: `src/gridder/SnapshotReader.cpp` →
`src/common/SnapshotReader.cpp` (same for `SmoothingLength`).

---

## 7. Validation

**Unit tests (under `test/`):**
1. `test_kernel_lut.cpp` — compare `KernelLUT::eval(u)` against high-order
   numerical quadrature of `∫ W(√(u²+s²),1) ds`. RMS < 1e-4.
2. `test_kernel_lut.cpp` — mass normalization: `∫ F(u) · 2π u du = 1`.
3. `test_bvh.cpp` — build over 10k random particles; brute-force ray-particle
   accumulator must match BVH-traversed accumulator to float precision.
4. `test_ray_particle.cpp` — hand-crafted rays vs. known particle positions.

**Integration test:**
- Single subfile (~500k gas particles). Render column density with both
  `renderer` (grid 2048³) and `sph_renderer`. Image RMS diff → 0 as grid
  resolution grows.

**Physical self-check:**
- `Σ pixel · pixel_area` for column density equals `Σ particle_mass` in the
  slab to < 1 % for a well-resolved region.

---

## 8. Phased rollout

### Phase A — Foundation
- **A1** Move `SnapshotReader` + `SmoothingLength` to `common/`. Gridder
  still builds.
- **A2** `KernelLUT` + unit tests.
- **A3** `RegionConfig` parser + example configs.
- **A4** `ParticleStore` + `ParticleLoader` (MPI, shared-mem, region cull,
  kNN for DM).
- **A5** `main.cpp` scaffolding: load particles, print bbox + h range, exit.

### Phase B — BVH + ortho column
- **B1** BVH build (single-threaded first, SAH binned).
- **B2** `SphRayTracer::traceColumn()` + orthographic camera.
- **B3** `ImageWriter`, first end-to-end SPH image.
- **B4** Validation: diff vs. grid path at 2048³.

### Phase C — Remaining modes + perspective
- **C1** Perspective camera (reuse `renderer/Camera`).
- **C2** `traceMassWeighted` (single and multi-channel, used for gas velocity as (vx, vy, vz)).
- **C3** DM BVH + `dm_density` column.
- **C4** LOS slab clipping.

### Phase D — Parallelism + batch
- **D1** OpenMP-parallel BVH build.
- **D2** MPI shared-mem BVH + particle buffers.
- **D3** `--camera-list` batch mode.
- **D4** Scaling benchmarks on Frontier (1, 2, 4 nodes).

### Phase E — Drivers + docs
- **E1** `run_sph_zoom16.sh` mirrors `run_zoom16_deep.sh`.
- **E2** `python/sph_zoom_frames.py` parallels `zoom_frames.py`, skipping
  the gridder stage.
- **E3** Add SPH pipeline section to `PLAN.md`.

### Phase F — Optional extensions (deferred)
- **F1** On-disk particle cache (binary per-region file, reused across
  camera batches and snapshots).
- **F2** Front-to-back traversal prep for future emission–absorption.
- **F3** Two-level BVH (per particle type) if profiling shows a win.

---

## 9. Estimated effort

- New C++: ~1500 lines across 12 new files.
- Moved C++: ~200 lines (SnapshotReader + SmoothingLength).
- CMake changes: ~15 lines.
- Python driver: ~300 lines (mirrors `zoom_frames.py`).
- Working Phase B (first image): ~1 focused day.
- Through Phase E (production-ready): ~2–3 focused days.

---

## 10. When to use which pipeline

| Use case | Recommended pipeline |
|---|---|
| Wide overviews, uniform resolution | `gridder` + `renderer` (grid path) |
| Many cameras over the same fixed grid | grid path (DDA is fastest) |
| Deep zooms where grid resolution is the bottleneck | `sph_renderer` |
| Regions with h varying by ≥10³ (voids + cores together) | `sph_renderer` |
| Stellar light / dust emission–absorption | neither yet — Phase F |
