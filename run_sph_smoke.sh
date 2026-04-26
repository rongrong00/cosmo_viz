#!/bin/bash
#SBATCH -A AST211
#SBATCH -J sph_smoke
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 56
#SBATCH -t 00:20:00
#SBATCH -p batch
#SBATCH -o output/sph_smoke_%j.out
#SBATCH -e output/sph_smoke_%j.err

# Phase-A smoke test for sph_renderer:
# - loads a zoom region from a real snapshot,
# - reports gas/dm particle counts, h extrema, and bounding boxes,
# - builds the BVH over the loaded particles (once integrated),
# - no rendering yet (Phases B onwards will add it).
#
# Uses a single rank with many OpenMP threads for the kNN and future BVH
# build; heavier MPI-scaled tests come in Phase D.

module load hdf5

cd /lustre/orion/ast211/proj-shared/rongrong/lumina_visualization/cosmo_viz

export OMP_NUM_THREADS=56
export OMP_PLACES=cores
export OMP_PROC_BIND=close

SNAP=/lustre/orion/ast211/proj-shared/EDE_Final/finalTest/50cMpc_2/output/snapdir_121/snap_121

echo "=== Unit tests ==="
./build/test_kernel_lut
./build/test_bvh

echo
echo "=== Phase A smoke: gas only ==="
srun -n 1 -c ${OMP_NUM_THREADS} --cpu-bind=threads \
    ./build/sph_renderer \
    --snapshot "$SNAP" \
    --region config/zoom_region_10mpc.yaml \
    --fields gas_density

echo
echo "=== Phase A smoke: gas + dm ==="
# Switch region to include DM and rerun.
# (zoom_region_10mpc.yaml currently lists gas only; override with a tmp file.)
TMP_REGION=$(mktemp --suffix=.yaml)
cat > "$TMP_REGION" <<EOF
name: "zoom_10mpc_gasdm"
center: [4093.16, 23556.48, 10679.74]
size: [10000, 10000, 10000]
particle_types: [gas, dm]
margin: 0.0
EOF

srun -n 1 -c ${OMP_NUM_THREADS} --cpu-bind=threads \
    ./build/sph_renderer \
    --snapshot "$SNAP" \
    --region "$TMP_REGION" \
    --fields gas_density,dm_density

rm -f "$TMP_REGION"

echo "=== DONE ==="
