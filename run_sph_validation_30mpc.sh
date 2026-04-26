#!/bin/bash
#SBATCH -A AST211
#SBATCH -J sph_valid30
#SBATCH -N 1
#SBATCH -n 8
#SBATCH -c 4
#SBATCH -t 02:00:00
#SBATCH -p batch
#SBATCH -o output/sph_valid30_%j.out
#SBATCH -e output/sph_valid30_%j.err

# Grid-path comparison for the 30 Mpc SPH render (output/sph_wide_test).

set -euo pipefail
module load hdf5

cd /lustre/orion/ast211/proj-shared/rongrong/lumina_visualization/cosmo_viz

SNAP=/lustre/orion/ast211/proj-shared/EDE_Final/finalTest/50cMpc_2/output/snapdir_121/snap_121
GRID_CFG=config/sph_validation_grid_30mpc.yaml
CAMERA=config/sph_cam_30mpc.yaml
GRID_OUT=output/grids/grid_sph_validation_30mpc_snap121.hdf5
PROJ_DIR=output/sph_valid30_proj

mkdir -p output/grids "$PROJ_DIR"

echo "=== GRIDDER (1024^3, 30 Mpc cube, 8 ranks) ==="
export OMP_NUM_THREADS=1
srun -n 8 ./build/gridder \
    --snapshot "$SNAP" \
    --config   "$GRID_CFG" \
    --output   output/grids/

echo "=== GRID RENDERER (same camera as SPH) ==="
OMP_NUM_THREADS=32 ./build/renderer \
    --grid   "$GRID_OUT" \
    --camera "$CAMERA" \
    --output "$PROJ_DIR"

ls -la "$PROJ_DIR"
echo "=== DONE ==="
