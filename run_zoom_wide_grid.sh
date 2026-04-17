#!/bin/bash
#SBATCH -A AST211
#SBATCH -J zoom_wide_grid
#SBATCH -N 1
#SBATCH -n 4
#SBATCH -c 4
#SBATCH -t 04:00:00
#SBATCH -p batch
#SBATCH -o output/zoom_wide_grid_%j.out
#SBATCH -e output/zoom_wide_grid_%j.err

set -euo pipefail

module load hdf5
source /autofs/nccs-svm1_sw/andes/miniforge3/23.11.0/etc/profile.d/conda.sh
conda activate chii

cd /lustre/orion/ast211/proj-shared/rongrong/lumina_visualization/cosmo_viz

SNAP=/lustre/orion/ast211/proj-shared/EDE_Final/finalTest/50cMpc_2/output/snapdir_121/snap_121
GRID_DIR=output/grids_10mpc

mkdir -p "$GRID_DIR"
export OMP_NUM_THREADS=4

echo "=== GRIDDER: zoom_grid_10mpc_wide ==="
srun -n 4 -c 4 ./build/gridder \
    --snapshot "$SNAP" \
    --config config/zoom_grid_10mpc_wide.yaml \
    --output "$GRID_DIR"

echo "=== DONE ==="