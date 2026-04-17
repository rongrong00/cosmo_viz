#!/bin/bash
#SBATCH -A AST211
#SBATCH -J cosmo_zoom
#SBATCH -N 1
#SBATCH -n 10
#SBATCH -t 02:00:00
#SBATCH -p batch
#SBATCH -o output/zoom_%j.out
#SBATCH -e output/zoom_%j.err

set -euo pipefail

module load hdf5
# conda for the python post-processing step
source /autofs/nccs-svm1_sw/andes/miniforge3/23.11.0/etc/profile.d/conda.sh
conda activate chii

cd /lustre/orion/ast211/proj-shared/rongrong/lumina_visualization/cosmo_viz

SNAP=/lustre/orion/ast211/proj-shared/EDE_Final/finalTest/50cMpc_2/output/snapdir_121/snap_121

echo "=== GRIDDER: zoom_wide (full box) ==="
srun -n 10 ./build/gridder --snapshot "$SNAP" \
    --config config/zoom_grid.yaml      --output output/grids/

echo "=== GRIDDER: zoom_mid (8 cMpc/h) ==="
srun -n 10 ./build/gridder --snapshot "$SNAP" \
    --config config/zoom_grid_mid.yaml  --output output/grids/

echo "=== GRIDDER: zoom_close (2 cMpc/h) ==="
srun -n 10 ./build/gridder --snapshot "$SNAP" \
    --config config/zoom_grid_close.yaml --output output/grids/

echo "=== RENDER + PLOT + ENCODE ==="
python3 python/zoom_frames.py --render --plot --encode \
    --frames 120 --image-size 720 \
    --video output/zoom_test.mp4

echo "=== DONE ==="
