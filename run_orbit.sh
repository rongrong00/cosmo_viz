#!/bin/bash
#SBATCH -A AST211
#SBATCH -J cosmo_orbit
#SBATCH -N 1
#SBATCH -n 4
#SBATCH -c 8
#SBATCH -t 04:00:00
#SBATCH -p batch
#SBATCH -o output/orbit_%j.out
#SBATCH -e output/orbit_%j.err

set -euo pipefail

module load hdf5
source /autofs/nccs-svm1_sw/andes/miniforge3/23.11.0/etc/profile.d/conda.sh
conda activate chii

cd /lustre/orion/ast211/proj-shared/rongrong/lumina_visualization/cosmo_viz

SNAP=/lustre/orion/ast211/proj-shared/EDE_Final/finalTest/50cMpc_2/output/snapdir_121/snap_121

export OMP_NUM_THREADS=8

echo "=== GRIDDER: orbit_grid (10 cMpc/h, 1024^3) ==="
srun -n 4 -c 8 ./build/gridder \
    --snapshot "$SNAP" \
    --config config/orbit_grid.yaml \
    --output output/grids/

echo "=== RENDER + PLOT + ENCODE (orbit, 16:9 1920x1080) ==="
rm -rf output/orbit_proj output/orbit_png
python3 python/orbit_frames.py --render --plot --encode \
    --frames 120 --video output/orbit.mp4

echo "=== DONE ==="
