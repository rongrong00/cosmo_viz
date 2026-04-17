#!/bin/bash
#SBATCH -A AST211
#SBATCH -J cosmo_zrend
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 32
#SBATCH -t 02:00:00
#SBATCH -p batch
#SBATCH -o output/zoom_render_%j.out
#SBATCH -e output/zoom_render_%j.err

set -euo pipefail

module load hdf5
source /autofs/nccs-svm1_sw/andes/miniforge3/23.11.0/etc/profile.d/conda.sh
conda activate chii

cd /lustre/orion/ast211/proj-shared/rongrong/lumina_visualization/cosmo_viz

# Renderer is OpenMP-parallel over image rows; give it a fat core count.
export OMP_NUM_THREADS=32

rm -rf output/zoom_proj output/zoom_png
python3 python/zoom_frames.py --render --plot --encode \
    --frames 120 --image-size 1440 --video output/zoom_test.mp4

echo "=== DONE ==="
