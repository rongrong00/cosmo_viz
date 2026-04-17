#!/bin/bash
#SBATCH -A AST211
#SBATCH -J cosmo_orbit_hires
#SBATCH -N 1
#SBATCH -n 4
#SBATCH -c 8
#SBATCH -t 08:00:00
#SBATCH -p batch
#SBATCH -o output/orbit_hires_%j.out
#SBATCH -e output/orbit_hires_%j.err

set -euo pipefail

module load hdf5
source /autofs/nccs-svm1_sw/andes/miniforge3/23.11.0/etc/profile.d/conda.sh
conda activate chii

cd /lustre/orion/ast211/proj-shared/rongrong/lumina_visualization/cosmo_viz

SNAP=/lustre/orion/ast211/proj-shared/EDE_Final/finalTest/50cMpc_2/output/snapdir_121/snap_121

# Default: reuse the existing orbit grid. Set REGRID=1 to rebuild it.
REGRID=${REGRID:-0}

# Higher-quality defaults for the rotation movie.
FRAMES=${FRAMES:-240}
IMAGE_WIDTH=${IMAGE_WIDTH:-3840}
IMAGE_HEIGHT=${IMAGE_HEIGHT:-2160}
FPS=${FPS:-30}

CONFIG_DIR=${CONFIG_DIR:-config/orbit_frames_hires}
PROJ_DIR=${PROJ_DIR:-output/orbit_proj_hires}
PNG_DIR=${PNG_DIR:-output/orbit_png_hires}
VIDEO=${VIDEO:-output/orbit_hires.mp4}

export OMP_NUM_THREADS=8

if [[ "$REGRID" == "1" ]]; then
    echo "=== GRIDDER: orbit_grid (10 cMpc/h, 1024^3) ==="
    srun -n 4 -c 8 ./build/gridder \
        --snapshot "$SNAP" \
        --config config/orbit_grid.yaml \
        --output output/grids/
else
    echo "=== REUSING EXISTING GRID: output/grids/grid_orbit_snap121.hdf5 ==="
fi

echo "=== RENDER + PLOT + ENCODE (orbit hires ${IMAGE_WIDTH}x${IMAGE_HEIGHT}, ${FRAMES} frames) ==="
rm -rf "$PROJ_DIR" "$PNG_DIR"
python3 python/orbit_frames.py --render --plot --encode \
    --frames "$FRAMES" \
    --image-width "$IMAGE_WIDTH" \
    --image-height "$IMAGE_HEIGHT" \
    --config-dir "$CONFIG_DIR" \
    --proj-dir "$PROJ_DIR" \
    --png-dir "$PNG_DIR" \
    --video "$VIDEO"

echo "=== DONE ==="