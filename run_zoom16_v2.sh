#!/bin/bash
#SBATCH -A AST211
#SBATCH -J cosmo_zoom16v2
#SBATCH -N 1
#SBATCH -n 4
#SBATCH -c 8
#SBATCH -t 06:00:00
#SBATCH -p batch
#SBATCH -o output/zoom16v2_%j.out
#SBATCH -e output/zoom16v2_%j.err

set -euo pipefail

module load hdf5
source /autofs/nccs-svm1_sw/andes/miniforge3/23.11.0/etc/profile.d/conda.sh
conda activate chii

cd /lustre/orion/ast211/proj-shared/rongrong/lumina_visualization/cosmo_viz

SNAP=/lustre/orion/ast211/proj-shared/EDE_Final/finalTest/50cMpc_2/output/snapdir_121/snap_121

# Set REGRID=1 to rebuild the grids first; default is to reuse the existing
# precomputed zoom grids in output/grids_v2.
export OMP_NUM_THREADS=8
REGRID=${REGRID:-0}

GRID_DIR=output/grids_v2
PROJ_DIR=output/zoom_proj_v2
PNG_DIR=output/zoom_png_v2
CFG_DIR=config/zoom_frames_v2

if [[ "$REGRID" == "1" ]]; then
    mkdir -p "$GRID_DIR"
    for cfg in zoom_grid zoom_grid_t2 zoom_grid_mid zoom_grid_t4 zoom_grid_close; do
        echo "=== GRIDDER: ${cfg} ==="
        srun -n 4 -c 8 ./build/gridder \
            --snapshot "$SNAP" \
            --config "config/${cfg}.yaml" \
            --output "$GRID_DIR"
    done
else
    echo "=== REUSING EXISTING GRIDS IN ${GRID_DIR} ==="
fi

echo "=== RENDER + PLOT + ENCODE (zoom 16:9, full-box LOS, v2) ==="
rm -rf "$PROJ_DIR" "$PNG_DIR"
python3 python/zoom_frames.py --render --plot --encode \
    --frames 120 --trim-start-frame 19 --image-width 1920 --image-height 1080 \
    --grid-dir "$GRID_DIR" \
    --config-dir "$CFG_DIR" \
    --proj-dir "$PROJ_DIR" \
    --png-dir "$PNG_DIR" \
    --video output/zoom_16x9_v2.mp4

echo "=== DONE ==="
