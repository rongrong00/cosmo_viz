#!/bin/bash
#SBATCH -A AST211
#SBATCH -J cosmo_zoom16deep
#SBATCH -N 4
#SBATCH -n 32
#SBATCH -c 4
#SBATCH -t 08:00:00
#SBATCH -p batch
#SBATCH -o output/zoom16deep_%j.out
#SBATCH -e output/zoom16deep_%j.err

set -euo pipefail

module load hdf5
source /autofs/nccs-svm1_sw/andes/miniforge3/23.11.0/etc/profile.d/conda.sh
conda activate chii

cd /lustre/orion/ast211/proj-shared/rongrong/lumina_visualization/cosmo_viz

SNAP=/lustre/orion/ast211/proj-shared/EDE_Final/finalTest/50cMpc_2/output/snapdir_121/snap_121

# Set REGRID=0 to reuse existing grids; default is to rebuild this deep run's
# extra zoom tiers.
export OMP_NUM_THREADS=4
REGRID=${REGRID:-1}
LOS_DEPTH_MODE=${LOS_DEPTH_MODE:-same-as-y}
LOS_SLAB=${LOS_SLAB:-10000}
FRAMES=${FRAMES:-200}
WIDTH_END=${WIDTH_END:-$(python3 - <<'PY'
import math
print(0.5 * 361.63)
PY
)}
VIDEO=${VIDEO:-output/zoom_16x9_10mpc.mp4}
PROJ_DIR=${PROJ_DIR:-output/zoom_proj_10mpc}
PNG_DIR=${PNG_DIR:-output/zoom_png_10mpc}
GRID_DIR=${GRID_DIR:-output/grids_10mpc}
CFG_DIR=${CFG_DIR:-config/zoom_frames_10mpc}
RENDER_PARALLELISM=${RENDER_PARALLELISM:-4}
RENDER_LAUNCHER=${RENDER_LAUNCHER:-"srun --exclusive -N 1 -n 8 -c 4"}
TIER_OVERFLOW_TOL=${TIER_OVERFLOW_TOL:-0.02}
GRID_CFGS=${GRID_CFGS:-"zoom_grid_10mpc_wide zoom_grid_10mpc_t2 zoom_grid_10mpc_mid zoom_grid_10mpc_t4 zoom_grid_10mpc_close zoom_grid_10mpc_t5 zoom_grid_10mpc_t6"}

if [[ "$REGRID" == "1" ]]; then
    mkdir -p "$GRID_DIR"
    for cfg in $GRID_CFGS; do
        echo "=== GRIDDER: ${cfg} ==="
        srun -n 4 -c 4 ./build/gridder \
            --snapshot "$SNAP" \
            --config "config/${cfg}.yaml" \
            --output "$GRID_DIR"
    done
else
    echo "=== REUSING EXISTING GRIDS IN ${GRID_DIR} ==="
fi

TRIM_START_FRAME=$(python3 - <<PY
import numpy as np

frames = int(${FRAMES})
width_start = 36245.0
width_end = float(${WIDTH_END})
aspect = 1920 / 1080
wide_side = 36245.0

widths = width_start * (width_end / width_start) ** (np.arange(frames) / max(1, frames - 1))
required = widths * aspect
trim = 0
for i, req in enumerate(required):
    if req <= wide_side:
        trim = i
        break
print(trim)
PY
)

echo "=== RENDER + PLOT + ENCODE (zoom 16:9, ${LOS_DEPTH_MODE} depth, deep tiers) ==="
rm -rf "$PROJ_DIR" "$PNG_DIR"
python3 python/zoom_frames.py --render --plot --encode \
    --tier-set deep \
    --los-depth "$LOS_DEPTH_MODE" \
    --los-slab "$LOS_SLAB" \
    --full-bleed \
    --frames "$FRAMES" \
    --width-end "$WIDTH_END" \
    --tier-overflow-tol "$TIER_OVERFLOW_TOL" \
    --trim-start-frame "$TRIM_START_FRAME" \
    --image-width 1920 --image-height 1080 \
    --grid-dir "$GRID_DIR" \
    --config-dir "$CFG_DIR" \
    --proj-dir "$PROJ_DIR" \
    --png-dir "$PNG_DIR" \
    --video "$VIDEO" \
    --launcher "$RENDER_LAUNCHER" \
    --render-parallelism "$RENDER_PARALLELISM"

echo "=== DONE ==="