#!/bin/bash
#SBATCH -A AST211
#SBATCH -J cosmo_zoom16_10k
#SBATCH -N 4
#SBATCH -n 32
#SBATCH -c 4
#SBATCH -t 08:00:00
#SBATCH -p batch
#SBATCH -o output/zoom16_10k_%j.out
#SBATCH -e output/zoom16_10k_%j.err

set -euo pipefail

export REGRID=${REGRID:-0}
export LOS_DEPTH_MODE=fixed
export LOS_SLAB=10000
export VIDEO=output/zoom_16x9_fixed10k.mp4
export PROJ_DIR=output/zoom_proj_fixed10k
export PNG_DIR=output/zoom_png_fixed10k
export CFG_DIR=config/zoom_frames_fixed10k

bash run_zoom16_deep.sh
