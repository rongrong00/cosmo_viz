#!/bin/bash
#SBATCH -A AST211
#SBATCH -J cosmo_zoom16_samey
#SBATCH -N 4
#SBATCH -n 32
#SBATCH -c 4
#SBATCH -t 08:00:00
#SBATCH -p batch
#SBATCH -o output/zoom16_samey_%j.out
#SBATCH -e output/zoom16_samey_%j.err

set -euo pipefail

export REGRID=${REGRID:-0}
export LOS_DEPTH_MODE=same-as-y
export VIDEO=output/zoom_16x9_samey.mp4
export PROJ_DIR=output/zoom_proj_samey
export PNG_DIR=output/zoom_png_samey
export CFG_DIR=config/zoom_frames_samey
export GRID_DIR=output/grids_samey
export GRID_CFGS="zoom_grid_samey_wide zoom_grid_samey_t2 zoom_grid_samey_mid zoom_grid_samey_t4 zoom_grid_samey_close zoom_grid_samey_t5 zoom_grid_samey_t6"

bash run_zoom16_deep.sh
