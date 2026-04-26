#!/bin/bash
#SBATCH -A AST211
#SBATCH -J stars_h8l
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 16
#SBATCH -t 00:30:00
#SBATCH -p batch
#SBATCH -o output/stars_halo8_last_%j.out
#SBATCH -e output/stars_halo8_last_%j.err

# Dust-free stellar light ray trace (python) for halo 8 last frame.
module load hdf5
source ~/.bashrc
conda activate chii
cd /lustre/orion/ast211/proj-shared/rongrong/lumina_visualization/cosmo_viz

SNAP=/lustre/orion/ast211/proj-shared/rongrong/extract_snap158/output_halo8/snap_158_cube.hdf5
CAMERA=config/snap158_halo8_flyin/frame_last.yaml
OUTDIR=output/snap158_halo8_stars_last
mkdir -p "$OUTDIR"

python python/stellar_trace.py \
    --snapshot "$SNAP" \
    --camera   "$CAMERA" \
    --output   "$OUTDIR/stars_ugr.h5"

python python/lupton_rgb.py \
    --input  "$OUTDIR/stars_ugr.h5" \
    --output "$OUTDIR/stars_rgb.png" \
    --Q 8 --pct 99.5

echo "=== DONE ==="
ls -l "$OUTDIR"
