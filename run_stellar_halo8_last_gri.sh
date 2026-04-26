#!/bin/bash
#SBATCH -A AST211
#SBATCH -J stars_h8gri
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 16
#SBATCH -t 00:30:00
#SBATCH -p batch
#SBATCH -o output/stars_halo8_last_gri_%j.out
#SBATCH -e output/stars_halo8_last_gri_%j.err
#SBATCH --dependency=singleton

# Stellar ray trace with Illustris-style g,r,i -> B,G,R composite.
module load hdf5
source ~/.bashrc
conda activate chii
cd /lustre/orion/ast211/proj-shared/rongrong/lumina_visualization/cosmo_viz

SNAP=/lustre/orion/ast211/proj-shared/rongrong/extract_snap158/output_halo8/snap_158_cube.hdf5
CAMERA=config/snap158_halo8_flyin/frame_last.yaml
OUTDIR=output/snap158_halo8_stars_last_gri
mkdir -p "$OUTDIR"

python python/stellar_trace.py \
    --snapshot "$SNAP" \
    --camera   "$CAMERA" \
    --fsps-grid config/fsps_gri_ssp.h5 \
    --clip-center 233794.84 60858.605 146579.95 \
    --clip-radius 20.0 \
    --clip-dim    0.10 \
    --output   "$OUTDIR/stars_gri.h5"

python python/lupton_rgb.py \
    --input  "$OUTDIR/stars_gri.h5" \
    --output "$OUTDIR/stars_rgb_gri.png" \
    --channels L_i,L_r,L_g \
    --Q 8 --pct 99.5

echo "=== DONE ==="
ls -l "$OUTDIR"
