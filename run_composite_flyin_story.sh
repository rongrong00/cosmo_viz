#!/bin/bash
#SBATCH -A AST211
#SBATCH -J story_h8
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 16
#SBATCH -t 01:00:00
#SBATCH -p batch
#SBATCH -o output/story_halo8_%j.out
#SBATCH -e output/story_halo8_%j.err

# Composite the 5 halo-8 fly-in tracks (dm, gas, temp, metal, stars) into
# one 960-frame story video with labeled crossfades at segment boundaries.

source ~/.bashrc
conda activate chii
cd /lustre/orion/ast211/proj-shared/rongrong/lumina_visualization/cosmo_viz

OUTDIR=output/snap158_halo8_flyin_story_png
VIDEO=output/snap158_halo8_flyin_story.mp4

python python/composite_flyin_story.py

ffmpeg -y -framerate 30 \
    -i "${OUTDIR}/frame_%04d.png" \
    -c:v libx264 -pix_fmt yuv420p -crf 18 \
    -vf 'pad=ceil(iw/2)*2:ceil(ih/2)*2' \
    "$VIDEO"

ls -l "$VIDEO"
echo "=== DONE ==="
