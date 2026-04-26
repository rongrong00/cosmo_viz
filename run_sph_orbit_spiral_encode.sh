#!/bin/bash
#SBATCH -A AST211
#SBATCH -J sph_spi_enc
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 16
#SBATCH -t 01:00:00
#SBATCH -p batch
#SBATCH -o output/sph_orbit_spiral_encode_%j.out
#SBATCH -e output/sph_orbit_spiral_encode_%j.err

# Plot bare PNGs + encode the spiral zoom-in frames at native 1920x1080.
source ~/.bashrc
conda activate chii
cd /lustre/orion/ast211/proj-shared/rongrong/lumina_visualization/cosmo_viz

python python/orbit_frames_sph.py --plot --encode \
    --frames  480 \
    --out-dir output/orbit_sph_spiral \
    --png-dir output/orbit_sph_spiral_png_bare \
    --video   output/orbit_sph_spiral/orbit_sph_spiral.mp4

echo "=== DONE ==="
ls -l output/orbit_sph_spiral/orbit_sph_spiral.mp4
