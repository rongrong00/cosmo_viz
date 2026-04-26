#!/bin/bash
#SBATCH -A AST211
#SBATCH -J sph_orb_enc
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 16
#SBATCH -t 00:30:00
#SBATCH -p batch
#SBATCH -o output/sph_orbit_encode_%j.out
#SBATCH -e output/sph_orbit_encode_%j.err

# Plot + encode the already-rendered circle-orbit frames.
source ~/.bashrc
conda activate chii
cd /lustre/orion/ast211/proj-shared/rongrong/lumina_visualization/cosmo_viz

python python/orbit_frames_sph.py --plot --encode \
    --frames  120 \
    --out-dir output/orbit_sph \
    --png-dir output/orbit_sph_png_bare \
    --video   output/orbit_sph/orbit_sph.mp4

echo "=== DONE ==="
ls -l output/orbit_sph/orbit_sph.mp4
