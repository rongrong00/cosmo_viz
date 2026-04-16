#!/bin/bash
#SBATCH -A AST211
#SBATCH -J cosmo_render
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 00:30:00
#SBATCH -p batch
#SBATCH -o output/render_hires_%j.out
#SBATCH -e output/render_hires_%j.err

module load hdf5

cd /lustre/orion/ast211/proj-shared/rongrong/lumina_visualization/cosmo_viz

./build/renderer \
    --grid output/grids/grid_hires_snap121.hdf5 \
    --camera config/hires_camera.yaml \
    --output output/projections/

echo "=== DONE ==="
