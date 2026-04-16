#!/bin/bash
#SBATCH -A AST211
#SBATCH -J cosmo_grid
#SBATCH -N 1
#SBATCH -n 10
#SBATCH -t 01:00:00
#SBATCH -p batch
#SBATCH -o output/gridder_hires_%j.out
#SBATCH -e output/gridder_hires_%j.err

module load hdf5

cd /lustre/orion/ast211/proj-shared/rongrong/lumina_visualization/cosmo_viz

mkdir -p output/grids output/projections output/figures

echo "=== GRIDDER ==="
srun -n 10 ./build/gridder \
    --snapshot /lustre/orion/ast211/proj-shared/EDE_Final/finalTest/50cMpc_2/output/snapdir_121/snap_121 \
    --config config/hires_grid.yaml \
    --output output/grids/

echo "=== RENDERER ==="
./build/renderer \
    --grid output/grids/grid_hires_snap121.hdf5 \
    --camera config/hires_camera.yaml \
    --output output/projections/

echo "=== DONE ==="
