#!/bin/bash
#SBATCH -A AST211
#SBATCH -J cosmo_p2
#SBATCH -N 1
#SBATCH -n 10
#SBATCH -t 01:00:00
#SBATCH -p batch
#SBATCH -o output/phase2_%j.out
#SBATCH -e output/phase2_%j.err

module load hdf5

cd /lustre/orion/ast211/proj-shared/rongrong/lumina_visualization/cosmo_viz

export OMP_NUM_THREADS=1

echo "=== GRIDDER ==="
srun -n 10 ./build/gridder \
    --snapshot /lustre/orion/ast211/proj-shared/EDE_Final/finalTest/50cMpc_2/output/snapdir_121/snap_121 \
    --config config/phase2_grid.yaml \
    --output output/grids/

echo "=== RENDERER ==="
OMP_NUM_THREADS=32 ./build/renderer \
    --grid output/grids/grid_phase2_snap121.hdf5 \
    --camera config/phase2_camera.yaml \
    --output output/projections/

echo "=== DONE ==="
