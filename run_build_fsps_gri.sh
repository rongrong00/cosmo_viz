#!/bin/bash
#SBATCH -A AST211
#SBATCH -J fsps_gri
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 8
#SBATCH -t 01:00:00
#SBATCH -p batch
#SBATCH -o output/fsps_gri_%j.out
#SBATCH -e output/fsps_gri_%j.err

source ~/.bashrc
conda activate chii
cd /lustre/orion/ast211/proj-shared/rongrong/lumina_visualization/cosmo_viz
export SPS_HOME=/lustre/orion/ast211/proj-shared/rongrong/fsps

python python/build_fsps_gri_grid.py config/fsps_gri_ssp.h5

echo "=== DONE ==="
ls -l config/fsps_gri_ssp.h5
