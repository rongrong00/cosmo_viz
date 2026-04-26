#!/bin/bash
#SBATCH -A AST211
#SBATCH -J sph_metf0
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 32
#SBATCH -t 01:00:00
#SBATCH -p batch
#SBATCH -o output/sph_metal_f0_%j.out
#SBATCH -e output/sph_metal_f0_%j.err

# Render gas metal column density for spiral-orbit frame 0.
module load hdf5
source ~/.bashrc
conda activate chii
cd /lustre/orion/ast211/proj-shared/rongrong/lumina_visualization/cosmo_viz

export OMP_NUM_THREADS=32
export OMP_PLACES=cores
export OMP_PROC_BIND=close

SNAP=/lustre/orion/ast211/proj-shared/EDE_Final/finalTest/50cMpc_2/output/snapdir_121/snap_121
REGION=config/orbit_sph_spiral/region.yaml
CAMERA=config/orbit_sph_spiral/frame_0000.yaml
OUTDIR=output/orbit_sph_spiral_metal_f0

mkdir -p "$OUTDIR"

srun -n 1 -c ${OMP_NUM_THREADS} --cpu-bind=threads \
    ./build/sph_renderer \
    --snapshot "$SNAP" \
    --region   "$REGION" \
    --camera   "$CAMERA" \
    --output   "$OUTDIR" \
    --fields   gas_density,metallicity_dw,temperature_dw \
    --no-volume

echo "=== DONE ==="
ls -l "$OUTDIR"
