#!/bin/bash
#SBATCH -A AST211
#SBATCH -J sph_phf0
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 32
#SBATCH -t 01:00:00
#SBATCH -p batch
#SBATCH -o output/sph_phase_f0_%j.out
#SBATCH -e output/sph_phase_f0_%j.err

# Tri-phase (cold / warm / hot) mass-column maps for spiral-orbit frame 0,
# then composite to a single tri-color PNG (Illustris-style).
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
OUTDIR=output/orbit_sph_spiral_phase_f0

mkdir -p "$OUTDIR"

srun -n 1 -c ${OMP_NUM_THREADS} --cpu-bind=threads \
    ./build/sph_renderer \
    --snapshot "$SNAP" \
    --region   "$REGION" \
    --camera   "$CAMERA" \
    --output   "$OUTDIR" \
    --fields   gas_density,cold_col,warm_col,hot_col \
    --no-volume

python python/composite_phases.py \
    --indir  "$OUTDIR" \
    --output "$OUTDIR/tri_phase.png" \
    --cold-color '#1a4fff' --warm-color '#30e0ff' --hot-color '#ff3020' \
    --cold-gain 1.0 --warm-gain 1.1 --hot-gain 1.3 \
    --pmin 40 --pmax 99.9

echo "=== DONE ==="
ls -l "$OUTDIR"
