#!/bin/bash
#SBATCH -A AST211
#SBATCH -J sph_ph158
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 32
#SBATCH -t 02:00:00
#SBATCH -p batch
#SBATCH -o output/sph_phase_snap158_%j.out
#SBATCH -e output/sph_phase_snap158_%j.err

# Tri-phase temperature projection on the extracted snap_158 cube
# (50 cMpc physical at z=3). Frame 0 = default perspective view of
# the whole cube.
module load hdf5
source ~/.bashrc
conda activate chii
cd /lustre/orion/ast211/proj-shared/rongrong/lumina_visualization/cosmo_viz

export OMP_NUM_THREADS=32
export OMP_PLACES=cores
export OMP_PROC_BIND=close

SNAP=/lustre/orion/ast211/proj-shared/rongrong/extract_snap158/output/snap_158_cube.hdf5
REGION=config/snap158_phase/region.yaml
CAMERA=config/snap158_phase/frame_0000.yaml
OUTDIR=output/snap158_phase_f0

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
    --cold-gain 0.55 --warm-gain 1.1 --hot-gain 0.7 \
    --pmin 55 --pmax 99.9

echo "=== DONE ==="
ls -l "$OUTDIR"
