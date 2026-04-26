#!/bin/bash
#SBATCH -A AST211
#SBATCH -J sph_phvol
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 32
#SBATCH -t 02:00:00
#SBATCH -p batch
#SBATCH -o output/sph_phasevol_f0_%j.out
#SBATCH -e output/sph_phasevol_f0_%j.err

# Volume-rendered density + tri-phase temperature (cold/warm/hot) for
# spiral-orbit frame 0. Each phase uses emission = ρ·𝟙[T∈bin] so the
# front-to-back emission/absorption integral gives real occlusion. Python
# then composites the 3 (E,T) pairs into one tri-color image.
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
OUTDIR=output/orbit_sph_spiral_phasevol_f0

mkdir -p "$OUTDIR"

srun -n 1 -c ${OMP_NUM_THREADS} --cpu-bind=threads \
    ./build/sph_renderer \
    --snapshot "$SNAP" \
    --region   "$REGION" \
    --camera   "$CAMERA" \
    --output   "$OUTDIR" \
    --supersample 2 \
    --fields   gas_density,gas_cold,gas_warm,gas_hot

# Density volume → inferno
python python/plot_vol_composite.py \
    --input  "$OUTDIR/gas_density_vol.h5" \
    --output "$OUTDIR/gas_density_vol.png" \
    --cmap inferno --alpha-norm --alpha-gamma 0.8

# Tri-phase temperature composite: cold=deep blue, warm=green, hot=red
# to match the Illustris-style halo reference. Lower pmin pulls more of
# the diffuse gas into visibility.
python python/composite_phase_vol.py \
    --indir  "$OUTDIR" \
    --output "$OUTDIR/tri_phase_vol.png" \
    --cold-color '#2060ff' --warm-color '#3ddf8f' --hot-color '#ff4040' \
    --cold-gain 1.5 --warm-gain 1.3 --hot-gain 1.6 \
    --pmin 30 --pmax 99.7 --alpha-gamma 0.8

echo "=== DONE ==="
ls -l "$OUTDIR"
