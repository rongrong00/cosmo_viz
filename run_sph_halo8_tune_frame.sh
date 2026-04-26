#!/bin/bash
#SBATCH -A AST211
#SBATCH -J h8_tune
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 32
#SBATCH -t 00:15:00
#SBATCH -p batch
#SBATCH -o output/sph_halo8_tune_%j.out
#SBATCH -e output/sph_halo8_tune_%j.err

# Single-frame halo-8 render for metallicity colormap tuning.
# Uses the existing frame_first.yaml (camera on +x at sphere boundary, fov=60).
module load hdf5
source ~/.bashrc
conda activate chii
cd /lustre/orion/ast211/proj-shared/rongrong/lumina_visualization/cosmo_viz

export OMP_NUM_THREADS=32
export OMP_PLACES=cores
export OMP_PROC_BIND=close

SNAP=/lustre/orion/ast211/proj-shared/rongrong/extract_snap158/output_halo8/snap_158_cube.hdf5
CAMERA=config/snap158_halo8_flyin/frame_first.yaml
REGION=config/snap158_halo8_flyin/region.yaml
OUTDIR=output/snap158_halo8_tune_frame

mkdir -p "$OUTDIR"

srun -n 1 -c ${OMP_NUM_THREADS} --cpu-bind=threads \
    ./build/sph_renderer \
    --snapshot "$SNAP" \
    --region   "$REGION" \
    --camera   "$CAMERA" \
    --output   "$OUTDIR" \
    --fields   gas_density,temperature,metallicity \
    --no-volume

ls -l "$OUTDIR"
echo "=== DONE ==="
