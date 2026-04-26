#!/bin/bash
#SBATCH -A AST211
#SBATCH -J sph_meth8
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 32
#SBATCH -t 01:00:00
#SBATCH -p batch
#SBATCH -o output/sph_metal_halo8_%j.out
#SBATCH -e output/sph_metal_halo8_%j.err

# Mass-weighted gas metallicity projection on snap_158 halo 8 cube, frame 0.
module load hdf5
source ~/.bashrc
conda activate chii
cd /lustre/orion/ast211/proj-shared/rongrong/lumina_visualization/cosmo_viz

export OMP_NUM_THREADS=32
export OMP_PLACES=cores
export OMP_PROC_BIND=close

SNAP=/lustre/orion/ast211/proj-shared/rongrong/extract_snap158/output_halo8/snap_158_cube.hdf5
REGION=config/snap158_halo8/region.yaml
CAMERA=config/snap158_halo8/frame_0000.yaml
OUTDIR=output/snap158_halo8_metal_f0

mkdir -p "$OUTDIR"

srun -n 1 -c ${OMP_NUM_THREADS} --cpu-bind=threads \
    ./build/sph_renderer \
    --snapshot "$SNAP" \
    --region   "$REGION" \
    --camera   "$CAMERA" \
    --output   "$OUTDIR" \
    --fields   metallicity \
    --no-volume

python python/plotter.py \
    --input  "$OUTDIR/metallicity.h5" \
    --output "$OUTDIR/gas_metallicity.png" \
    --field gas_metallicity_mw \
    --bare

echo "=== DONE ==="
ls -l "$OUTDIR"
