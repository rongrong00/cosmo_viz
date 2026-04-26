#!/bin/bash
#SBATCH -A AST211
#SBATCH -J sph_den158
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 32
#SBATCH -t 01:00:00
#SBATCH -p batch
#SBATCH -o output/sph_density_snap158_%j.out
#SBATCH -e output/sph_density_snap158_%j.err

# Gas column density projection on the snap_158 cube, frame 0.
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
OUTDIR=output/snap158_density_f0

mkdir -p "$OUTDIR"

srun -n 1 -c ${OMP_NUM_THREADS} --cpu-bind=threads \
    ./build/sph_renderer \
    --snapshot "$SNAP" \
    --region   "$REGION" \
    --camera   "$CAMERA" \
    --output   "$OUTDIR" \
    --fields   gas_density \
    --no-volume

python python/plotter.py \
    --input  "$OUTDIR/gas_column.h5" \
    --output "$OUTDIR/gas_column.png" \
    --field gas_column_density \
    --cmap inferno --bare

echo "=== DONE ==="
ls -l "$OUTDIR"
