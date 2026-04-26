#!/bin/bash
#SBATCH -A AST211
#SBATCH -J sph_dm158
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 32
#SBATCH -t 01:00:00
#SBATCH -p batch
#SBATCH -o output/sph_dm_snap158_%j.out
#SBATCH -e output/sph_dm_snap158_%j.err

# DM column density test on the snap_158 cube, frame 0.
module load hdf5
source ~/.bashrc
conda activate chii
cd /lustre/orion/ast211/proj-shared/rongrong/lumina_visualization/cosmo_viz

export OMP_NUM_THREADS=32
export OMP_PLACES=cores
export OMP_PROC_BIND=close

SNAP=/lustre/orion/ast211/proj-shared/rongrong/extract_snap158/output/snap_158_cube.hdf5
REGION=config/snap158_dm/region.yaml
CAMERA=config/snap158_dm/frame_0000.yaml
OUTDIR=output/snap158_dm_f0

mkdir -p "$OUTDIR"

srun -n 1 -c ${OMP_NUM_THREADS} --cpu-bind=threads \
    ./build/sph_renderer \
    --snapshot "$SNAP" \
    --region   "$REGION" \
    --camera   "$CAMERA" \
    --output   "$OUTDIR" \
    --fields   dm_density \
    --no-volume

python python/plot_density_matter_fade.py \
    --input  "$OUTDIR/dm_column.h5" \
    --output "$OUTDIR/dm_column_matter_fade.png" \
    --field  dm_column_density \
    --cmap   matter_r_fade \
    --alpha-gamma 0.4 \
    --vmin-pct 2 --vmax-pct 99.95 \
    --bg black

echo "=== DONE ==="
ls -l "$OUTDIR"
