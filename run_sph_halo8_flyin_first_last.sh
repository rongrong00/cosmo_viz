#!/bin/bash
#SBATCH -A AST211
#SBATCH -J sph_h8fl
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 32
#SBATCH -t 01:30:00
#SBATCH -p batch
#SBATCH -o output/sph_halo8_flyin_%j.out
#SBATCH -e output/sph_halo8_flyin_%j.err

# First + last gas density frames of the halo 8 flyin (snap_158 cube).
module load hdf5
source ~/.bashrc
conda activate chii
cd /lustre/orion/ast211/proj-shared/rongrong/lumina_visualization/cosmo_viz

export OMP_NUM_THREADS=32
export OMP_PLACES=cores
export OMP_PROC_BIND=close

SNAP=/lustre/orion/ast211/proj-shared/rongrong/extract_snap158/output_halo8/snap_158_cube.hdf5
REGION=config/snap158_halo8_flyin/region.yaml

for TAG in first last; do
    CAMERA=config/snap158_halo8_flyin/frame_${TAG}.yaml
    OUTDIR=output/snap158_halo8_flyin_${TAG}

    mkdir -p "$OUTDIR"

    srun -n 1 -c ${OMP_NUM_THREADS} --cpu-bind=threads \
        ./build/sph_renderer \
        --snapshot "$SNAP" \
        --region   "$REGION" \
        --camera   "$CAMERA" \
        --output   "$OUTDIR" \
        --fields   gas_density \
        --no-volume

    python python/plot_density_matter_fade.py \
        --input  "$OUTDIR/gas_column.h5" \
        --output "$OUTDIR/gas_column_matter_fade.png" \
        --cmap   matter_r_fade \
        --alpha-gamma 0.4 \
        --vmin-pct 2 --vmax-pct 99.95 \
        --bg black
done

echo "=== DONE ==="
