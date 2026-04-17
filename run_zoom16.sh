#!/bin/bash
#SBATCH -A AST211
#SBATCH -J cosmo_zoom16
#SBATCH -N 1
#SBATCH -n 4
#SBATCH -c 8
#SBATCH -t 06:00:00
#SBATCH -p batch
#SBATCH -o output/zoom16_%j.out
#SBATCH -e output/zoom16_%j.err

set -euo pipefail

module load hdf5
source /autofs/nccs-svm1_sw/andes/miniforge3/23.11.0/etc/profile.d/conda.sh
conda activate chii

cd /lustre/orion/ast211/proj-shared/rongrong/lumina_visualization/cosmo_viz

SNAP=/lustre/orion/ast211/proj-shared/EDE_Final/finalTest/50cMpc_2/output/snapdir_121/snap_121

export OMP_NUM_THREADS=8

# Regrid the 4 inner zoom tiers as tall-in-z slabs (xy narrow, z = BoxSize).
# zoom_wide is already a full-box cube and doesn't need to change, but we
# rebuild it too so every tier is written with the new header format.
for cfg in zoom_grid zoom_grid_t2 zoom_grid_mid zoom_grid_t4 zoom_grid_close; do
    echo "=== GRIDDER: ${cfg} ==="
    srun -n 4 -c 8 ./build/gridder \
        --snapshot "$SNAP" \
        --config "config/${cfg}.yaml" \
        --output output/grids/
done

echo "=== RENDER + PLOT + ENCODE (zoom 16:9, full-box LOS) ==="
rm -rf output/zoom_proj output/zoom_png
python3 python/zoom_frames.py --render --plot --encode \
    --frames 120 --image-width 1920 --image-height 1080 \
    --video output/zoom_16x9.mp4

echo "=== DONE ==="
