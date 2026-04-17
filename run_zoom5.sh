#!/bin/bash
#SBATCH -A AST211
#SBATCH -J cosmo_zoom5
#SBATCH -N 1
#SBATCH -n 4
#SBATCH -c 8
#SBATCH -t 06:00:00
#SBATCH -p batch
#SBATCH -o output/zoom5_%j.out
#SBATCH -e output/zoom5_%j.err

set -euo pipefail

module load hdf5
source /autofs/nccs-svm1_sw/andes/miniforge3/23.11.0/etc/profile.d/conda.sh
conda activate chii

cd /lustre/orion/ast211/proj-shared/rongrong/lumina_visualization/cosmo_viz

SNAP=/lustre/orion/ast211/proj-shared/EDE_Final/finalTest/50cMpc_2/output/snapdir_121/snap_121

# All 5 tiers at 1024^3 with the new h_min = 0.5 * cell_size floor in the
# depositor. 4 ranks * 8 OMP threads fits 1024^3 grids comfortably in RAM.
export OMP_NUM_THREADS=8

for cfg in zoom_grid zoom_grid_t2 zoom_grid_mid zoom_grid_t4 zoom_grid_close; do
    echo "=== GRIDDER: ${cfg} ==="
    srun -n 4 -c 8 ./build/gridder \
        --snapshot "$SNAP" \
        --config "config/${cfg}.yaml" \
        --output output/grids/
done

echo "=== RENDER + PLOT + ENCODE (5 tiers, 1440px) ==="
rm -rf output/zoom_proj output/zoom_png
python3 python/zoom_frames.py --render --plot --encode \
    --frames 120 --image-size 1440 --video output/zoom_test.mp4

echo "=== DONE ==="
