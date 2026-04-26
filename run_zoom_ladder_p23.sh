#!/bin/bash
#SBATCH -A AST211
#SBATCH -J zoom_p23
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 32
#SBATCH -t 01:00:00
#SBATCH -p batch
#SBATCH -o output/zoom_ladder_p23_%j.out
#SBATCH -e output/zoom_ladder_p23_%j.err

# Panels 2 (50 cMpc) and 3 (5 cMpc) with the same baryon+render params
# as panel 4.
module load hdf5
source ~/.bashrc
conda activate chii
cd /lustre/orion/ast211/proj-shared/rongrong/lumina_visualization/cosmo_viz

export OMP_NUM_THREADS=32
export OMP_PLACES=cores
export OMP_PROC_BIND=close

SNAP=/lustre/orion/ast211/proj-shared/rongrong/extract_snap158/output/snap_158_cube.hdf5

for P in panel2 panel3; do
    REGION=config/zoom_ladder/${P}.yaml
    CAMERA=config/zoom_ladder/cam_${P}.yaml
    OUTDIR=output/zoom_ladder_${P}
    mkdir -p "$OUTDIR"

    echo "=== Rendering $P ==="
    srun -n 1 -c ${OMP_NUM_THREADS} --cpu-bind=threads \
        ./build/sph_renderer \
        --snapshot "$SNAP" \
        --region   "$REGION" \
        --camera   "$CAMERA" \
        --output   "$OUTDIR" \
        --fields   gas_density,star_density \
        --gas-h-scale 2.0 \
        --star-nn-k   16 \
        --star-h-max  0 \
        --star-h-min  1.77 \
        --no-volume

    python python/baryon_composite.py \
        --gas     "$OUTDIR/gas_column.h5" \
        --star    "$OUTDIR/star_column.h5" \
        --h5-out  "$OUTDIR/baryon_column.h5" \
        --png-out "$OUTDIR/baryon.png" \
        --cmap matter_r_fade --vmax-pct 99.9
done

echo "=== DONE ==="
ls -l output/zoom_ladder_panel2 output/zoom_ladder_panel3
