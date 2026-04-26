#!/bin/bash
#SBATCH -A AST211
#SBATCH -J zoom_ladder
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 32
#SBATCH -t 02:00:00
#SBATCH -p batch
#SBATCH -o output/zoom_ladder_%j.out
#SBATCH -e output/zoom_ladder_%j.err

# Ray-traced baryonic (gas + stars) density for zoom-ladder panels 2, 3, 4.
# All renders run against the halo 0 cube (50 cMpc wide). Panel sizes:
#   p2: 50 cMpc (33830 ckpc/h)   - bar 10 cMpc
#   p3:  5 cMpc ( 3383 ckpc/h)   - bar  1 cMpc
#   p4: 500 ckpc ( 338.3 ckpc/h) - bar 100 ckpc
module load hdf5
source ~/.bashrc
conda activate chii
cd /lustre/orion/ast211/proj-shared/rongrong/lumina_visualization/cosmo_viz

export OMP_NUM_THREADS=32
export OMP_PLACES=cores
export OMP_PROC_BIND=close

SNAP=/lustre/orion/ast211/proj-shared/rongrong/extract_snap158/output/snap_158_cube.hdf5

for P in panel2 panel3 panel4; do
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
        --no-volume
done

echo "=== DONE ==="
ls -l output/zoom_ladder_*/
