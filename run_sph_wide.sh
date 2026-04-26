#!/bin/bash
#SBATCH -A AST211
#SBATCH -J sph_wide
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 32
#SBATCH -t 00:45:00
#SBATCH -p batch
#SBATCH -o output/sph_wide_%j.out
#SBATCH -e output/sph_wide_%j.err

# Wide-field SPH render over the 30 Mpc cube.

module load hdf5

cd /lustre/orion/ast211/proj-shared/rongrong/lumina_visualization/cosmo_viz

export OMP_NUM_THREADS=32
export OMP_PLACES=cores
export OMP_PROC_BIND=close

SNAP=/lustre/orion/ast211/proj-shared/EDE_Final/finalTest/50cMpc_2/output/snapdir_121/snap_121
REGION=config/zoom_region_30mpc.yaml
CAMERA=config/sph_cam_30mpc.yaml
OUTDIR=output/sph_wide_test

mkdir -p "$OUTDIR"

srun -n 1 -c ${OMP_NUM_THREADS} --cpu-bind=threads \
    ./build/sph_renderer \
    --snapshot "$SNAP" \
    --region   "$REGION" \
    --camera   "$CAMERA" \
    --output   "$OUTDIR" \
    --fields   gas_density

echo "=== DONE ==="
ls -l "$OUTDIR"
