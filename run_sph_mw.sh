#!/bin/bash
#SBATCH -A AST211
#SBATCH -J sph_mw
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 32
#SBATCH -t 01:00:00
#SBATCH -p batch
#SBATCH -o output/sph_mw_%j.out
#SBATCH -e output/sph_mw_%j.err

module load hdf5
cd /lustre/orion/ast211/proj-shared/rongrong/lumina_visualization/cosmo_viz

export OMP_NUM_THREADS=32
export OMP_PLACES=cores
export OMP_PROC_BIND=close

SNAP=/lustre/orion/ast211/proj-shared/EDE_Final/finalTest/50cMpc_2/output/snapdir_121/snap_121
REGION=config/zoom_region_10mpc.yaml
CAMERA=config/sph_cam_4k.yaml
OUTDIR=output/sph_mw_test

mkdir -p "$OUTDIR"

srun -n 1 -c ${OMP_NUM_THREADS} --cpu-bind=threads \
    ./build/sph_renderer \
    --snapshot "$SNAP" \
    --region   "$REGION" \
    --camera   "$CAMERA" \
    --output   "$OUTDIR" \
    --fields   gas_density,temperature,metallicity,gas_velocity

echo "=== DONE ==="
ls -la "$OUTDIR"
