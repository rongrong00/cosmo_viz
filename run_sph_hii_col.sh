#!/bin/bash
#SBATCH -A AST211
#SBATCH -J sph_hii_col
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 32
#SBATCH -t 00:45:00
#SBATCH -p batch
#SBATCH -o output/sph_hii_col_%j.out
#SBATCH -e output/sph_hii_col_%j.err

module load hdf5
cd /lustre/orion/ast211/proj-shared/rongrong/lumina_visualization/cosmo_viz

export OMP_NUM_THREADS=32
export OMP_PLACES=cores
export OMP_PROC_BIND=close

# z=10 snapshot (mid-reionization) — visible bubbles.
SNAP=/lustre/orion/ast211/proj-shared/EDE_Final/finalTest/50cMpc_2/output/snapdir_050/snap_050
REGION=config/zoom_region_10mpc.yaml
CAMERA=config/sph_cam_4k.yaml
OUTDIR=output/sph_hii_col_z10

mkdir -p "$OUTDIR"

srun -n 1 -c ${OMP_NUM_THREADS} --cpu-bind=threads \
    ./build/sph_renderer \
    --snapshot "$SNAP" \
    --region   "$REGION" \
    --camera   "$CAMERA" \
    --output   "$OUTDIR" \
    --fields   hii_column

echo "=== DONE ==="
ls -la "$OUTDIR"
