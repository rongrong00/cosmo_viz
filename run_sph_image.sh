#!/bin/bash
#SBATCH -A AST211
#SBATCH -J sph_image
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 32
#SBATCH -t 00:30:00
#SBATCH -p batch
#SBATCH -o output/sph_image_%j.out
#SBATCH -e output/sph_image_%j.err

# Phase-B3 first end-to-end SPH image:
# - loads the 10 Mpc zoom region,
# - builds gas BVH,
# - ray-traces gas column density into a 1920x1080 image,
# - writes HDF5 with /Header + /gas_column_density.

module load hdf5

cd /lustre/orion/ast211/proj-shared/rongrong/lumina_visualization/cosmo_viz

export OMP_NUM_THREADS=32
export OMP_PLACES=cores
export OMP_PROC_BIND=close

SNAP=/lustre/orion/ast211/proj-shared/EDE_Final/finalTest/50cMpc_2/output/snapdir_121/snap_121
REGION=config/zoom_region_10mpc.yaml
CAMERA=config/sph_cam_4k.yaml
OUTDIR=output/sph_image_test_4k

mkdir -p "$OUTDIR"

echo "=== Phase B3 end-to-end SPH image ==="
srun -n 1 -c ${OMP_NUM_THREADS} --cpu-bind=threads \
    ./build/sph_renderer \
    --snapshot "$SNAP" \
    --region   "$REGION" \
    --camera   "$CAMERA" \
    --output   "$OUTDIR" \
    --fields   gas_density

echo "=== DONE ==="
ls -l "$OUTDIR"
