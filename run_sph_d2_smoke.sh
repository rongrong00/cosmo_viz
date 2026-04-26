#!/bin/bash
#SBATCH -A AST211
#SBATCH -J sph_d2
#SBATCH -N 1
#SBATCH -n 2
#SBATCH -c 16
#SBATCH -t 00:30:00
#SBATCH -p batch
#SBATCH -o output/sph_d2_smoke_%j.out
#SBATCH -e output/sph_d2_smoke_%j.err

# D2 smoke test: 2 ranks on one node. Exercises the MPI shared-memory
# replication code path (node_rank==0 memcpy, followers MPI_Win_shared_query).
# Each rank round-robins a subset of camera frames — here we pass a single
# --camera, so rank 0 renders it and rank 1 idles (still proves the shared
# buffers are readable from the follower rank).

module load hdf5
cd /lustre/orion/ast211/proj-shared/rongrong/lumina_visualization/cosmo_viz

export OMP_NUM_THREADS=16
export OMP_PLACES=cores
export OMP_PROC_BIND=close

SNAP=/lustre/orion/ast211/proj-shared/EDE_Final/finalTest/50cMpc_2/output/snapdir_050/snap_050
REGION=config/zoom_region_10mpc.yaml
CAMERA=config/sph_cam_4k.yaml
OUTDIR=output/sph_d2_smoke

mkdir -p "$OUTDIR"

# Two cameras so both ranks render.
CAM_LIST=$(mktemp --suffix=.txt)
echo "$CAMERA"  > "$CAM_LIST"
echo "$CAMERA" >> "$CAM_LIST"

srun -n 2 -c ${OMP_NUM_THREADS} --cpu-bind=threads \
    ./build/sph_renderer \
    --snapshot "$SNAP" \
    --region   "$REGION" \
    --camera-list "$CAM_LIST" \
    --output   "$OUTDIR" \
    --fields   neutral_H,ionized_H

rm -f "$CAM_LIST"
echo "=== DONE ==="
ls -la "$OUTDIR"/frame_*/ 2>/dev/null
