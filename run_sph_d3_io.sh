#!/bin/bash
#SBATCH -A AST211
#SBATCH -J sph_d3_io
#SBATCH -N 2
#SBATCH -n 2
#SBATCH -c 32
#SBATCH -t 00:30:00
#SBATCH -p batch
#SBATCH -o output/sph_d3_io_%j.out
#SBATCH -e output/sph_d3_io_%j.err

# D3 parallel-I/O smoke: 2 nodes × 1 leader rank each → subfile reads
# round-robin across the 2 node leaders via ParticleLoader::loadMPI, then
# Allgatherv consolidates. Followers on each node use shared-mem replicated
# buffers (D2 path).

module load hdf5
cd /lustre/orion/ast211/proj-shared/rongrong/lumina_visualization/cosmo_viz

export OMP_NUM_THREADS=32
export OMP_PLACES=cores
export OMP_PROC_BIND=close

SNAP=/lustre/orion/ast211/proj-shared/EDE_Final/finalTest/50cMpc_2/output/snapdir_050/snap_050
REGION=config/zoom_region_10mpc.yaml
CAMERA=config/sph_cam_4k.yaml
OUTDIR=output/sph_d3_io

mkdir -p "$OUTDIR"

CAM_LIST=$(mktemp --suffix=.txt)
echo "$CAMERA"  > "$CAM_LIST"
echo "$CAMERA" >> "$CAM_LIST"

srun -N 2 -n 2 --ntasks-per-node=1 -c ${OMP_NUM_THREADS} --cpu-bind=threads \
    ./build/sph_renderer \
    --snapshot "$SNAP" \
    --region   "$REGION" \
    --camera-list "$CAM_LIST" \
    --output   "$OUTDIR" \
    --fields   neutral_H,ionized_H

rm -f "$CAM_LIST"
echo "=== DONE ==="
ls -la "$OUTDIR"/frame_*/ 2>/dev/null
