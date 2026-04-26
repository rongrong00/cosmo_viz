#!/bin/bash
#SBATCH -A AST211
#SBATCH -J sph_orbit
#SBATCH -N 2
#SBATCH -n 4
#SBATCH -c 16
#SBATCH -t 02:00:00
#SBATCH -p batch
#SBATCH -o output/sph_orbit_%j.out
#SBATCH -e output/sph_orbit_%j.err

# Perspective orbit movie around the largest FOF halo in snap_121 of 50cMpc_2.
# Camera orbits at R=5000 ckpc/h; spherical region has the same radius so the
# camera sits exactly on the sphere boundary. 4 MPI ranks across 2 nodes
# (2 ranks/node) render 120 frames round-robin; each rank has 16 OMP threads
# (32 cores/node on andes). Volume render is skipped — only gas column.

module load hdf5
source ~/.bashrc
conda activate chii
cd /lustre/orion/ast211/proj-shared/rongrong/lumina_visualization/cosmo_viz

export OMP_NUM_THREADS=16
export OMP_PLACES=cores
export OMP_PROC_BIND=close

SNAP=/lustre/orion/ast211/proj-shared/EDE_Final/finalTest/50cMpc_2/output/snapdir_121/snap_121
FRAMES=120
OUTDIR=output/orbit_sph
CFGDIR=config/orbit_sph
REGION=${CFGDIR}/region.yaml
CAMLIST=${CFGDIR}/camera_list.txt
PNGDIR=output/orbit_sph_png
VIDEO=output/orbit_sph.mp4

mkdir -p "$OUTDIR" "$CFGDIR"

# 1) Generate region + per-frame camera YAMLs + camera-list.
python python/orbit_frames_sph.py --prep \
    --frames       "$FRAMES" \
    --orbit-radius 5000.0 \
    --fov          40.0 \
    --image-width  1920 \
    --image-height 1080 \
    --config-dir   "$CFGDIR" \
    --region-file  "$REGION" \
    --camera-list  "$CAMLIST"

# 2) Render all frames across MPI ranks (batch mode: frame_NNNN/ subdirs).
srun -n ${SLURM_NTASKS} -c ${OMP_NUM_THREADS} --cpu-bind=threads \
    ./build/sph_renderer \
    --snapshot    "$SNAP" \
    --region      "$REGION" \
    --camera-list "$CAMLIST" \
    --output      "$OUTDIR" \
    --fields      gas_density \
    --no-volume

# 3) Plot each frame's gas_column.h5 -> PNG, then ffmpeg to mp4.
python python/orbit_frames_sph.py --plot --encode \
    --frames   "$FRAMES" \
    --out-dir  "$OUTDIR" \
    --png-dir  "$PNGDIR" \
    --video    "$VIDEO"

echo "=== DONE ==="
ls -l "$VIDEO"
