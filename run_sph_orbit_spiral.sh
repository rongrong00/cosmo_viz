#!/bin/bash
#SBATCH -A AST211
#SBATCH -J sph_orb_sp
#SBATCH -N 2
#SBATCH -n 4
#SBATCH -c 16
#SBATCH -t 03:00:00
#SBATCH -p batch
#SBATCH -o output/sph_orbit_spiral_%j.out
#SBATCH -e output/sph_orbit_spiral_%j.err

# Spiral zoom-in: camera orbits the largest FOF halo while the orbit radius
# decays log-linearly from 20 Mpc to 1 Mpc over 2 full revolutions. The
# spherical selection region has radius = r-outer so every camera position
# lies inside (outermost on the boundary). 4 MPI ranks × 16 OMP threads on
# 2 andes nodes; 480 frames round-robin. Volume render skipped.

module load hdf5
source ~/.bashrc
conda activate chii
cd /lustre/orion/ast211/proj-shared/rongrong/lumina_visualization/cosmo_viz

export OMP_NUM_THREADS=16
export OMP_PLACES=cores
export OMP_PROC_BIND=close

SNAP=/lustre/orion/ast211/proj-shared/EDE_Final/finalTest/50cMpc_2/output/snapdir_121/snap_121
FRAMES=480
OUTDIR=output/orbit_sph_spiral
CFGDIR=config/orbit_sph_spiral
REGION=${CFGDIR}/region.yaml
CAMLIST=${CFGDIR}/camera_list.txt
PNGDIR=output/orbit_sph_spiral_png
VIDEO=output/orbit_sph_spiral.mp4

R_OUTER=20000.0
R_INNER=1000.0
TURNS=2.0

mkdir -p "$OUTDIR" "$CFGDIR"

python python/orbit_frames_sph.py --prep \
    --mode         spiral \
    --frames       "$FRAMES" \
    --r-outer      "$R_OUTER" \
    --r-inner      "$R_INNER" \
    --turns        "$TURNS" \
    --fov          40.0 \
    --image-width  1920 \
    --image-height 1080 \
    --config-dir   "$CFGDIR" \
    --region-file  "$REGION" \
    --camera-list  "$CAMLIST"

srun -n ${SLURM_NTASKS} -c ${OMP_NUM_THREADS} --cpu-bind=threads \
    ./build/sph_renderer \
    --snapshot    "$SNAP" \
    --region      "$REGION" \
    --camera-list "$CAMLIST" \
    --output      "$OUTDIR" \
    --fields      gas_density \
    --no-volume

python python/orbit_frames_sph.py --plot --encode \
    --frames   "$FRAMES" \
    --out-dir  "$OUTDIR" \
    --png-dir  "$PNGDIR" \
    --video    "$VIDEO"

echo "=== DONE ==="
ls -l "$VIDEO"
