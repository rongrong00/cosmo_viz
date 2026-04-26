#!/bin/bash
#SBATCH -A AST211
#SBATCH -J stars_h8fly
#SBATCH -N 4
#SBATCH -n 32
#SBATCH --ntasks-per-node=8
#SBATCH -c 4
#SBATCH -t 01:30:00
#SBATCH -p batch
#SBATCH -o output/stars_halo8_flyin_%j.out
#SBATCH -e output/stars_halo8_flyin_%j.err

# Halo 8 stellar flyin, 960 frames. Camera spirals from
# r_outer=16915 -> r_inner=250 around halo center, exactly 2 turns in xy
# starting/ending at frame-803 azimuth (4.23854 rad ≈ 242.85°).
# Parallelized via SLURM: each rank renders frames where (i % WORLD_SIZE) == RANK.
module load hdf5
source ~/.bashrc
conda activate chii
cd /lustre/orion/ast211/proj-shared/rongrong/lumina_visualization/cosmo_viz

SNAP=/lustre/orion/ast211/proj-shared/rongrong/extract_snap158/output_halo8/snap_158_cube.hdf5
GRID=config/fsps_gri_ssp.h5
OUTDIR=output/snap158_halo8_stars_flyin
VIDEO=$OUTDIR/flyin.mp4
FRAMES=960

mkdir -p "$OUTDIR"

srun -n ${SLURM_NTASKS} -c ${SLURM_CPUS_PER_TASK} --cpu-bind=cores \
    python python/stellar_flyin.py \
    --snapshot          "$SNAP" \
    --fsps-grid         "$GRID" \
    --center            233794.84 60858.605 146579.95 \
    --r-outer           16915.0 \
    --r-inner           250.0 \
    --frames            $FRAMES \
    --fov               60.0 \
    --mode              spiral \
    --turns             2.0 \
    --theta-start       4.23854 \
    --spiral-plane      xy \
    --out-dir           "$OUTDIR" \
    --channels          L_i,L_r,L_g \
    --Q                 8 \
    --stretch-pct       99.5 \
    --clip-center       233794.84 60858.605 146579.95 \
    --clip-radius-start 16915.0 \
    --clip-radius-end   150.0 \
    --clip-dim          0.10 \
    --splat-sigma-pix   0.8

ffmpeg -y -framerate 30 \
    -i "${OUTDIR}/frame_%04d.png" \
    -c:v libx264 -pix_fmt yuv420p -crf 18 \
    -vf 'pad=ceil(iw/2)*2:ceil(ih/2)*2' \
    "$VIDEO"

echo "=== DONE ==="
ls -l "$VIDEO"
