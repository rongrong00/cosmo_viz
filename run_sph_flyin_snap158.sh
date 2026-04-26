#!/bin/bash
#SBATCH -A AST211
#SBATCH -J sph_flyin158
#SBATCH -N 16
#SBATCH -n 32
#SBATCH -c 16
#SBATCH -t 02:00:00
#SBATCH -p batch
#SBATCH -o output/sph_flyin_snap158_%j.out
#SBATCH -e output/sph_flyin_snap158_%j.err

# 960-frame spiral fly-in on the snap_158 extracted cube. Camera spirals
# log-linearly from r_outer (on the 16915 ckpc/h sphere boundary) down
# toward the halo center with TURNS full revolutions. Renderer writes
# raw gas_column.h5 (no vmin/vmax clip); PNG step picks one global
# (vmin, vmax) so brightness is stable across the movie.

module load hdf5
source ~/.bashrc
conda activate chii
cd /lustre/orion/ast211/proj-shared/rongrong/lumina_visualization/cosmo_viz

export OMP_NUM_THREADS=16
export OMP_PLACES=cores
export OMP_PROC_BIND=close

SNAP=/lustre/orion/ast211/proj-shared/rongrong/extract_snap158/output/snap_158_cube.hdf5
FRAMES=960
OUTDIR=output/snap158_flyin
CFGDIR=config/snap158_flyin
REGION=${CFGDIR}/region.yaml
CAMLIST=${CFGDIR}/camera_list.txt
PNGDIR=output/snap158_flyin_png
VIDEO=output/snap158_flyin.mp4

CX=178875.94
CY=128443.23
CZ=54431.81
R_OUTER=16915.0
R_INNER=100.0
TURNS=2.0

mkdir -p "$OUTDIR" "$CFGDIR" "$PNGDIR"

python python/orbit_frames_sph.py --prep \
    --mode         spiral \
    --frames       "$FRAMES" \
    --r-outer      "$R_OUTER" \
    --r-inner      "$R_INNER" \
    --turns        "$TURNS" \
    --sphere-radius "$R_OUTER" \
    --center       "$CX" "$CY" "$CZ" \
    --fov          60.0 \
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

python python/plot_flyin_matter_fade.py \
    --frame-dir "$OUTDIR" \
    --png-dir   "$PNGDIR" \
    --frames    "$FRAMES" \
    --field     gas_column_density \
    --vmin-pct  50 \
    --vmax-pct  99.7 \
    --alpha-gamma 0.7 \
    --bg        black

ffmpeg -y -framerate 30 \
    -i "${PNGDIR}/frame_%04d.png" \
    -c:v libx264 -pix_fmt yuv420p -crf 18 \
    -vf 'pad=ceil(iw/2)*2:ceil(ih/2)*2' \
    "$VIDEO"

echo "=== DONE ==="
ls -l "$VIDEO"
