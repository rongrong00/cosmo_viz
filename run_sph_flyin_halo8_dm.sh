#!/bin/bash
#SBATCH -A AST211
#SBATCH -J sph_h8dm
#SBATCH -N 16
#SBATCH -n 32
#SBATCH -c 16
#SBATCH -t 03:00:00
#SBATCH -p batch
#SBATCH -o output/sph_halo8_flyin_dm_%j.out
#SBATCH -e output/sph_halo8_flyin_dm_%j.err

# 960-frame halo-8 spiral fly-in, DM density column only.
# Same path as stellar/gas flyins: r_outer=16915 -> r_inner=250, exactly
# 2 turns in xy starting/ending at frame-803 azimuth (4.23854 rad ≈ 242.85°).

module load hdf5
source ~/.bashrc
conda activate chii
cd /lustre/orion/ast211/proj-shared/rongrong/lumina_visualization/cosmo_viz

export OMP_NUM_THREADS=16
export OMP_PLACES=cores
export OMP_PROC_BIND=close

SNAP=/lustre/orion/ast211/proj-shared/rongrong/extract_snap158/output_halo8/snap_158_cube.hdf5
FRAMES=960
OUTDIR=output/snap158_halo8_flyin_dm
CFGDIR=config/snap158_halo8_flyin_dm
REGION=${CFGDIR}/region.yaml
CAMLIST=${CFGDIR}/camera_list.txt

CX=233794.84
CY=60858.605
CZ=146579.95
R_OUTER=16915.0
R_INNER=250.0
TURNS=2.0
THETA_START=4.23854

mkdir -p "$OUTDIR" "$CFGDIR"

python python/orbit_frames_sph.py --prep \
    --mode         spiral \
    --frames       "$FRAMES" \
    --r-outer      "$R_OUTER" \
    --r-inner      "$R_INNER" \
    --turns        "$TURNS" \
    --theta-start  "$THETA_START" \
    --sphere-radius "$R_OUTER" \
    --center       "$CX" "$CY" "$CZ" \
    --fov          60.0 \
    --image-width  1920 \
    --image-height 1080 \
    --config-dir   "$CFGDIR" \
    --region-file  "$REGION" \
    --camera-list  "$CAMLIST"

sed -i 's/^particle_types:.*/particle_types: [dm]/' "$REGION"
echo "=== region.yaml ==="
cat "$REGION"

srun -n ${SLURM_NTASKS} -c ${OMP_NUM_THREADS} --cpu-bind=threads \
    ./build/sph_renderer \
    --snapshot    "$SNAP" \
    --region      "$REGION" \
    --camera-list "$CAMLIST" \
    --output      "$OUTDIR" \
    --fields      dm_density \
    --no-volume

PNGDIR=output/snap158_halo8_flyin_dm_png
VIDEO=output/snap158_halo8_flyin_dm.mp4
mkdir -p "$PNGDIR"

python python/plot_flyin_matter_fade.py \
    --frame-dir "$OUTDIR" \
    --png-dir   "$PNGDIR" \
    --frames    "$FRAMES" \
    --h5-name   dm_column.h5 \
    --field     dm_column_density \
    --cmap      plasma \
    --vmin-pct  50 \
    --vmax-pct  99.7 \
    --adaptive --smooth-sigma 60 \
    --alpha-gamma 0.7 \
    --bg        black

ffmpeg -y -framerate 30 \
    -i "${PNGDIR}/frame_%04d.png" \
    -c:v libx264 -pix_fmt yuv420p -crf 18 \
    -vf 'pad=ceil(iw/2)*2:ceil(ih/2)*2' \
    "$VIDEO"
ls -l "$VIDEO"

echo "=== DONE ==="
