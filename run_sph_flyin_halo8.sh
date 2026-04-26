#!/bin/bash
#SBATCH -A AST211
#SBATCH -J sph_h8fly
#SBATCH -N 16
#SBATCH -n 32
#SBATCH -c 16
#SBATCH -t 04:00:00
#SBATCH -p batch
#SBATCH -o output/sph_halo8_flyin_%j.out
#SBATCH -e output/sph_halo8_flyin_%j.err

# 960-frame spiral fly-in on halo 8 (snap_158 cube), same path as the stellar
# flyin: center=(233794.84, 60858.605, 146579.95), r_outer=16915, r_inner=40,
# 2 turns, xy plane, fov=60.
# Renders four fields in a single pass: gas_density (column), dm_density
# (column), temperature (mass-weighted), metallicity (mass-weighted). Then the
# PNG/video stage is run once per field.

module load hdf5
source ~/.bashrc
conda activate chii
cd /lustre/orion/ast211/proj-shared/rongrong/lumina_visualization/cosmo_viz

export OMP_NUM_THREADS=16
export OMP_PLACES=cores
export OMP_PROC_BIND=close

SNAP=/lustre/orion/ast211/proj-shared/rongrong/extract_snap158/output_halo8/snap_158_cube.hdf5
FRAMES=960
OUTDIR=output/snap158_halo8_flyin_spiral
CFGDIR=config/snap158_halo8_flyin_spiral
REGION=${CFGDIR}/region.yaml
CAMLIST=${CFGDIR}/camera_list.txt

CX=233794.84
CY=60858.605
CZ=146579.95
R_OUTER=16915.0
R_INNER=40.0
TURNS=2.0

mkdir -p "$OUTDIR" "$CFGDIR"

# Generate region + 960 per-frame camera YAMLs. orbit_frames_sph writes
# particle_types: [gas] — patch it to include dm so dm_column also renders.
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

sed -i 's/^particle_types:.*/particle_types: [gas, dm]/' "$REGION"
echo "=== region.yaml ==="
cat "$REGION"

srun -n ${SLURM_NTASKS} -c ${OMP_NUM_THREADS} --cpu-bind=threads \
    ./build/sph_renderer \
    --snapshot    "$SNAP" \
    --region      "$REGION" \
    --camera-list "$CAMLIST" \
    --output      "$OUTDIR" \
    --fields      gas_density,dm_density,temperature,metallicity \
    --no-volume

# --- PNGs + video per field ----------------------------------------------
render_field () {
    local tag=$1 h5_name=$2 dataset=$3 vmin_pct=$4 vmax_pct=$5 alpha_gamma=$6
    local PNGDIR=output/snap158_halo8_flyin_${tag}_png
    local VIDEO=output/snap158_halo8_flyin_${tag}.mp4
    mkdir -p "$PNGDIR"

    python python/plot_flyin_matter_fade.py \
        --frame-dir "$OUTDIR" \
        --png-dir   "$PNGDIR" \
        --frames    "$FRAMES" \
        --h5-name   "$h5_name" \
        --field     "$dataset" \
        --vmin-pct  "$vmin_pct" \
        --vmax-pct  "$vmax_pct" \
        --alpha-gamma "$alpha_gamma" \
        --bg        black

    ffmpeg -y -framerate 30 \
        -i "${PNGDIR}/frame_%04d.png" \
        -c:v libx264 -pix_fmt yuv420p -crf 18 \
        -vf 'pad=ceil(iw/2)*2:ceil(ih/2)*2' \
        "$VIDEO"
    ls -l "$VIDEO"
}

render_field gas   gas_column.h5   gas_column_density 50 99.7 0.7
render_field dm    dm_column.h5    dm_column_density  50 99.7 0.7
render_field temp  temperature.h5  gas_temperature_mw 40 99.5 0.6
render_field metal metallicity.h5  gas_metallicity_mw 40 99.5 0.6

echo "=== DONE ==="
