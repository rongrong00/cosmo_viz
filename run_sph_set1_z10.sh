#!/bin/bash
#SBATCH -A AST211
#SBATCH -J sph_s1p_z10
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 32
#SBATCH -t 01:00:00
#SBATCH -p batch
#SBATCH -o output/sph_set1_persp_z10_%j.out
#SBATCH -e output/sph_set1_persp_z10_%j.err

# Set-2-style (emission-absorption volume render) for gas_density,
# gas_temperature, gas_metallicity — matches the perspective camera used in
# the existing sph_set2_persp_z10 Set-2 outputs (neutral_H / ionized_H /
# heated_gas / heavy_elements).

module load hdf5
cd /lustre/orion/ast211/proj-shared/rongrong/lumina_visualization/cosmo_viz

export OMP_NUM_THREADS=32
export OMP_PLACES=cores
export OMP_PROC_BIND=close

SNAP=/lustre/orion/ast211/proj-shared/EDE_Final/finalTest/50cMpc_2/output/snapdir_050/snap_050
REGION=config/zoom_region_10mpc.yaml
CAMERA=config/sph_cam_persp.yaml
OUTDIR=output/sph_set2_persp_z10

mkdir -p "$OUTDIR"

srun -n 1 -c ${OMP_NUM_THREADS} --cpu-bind=threads \
    ./build/sph_renderer \
    --snapshot "$SNAP" \
    --region   "$REGION" \
    --camera   "$CAMERA" \
    --output   "$OUTDIR" \
    --fields   gas_density,gas_temperature,gas_metallicity

echo "=== RENDER DONE ==="

source ~/.bashrc
conda activate chii

# Use Set-1 Illustris palettes to tone-map the volume-rendered emission.
python python/plot_illustris_style.py --input "$OUTDIR/gas_density_vol.h5" \
    --output "$OUTDIR/gas_density_ill.png" \
    --field emission --style density_dark --label "Gas Density"
python python/plot_illustris_style.py --input "$OUTDIR/gas_temperature_vol.h5" \
    --output "$OUTDIR/gas_temperature_ill.png" \
    --field emission --style velocity --label "Gas Temperature"
python python/plot_illustris_style.py --input "$OUTDIR/gas_metallicity_vol.h5" \
    --output "$OUTDIR/gas_metallicity_ill.png" \
    --field emission --style metals_dark --label "Gas Metallicity"

echo "=== ALL DONE ==="
ls -la "$OUTDIR"/*_ill.png
