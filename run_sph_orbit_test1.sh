#!/bin/bash
#SBATCH -A AST211
#SBATCH -J sph_orb1
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 32
#SBATCH -t 00:30:00
#SBATCH -p batch
#SBATCH -o output/sph_orbit_test1_%j.out
#SBATCH -e output/sph_orbit_test1_%j.err

# Single-frame smoke test of the perspective orbit setup: theta=0 only.
module load hdf5
source ~/.bashrc
conda activate chii
cd /lustre/orion/ast211/proj-shared/rongrong/lumina_visualization/cosmo_viz

export OMP_NUM_THREADS=32
export OMP_PLACES=cores
export OMP_PROC_BIND=close

SNAP=/lustre/orion/ast211/proj-shared/EDE_Final/finalTest/50cMpc_2/output/snapdir_121/snap_121
CFGDIR=config/orbit_sph_test1
OUTDIR=output/orbit_sph_test1
REGION=${CFGDIR}/region.yaml
CAMERA=${CFGDIR}/frame_0000.yaml
PNG=${OUTDIR}/frame_0000.png

mkdir -p "$CFGDIR" "$OUTDIR"

# Generate region + 1 camera YAML (frame 0 only).
python python/orbit_frames_sph.py --prep \
    --frames       1 \
    --orbit-radius 5000.0 \
    --fov          40.0 \
    --image-width  1920 \
    --image-height 1080 \
    --config-dir   "$CFGDIR" \
    --region-file  "$REGION" \
    --camera-list  "$CFGDIR/camera_list.txt"

# Single-camera mode (no frame_NNNN subdir; writes directly to OUTDIR).
srun -n 1 -c ${OMP_NUM_THREADS} --cpu-bind=threads \
    ./build/sph_renderer \
    --snapshot "$SNAP" \
    --region   "$REGION" \
    --camera   "$CAMERA" \
    --output   "$OUTDIR" \
    --fields   gas_density

# Plot.
python python/plotter.py \
    --input  "$OUTDIR/gas_column.h5" \
    --output "$PNG" \
    --field  gas_column_density \
    --cmap   inferno

echo "=== DONE ==="
ls -l "$OUTDIR"
