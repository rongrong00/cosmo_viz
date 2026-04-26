#!/bin/bash
#SBATCH -A AST211
#SBATCH -J sph_dm1
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 32
#SBATCH -t 00:45:00
#SBATCH -p batch
#SBATCH -o output/sph_dm_test1_%j.out
#SBATCH -e output/sph_dm_test1_%j.err

# Single-frame DM column density render, same camera as orbit test1.
module load hdf5
source ~/.bashrc
conda activate chii
cd /lustre/orion/ast211/proj-shared/rongrong/lumina_visualization/cosmo_viz

export OMP_NUM_THREADS=32
export OMP_PLACES=cores
export OMP_PROC_BIND=close

SNAP=/lustre/orion/ast211/proj-shared/EDE_Final/finalTest/50cMpc_2/output/snapdir_121/snap_121
CFGDIR=config/dm_test1
OUTDIR=output/dm_test1
REGION=${CFGDIR}/region.yaml
CAMERA=${CFGDIR}/frame_0000.yaml
PNG=${OUTDIR}/frame_0000.png

mkdir -p "$CFGDIR" "$OUTDIR"

# Region: sphere of radius 5000 ckpc/h around the halo, dm-only.
cat > "$REGION" <<EOF
name: dm_test1
center: [4093.16, 23556.48, 10679.74]
radius: 5000.0
particle_types: [dm]
EOF

# Camera: same geometry as the orbit frame 0 (perspective, on sphere boundary).
cat > "$CAMERA" <<EOF
camera:
  type: perspective
  position: [9093.16, 23556.48, 10679.74]
  look_at:  [4093.16, 23556.48, 10679.74]
  up:       [0, 0, 1]
  fov: 40.0
  ortho_width: 0.0
  los_slab: 0.0
  image_width: 1920
  image_height: 1080

projections:
  - field: dm_density
    mode: column
EOF

srun -n 1 -c ${OMP_NUM_THREADS} --cpu-bind=threads \
    ./build/sph_renderer \
    --snapshot "$SNAP" \
    --region   "$REGION" \
    --camera   "$CAMERA" \
    --output   "$OUTDIR" \
    --fields   dm_density \
    --no-volume

python python/plotter.py \
    --input  "$OUTDIR/dm_column.h5" \
    --output "$PNG" \
    --field  dm_column_density \
    --cmap   magma \
    --bare

echo "=== DONE ==="
ls -l "$OUTDIR"
