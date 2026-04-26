#!/usr/bin/env python3
"""
Perspective orbit movie around the largest FOF halo, driven by sph_renderer
(direct SPH ray tracer — no grid intermediate).

The camera orbits in the xy plane at radius R around the halo center, always
looking at the center, with +z up. The region is a sphere of the same radius
R centered on the halo, so the camera sits exactly on the sphere boundary.

Steps:
  --prep     write region + per-frame camera YAMLs + camera-list
  (render)   run ./build/sph_renderer --camera-list under srun (done by SLURM)
  --plot     plotter.py each frame_NNNN/gas_column.h5 -> PNG
  --encode   ffmpeg PNGs -> mp4
"""

import argparse
import math
import os
import subprocess
import sys
from pathlib import Path
import numpy as np


CLUSTER_POS = (4093.16, 23556.48, 10679.74)   # largest FOF halo in snap_121


def write_region_yaml(path, center, radius, name):
    content = f"""# Spherical orbit region. Radius matches the camera orbit radius, so the
# camera sits exactly on the sphere boundary.
name: {name}
center: [{center[0]}, {center[1]}, {center[2]}]
radius: {radius}
particle_types: [gas]
"""
    Path(path).write_text(content)


def write_camera_yaml(path, cam_xyz, look_at, up, fov, image_w, image_h):
    content = f"""# orbit frame (perspective): pos=({cam_xyz[0]:.3f}, {cam_xyz[1]:.3f}, {cam_xyz[2]:.3f})
camera:
  type: perspective
  position: [{cam_xyz[0]}, {cam_xyz[1]}, {cam_xyz[2]}]
  look_at:  [{look_at[0]}, {look_at[1]}, {look_at[2]}]
  up:       [{up[0]}, {up[1]}, {up[2]}]
  fov: {fov}
  ortho_width: 0.0
  los_slab: 0.0
  image_width: {image_w}
  image_height: {image_h}

projections:
  - field: gas_density
    mode: column
"""
    Path(path).write_text(content)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--frames',       type=int,   default=120)
    p.add_argument('--orbit-radius', type=float, default=5000.0,
                   help='camera distance from halo center in xy plane, ckpc/h. '
                        'Spherical region uses the same radius in circle mode.')
    p.add_argument('--mode', choices=['circle', 'spiral'], default='circle',
                   help='circle: constant radius; spiral: zoom from R-outer to R-inner')
    p.add_argument('--r-outer', type=float, default=15000.0,
                   help='spiral mode: starting (outer) orbit radius, ckpc/h')
    p.add_argument('--r-inner', type=float, default=1500.0,
                   help='spiral mode: ending (inner) orbit radius, ckpc/h')
    p.add_argument('--turns',   type=float, default=3.0,
                   help='spiral mode: number of full revolutions across all frames')
    p.add_argument('--theta-start', type=float, default=0.0,
                   help='starting azimuth in radians (xy plane). Default 0 = +x axis.')
    p.add_argument('--sphere-radius', type=float, default=None,
                   help='explicit spherical region radius; defaults to orbit R (circle) '
                        'or R-outer (spiral).')
    p.add_argument('--fov',          type=float, default=40.0)
    p.add_argument('--image-width',  type=int,   default=1920)
    p.add_argument('--image-height', type=int,   default=1080)
    p.add_argument('--config-dir',   default='config/orbit_sph')
    p.add_argument('--out-dir',      default='output/orbit_sph')
    p.add_argument('--png-dir',      default='output/orbit_sph_png')
    p.add_argument('--video',        default='output/orbit_sph.mp4')
    p.add_argument('--camera-list',  default='config/orbit_sph/camera_list.txt')
    p.add_argument('--region-file',  default='config/orbit_sph/region.yaml')
    p.add_argument('--center',       type=float, nargs=3, default=list(CLUSTER_POS))
    p.add_argument('--field',        default='gas_density',
                   help='dataset field to plot from gas_column.h5')
    p.add_argument('--cmap',         default='inferno')
    p.add_argument('--prep',   action='store_true', help='generate region + camera YAMLs')
    p.add_argument('--plot',   action='store_true', help='plot frame_*/gas_column.h5 -> PNG')
    p.add_argument('--encode', action='store_true', help='ffmpeg PNGs -> mp4')
    args = p.parse_args()

    cx, cy, cz = args.center
    Path(args.config_dir).mkdir(parents=True, exist_ok=True)

    # Per-frame (theta, radius). Circle: fixed R, one revolution.
    # Spiral: radius decays log-linearly from r-outer to r-inner over `turns`
    # revolutions — constant fractional zoom per frame, so the perceived
    # zoom speed is uniform.
    N = args.frames
    if args.mode == 'circle':
        R_region = args.sphere_radius if args.sphere_radius is not None else args.orbit_radius
        radii  = [args.orbit_radius] * N
        thetas = [args.theta_start + 2.0 * math.pi * i / N for i in range(N)]
    else:  # spiral
        R_region = args.sphere_radius if args.sphere_radius is not None else args.r_outer
        if N == 1:
            radii = [args.r_outer]
            thetas = [args.theta_start]
        else:
            ratio = args.r_inner / args.r_outer
            radii  = [args.r_outer * (ratio ** (i / (N - 1))) for i in range(N)]
            thetas = [args.theta_start + 2.0 * math.pi * args.turns * i / (N - 1) for i in range(N)]

    # --- Prep: write region + per-frame camera YAMLs + camera-list ----------
    if args.prep:
        write_region_yaml(args.region_file, (cx, cy, cz), R_region, f'orbit_sph_{args.mode}')
        up = (0.0, 0.0, 1.0)
        cam_paths = []
        for i in range(N):
            theta = thetas[i]
            R_i   = radii[i]
            cam_x = cx + R_i * math.cos(theta)
            cam_y = cy + R_i * math.sin(theta)
            cam_z = cz
            cam_path = f'{args.config_dir}/frame_{i:04d}.yaml'
            write_camera_yaml(cam_path,
                              (cam_x, cam_y, cam_z),
                              (cx, cy, cz),
                              up,
                              args.fov,
                              args.image_width, args.image_height)
            cam_paths.append(cam_path)
        Path(args.camera_list).write_text('\n'.join(cam_paths) + '\n')
        print(f'[prep] mode={args.mode} R_region={R_region:.1f} '
              f'R=[{radii[0]:.1f}..{radii[-1]:.1f}] turns≈{thetas[-1]/(2*math.pi):.2f}')
        print(f'[prep] wrote {args.frames} cameras, region={args.region_file}, '
              f'camera-list={args.camera_list}')

    # --- Plot: gas_column.h5 -> PNG per frame -------------------------------
    if args.plot:
        Path(args.png_dir).mkdir(parents=True, exist_ok=True)
        plotter = Path(__file__).parent / 'plotter.py'
        missing = 0
        for i in range(args.frames):
            h5 = f'{args.out_dir}/frame_{i:04d}/gas_column.h5'
            png = f'{args.png_dir}/frame_{i:04d}.png'
            if not os.path.exists(h5):
                missing += 1
                print(f'  [plot] missing {h5}', file=sys.stderr)
                continue
            cmd = ['python', str(plotter),
                   '--input', h5,
                   '--output', png,
                   '--field', 'gas_column_density',
                   '--cmap',  args.cmap,
                   '--bare']
            subprocess.run(cmd, check=True)
        if missing:
            print(f'[plot] {missing} frames missing; video will be incomplete',
                  file=sys.stderr)

    # --- Encode: PNGs -> mp4 ------------------------------------------------
    if args.encode:
        Path(os.path.dirname(args.video) or '.').mkdir(parents=True, exist_ok=True)
        cmd = [
            'ffmpeg', '-y',
            '-framerate', '30',
            '-i', f'{args.png_dir}/frame_%04d.png',
            '-c:v', 'libx264',
            '-pix_fmt', 'yuv420p',
            '-crf', '18',
            # Preserve input pixel dimensions: no scaling filter. libx264
            # requires even dimensions so pad odd sizes up by 1.
            '-vf', 'pad=ceil(iw/2)*2:ceil(ih/2)*2',
            args.video,
        ]
        subprocess.run(cmd, check=True)
        print(f'[encode] wrote {args.video}')


if __name__ == '__main__':
    main()
