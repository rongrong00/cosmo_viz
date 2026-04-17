#!/usr/bin/env python3
"""
Camera orbit around the largest FOF halo: one full revolution in the xy plane
with +z as up, always looking at the cluster center. 16:9 frame by default.
"""

import argparse
import math
import os
import subprocess
import sys
from pathlib import Path
import numpy as np

CLUSTER_POS = (4093.16, 23556.48, 10679.74)   # ckpc/h
GRID_FILE   = 'output/grids/grid_orbit_snap121.hdf5'
GRID_SIDE   = 10000.0                          # must match orbit_grid.yaml

def write_camera_yaml(path, cam_xyz, look_at, up, ortho_width, los_slab,
                      image_w, image_h):
    content = f"""# orbit frame: pos=({cam_xyz[0]:.3f}, {cam_xyz[1]:.3f}, {cam_xyz[2]:.3f})
camera:
  type: orthographic
  position: [{cam_xyz[0]}, {cam_xyz[1]}, {cam_xyz[2]}]
  look_at:  [{look_at[0]}, {look_at[1]}, {look_at[2]}]
  up:       [{up[0]}, {up[1]}, {up[2]}]
  fov: 30.0
  ortho_width: {ortho_width}
  los_slab: {los_slab}
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
    p.add_argument('--ortho-width',  type=float, default=3000.0,
                   help='vertical extent of the ortho view in ckpc/h')
    p.add_argument('--los-slab',     type=float, default=4000.0,
                   help='slab depth around cluster center in ckpc/h')
    p.add_argument('--orbit-radius', type=float, default=8000.0,
                   help='camera distance from cluster in xy plane')
    p.add_argument('--image-width',  type=int,   default=1920)
    p.add_argument('--image-height', type=int,   default=1080)
    p.add_argument('--config-dir',   default='config/orbit_frames')
    p.add_argument('--proj-dir',     default='output/orbit_proj')
    p.add_argument('--png-dir',      default='output/orbit_png')
    p.add_argument('--video',        default='output/orbit.mp4')
    p.add_argument('--grid',         default=GRID_FILE)
    p.add_argument('--renderer',     default='./build/renderer')
    p.add_argument('--render', action='store_true')
    p.add_argument('--plot',   action='store_true')
    p.add_argument('--encode', action='store_true')
    args = p.parse_args()

    cx, cy, cz = CLUSTER_POS
    for d in (args.config_dir, args.proj_dir, args.png_dir):
        os.makedirs(d, exist_ok=True)
    os.makedirs(os.path.dirname(args.video) or '.', exist_ok=True)

    # One revolution over `frames` frames, circling in the xy plane, cluster at center.
    thetas = 2.0 * math.pi * np.arange(args.frames) / args.frames
    up = (0.0, 0.0, 1.0)

    cam_paths = []
    for i, theta in enumerate(thetas):
        cam_x = cx + args.orbit_radius * math.cos(theta)
        cam_y = cy + args.orbit_radius * math.sin(theta)
        cam_z = cz
        cam_path = f'{args.config_dir}/frame_{i:04d}.yaml'
        write_camera_yaml(cam_path,
                          (cam_x, cam_y, cam_z),
                          (cx, cy, cz),
                          up,
                          args.ortho_width, args.los_slab,
                          args.image_width, args.image_height)
        cam_paths.append(cam_path)

    print(f'Wrote {len(cam_paths)} camera configs')
    print(f'  radius={args.orbit_radius} ckpc/h, ortho_width={args.ortho_width}, '
          f'slab={args.los_slab}, image={args.image_width}x{args.image_height}')

    if args.render:
        for i, cam_path in enumerate(cam_paths):
            frame_dir = f'{args.proj_dir}/frame_{i:04d}'
            os.makedirs(frame_dir, exist_ok=True)
            cmd = [args.renderer, '--grid', args.grid,
                   '--camera', cam_path, '--output', frame_dir]
            print(f'[{i+1:04d}/{args.frames}] theta={math.degrees(thetas[i]):6.1f}',
                  flush=True)
            subprocess.run(cmd, check=True)

    if args.plot:
        import h5py
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        from matplotlib.colors import LogNorm

        # Mean density along LOS (column / slab) so units match zoom movie.
        slab = args.los_slab

        # One global vmin/vmax from a handful of frames across the orbit —
        # the cluster doesn't change magnitude much as we rotate, so a single
        # color scale is fine here (unlike the zoom where magnitude varies).
        samples = []
        probe_idxs = sorted(set([0, args.frames // 4, args.frames // 2,
                                 (3 * args.frames) // 4]))
        for pi in probe_idxs:
            pf = f'{args.proj_dir}/frame_{pi:04d}/projection_grid_orbit_snap121_gas_density.hdf5'
            if not os.path.exists(pf):
                continue
            with h5py.File(pf, 'r') as f:
                d = f['gas_density'][:].astype(np.float64) / slab
            pos = d[d > 0]
            if pos.size:
                samples.append(pos)
        vmin = vmax = None
        if samples:
            pooled = np.concatenate(samples)
            vmin = np.percentile(pooled, 5)
            vmax = np.percentile(pooled, 99.7)
            print(f'Global color range: {vmin:.3e} .. {vmax:.3e}')

        for i in range(args.frames):
            proj = f'{args.proj_dir}/frame_{i:04d}/projection_grid_orbit_snap121_gas_density.hdf5'
            if not os.path.exists(proj):
                print(f'missing: {proj}', file=sys.stderr); continue
            with h5py.File(proj, 'r') as f:
                data = f['gas_density'][:].astype(np.float64) / slab
            dp = data.copy()
            dp[dp <= 0] = np.nan
            if vmin is None or vmax is None or vmin <= 0:
                pos = data[data > 0]
                lvmin = np.percentile(pos, 5)  if pos.size else 1e-10
                lvmax = np.percentile(pos, 99.7) if pos.size else 1.0
            else:
                lvmin, lvmax = vmin, vmax

            cmap = plt.get_cmap('inferno').copy(); cmap.set_bad('black')
            dpi = 120
            figsize = (data.shape[1] / dpi, data.shape[0] / dpi)
            fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
            ax.imshow(dp, cmap=cmap, norm=LogNorm(vmin=lvmin, vmax=lvmax),
                      origin='upper', interpolation='nearest')
            theta_deg = math.degrees(2.0 * math.pi * i / args.frames)
            ax.set_title(f'orbit: {theta_deg:5.1f} deg', color='white', fontsize=12)
            ax.set_xticks([]); ax.set_yticks([])
            ax.set_facecolor('black'); fig.patch.set_facecolor('black')

            # Scale bar: 20% of image width in world units
            img_w = data.shape[1]; img_h = data.shape[0]
            aspect = img_w / img_h
            x_extent = args.ortho_width * aspect
            bar_frac = 0.2
            x0 = img_w * 0.05; x1 = x0 + img_w * bar_frac
            y  = img_h * 0.92
            ax.plot([x0, x1], [y, y], color='white', lw=2)
            bar_len = x_extent * bar_frac
            bar_lbl = (f'{bar_len/1000:.2f} cMpc/h' if bar_len >= 1000
                       else f'{bar_len:.0f} ckpc/h')
            ax.text((x0 + x1)/2, y*0.97, bar_lbl,
                    color='white', fontsize=11, ha='center', va='bottom')

            out = f'{args.png_dir}/frame_{i:04d}.png'
            plt.savefig(out, facecolor='black')
            plt.close(fig)
            if (i + 1) % 10 == 0 or i == args.frames - 1:
                print(f'plotted {i+1}/{args.frames}', flush=True)

    if args.encode:
        cmd = ['ffmpeg', '-y', '-framerate', '30',
               '-i', f'{args.png_dir}/frame_%04d.png',
               '-c:v', 'libx264', '-pix_fmt', 'yuv420p',
               '-vf', 'scale=trunc(iw/2)*2:trunc(ih/2)*2',
               args.video]
        print(' '.join(cmd))
        subprocess.run(cmd, check=True)
        print(f'wrote {args.video}')

if __name__ == '__main__':
    main()
