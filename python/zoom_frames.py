#!/usr/bin/env python3
"""
Adaptive-resolution zoom-in movie into the largest FOF halo at snap 121.

The script supports a standard 5-tier run and a deeper 7-tier run. Per frame
we pick the smallest grid whose side still contains the current orthographic
view. When full-bleed mode is enabled, the plotted image fills the canvas so
the movie has no dark rim.
"""

import argparse
import os
import subprocess
import sys
from pathlib import Path
import numpy as np

CLUSTER_POS = (4093.16, 23556.48, 10679.74)   # ckpc/h, GroupPos[0]
BOX_SIZE    = 36245.0                          # ckpc/h
R200        = 361.63                           # ckpc/h

# (grid_name, xy_side_ckpc/h). Grid file path = f'{grid_dir}/grid_{name}_snap121.hdf5'.
TIERS_STANDARD = [
    ('zoom_wide',  36245.0),
    ('zoom_t2',    15000.0),
    ('zoom_mid',    8000.0),
    ('zoom_t4',     3500.0),
    ('zoom_close',  2000.0),
]

TIERS_DEEP = [
    ('zoom_wide',   36245.0),
    ('zoom_t2',     15000.0),
    ('zoom_mid',     8000.0),
    ('zoom_t4',      3500.0),
    ('zoom_close',   2000.0),
    ('zoom_t5',      1200.0),
    ('zoom_t6',       600.0),
]

TIERS_BY_SET = {
    'standard': TIERS_STANDARD,
    'deep': TIERS_DEEP,
}

def build_tiers(grid_dir, tier_set):
    return [(name, xy, f'{grid_dir}/grid_{name}_snap121.hdf5')
            for name, xy in TIERS_BY_SET[tier_set]]

def pick_tier(tiers, ortho_width, aspect, overflow_tol=0.0):
    """Smallest tier whose xy side still contains the full orthographic view."""
    required_side = max(ortho_width, ortho_width * aspect)
    tol = max(0.0, float(overflow_tol))
    chosen = tiers[0]
    for name, side, path in tiers:
        if side * (1.0 + tol) >= required_side:
            chosen = (name, side, path)
    return chosen

def write_camera_yaml(path, cx, cy, cz, ortho_width, los_slab, image_w, image_h):
    cam_z = cz - max(BOX_SIZE, los_slab)  # well outside any grid along -z
    Path(path).write_text(f"""# zoom frame: ortho_width={ortho_width:.3f}, los_slab={los_slab:.3f} ckpc/h
camera:
  type: orthographic
  position: [{cx}, {cy}, {cam_z}]
  look_at: [{cx}, {cy}, {cz}]
  up: [0, 1, 0]
  fov: 30.0
  ortho_width: {ortho_width}
  los_slab: {los_slab}
  image_width: {image_w}
  image_height: {image_h}

projections:
  - field: gas_density
    mode: column
""")

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--frames',      type=int,   default=120)
    p.add_argument('--width-start', type=float, default=BOX_SIZE)
    p.add_argument('--width-end',   type=float, default=2.5 * R200)
    p.add_argument('--los-slab',    type=float, default=BOX_SIZE,
                   help='line-of-sight depth in ckpc/h when --los-depth is fixed')
    p.add_argument('--los-depth', choices=['fixed', 'same-as-y'], default='fixed',
                   help='set the LOS depth to a fixed value or match the vertical extent')
    p.add_argument('--tier-set',    choices=sorted(TIERS_BY_SET), default='standard')
    p.add_argument('--tier-overflow-tol', type=float, default=0.0,
                   help='allow a fractional xy overflow when selecting smaller tiers (e.g. 0.02)')
    p.add_argument('--trim-start-frame', type=int, default=0,
                   help='skip frames before this index when plotting/encoding')
    p.add_argument('--full-bleed', action='store_true',
                   help='fill the canvas with the image and overlay labels in-frame')
    p.add_argument('--image-size',  type=int,   default=0,
                   help='if >0, render a square image of this size (overrides width/height)')
    p.add_argument('--image-width', type=int,   default=1920)
    p.add_argument('--image-height',type=int,   default=1080)
    p.add_argument('--grid-dir',    default='output/grids')
    p.add_argument('--config-dir',  default='config/zoom_frames')
    p.add_argument('--proj-dir',    default='output/zoom_proj')
    p.add_argument('--png-dir',     default='output/zoom_png')
    p.add_argument('--video',       default='output/zoom_test.mp4')
    p.add_argument('--renderer',    default='./build/renderer')
    p.add_argument('--launcher',    default='',
                   help='prefix command for renderer, e.g. "srun -n 4 -c 8"')
    p.add_argument('--render-parallelism', type=int, default=1,
                   help='maximum number of renderer jobs to run concurrently')
    p.add_argument('--render', action='store_true')
    p.add_argument('--plot',   action='store_true')
    p.add_argument('--encode', action='store_true')
    args = p.parse_args()

    cx, cy, cz = CLUSTER_POS
    for d in (args.config_dir, args.proj_dir, args.png_dir):
        os.makedirs(d, exist_ok=True)
    os.makedirs(os.path.dirname(args.video) or '.', exist_ok=True)

    w0, w1 = args.width_start, args.width_end
    widths = w0 * (w1 / w0) ** (np.arange(args.frames) / max(1, args.frames - 1))
    start_frame = max(0, min(args.trim_start_frame, args.frames - 1))

    # LOS slab = ortho_width per frame. Each frame integrates a cube of side
    # ortho_width around the cluster (same physical volume in xy and z). Across
    # a tier boundary, ortho_width is continuous, so the integrated content is
    # continuous too — the switch is purely a resolution change that a short
    # crossfade (optional) can mask.
    if args.image_size > 0:
        img_w = img_h = args.image_size
    else:
        img_w, img_h = args.image_width, args.image_height

    # Build the requested tier ladder. The deep tier set adds two smaller
    # grids so the movie can keep zooming after the original close view.
    tiers = build_tiers(args.grid_dir, args.tier_set)
    frame_plan = []  # (i, width, slab, tier_name, tier_grid_path, cam_path)
    for i, w in enumerate(widths):
        aspect = img_w / img_h
        tname, txy, tpath = pick_tier(tiers, w, aspect, args.tier_overflow_tol)
        slab = w if args.los_depth == 'same-as-y' else args.los_slab
        cam_path = f'{args.config_dir}/frame_{i:04d}.yaml'
        write_camera_yaml(cam_path, cx, cy, cz, w, slab, img_w, img_h)
        frame_plan.append((i, w, slab, tname, tpath, cam_path))

    if start_frame > 0:
        frame_plan = [frame for frame in frame_plan if frame[0] >= start_frame]

    # Report tier breakdown so the user can see the boundaries
    counts = {}
    for _, _, _, t, _, _ in frame_plan:
        counts[t] = counts.get(t, 0) + 1
    print(f'Frames per tier: {counts}')
    if start_frame > 0:
        print(f'Trimmed leading frames: 0..{start_frame - 1}')
    print(f'Width range: {widths[0]:.1f} -> {widths[-1]:.1f} ckpc/h')

    if args.render:
        # Group frames by tier (they're contiguous since we zoom monotonically)
        # and issue one renderer call per tier so the grid is loaded once.
        from itertools import groupby
        launch_prefix = args.launcher.split() if args.launcher else []
        max_parallel = max(1, args.render_parallelism)
        active = []

        def reap_finished(block=False):
            nonlocal active
            while active:
                for idx, (tier_name, proc) in enumerate(list(active)):
                    rc = proc.poll()
                    if rc is None:
                        continue
                    active.pop(idx)
                    if rc != 0:
                        raise subprocess.CalledProcessError(rc, proc.args)
                    return True
                if not block:
                    return False
                tier_name, proc = active[0]
                rc = proc.wait()
                active.pop(0)
                if rc != 0:
                    raise subprocess.CalledProcessError(rc, proc.args)
                return True
            return False

        for tname, tgroup in groupby(frame_plan, key=lambda r: r[3]):
            group = list(tgroup)
            tpath = group[0][4]
            if not os.path.exists(tpath):
                sys.exit(f'missing grid file: {tpath} (run the gridder first)')
            list_file = f'{args.config_dir}/cams_{tname}.txt'
            with open(list_file, 'w') as f:
                for i, w, slab, _tn, _tp, cam_path in group:
                    frame_dir = f'{args.proj_dir}/frame_{i:04d}'
                    os.makedirs(frame_dir, exist_ok=True)
                    f.write(f'{cam_path} {frame_dir}\n')
            cmd = launch_prefix + [args.renderer, '--grid', tpath,
                                   '--camera-list', list_file,
                                   '--fields', 'gas_density']
            print(f'[{tname}] {len(group)} frames -> one renderer call')
            print('  ' + ' '.join(cmd), flush=True)
            while len(active) >= max_parallel:
                reap_finished(block=True)
            proc = subprocess.Popen(cmd)
            active.append((tname, proc))

        while active:
            reap_finished(block=True)

    if args.plot:
        import h5py
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        from matplotlib.colors import LogNorm

        # Each frame integrates over its own slab (= ortho_width). Because the
        # cluster dominates a small volume, mean density along LOS drops by
        # orders of magnitude as we zoom out. A single global color range would
        # then make wide views mostly dark. Instead we pick per-frame
        # percentiles and smooth them over time to avoid flicker / jumps.

        # First pass: measure p_lo / p_hi for every frame in log space.
        log_lo = np.full(args.frames, np.nan)
        log_hi = np.full(args.frames, np.nan)
        P_LO, P_HI = 20.0, 99.7
        for i, w, slab, tname, _, _ in frame_plan:
            proj = f'{args.proj_dir}/frame_{i:04d}/projection_grid_{tname}_snap121_gas_density.hdf5'
            if not os.path.exists(proj):
                continue
            with h5py.File(proj, 'r') as f:
                d = f['gas_density'][:].astype(np.float64) / slab
            pos = d[d > 0]
            if pos.size == 0:
                continue
            log_lo[i] = np.log10(np.percentile(pos, P_LO))
            log_hi[i] = np.log10(np.percentile(pos, P_HI))

        # Temporal smoothing (rolling mean, window ~5 % of frames) in log space.
        win = max(5, args.frames // 20)
        if win % 2 == 0:
            win += 1
        def smooth(x):
            # Fill NaNs first
            idx = np.arange(len(x))
            mask = np.isfinite(x)
            if mask.sum() < 2:
                return x
            x = np.interp(idx, idx[mask], x[mask])
            kernel = np.ones(win) / win
            # Reflect-pad so endpoints aren't biased toward zero
            pad = win // 2
            xp = np.pad(x, pad, mode='edge')
            return np.convolve(xp, kernel, mode='valid')
        log_lo_s = smooth(log_lo)
        log_hi_s = smooth(log_hi)

        print(f'Per-frame log color range (smoothed, window={win}):')
        print(f'  start: {10**log_lo_s[0]:.3e} .. {10**log_hi_s[0]:.3e}')
        print(f'  end:   {10**log_lo_s[-1]:.3e} .. {10**log_hi_s[-1]:.3e}')

        for i, w, slab, tname, _, _ in frame_plan:
            proj = f'{args.proj_dir}/frame_{i:04d}/projection_grid_{tname}_snap121_gas_density.hdf5'
            if not os.path.exists(proj):
                print(f'missing: {proj}', file=sys.stderr); continue
            with h5py.File(proj, 'r') as f:
                data = f['gas_density'][:].astype(np.float64) / slab
            dp = data.copy()
            dp[dp <= 0] = np.nan
            lvmin = 10 ** log_lo_s[i]
            lvmax = 10 ** log_hi_s[i]
            if not (np.isfinite(lvmin) and np.isfinite(lvmax) and lvmin > 0 and lvmax > lvmin):
                pos = data[data > 0]
                lvmin = np.percentile(pos, P_LO) if pos.size else 1e-10
                lvmax = np.percentile(pos, P_HI) if pos.size else 1.0
            cmap = plt.get_cmap('inferno').copy(); cmap.set_bad('black')

            # Size the figure so the pixel grid is 1:1 with the data.
            dpi = 120
            figsize = (data.shape[1] / dpi, data.shape[0] / dpi)
            fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
            fig.patch.set_facecolor('black')
            ax.set_facecolor('black')
            if args.full_bleed:
                ax.set_position([0.0, 0.0, 1.0, 1.0])
                ax.set_axis_off()
            ax.imshow(dp, cmap=cmap, norm=LogNorm(vmin=lvmin, vmax=lvmax),
                      origin='upper', interpolation='nearest')
            label = f'{w/1000:.2f} cMpc/h' if w >= 1000 else f'{w:.0f} ckpc/h'
            if args.full_bleed:
                ax.text(0.015, 0.985, f'zoom width = {label}   (grid: {tname})',
                        transform=ax.transAxes, color='white', fontsize=10,
                        ha='left', va='top',
                        bbox=dict(boxstyle='round,pad=0.25', facecolor='black',
                                  alpha=0.35, edgecolor='none'))
            else:
                ax.set_title(f'zoom width = {label}   (grid: {tname})',
                             fontsize=10, color='white')
                ax.set_xticks([]); ax.set_yticks([])

            # Scale bar: 25 % of image width in data units.
            # ortho_width sets the vertical extent; horizontal extent scales
            # with aspect ratio.
            pw = data.shape[1]; ph = data.shape[0]
            x_extent = w * (pw / ph)
            bar_frac = 0.25
            x0 = pw * 0.05
            x1 = x0 + pw * bar_frac
            y  = ph * 0.92
            ax.plot([x0, x1], [y, y], color='white', lw=2)
            bar_len = x_extent * bar_frac
            bar_lbl = (f'{bar_len/1000:.2f} cMpc/h' if bar_len >= 1000
                       else f'{bar_len:.0f} ckpc/h')
            ax.text((x0 + x1)/2, y*0.97, bar_lbl,
                    color='white', fontsize=9, ha='center', va='bottom')

            out = f'{args.png_dir}/frame_{i:04d}.png'
            plt.savefig(out, facecolor='black')
            plt.close(fig)
            if (i + 1) % 10 == 0 or i == args.frames - 1:
                print(f'plotted {i+1}/{args.frames}', flush=True)

    if args.encode:
        cmd = ['ffmpeg', '-y', '-framerate', '24',
               '-start_number', str(start_frame),
               '-i', f'{args.png_dir}/frame_%04d.png',
               '-c:v', 'libx264', '-pix_fmt', 'yuv420p',
               '-vf', 'scale=trunc(iw/2)*2:trunc(ih/2)*2',
               args.video]
        print(' '.join(cmd))
        subprocess.run(cmd, check=True)
        print(f'wrote {args.video}')

if __name__ == '__main__':
    main()
