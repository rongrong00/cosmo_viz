#!/usr/bin/env python3
"""
Quick colormap tuning for metallicity mass-weighted column.
Reads a single <frame>/metallicity.h5, renders a grid of candidate colormaps
(log norm, fixed vmin/vmax from percentiles) into one PNG for side-by-side
evaluation.

Usage:
  python python/tune_metallicity_cmap.py \
      --h5 output/orbit_sph_spiral_mw_f0/metallicity.h5 \
      --out output/metal_cmap_tune.png
"""
import argparse, os
import numpy as np
import h5py
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm, LinearSegmentedColormap


def matter_r_fade():
    stops = [
        (0.00, '#2d0f3a'), (0.12, '#5a1552'), (0.28, '#932a5f'),
        (0.45, '#c94761'), (0.62, '#e67b56'), (0.78, '#f0b07a'),
        (0.90, '#f8dcb0'), (1.00, '#fef4df'),
    ]
    return LinearSegmentedColormap.from_list('matter_r_fade', stops, N=512)


def metal_gold():
    """Dark teal -> rust -> gold -> pale cream. Reads as metal-rich."""
    stops = [
        (0.00, '#0a1d2a'), (0.18, '#1f3a4a'), (0.38, '#7a3a1e'),
        (0.58, '#c57a1f'), (0.78, '#e8c25a'), (1.00, '#fff2c8'),
    ]
    return LinearSegmentedColormap.from_list('metal_gold', stops, N=512)


def metal_cu():
    """Near-black -> deep copper -> bright copper -> ivory."""
    stops = [
        (0.00, '#05070a'), (0.22, '#3c1410'), (0.45, '#8f2d14'),
        (0.65, '#d0691e'), (0.82, '#f0b065'), (1.00, '#fff6e0'),
    ]
    return LinearSegmentedColormap.from_list('metal_cu', stops, N=512)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--h5', required=True)
    ap.add_argument('--field', default='gas_metallicity_mw')
    ap.add_argument('--out', default='output/metal_cmap_tune/metal_cmap_tune.png')
    ap.add_argument('--vmin-pct', type=float, default=60.0)
    ap.add_argument('--vmax-pct', type=float, default=99.7)
    ap.add_argument('--bg', default='black')
    args = ap.parse_args()

    with h5py.File(args.h5) as f:
        d = f[args.field][:].astype(np.float64)
    pos = d[d > 0]
    if pos.size == 0:
        raise SystemExit('no positive pixels')
    vmin = float(np.percentile(pos, args.vmin_pct))
    vmax = float(np.percentile(pos, args.vmax_pct))
    print(f'vmin={vmin:.3e} vmax={vmax:.3e} (pct {args.vmin_pct}/{args.vmax_pct})')

    norm = LogNorm(vmin=vmin, vmax=vmax, clip=True)
    H, W = d.shape
    data = np.where(d > 0, d, vmin * 0.1)

    cmaps = [
        ('matter_r_fade (current)', matter_r_fade()),
        ('metal_gold',              metal_gold()),
        ('metal_cu',                metal_cu()),
        ('plasma',                  plt.get_cmap('plasma')),
        ('inferno',                 plt.get_cmap('inferno')),
        ('cividis',                 plt.get_cmap('cividis')),
        ('magma',                   plt.get_cmap('magma')),
        ('viridis',                 plt.get_cmap('viridis')),
    ]

    nc = 2
    nr = (len(cmaps) + nc - 1) // nc
    fig, axes = plt.subplots(nr, nc, figsize=(nc * 7, nr * 7 * H / W),
                             facecolor=args.bg)
    for ax, (name, cmap) in zip(axes.flat, cmaps):
        ax.imshow(data, cmap=cmap, norm=norm, origin='lower',
                  interpolation='nearest')
        ax.set_title(name, color='white', fontsize=11)
        ax.set_facecolor(args.bg)
        ax.set_xticks([]); ax.set_yticks([])
    for ax in axes.flat[len(cmaps):]:
        ax.axis('off')

    os.makedirs(os.path.dirname(args.out) or '.', exist_ok=True)
    fig.tight_layout()
    fig.savefig(args.out, dpi=120, facecolor=args.bg)
    print('wrote', args.out)


if __name__ == '__main__':
    main()
