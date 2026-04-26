#!/usr/bin/env python3
"""Diff the SPH direct-trace image vs the grid-path image.

Reports per-pixel statistics and writes:
  - side-by-side PNG
  - relative-difference PNG
"""
import argparse
import numpy as np
import h5py
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm


def read(path, field):
    with h5py.File(path, 'r') as f:
        return f[field][:]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--sph',  required=True)
    ap.add_argument('--grid', required=True)
    ap.add_argument('--sph-field',  default='gas_column_density')
    ap.add_argument('--grid-field', default='gas_density')
    ap.add_argument('--out-prefix', default='output/sph_vs_grid')
    args = ap.parse_args()

    sph  = read(args.sph,  args.sph_field ).astype(np.float64)
    grid = read(args.grid, args.grid_field).astype(np.float64)
    assert sph.shape == grid.shape, f"{sph.shape} vs {grid.shape}"

    both = (sph > 0) & (grid > 0)
    ratio = np.where(both, sph / grid, np.nan)
    rel   = np.where(both, (sph - grid) / np.maximum(np.abs(grid), 1e-30), np.nan)

    def stats(name, a):
        a = a[np.isfinite(a)]
        print(f"  {name}: mean={a.mean():.4g}  median={np.median(a):.4g}  "
              f"std={a.std():.4g}  min={a.min():.4g}  max={a.max():.4g}")

    print(f"SPH  : min={sph.min():.3e} max={sph.max():.3e} sum={sph.sum():.3e}")
    print(f"Grid : min={grid.min():.3e} max={grid.max():.3e} sum={grid.sum():.3e}")
    print(f"Integrated-sum ratio (sph/grid): {sph.sum() / max(grid.sum(), 1e-30):.4f}")
    print("Per-pixel stats (where both > 0):")
    stats("ratio  sph/grid", ratio)
    stats("rel diff",        rel)

    H, W = sph.shape
    fw = 18.0
    fh = fw * (H / W) * 0.5
    fig, axes = plt.subplots(1, 3, figsize=(fw, fh))

    positive = np.concatenate([sph[sph > 0], grid[grid > 0]])
    vmin = np.percentile(positive, 1)
    vmax = np.percentile(positive, 99.5)
    cmap = plt.get_cmap('inferno').copy(); cmap.set_bad('black')

    for ax, img, title in [(axes[0], sph, 'SPH direct'),
                           (axes[1], grid, 'Grid path')]:
        arr = img.copy(); arr[arr <= 0] = np.nan
        im = ax.imshow(arr, cmap=cmap, norm=LogNorm(vmin=vmin, vmax=vmax),
                       origin='upper')
        ax.set_title(title); ax.set_xticks([]); ax.set_yticks([])
    plt.colorbar(im, ax=axes[:2].tolist(), shrink=0.8, label='column density')

    r = rel.copy()
    im2 = axes[2].imshow(r, cmap='RdBu_r', vmin=-0.5, vmax=0.5, origin='upper')
    axes[2].set_title('(SPH - Grid) / Grid')
    axes[2].set_xticks([]); axes[2].set_yticks([])
    plt.colorbar(im2, ax=axes[2], shrink=0.8)

    out_png = args.out_prefix + '.png'
    plt.savefig(out_png, dpi=150, bbox_inches='tight')
    print(f"Wrote {out_png}")


if __name__ == '__main__':
    main()
