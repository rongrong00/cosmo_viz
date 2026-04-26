#!/usr/bin/env python3
"""Diverging-cmap comparison grid for gas metallicity, on log10(Z) with a
TwoSlopeNorm pivot (metal-poor IGM vs enriched gas)."""
import argparse, os
import numpy as np
import h5py
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm, LinearSegmentedColormap


def metal_diverge_custom():
    """Black (pristine) -> navy -> teal -> cream (pivot) -> gold -> copper."""
    stops = [
        (0.00, '#000000'), (0.12, '#050b1c'), (0.28, '#0a1630'),
        (0.40, '#1f5560'), (0.48, '#f2ead2'), (0.50, '#fff4cf'),
        (0.72, '#d19040'), (1.00, '#5a1808'),
    ]
    return LinearSegmentedColormap.from_list('metal_diverge_custom', stops, N=512)


def metal_dark_diverge():
    """Pure black (pristine) -> deep indigo -> teal -> neutral pivot ->
    gold -> deep copper -> bright saturated yellow (enriched). Low end is
    true #000000; pivot is a neutral warm-gray so cool/warm sides read
    distinctly; high end is saturated yellow, visually different from pivot."""
    stops = [
        (0.00, '#000000'),
        (0.12, '#0a1028'),
        (0.28, '#18365c'),
        (0.42, '#2e6b72'),
        (0.50, '#c8c0ac'),
        (0.62, '#e6a23a'),
        (0.80, '#c14c12'),
        (1.00, '#5a1808'),
    ]
    return LinearSegmentedColormap.from_list('metal_dark_diverge', stops, N=512)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--h5', required=True)
    ap.add_argument('--field', default='gas_metallicity_mw')
    ap.add_argument('--out', default='output/metal_cmap_tune/metal_diverging_tune.png')
    ap.add_argument('--log-pivot', type=float, default=-3.0,
                    help='log10(Z) value at the neutral center of the cmap')
    ap.add_argument('--log-low', type=float, default=None,
                    help='log10(Z) low end (default: p2 of positive log Z)')
    ap.add_argument('--log-high', type=float, default=None,
                    help='log10(Z) high end (default: p99.5 of positive log Z)')
    ap.add_argument('--bg', default='black')
    args = ap.parse_args()

    with h5py.File(args.h5) as f:
        d = f[args.field][:].astype(np.float64)
    pos = d > 0
    logZ = np.where(pos, np.log10(np.where(pos, d, 1e-30)), np.nan)
    lp = logZ[~np.isnan(logZ)]
    lo = args.log_low  if args.log_low  is not None else float(np.percentile(lp, 2))
    hi = args.log_high if args.log_high is not None else float(np.percentile(lp, 99.5))
    pivot = args.log_pivot
    if not (lo < pivot < hi):
        # clamp pivot into range so TwoSlopeNorm works
        pivot = 0.5 * (lo + hi)
    print(f'log10(Z): lo={lo:.2f} pivot={pivot:.2f} hi={hi:.2f}')

    norm = TwoSlopeNorm(vmin=lo, vcenter=pivot, vmax=hi)
    H, W = logZ.shape
    img = np.where(np.isnan(logZ), lo, logZ)

    cmaps = [
        ('RdBu_r (blue=poor, red=rich)',      plt.get_cmap('RdBu_r')),
        ('coolwarm',                          plt.get_cmap('coolwarm')),
        ('Spectral_r',                        plt.get_cmap('Spectral_r')),
        ('PuOr_r (purple=poor, orange=rich)', plt.get_cmap('PuOr_r')),
        ('BrBG_r (teal=poor, brown=rich)',    plt.get_cmap('BrBG_r')),
        ('PiYG_r',                            plt.get_cmap('PiYG_r')),
        ('seismic',                           plt.get_cmap('seismic')),
        ('metal_diverge_custom (navy→copper)', metal_diverge_custom()),
        ('metal_dark_diverge (black→cream→copper)', metal_dark_diverge()),
    ]

    nc = 2
    nr = (len(cmaps) + nc - 1) // nc
    fig, axes = plt.subplots(nr, nc, figsize=(nc * 7, nr * 7 * H / W),
                             facecolor=args.bg)
    for ax, (name, cmap) in zip(axes.flat, cmaps):
        ax.imshow(img, cmap=cmap, norm=norm, origin='lower',
                  interpolation='nearest')
        ax.set_title(f'{name}  | pivot={pivot:.1f}', color='white', fontsize=10)
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
