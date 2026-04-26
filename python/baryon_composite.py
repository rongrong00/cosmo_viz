#!/usr/bin/env python3
"""Sum gas_column.h5 + star_column.h5 into baryon_column.h5 and write a PNG."""
import argparse, numpy as np, h5py, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm, LinearSegmentedColormap


def _register_matter_r_fade():
    stops = [
        (0.00, '#2d0f3a'),
        (0.12, '#5a1552'),
        (0.28, '#932a5f'),
        (0.45, '#c94761'),
        (0.62, '#e67b56'),
        (0.78, '#f0b07a'),
        (0.90, '#f8dcb0'),
        (1.00, '#fef4df'),
    ]
    cmap = LinearSegmentedColormap.from_list(
        'matter_r_fade', [(s, c) for s, c in stops], N=512)
    try:
        matplotlib.colormaps.register(cmap=cmap, name='matter_r_fade')
    except ValueError:
        pass


_register_matter_r_fade()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--gas', required=True)
    ap.add_argument('--star', required=True)
    ap.add_argument('--h5-out', required=True)
    ap.add_argument('--png-out', required=True)
    ap.add_argument('--cmap', default='matter_r_fade')
    ap.add_argument('--gas-weight', type=float, default=1.0)
    ap.add_argument('--star-weight', type=float, default=1.0,
                    help='Multiplier on star column before summing. Stars are '
                         'physically denser than gas in galactic nuclei so the '
                         'sum is star-dominated at 1.0; try 0.1-0.3 to let gas show.')
    ap.add_argument('--vmin-pct', type=float, default=2.0)
    ap.add_argument('--vmax-pct', type=float, default=99.5)
    args = ap.parse_args()

    with h5py.File(args.gas, 'r') as f:
        g = f['gas_column_density'][:]
    with h5py.File(args.star, 'r') as f:
        s = f['star_column_density'][:]

    b = args.gas_weight * g.astype(np.float64) + args.star_weight * s.astype(np.float64)

    with h5py.File(args.h5_out, 'w') as f:
        f.create_dataset('baryon_column_density', data=b.astype(np.float32),
                         compression='gzip')
        f.attrs['fields_summed'] = 'gas_column_density + star_column_density'

    pos = b[b > 0]
    vmin = float(np.percentile(pos, args.vmin_pct))
    vmax = float(np.percentile(pos, args.vmax_pct))
    print(f'baryon min={b.min():.3e} max={b.max():.3e} '
          f'vmin={vmin:.3e} vmax={vmax:.3e}  gas_max={g.max():.3e} '
          f'star_max={s.max():.3e}')

    H, W = b.shape
    fig, ax = plt.subplots(figsize=(W/100, H/100), dpi=100)
    ax.set_position([0, 0, 1, 1]); ax.set_axis_off()
    fig.patch.set_facecolor('black')
    bp = b.astype(np.float64); bp[bp <= 0] = np.nan
    cmap = plt.get_cmap(args.cmap).copy(); cmap.set_bad('black')
    ax.imshow(bp, cmap=cmap, norm=LogNorm(vmin=vmin, vmax=vmax), origin='upper')
    plt.savefig(args.png_out, dpi=100, facecolor='black', pad_inches=0)
    print(f'wrote {args.h5_out}, {args.png_out}')


if __name__ == '__main__':
    main()
