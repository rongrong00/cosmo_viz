#!/usr/bin/env python3
"""
Plot a gas column-density map with a matter_r colormap and a low-density
alpha fade-out so voids are transparent rather than hard black. Saves both
an RGBA PNG (compositable over other frames) and an RGB PNG flattened over
a chosen background color.
"""
import argparse, h5py, numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm, LinearSegmentedColormap, to_rgb
import matplotlib.image as mpimg


def blue_orange_cmap():
    # Diverging-style: very dark blue low end, bright neutral midpoint,
    # moderate orange high end (warm but not as dark as the blue tail).
    stops = [
        (0.00, '#02041e'),  # near-black navy
        (0.18, '#0a1a6a'),  # deep blue
        (0.36, '#2a56c8'),  # medium blue
        (0.50, '#e8eaec'),  # neutral pale (midpoint)
        (0.66, '#f4c07a'),  # soft orange
        (0.85, '#ec8a30'),  # orange
        (1.00, '#c25410'),  # deeper amber (not as dark as blue end)
    ]
    return LinearSegmentedColormap.from_list(
        'blue_orange', [(s, c) for s, c in stops], N=512)


def ocean_glow_cmap():
    # Deep-ocean navy -> teal -> cyan -> pale aqua -> luminous white-cyan.
    stops = [
        (0.00, '#02061c'),
        (0.15, '#07224d'),
        (0.32, '#0d4e7a'),
        (0.50, '#1a8aa6'),
        (0.68, '#3cc9c4'),
        (0.85, '#b6f1e6'),
        (1.00, '#f4fffb'),
    ]
    return LinearSegmentedColormap.from_list(
        'ocean_glow', [(s, c) for s, c in stops], N=512)


def ice_cmap():
    # Near-black -> deep navy -> steel blue -> frost -> near-white icy cyan.
    stops = [
        (0.00, '#02030f'),
        (0.18, '#0b1e4a'),
        (0.38, '#1f4f8f'),
        (0.58, '#4d8ec8'),
        (0.78, '#a9d6ef'),
        (0.92, '#e2f3fb'),
        (1.00, '#ffffff'),
    ]
    return LinearSegmentedColormap.from_list(
        'ice', [(s, c) for s, c in stops], N=512)


def matter_r_cmap():
    # Hand-tuned approximation of cmocean 'matter' reversed: goes from deep
    # aubergine through plum/magenta and orange-red up to pale cream.
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
    return LinearSegmentedColormap.from_list(
        'matter_r_fade', [(s, c) for s, c in stops], N=512)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--input', required=True)
    ap.add_argument('--output', required=True)
    ap.add_argument('--field', default='gas_column_density')
    ap.add_argument('--vmin-pct', type=float, default=40.0)
    ap.add_argument('--vmax-pct', type=float, default=99.7)
    ap.add_argument('--vmin', type=float, default=None)
    ap.add_argument('--vmax', type=float, default=None)
    ap.add_argument('--alpha-floor', type=float, default=0.0,
                    help='alpha at vmin')
    ap.add_argument('--alpha-gamma', type=float, default=1.0,
                    help='alpha = norm**gamma (gamma<1 makes fade steeper)')
    ap.add_argument('--bg', default='black',
                    help='background color for flattened RGB png')
    ap.add_argument('--cmap', default='matter_r_fade',
                    help='matter_r_fade (custom) or any matplotlib cmap name')
    args = ap.parse_args()

    with h5py.File(args.input, 'r') as f:
        data = f[args.field][:].astype(np.float64)

    positive = data[data > 0]
    vmin = args.vmin if args.vmin is not None else np.percentile(positive, args.vmin_pct)
    vmax = args.vmax if args.vmax is not None else np.percentile(positive, args.vmax_pct)
    print(f'vmin={vmin:.3e}  vmax={vmax:.3e}')

    norm = LogNorm(vmin=vmin, vmax=vmax, clip=True)
    x = norm(np.where(data > 0, data, vmin)).filled(0.0) if hasattr(norm(data), 'filled') else norm(np.clip(data, vmin, vmax))
    x = np.asarray(x, dtype=np.float64)

    if args.cmap == 'matter_r_fade':
        cmap = matter_r_cmap()
    elif args.cmap == 'blue_orange':
        cmap = blue_orange_cmap()
    elif args.cmap == 'ocean_glow':
        cmap = ocean_glow_cmap()
    elif args.cmap == 'ice':
        cmap = ice_cmap()
    else:
        cmap = plt.get_cmap(args.cmap)
    rgba = cmap(x)  # H x W x 4
    rgba = np.array(rgba)
    alpha = args.alpha_floor + (1.0 - args.alpha_floor) * np.power(x, args.alpha_gamma)
    # fully transparent for non-positive pixels
    alpha[data <= 0] = 0.0
    rgba[..., 3] = alpha

    # flattened RGB over bg (written directly to --output; no RGBA file)
    bg = np.array(to_rgb(args.bg))
    flat = rgba[..., :3] * rgba[..., 3:4] + bg * (1.0 - rgba[..., 3:4])
    mpimg.imsave(args.output, np.clip(flat, 0, 1), origin='upper')
    print(f'Saved -> {args.output}')


if __name__ == '__main__':
    main()
