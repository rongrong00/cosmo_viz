#!/usr/bin/env python3
"""
Hue = mass-weighted temperature, brightness = log gas column.

Input:
  --col   HDF5 with mass column density   (dataset 'gas_column_density')
  --temp  HDF5 with mass-weighted T map   (dataset 'gas_temperature_mw')

Per pixel:
  hue   = cmap_T(log T)       # where T sits in the cold/warm/hot rainbow
  value = norm(log rho_col)   # how bright the pixel gets

Final RGB = hue * value. Cold+bright → saturated blue knots. Hot+bright →
red halo cores. Low density → dark regardless of T. This is the standard
"temperature-colored density" cosmo-viz palette.
"""
import argparse, os, numpy as np, h5py
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap


def load(path, field):
    with h5py.File(path, 'r') as f:
        return f[field][:].astype(np.float64)


def log_norm01(x, pmin, pmax, vmin=None, vmax=None):
    pos = x[x > 0]
    if pos.size == 0:
        return np.zeros_like(x)
    if vmin is None: vmin = np.percentile(pos, pmin)
    if vmax is None: vmax = np.percentile(pos, pmax)
    vmin = max(vmin, vmax * 1e-6)
    lx = np.log10(np.clip(x, vmin, vmax))
    t = (lx - np.log10(vmin)) / (np.log10(vmax) - np.log10(vmin))
    t[x <= 0] = 0.0
    return np.clip(t, 0.0, 1.0)


def make_temp_cmap():
    # Cold (deep blue) → warm (purple/magenta) → hot (red), monotonic hue
    # with no white in the middle. Any white in the final image has to
    # come from --white-boost at actually bright+hot pixels, so warm
    # filaments at ~10^4.7 K stay red-tinted instead of washing white.
    stops = [
        (0.00, '#0820a0'),  # cold          deep blue
        (0.20, '#3050d8'),  # cool IGM      blue
        (0.45, '#7030c0'),  # warm filament violet
        (0.70, '#c03060'),  # shocked warm  magenta
        (0.90, '#e84020'),  # hot halo edge orange-red
        (1.00, '#b01010'),  # very hot core deep red
    ]
    return LinearSegmentedColormap.from_list(
        'temp_brm', [(s, c) for s, c in stops], N=256)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--col',    required=True)
    p.add_argument('--temp',   required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--col-field',  default='gas_column_density')
    p.add_argument('--temp-field', default='gas_temperature_mw')
    p.add_argument('--pmin', type=float, default=1.0,
                   help='col percentile for brightness floor')
    p.add_argument('--pmax', type=float, default=99.7,
                   help='col percentile for brightness top')
    p.add_argument('--logT-min', type=float, default=3.8)
    p.add_argument('--logT-max', type=float, default=6.0)
    p.add_argument('--gamma', type=float, default=1.0,
                   help='brightness gamma. >1 darkens mids, <1 lifts them')
    p.add_argument('--white-boost', type=float, default=0.0,
                   help='desaturate toward white as brightness→1. '
                        '0=off, ~0.6 burns halo cores to white.')
    p.add_argument('--white-power', type=float, default=4.0,
                   help='how sharply the white-boost kicks in. Larger '
                        'values confine the burn to the brightest tail.')
    args = p.parse_args()

    col  = load(args.col,  args.col_field)
    temp = load(args.temp, args.temp_field)
    assert col.shape == temp.shape, f'shape mismatch {col.shape} vs {temp.shape}'

    bright = log_norm01(col, args.pmin, args.pmax)
    if args.gamma != 1.0:
        bright = np.power(bright, args.gamma)

    # Map log T → [0,1] for the temperature cmap. Pixels with no gas get
    # no hue and (via the brightness=0) stay black.
    logT = np.log10(np.clip(temp, 10**args.logT_min, 10**args.logT_max))
    hue_t = (logT - args.logT_min) / (args.logT_max - args.logT_min)
    hue_t[temp <= 0] = 0.5  # irrelevant: bright=0 there anyway

    cmap_T = make_temp_cmap()
    rgb = cmap_T(hue_t)[..., :3]
    rgb = rgb * bright[..., None]
    if args.white_boost > 0:
        w = args.white_boost * np.power(bright, args.white_power)
        rgb = rgb + w[..., None] * (1.0 - rgb)
    rgb = np.clip(rgb, 0.0, 1.0)

    print(f'col:  p1={np.percentile(col[col>0], 1):.3e}  '
          f'p99.7={np.percentile(col[col>0], 99.7):.3e}')
    print(f'temp: min={temp.min():.3e} max={temp.max():.3e} '
          f'median={np.median(temp[temp>0]):.3e}')

    H, W = col.shape
    os.makedirs(os.path.dirname(args.output) or '.', exist_ok=True)
    dpi = 100
    fig, ax = plt.subplots(figsize=(W/dpi, H/dpi), dpi=dpi)
    ax.imshow(rgb, origin='upper')
    ax.set_position([0, 0, 1, 1]); ax.set_axis_off()
    fig.patch.set_facecolor('black')
    plt.savefig(args.output, dpi=dpi, facecolor='black', pad_inches=0)
    plt.close(fig)
    print(f'wrote {args.output}')


if __name__ == '__main__':
    main()
