#!/usr/bin/env python3
"""
Composite three volume-rendered (emission, transmittance) phase maps
(cold/warm/hot) into a single tri-color RGB image.

Input:  <indir>/gas_cold_vol.h5, gas_warm_vol.h5, gas_hot_vol.h5
        each with datasets 'emission' and 'transmittance'.
Output: a PNG where each phase contributes its own color, modulated by
        its per-pixel opacity (1 - T), so occlusion is physically real.
"""
import argparse, os, numpy as np, h5py
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgb


def load_vol(path):
    with h5py.File(path, 'r') as f:
        return f['emission'][:].astype(np.float64), f['transmittance'][:].astype(np.float64)


def shade(E, T, color_rgb, pmin=50.0, pmax=99.5, gain=1.0,
          alpha_gamma=1.0, alpha_norm=True):
    pos = E[E > 0]
    if pos.size == 0:
        return np.zeros(E.shape + (3,)), np.zeros(E.shape)
    vmin = max(np.percentile(pos, pmin), 1e-30)
    vmax = np.percentile(pos, pmax)
    vmin = max(vmin, vmax * 1e-4)
    lE = np.log10(np.clip(E, vmin, vmax))
    t = (lE - np.log10(vmin)) / (np.log10(vmax) - np.log10(vmin))
    t[E <= 0] = 0.0
    t = np.clip(t, 0.0, 1.0) * gain

    opacity = np.clip(1.0 - T, 0.0, 1.0)
    if alpha_norm:
        p99 = np.percentile(opacity, 99)
        if p99 > 0:
            opacity = np.clip(opacity / p99, 0.0, 1.0)
    if alpha_gamma != 1.0:
        opacity = np.power(opacity, alpha_gamma)

    intensity = t * opacity
    rgb = intensity[..., None] * np.asarray(color_rgb)[None, None, :]
    return rgb, opacity


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--indir',  required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--cold-color', default='#3ab6ff')
    p.add_argument('--warm-color', default='#3ddf8f')
    p.add_argument('--hot-color',  default='#ff5030')
    p.add_argument('--cold-gain', type=float, default=1.0)
    p.add_argument('--warm-gain', type=float, default=1.0)
    p.add_argument('--hot-gain',  type=float, default=1.2)
    p.add_argument('--pmin', type=float, default=50.0)
    p.add_argument('--pmax', type=float, default=99.5)
    p.add_argument('--alpha-gamma', type=float, default=1.0)
    args = p.parse_args()

    def load(name):
        path = os.path.join(args.indir, f'gas_{name}_vol.h5')
        E, T = load_vol(path)
        pos = E[E > 0]
        stat = f'nz={pos.size}' if pos.size == 0 else (
            f'nz={pos.size} Emed={np.percentile(pos,50):.2e} '
            f'Emax={E.max():.2e} 1-Tmax={(1-T).max():.3f}')
        print(f'{name}: {stat}')
        return E, T

    Ec, Tc = load('cold')
    Ew, Tw = load('warm')
    Eh, Th = load('hot')

    rc, _ = shade(Ec, Tc, to_rgb(args.cold_color), args.pmin, args.pmax,
                  args.cold_gain, args.alpha_gamma)
    rw, _ = shade(Ew, Tw, to_rgb(args.warm_color), args.pmin, args.pmax,
                  args.warm_gain, args.alpha_gamma)
    rh, _ = shade(Eh, Th, to_rgb(args.hot_color),  args.pmin, args.pmax,
                  args.hot_gain,  args.alpha_gamma)

    rgb = np.clip(rc + rw + rh, 0.0, 1.0)

    H, W = Ec.shape
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
