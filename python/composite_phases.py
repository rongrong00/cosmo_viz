#!/usr/bin/env python3
"""
Composite cold/warm/hot mass-column maps into a single tri-color image
a la Illustris. Each channel gets its own color + log normalization +
alpha; channels are added in linear light so overlapping phases blend.
"""
import argparse, os, numpy as np, h5py
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgb


def load_col(path, field):
    with h5py.File(path, 'r') as f:
        return f[field][:].astype(np.float64)


def norm_log(x, pmin=30.0, pmax=99.7, vmin=None, vmax=None):
    pos = x[x > 0]
    if pos.size == 0:
        return np.zeros_like(x)
    if vmin is None: vmin = np.percentile(pos, pmin)
    if vmax is None: vmax = np.percentile(pos, pmax)
    vmin = max(vmin, vmax * 1e-4)
    lx = np.log10(np.clip(x, vmin, vmax))
    t = (lx - np.log10(vmin)) / (np.log10(vmax) - np.log10(vmin))
    t[x <= 0] = 0.0
    return np.clip(t, 0.0, 1.0)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--indir', required=True, help='dir with cold_col.h5 warm_col.h5 hot_col.h5')
    p.add_argument('--output', required=True)
    p.add_argument('--cold-color', default='#3ab6ff')
    p.add_argument('--warm-color', default='#3ddf8f')
    p.add_argument('--hot-color',  default='#ff5030')
    p.add_argument('--cold-gain', type=float, default=1.0)
    p.add_argument('--warm-gain', type=float, default=1.0)
    p.add_argument('--hot-gain',  type=float, default=1.2)
    p.add_argument('--pmin', type=float, default=40.0)
    p.add_argument('--pmax', type=float, default=99.7)
    args = p.parse_args()

    cold = load_col(os.path.join(args.indir, 'cold_col.h5'), 'gas_cold_column')
    warm = load_col(os.path.join(args.indir, 'warm_col.h5'), 'gas_warm_column')
    hot  = load_col(os.path.join(args.indir, 'hot_col.h5'),  'gas_hot_column')

    for name, a in [('cold', cold), ('warm', warm), ('hot', hot)]:
        pos = a[a > 0]
        if pos.size:
            print(f'{name}: nz={pos.size}  min={pos.min():.3e}  '
                  f'p50={np.percentile(pos,50):.3e}  max={a.max():.3e}')
        else:
            print(f'{name}: EMPTY')

    tc = norm_log(cold, args.pmin, args.pmax) * args.cold_gain
    tw = norm_log(warm, args.pmin, args.pmax) * args.warm_gain
    th = norm_log(hot,  args.pmin, args.pmax) * args.hot_gain

    c_cold = np.array(to_rgb(args.cold_color))
    c_warm = np.array(to_rgb(args.warm_color))
    c_hot  = np.array(to_rgb(args.hot_color))

    H, W = cold.shape
    rgb = (tc[..., None] * c_cold
         + tw[..., None] * c_warm
         + th[..., None] * c_hot)
    rgb = np.clip(rgb, 0.0, 1.0)

    os.makedirs(os.path.dirname(args.output) or '.', exist_ok=True)
    dpi = 100
    fig, ax = plt.subplots(figsize=(W/dpi, H/dpi), dpi=dpi)
    ax.imshow(rgb, origin='upper')
    ax.set_position([0,0,1,1]); ax.set_axis_off()
    fig.patch.set_facecolor('black')
    plt.savefig(args.output, dpi=dpi, facecolor='black', pad_inches=0)
    plt.close(fig)
    print(f'wrote {args.output}')


if __name__ == '__main__':
    main()
