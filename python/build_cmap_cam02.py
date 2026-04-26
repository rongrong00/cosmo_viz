#!/usr/bin/env python3
"""
Design a perceptually-uniform colormap from colors you pick.

How it works:
  1. Your colors are converted from sRGB to CAM02-UCS (J', a', b').
  2. J' (lightness) is overridden with a STRICTLY LINEAR ramp from the
     first anchor's J' to the last, so perceived brightness rises at a
     constant rate across the colormap (the same guarantee viridis gives).
  3. (a', b') — the hue/chroma path you designed — is interpolated along
     the same linear-in-J' parameter, so the hue walk is preserved while
     brightness is forced uniform.
  4. Converted back to sRGB; any out-of-gamut samples are clipped and
     reported.

Usage:
  python python/build_cmap_cam02.py \\
      --input  output/orbit_sph_spiral/frame_0000/gas_column.h5 \\
      --output output/cam02_preview/sunset.png \\
      --name   sunset \\
      --colors '#03000a' '#2d1a6b' '#b83b82' '#f07035' '#fff4c2'

Color order = low-density → high-density. For a monotonic-brightness
colormap, pick colors that are already roughly dark→bright (the script
will then enforce exact linearity).
"""

import argparse
import os
import numpy as np
import h5py
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm, LinearSegmentedColormap, to_rgb
from colorspacious import cspace_convert


def _alpha_ramp(N, alpha_low=0.0, alpha_high=1.0, alpha_gamma=1.0):
    # Ramp alpha from alpha_low (bottom of cmap) to alpha_high (top). A
    # gamma > 1 keeps the low end transparent for longer before ramping up;
    # gamma < 1 fades in sharply near the bottom then holds high alpha.
    t = np.linspace(0, 1, N) ** alpha_gamma
    return alpha_low + t * (alpha_high - alpha_low)


def build_cmap(colors, name='custom_cam02', N=256, enforce_linear_J=True,
               alpha_low=1.0, alpha_high=1.0, alpha_gamma=1.0):
    rgbs = np.array([to_rgb(c) for c in colors])
    jab = cspace_convert(rgbs, 'sRGB1', 'CAM02-UCS')
    Jp, ap, bp = jab[:, 0], jab[:, 1], jab[:, 2]

    # Parameterize anchors by their J' value — this way, interpolation at
    # a given J' target lands at the right anchor. If input J' isn't
    # monotonic, sort anchors by J' so the ramp is well-defined.
    order = np.argsort(Jp)
    Jp_s, ap_s, bp_s = Jp[order], ap[order], bp[order]

    # Target J' ramp: strictly linear from min(Jp) to max(Jp).
    t = np.linspace(0, 1, N)
    if enforce_linear_J:
        Jp_i = Jp_s[0] + t * (Jp_s[-1] - Jp_s[0])
    else:
        Jp_i = np.interp(t, np.linspace(0, 1, len(Jp_s)), Jp_s)

    # Interpolate (a', b') along the sorted anchor sequence, using J' as
    # the independent variable so hue at a given lightness matches what
    # the user picked.
    ap_i = np.interp(Jp_i, Jp_s, ap_s)
    bp_i = np.interp(Jp_i, Jp_s, bp_s)
    jab_i = np.column_stack([Jp_i, ap_i, bp_i])

    rgb_i = cspace_convert(jab_i, 'CAM02-UCS', 'sRGB1')
    oog = np.any((rgb_i < 0) | (rgb_i > 1), axis=1).sum()
    rgb_i = np.clip(rgb_i, 0, 1)

    print(f'[{name}] anchors={len(colors)}  out-of-sRGB-gamut clipped: {oog}/{N}')
    print(f'[{name}] J\' anchors (sorted) = {np.round(Jp_s, 1)}')
    print(f'[{name}] J\' ramp: {Jp_i[0]:.1f} → {Jp_i[-1]:.1f} (linear)')

    # Attach an alpha ramp so low-density pixels can be made transparent.
    alpha = _alpha_ramp(N, alpha_low, alpha_high, alpha_gamma)
    rgba = np.column_stack([rgb_i, alpha])
    print(f'[{name}] alpha ramp: {alpha[0]:.2f} → {alpha[-1]:.2f} '
          f'(gamma={alpha_gamma})')

    return LinearSegmentedColormap.from_list(name, rgba, N=N)


def render_swatch(cmap, out_path):
    os.makedirs(os.path.dirname(out_path) or '.', exist_ok=True)
    grad = np.linspace(0, 1, 512).reshape(1, -1)
    fig, ax = plt.subplots(figsize=(8, 1), dpi=100)
    ax.imshow(grad, aspect='auto', cmap=cmap)
    ax.set_axis_off()
    plt.subplots_adjust(0, 0, 1, 1)
    plt.savefig(out_path, dpi=100, pad_inches=0)
    plt.close(fig)
    print(f'wrote {out_path}')


def render_preview(cmap, h5_path, out_path, field='gas_column_density',
                   vmin_pct=1.0, vmax_pct=99.5):
    with h5py.File(h5_path, 'r') as f:
        d = f[field][:].astype(np.float64)
    pos = d[d > 0]
    vmin = np.percentile(pos, vmin_pct)
    vmax = np.percentile(pos, vmax_pct)
    d[d <= 0] = np.nan

    cm = cmap.copy()
    cm.set_bad('black')
    h, w = d.shape
    fig, ax = plt.subplots(figsize=(w / 100, h / 100), dpi=100)
    ax.imshow(d, cmap=cm, norm=LogNorm(vmin=vmin, vmax=vmax), origin='upper')
    ax.set_position([0, 0, 1, 1])
    ax.set_axis_off()
    fig.patch.set_facecolor('black')
    os.makedirs(os.path.dirname(out_path) or '.', exist_ok=True)
    plt.savefig(out_path, dpi=100, facecolor='black', pad_inches=0)
    plt.close(fig)
    print(f'wrote {out_path}')


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--input',  required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--name',   default='custom_cam02')
    p.add_argument('--colors', nargs='+', required=True,
                   help='Hex or named colors (order: low→high density).')
    p.add_argument('--field',    default='gas_column_density')
    p.add_argument('--vmin-pct', type=float, default=1.0)
    p.add_argument('--vmax-pct', type=float, default=99.5)
    p.add_argument('--alpha-low',   type=float, default=1.0,
                   help='alpha at low-density end (0=fully transparent)')
    p.add_argument('--alpha-high',  type=float, default=1.0,
                   help='alpha at high-density end')
    p.add_argument('--alpha-gamma', type=float, default=1.0,
                   help='>1 keeps low end transparent longer; <1 fades in fast')
    args = p.parse_args()

    cmap = build_cmap(args.colors, name=args.name,
                      alpha_low=args.alpha_low,
                      alpha_high=args.alpha_high,
                      alpha_gamma=args.alpha_gamma)
    render_swatch(cmap, os.path.splitext(args.output)[0] + '_swatch.png')
    render_preview(cmap, args.input, args.output,
                   field=args.field,
                   vmin_pct=args.vmin_pct, vmax_pct=args.vmax_pct)


if __name__ == '__main__':
    main()
