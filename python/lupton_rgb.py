#!/usr/bin/env python3
"""Lupton-style asinh RGB compositor for stellar L_u,L_g,L_r maps."""
import argparse, numpy as np, h5py
from PIL import Image


def lupton_rgb(L_u, L_g, L_r, Q=8.0, stretch=None, pct=99.5):
    # Channel assignment: r = r, g = g, b = u. Scale each so that the per-image
    # high percentile sits near 1 before the asinh stretch is applied.
    chans = [L_r, L_g, L_u]
    if stretch is None:
        stretch = np.percentile(np.stack(chans), pct) + 1e-30
    r = chans[0] / stretch
    g = chans[1] / stretch
    b = chans[2] / stretch
    I = (r + g + b) / 3.0 + 1e-30
    f = np.arcsinh(Q * I) / (Q * I)
    R = np.clip(r * f, 0, 1)
    G = np.clip(g * f, 0, 1)
    B = np.clip(b * f, 0, 1)
    return np.stack([R, G, B], axis=-1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--input', required=True, help='HDF5 with per-band L_* maps')
    ap.add_argument('--output', required=True)
    ap.add_argument('--channels', default='L_r,L_g,L_u',
                    help='Comma-separated dataset names assigned to R,G,B '
                         '(Illustris g,r,i -> B,G,R means --channels L_i,L_r,L_g)')
    ap.add_argument('--Q', type=float, default=8.0)
    ap.add_argument('--pct', type=float, default=99.5)
    ap.add_argument('--stretch', type=float, default=None)
    args = ap.parse_args()

    R_key, G_key, B_key = [s.strip() for s in args.channels.split(',')]
    with h5py.File(args.input, 'r') as f:
        R_img = f[R_key][:]; G_img = f[G_key][:]; B_img = f[B_key][:]
    # lupton_rgb takes (u,g,r) with channel order (r,g,b) = (r,g,u) internally.
    # Here we pass (B,G,R) positionally so the internal assignment lines up.
    rgb = lupton_rgb(B_img, G_img, R_img, Q=args.Q, stretch=args.stretch, pct=args.pct)
    img = (rgb * 255).astype(np.uint8)
    Image.fromarray(img).save(args.output)
    print(f'Saved {args.output}  shape={img.shape}  stretch_pct={args.pct}  Q={args.Q}')


if __name__ == '__main__':
    main()
