#!/usr/bin/env python3
"""Composite an sph_renderer volume output (emission + transmittance) into
an RGB image with correct depth cue:

    RGB = cmap(log E) * (1 - T) + bg * T

where (1 - T) is the per-pixel opacity accumulated along the ray.
"""
import argparse
import h5py
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input",  required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--cmap",   default="inferno")
    ap.add_argument("--vmin-pct", type=float, default=1.0)
    ap.add_argument("--vmax-pct", type=float, default=99.5)
    ap.add_argument("--vmin", type=float, default=None)
    ap.add_argument("--vmax", type=float, default=None)
    ap.add_argument("--bg", type=float, nargs=3, default=[0.0, 0.0, 0.0],
                    help="background RGB in [0,1]")
    ap.add_argument("--alpha-gamma", type=float, default=1.0,
                    help="gamma<1 boosts thin regions: alpha = (1-T)**gamma")
    ap.add_argument("--alpha-norm", action="store_true",
                    help="rescale (1-T) so its 99th percentile = 1")
    args = ap.parse_args()

    with h5py.File(args.input, "r") as f:
        E = f["emission"][:].astype(np.float64)
        T = f["transmittance"][:].astype(np.float64)

    pos = E[E > 0]
    vmin = args.vmin if args.vmin is not None else np.percentile(pos, args.vmin_pct)
    vmax = args.vmax if args.vmax is not None else np.percentile(pos, args.vmax_pct)
    print(f"E: min={E.min():.3e} max={E.max():.3e} vmin={vmin:.3e} vmax={vmax:.3e}")
    print(f"T: min={T.min():.3f} max={T.max():.3f} (1-T): min={(1-T).min():.3f} max={(1-T).max():.3f}")

    # Map E -> RGBA with LogNorm + cmap.
    E_plot = np.where(E > 0, E, vmin)
    norm = LogNorm(vmin=vmin, vmax=vmax, clip=True)
    cmap = plt.get_cmap(args.cmap)
    rgba = cmap(norm(E_plot))        # (H, W, 4)
    rgb  = rgba[..., :3]

    # Composite: opacity = (1 - T), optionally gamma-boosted / rescaled.
    alpha_raw = (1.0 - T)
    if args.alpha_norm:
        p99 = np.percentile(alpha_raw, 99)
        if p99 > 0:
            alpha_raw = np.clip(alpha_raw / p99, 0.0, 1.0)
    if args.alpha_gamma != 1.0:
        alpha_raw = np.power(np.clip(alpha_raw, 0.0, 1.0), args.alpha_gamma)
    print(f"alpha: min={alpha_raw.min():.3f} mean={alpha_raw.mean():.3f} max={alpha_raw.max():.3f}")
    alpha = alpha_raw[..., None]     # (H, W, 1)
    bg = np.array(args.bg, dtype=np.float64)[None, None, :]
    out = rgb * alpha + bg * (1.0 - alpha)
    out = np.clip(out, 0.0, 1.0)

    h_pix, w_pix = E.shape
    dpi = 100
    fig = plt.figure(figsize=(w_pix / dpi, h_pix / dpi), dpi=dpi)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.imshow(out, origin="upper", interpolation="nearest")
    ax.set_axis_off()
    fig.savefig(args.output, dpi=dpi, facecolor="black")
    print(f"Saved {args.output}")


if __name__ == "__main__":
    main()
