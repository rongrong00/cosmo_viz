#!/usr/bin/env python3
"""Plot gas speed |<v>| from a 3-channel sph_renderer velocity HDF5.

Reads mass-weighted (vx, vy, vz) planes and renders their vector magnitude —
i.e. the speed of the coherent bulk motion, not mass-weighted speed.
"""
import argparse
import h5py
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--log", action="store_true")
    ap.add_argument("--vmin", type=float, default=None)
    ap.add_argument("--vmax", type=float, default=None)
    args = ap.parse_args()

    with h5py.File(args.input, "r") as f:
        vx = f["gas_vx_mw"][:]
        vy = f["gas_vy_mw"][:]
        vz = f["gas_vz_mw"][:]
    vmag = np.sqrt(vx * vx + vy * vy + vz * vz)

    h_pix, w_pix = vmag.shape
    fig_h = 10.0
    fig_w = fig_h * (w_pix / h_pix)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    if args.log:
        data = np.log10(np.maximum(vmag, 1e-6))
        label = r"log$_{10}$ |v| [km/s]"
    else:
        data = vmag
        label = r"|v| [km/s]"
    im = ax.imshow(data, origin="lower", cmap="cividis",
                   vmin=args.vmin, vmax=args.vmax)
    ax.set_axis_off()
    cb = plt.colorbar(im, ax=ax, fraction=0.035, pad=0.02)
    cb.set_label(label)
    fig.tight_layout()
    fig.savefig(args.output, dpi=150, bbox_inches="tight")
    print(f"min={vmag.min():.3e}  max={vmag.max():.3e}  mean={vmag.mean():.3e}")
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
