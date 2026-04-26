#!/usr/bin/env python3
"""Plot SPH projection HDF5 files with Illustris-style colormaps.

Approximates the palette used in the public "Most Detailed Simulation of our
Universe" visualizations (green→blue→white for gas density, tan→green→brown
for metals, black→red→orange→white for velocity).
"""
import argparse
import h5py
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, LogNorm


ILL_CMAPS = {
    "density": LinearSegmentedColormap.from_list("ill_density", [
        (0.00, "#c4d88f"),  # pale yellow-green
        (0.30, "#6ea85a"),  # mid green
        (0.60, "#2a5da8"),  # dark blue
        (0.85, "#7ab7ff"),  # light blue
        (1.00, "#ffffff"),  # white cores
    ]),
    "density_dark": LinearSegmentedColormap.from_list("ill_density_dark", [
        (0.00, "#000010"),  # near-black void
        (0.25, "#0a1a4a"),  # deep blue
        (0.55, "#1e5fb0"),  # mid blue
        (0.80, "#7ac0ff"),  # light blue
        (1.00, "#ffffff"),  # white cores
    ]),
    "metals": LinearSegmentedColormap.from_list("ill_metals", [
        (0.00, "#e6d9a2"),  # cream / tan
        (0.40, "#b6c77a"),  # yellow-green
        (0.65, "#5e8a4c"),  # green
        (0.85, "#6b3a1a"),  # brown
        (1.00, "#2b1205"),  # dark brown
    ]),
    "metals_dark": LinearSegmentedColormap.from_list("ill_metals_dark", [
        (0.00, "#050208"),  # near-black, pristine gas
        (0.25, "#2a1030"),  # deep purple
        (0.50, "#8a2a20"),  # dark red-brown
        (0.75, "#e08a2a"),  # warm orange
        (0.90, "#ffd66a"),  # yellow
        (1.00, "#ffffff"),  # white, metal-rich cores
    ]),
    "velocity": LinearSegmentedColormap.from_list("ill_velocity", [
        (0.00, "#1a0500"),  # near-black
        (0.25, "#5a1a04"),  # dark red
        (0.50, "#c14a1a"),  # red-orange
        (0.80, "#ffb04a"),  # orange-yellow
        (1.00, "#ffffd8"),  # pale yellow/white
    ]),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input",  required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--field",  required=True)
    ap.add_argument("--style",  choices=list(ILL_CMAPS.keys()), required=True)
    ap.add_argument("--pmin",   type=float, default=1.0,  help="low percentile")
    ap.add_argument("--pmax",   type=float, default=99.8, help="high percentile")
    ap.add_argument("--label",  default=None)
    args = ap.parse_args()

    with h5py.File(args.input, "r") as f:
        data = f[args.field][:].astype(np.float64)

    pos = data[data > 0]
    vmin = np.percentile(pos, args.pmin)
    vmax = np.percentile(pos, args.pmax)
    data_plot = np.where(data > 0, data, np.nan)

    h_pix, w_pix = data.shape
    dpi = 100
    fig = plt.figure(figsize=(w_pix / dpi, h_pix / dpi), dpi=dpi)
    ax = fig.add_axes([0, 0, 1, 1])
    cmap = ILL_CMAPS[args.style].copy()
    # Bad/NaN pixels (zero or negative) get the cmap's low end instead of
    # black, so "no signal" blends into the background rather than punching
    # holes through the map.
    cmap.set_bad(cmap(0.0))
    im = ax.imshow(data_plot, cmap=cmap,
                   norm=LogNorm(vmin=vmin, vmax=vmax), origin="upper")
    ax.set_axis_off()
    label = args.label or args.field
    ax.text(0.02, 0.03, label, transform=ax.transAxes,
            color="white", fontsize=16, weight="bold")
    fig.savefig(args.output, dpi=dpi, facecolor="black")
    print(f"{label}: range [{vmin:.3e}, {vmax:.3e}] → {args.output}")


if __name__ == "__main__":
    main()
