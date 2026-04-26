#!/usr/bin/env python3
"""
Minimal plotter: reads a 2D projection HDF5 file and produces a PNG image.

Usage:
    python plotter.py --input projection_file.hdf5 --output figure.png [--field gas_density]
"""

import argparse
import numpy as np
import h5py
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm, LinearSegmentedColormap
try:
    import cmocean  # registers cmo.* colormaps
except ImportError:
    pass


def _register_custom_cmaps():
    # 'cosmic_layers': dark→indigo→violet→magenta→orange→gold→cream→white.
    # Each stop sits at a distinct hue so successive decades of log column
    # density read as visually separable bands — good for seeing structure
    # from diffuse outskirts all the way into a saturated halo core.
    # Smooth monotonic ramp in both brightness and hue. Low-density filaments
    # stay visible (not buried in black), midrange climbs through purple →
    # blue → teal → green → yellow → orange, and only the densest ~5% tips
    # over into red — a narrow peak rather than a saturated core.
    # Low half (0 → 0.30) holds the full cool-hue rainbow so filaments show
    # rich color variety; upper half (0.30 → 1.0) is a smooth warm ramp for
    # the halo core (green → yellow → orange → coral → salmon). Use with
    # --vmax-pct 100 so the core isn't clipped.
    stops = [
        (0.00, '#000008'),  # black (voids)
        (0.02, '#130733'),  # dark purple
        (0.05, '#241460'),  # indigo
        (0.08, '#2c2690'),  # blue-violet
        (0.11, '#2352b5'),  # deep blue
        (0.14, '#1cb8ae'),  # teal (brief)
        (0.18, '#3ecc56'),  # green
        (0.25, '#b8dc33'),  # yellow-green
        (0.38, '#f0cb1f'),  # yellow
        (0.55, '#f08a20'),  # orange (wide band)
        (0.75, '#f55850'),  # warm coral (wide band)
        (1.00, '#ff9480'),  # light salmon (peak)
    ]
    cmap = LinearSegmentedColormap.from_list(
        'cosmic_layers', [(s, c) for s, c in stops], N=512)

    # 'aurora': same design philosophy as inferno (monotonic luminance ramp,
    # one smooth hue rotation, no backtracking), but takes a cool-to-warm
    # path instead of inferno's purple→red→yellow warm-only path.
    # Stops are ~evenly spaced so perceptual brightness increases at a near-
    # constant rate, and hue rotates once from deep blue → teal → green →
    # yellow-green → cream. Distinct from viridis because the endpoints are
    # brighter and warmer; distinct from inferno because it stays cool through
    # the midtones.
    aurora_stops = [
        (0.000, '#03000a'),  # near-black
        (0.12,  '#1a1055'),  # deep indigo-blue
        (0.24,  '#1d3a92'),  # royal blue
        (0.36,  '#1e7ea6'),  # ocean blue
        (0.48,  '#1fb596'),  # teal-green
        (0.62,  '#72dc5a'),  # fresh green (core bulk starts reading)
        (0.78,  '#e5e860'),  # lime yellow
        (0.90,  '#fbf5d5'),  # pale cream
        (1.000, '#ffffff'),  # white (narrow peak tip)
    ]
    aurora = LinearSegmentedColormap.from_list(
        'aurora', [(s, c) for s, c in aurora_stops], N=512)

    try:
        matplotlib.colormaps.register(cmap=cmap, name='cosmic_layers')
        matplotlib.colormaps.register(cmap=cmap.reversed(), name='cosmic_layers_r')
        matplotlib.colormaps.register(cmap=aurora, name='aurora')
        matplotlib.colormaps.register(cmap=aurora.reversed(), name='aurora_r')
    except ValueError:
        pass  # already registered


_register_custom_cmaps()


def main():
    parser = argparse.ArgumentParser(description='Plot 2D projection maps')
    parser.add_argument('--input', required=True, help='Input HDF5 projection file')
    parser.add_argument('--output', required=True, help='Output PNG file')
    parser.add_argument('--field', default='gas_density', help='Field name to plot')
    parser.add_argument('--cmap', default=None, help='Colormap name (auto from field if omitted)')
    parser.add_argument('--log', action='store_true', default=True, help='Use log scale')
    parser.add_argument('--vmin', type=float, default=None,
                        help='Explicit vmin (overrides percentile)')
    parser.add_argument('--vmax', type=float, default=None,
                        help='Explicit vmax (overrides percentile)')
    parser.add_argument('--vmin-pct', type=float, default=1.0,
                        help='Percentile for vmin when --vmin not set')
    parser.add_argument('--vmax-pct', type=float, default=99.5,
                        help='Percentile for vmax when --vmax not set')
    parser.add_argument('--bare', action='store_true',
                        help='Full-bleed image only: no colorbar, title, or axes')
    args = parser.parse_args()

    # Per-field-type default colormap. Density stays on inferno; other
    # quantity types get distinct palettes so they're visually separable.
    if args.cmap is None:
        f = args.field.lower()
        if 'temperature' in f:
            args.cmap = 'cmo.thermal'
        elif 'metallicity' in f or 'metal' in f:
            args.cmap = 'viridis'
        elif 'vlos' in f or 'los_velocity' in f:
            args.cmap = 'RdBu_r'
        elif 'velocity' in f or f.startswith('gas_v') or 'vmag' in f:
            args.cmap = 'cividis'
        else:
            args.cmap = 'inferno'

    with h5py.File(args.input, 'r') as f:
        data = f[args.field][:]
        # Read header info if available
        header = {}
        if 'Header' in f:
            for key in f['Header'].attrs:
                val = f['Header'].attrs[key]
                if isinstance(val, bytes):
                    val = val.decode('utf-8')
                header[key] = val

    print(f"Data shape: {data.shape}")
    print(f"Min: {data.min():.6e}, Max: {data.max():.6e}")
    print(f"Non-zero fraction: {np.count_nonzero(data) / data.size:.4f}")

    # Size the figure to the data aspect so a 16:9 image isn't letterboxed
    # inside a square canvas (which otherwise compresses the colorbar).
    h_pix, w_pix = data.shape
    if args.bare:
        # 1 pixel == 1 output pixel. dpi=100 + figsize=(w/100, h/100) → exact.
        save_dpi = 100
        fig, ax = plt.subplots(1, 1, figsize=(w_pix / save_dpi, h_pix / save_dpi),
                               dpi=save_dpi)
    else:
        save_dpi = 150
        fig_h = 10.0
        fig_w = fig_h * (w_pix / h_pix)
        fig, ax = plt.subplots(1, 1, figsize=(fig_w, fig_h))

    # LOS velocity is signed — needs a symmetric diverging scale, not log.
    is_diverging = 'vlos' in args.field.lower() or 'los_velocity' in args.field.lower()
    if is_diverging:
        vmax = np.percentile(np.abs(data), 99)
        im = ax.imshow(data, cmap=args.cmap, vmin=-vmax, vmax=vmax, origin='upper')
    elif args.log:
        data_plot = data.astype(np.float64)
        positive = data_plot[data_plot > 0]
        vmin = args.vmin if args.vmin is not None else np.percentile(positive, args.vmin_pct)
        vmax = args.vmax if args.vmax is not None else np.percentile(positive, args.vmax_pct)
        data_plot[data_plot <= 0] = np.nan
        cmap = plt.get_cmap(args.cmap).copy()
        cmap.set_bad('black')
        im = ax.imshow(data_plot, cmap=cmap, norm=LogNorm(vmin=vmin, vmax=vmax),
                       origin='upper')
    else:
        im = ax.imshow(data, cmap=args.cmap, origin='upper')

    if args.bare:
        # Full-bleed: image fills the canvas, no colorbar/axes/title.
        ax.set_position([0.0, 0.0, 1.0, 1.0])
        ax.set_axis_off()
        fig.patch.set_facecolor('black')
        plt.savefig(args.output, dpi=save_dpi, facecolor='black', pad_inches=0)
    else:
        cbar = plt.colorbar(im, ax=ax, shrink=0.8)
        cbar.set_label(f'{args.field} (column density)', fontsize=12)

        title = args.field
        if 'camera_type' in header:
            title += f" ({header['camera_type']})"
        ax.set_title(title, fontsize=14)
        ax.set_xlabel('pixels')
        ax.set_ylabel('pixels')

        plt.tight_layout()
        plt.savefig(args.output, dpi=save_dpi, bbox_inches='tight')
    print(f"Saved figure to {args.output}")

if __name__ == '__main__':
    main()
