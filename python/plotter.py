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
from matplotlib.colors import LogNorm

def main():
    parser = argparse.ArgumentParser(description='Plot 2D projection maps')
    parser.add_argument('--input', required=True, help='Input HDF5 projection file')
    parser.add_argument('--output', required=True, help='Output PNG file')
    parser.add_argument('--field', default='gas_density', help='Field name to plot')
    parser.add_argument('--cmap', default='inferno', help='Colormap name')
    parser.add_argument('--log', action='store_true', default=True, help='Use log scale')
    args = parser.parse_args()

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

    fig, ax = plt.subplots(1, 1, figsize=(10, 10))

    # Replace zeros with NaN so they render transparent / use background color
    if args.log:
        data_plot = data.astype(np.float64)
        positive = data_plot[data_plot > 0]
        vmin = np.percentile(positive, 1)
        vmax = np.percentile(positive, 99.5)
        data_plot[data_plot <= 0] = np.nan
        cmap = plt.get_cmap(args.cmap).copy()
        cmap.set_bad('black')
        im = ax.imshow(data_plot, cmap=cmap, norm=LogNorm(vmin=vmin, vmax=vmax),
                       origin='upper')
    else:
        im = ax.imshow(data, cmap=args.cmap, origin='upper')

    cbar = plt.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label(f'{args.field} (column density)', fontsize=12)

    # Add info
    title = args.field
    if 'camera_type' in header:
        title += f" ({header['camera_type']})"
    ax.set_title(title, fontsize=14)
    ax.set_xlabel('pixels')
    ax.set_ylabel('pixels')

    plt.tight_layout()
    plt.savefig(args.output, dpi=150, bbox_inches='tight')
    print(f"Saved figure to {args.output}")

if __name__ == '__main__':
    main()
