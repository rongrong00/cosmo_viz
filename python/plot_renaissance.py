#!/usr/bin/env python3
"""Composite a Renaissance-style layered volume render from one or more
(emission, transmittance) HDF5 outputs produced by traceGasEmission.

For a single layer: emission→colormap, alpha = 1 − T. Shown over black.
For multi-layer: front-to-back screen composite of the supplied layers in
the order given (same alpha compositing on RGBA).
"""
import argparse
import h5py
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap


# Renaissance-inspired palettes. Values near 0 are black (invisible);
# brightness ramps up toward 1.
CMAPS = {
    "neutral_H":      LinearSegmentedColormap.from_list("neutral_H", [
        "#000020", "#1a3a8a", "#5f7fd0", "#c0e0ff"]),
    "ionized_H":      LinearSegmentedColormap.from_list("ionized_H", [
        "#0a0000", "#802010", "#d05020", "#ffd090"]),
    "heated_gas":     LinearSegmentedColormap.from_list("heated_gas", [
        "#050005", "#2a1040", "#c05020", "#fff0b0"]),
    "heavy_elements": LinearSegmentedColormap.from_list("heavy_elements", [
        "#000a08", "#106050", "#70d0b0", "#e0ffe8"]),
}


def layer_rgba(path, style, pmin, pmax, alpha_gain):
    with h5py.File(path, "r") as f:
        E = f["emission"][:].astype(np.float64)
        T = f["transmittance"][:].astype(np.float64)
    # Log-scale the emission for colormap indexing; empty pixels stay dark.
    pos = E[E > 0]
    if pos.size == 0:
        norm = np.zeros_like(E)
    else:
        lo = np.log10(np.percentile(pos, pmin))
        hi = np.log10(np.percentile(pos, pmax))
        norm = (np.log10(np.maximum(E, 1e-300)) - lo) / max(hi - lo, 1e-6)
        norm = np.clip(norm, 0.0, 1.0)
    rgb = CMAPS[style](norm)[..., :3]
    # Translucent look: use log-normalized emission as the alpha ramp but
    # gamma-compress it so faint gas stays wispy instead of solid. The
    # (1−T) physical depth cue is added at reduced weight for the same
    # reason — we want diffuse structure to show through, not mask behind.
    alpha = np.clip(alpha_gain * (norm ** 1.8) + 0.3 * (1.0 - T), 0.0, 1.0)
    rgba = np.dstack([rgb, alpha])
    return rgba, E, T


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--layer", action="append", required=True,
                    help="repeat: --layer path:style[:alpha_gain]")
    ap.add_argument("--output", required=True)
    ap.add_argument("--pmin", type=float, default=2.0)
    ap.add_argument("--pmax", type=float, default=99.8)
    args = ap.parse_args()

    layers = []
    for spec in args.layer:
        parts = spec.split(":")
        path = parts[0]; style = parts[1]
        gain = float(parts[2]) if len(parts) > 2 else 1.0
        layers.append((path, style, gain))

    first_rgba, _, _ = layer_rgba(layers[0][0], layers[0][1], args.pmin, args.pmax, layers[0][2])
    H, W, _ = first_rgba.shape
    out = np.zeros((H, W, 3), dtype=np.float64)
    A   = np.zeros((H, W),    dtype=np.float64)

    def over(base_rgb, base_a, src):
        s_rgb = src[..., :3]
        s_a   = src[..., 3]
        new_a = s_a + base_a * (1.0 - s_a)
        new_rgb_num = s_rgb * s_a[..., None] + base_rgb * base_a[..., None] * (1.0 - s_a[..., None])
        safe = np.where(new_a > 1e-6, new_a, 1.0)
        new_rgb = new_rgb_num / safe[..., None]
        return new_rgb, new_a

    # Composite back-to-front so first --layer is behind; reverse iterate.
    for path, style, gain in reversed(layers):
        rgba, _, _ = layer_rgba(path, style, args.pmin, args.pmax, gain)
        out, A = over(out, A, rgba)
        print(f"  composited {path} ({style}, gain={gain})")

    # Final: put over black background.
    final = out * A[..., None]

    fig_h = 10.0
    fig_w = fig_h * (W / H)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    ax.imshow(np.clip(final, 0, 1), origin="upper")
    ax.set_axis_off()
    fig.tight_layout(pad=0)
    fig.savefig(args.output, dpi=150, bbox_inches="tight", facecolor="black")
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
