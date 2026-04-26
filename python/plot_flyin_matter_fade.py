#!/usr/bin/env python3
"""
Post-process a batch of gas_column.h5 frames into PNGs using the
matter_r_fade cmap + alpha fade, with a SINGLE global (vmin, vmax)
computed across all frames so brightness is stable through the movie.

The renderer stores raw column densities (no clipping). All vmin/vmax
decisions happen here.
"""
import argparse, glob, os, sys, h5py, numpy as np
from multiprocessing import Pool
import matplotlib
matplotlib.use('Agg')
import matplotlib.image as mpimg
from matplotlib.colors import LogNorm, TwoSlopeNorm, LinearSegmentedColormap, to_rgb
from PIL import Image, ImageDraw, ImageFont


def matter_r_cmap():
    stops = [
        (0.00, '#2d0f3a'),
        (0.12, '#5a1552'),
        (0.28, '#932a5f'),
        (0.45, '#c94761'),
        (0.62, '#e67b56'),
        (0.78, '#f0b07a'),
        (0.90, '#f8dcb0'),
        (1.00, '#fef4df'),
    ]
    return LinearSegmentedColormap.from_list(
        'matter_r_fade', [(s, c) for s, c in stops], N=512)


def metal_gold_cmap():
    stops = [
        (0.00, '#0a1d2a'), (0.18, '#1f3a4a'), (0.38, '#7a3a1e'),
        (0.58, '#c57a1f'), (0.78, '#e8c25a'), (1.00, '#fff2c8'),
    ]
    return LinearSegmentedColormap.from_list('metal_gold', stops, N=512)


def metal_cu_cmap():
    stops = [
        (0.00, '#05070a'), (0.22, '#3c1410'), (0.45, '#8f2d14'),
        (0.65, '#d0691e'), (0.82, '#f0b065'), (1.00, '#fff6e0'),
    ]
    return LinearSegmentedColormap.from_list('metal_cu', stops, N=512)


def metal_diverge_custom_cmap():
    stops = [
        (0.00, '#000000'), (0.12, '#050b1c'), (0.28, '#0a1630'),
        (0.40, '#1f5560'), (0.48, '#f2ead2'), (0.50, '#fff4cf'),
        (0.72, '#d19040'), (1.00, '#5a1808'),
    ]
    return LinearSegmentedColormap.from_list('metal_diverge_custom', stops, N=512)


def resolve_cmap(name):
    named = {
        'matter_r_fade':        matter_r_cmap,
        'metal_gold':           metal_gold_cmap,
        'metal_cu':             metal_cu_cmap,
        'metal_diverge_custom': metal_diverge_custom_cmap,
    }
    if name in named:
        return named[name]()
    import matplotlib.pyplot as plt
    try:
        return plt.get_cmap(name)
    except (ValueError, KeyError):
        import cmocean
        base = name[:-2] if name.endswith('_r') else name
        cm = getattr(cmocean.cm, base)
        return cm.reversed() if name.endswith('_r') else cm


def collect_range(h5s, field, vmin_pct, vmax_pct, sample_stride,
                  lo_frame_q=0.5, hi_frame_q=0.5, log_space=False):
    """Per-frame percentiles, then aggregate across frames. lo/hi_frame_q
    picks *which* frame's level to use (0.5 = median; smaller hi_frame_q
    biases vmax toward dimmer frames so they're not crushed).
    If log_space, percentiles are taken on log10(positive values)."""
    lo_vals, hi_vals = [], []
    for i, h5 in enumerate(h5s[::sample_stride]):
        try:
            with h5py.File(h5, 'r') as f:
                d = f[field][:]
        except Exception as e:
            print(f'  skip {h5}: {e}', file=sys.stderr)
            continue
        pos = d[d > 0]
        if pos.size == 0:
            continue
        vals = np.log10(pos) if log_space else pos
        lo_vals.append(np.percentile(vals, vmin_pct))
        hi_vals.append(np.percentile(vals, vmax_pct))
    if not lo_vals:
        raise RuntimeError('no positive pixels in any sampled frame')
    vmin = float(np.quantile(lo_vals, lo_frame_q))
    vmax = float(np.quantile(hi_vals, hi_frame_q))
    return vmin, vmax


def _gaussian_smooth_1d(y, sigma):
    y = np.asarray(y, dtype=np.float64)
    if sigma <= 0:
        return y.copy()
    radius = int(max(1, np.ceil(4.0 * sigma)))
    t = np.arange(-radius, radius + 1, dtype=np.float64)
    k = np.exp(-0.5 * (t / sigma) ** 2)
    k /= k.sum()
    # reflect pad to avoid end drift
    pad = radius
    yp = np.concatenate([y[pad:0:-1], y, y[-2:-pad-2:-1]])
    return np.convolve(yp, k, mode='valid')


def compute_adaptive_range(h5s, field, vmin_pct, vmax_pct, sigma,
                           log_space=False, log_display=True):
    """Per-frame (vmin, vmax) smoothed across frames.

    log_display=True => LogNorm downstream; smooth percentiles in log10 space
    so geometric interpolation reads as linear on screen.
    Returns {frame_index: (vmin, vmax)} for every frame with a readable h5.
    """
    n = len(h5s)
    lo = np.full(n, np.nan)
    hi = np.full(n, np.nan)
    for i, h5 in enumerate(h5s):
        try:
            with h5py.File(h5, 'r') as f:
                d = f[field][:]
        except Exception as e:
            print(f'  skip {h5}: {e}', file=sys.stderr)
            continue
        pos = d[d > 0]
        if pos.size == 0:
            continue
        vals = np.log10(pos) if log_space else pos
        lo[i] = np.percentile(vals, vmin_pct)
        hi[i] = np.percentile(vals, vmax_pct)
        if i % 100 == 0:
            print(f'  [{i}/{n}] lo={lo[i]:.3e} hi={hi[i]:.3e}', flush=True)

    # fill NaNs from neighbours
    mask = np.isfinite(lo)
    if not mask.any():
        raise RuntimeError('no frame yielded a readable percentile')
    idx = np.arange(n)
    lo = np.interp(idx, idx[mask], lo[mask])
    hi = np.interp(idx, idx[mask], hi[mask])

    if log_display and not log_space:
        # smooth geometric mean: operate in log10 of linear values
        lo_s = 10.0 ** _gaussian_smooth_1d(np.log10(np.maximum(lo, 1e-300)), sigma)
        hi_s = 10.0 ** _gaussian_smooth_1d(np.log10(np.maximum(hi, 1e-300)), sigma)
    else:
        lo_s = _gaussian_smooth_1d(lo, sigma)
        hi_s = _gaussian_smooth_1d(hi, sigma)

    # guarantee vmin < vmax
    eps = 1e-12
    hi_s = np.maximum(hi_s, lo_s * (1.0 + 1e-6) + eps) if log_display \
           else np.maximum(hi_s, lo_s + eps)
    return {i: (float(lo_s[i]), float(hi_s[i])) for i in range(n)}


def _nice_length(target):
    # round `target` down to 1, 2, or 5 * 10^k
    if target <= 0:
        return 1.0
    k = np.floor(np.log10(target))
    base = 10.0 ** k
    for m in (5.0, 2.0, 1.0):
        if target >= m * base:
            return m * base
    return base


def _draw_scale_bar(img_arr, h5_path, units):
    # H x W x 3 float [0,1]
    H, W = img_arr.shape[:2]
    try:
        with h5py.File(h5_path, 'r') as f:
            pos = np.array(f['Header'].attrs['camera_position'])
            look = np.array(f['Header'].attrs['camera_look_at'])
            fov_v = float(f['Header'].attrs['fov'])  # vertical FOV, degrees
    except Exception as e:
        print(f'  [scale] skip {h5_path}: {e}', file=sys.stderr)
        return img_arr
    d = float(np.linalg.norm(pos - look))
    # horizontal physical extent at the look-at depth
    fov_h_rad = 2.0 * np.arctan(np.tan(np.deg2rad(fov_v) / 2.0) * (W / H))
    extent_x = 2.0 * d * np.tan(fov_h_rad / 2.0)
    # target ~25% of image width
    bar_phys = _nice_length(extent_x * 0.25)
    bar_px = int(bar_phys / extent_x * W)
    if bar_px < 20:
        return img_arr
    # convert to uint8 PIL image, draw, convert back
    pil = Image.fromarray((np.clip(img_arr, 0, 1) * 255).astype(np.uint8))
    draw = ImageDraw.Draw(pil)
    margin_x = int(0.04 * W)
    margin_y = int(0.05 * H)
    y = H - margin_y
    x0 = margin_x
    x1 = margin_x + bar_px
    thickness = max(3, H // 300)
    draw.rectangle([x0, y - thickness // 2, x1, y + thickness // 2],
                   fill=(255, 255, 255))
    # end caps
    cap = max(6, H // 150)
    draw.rectangle([x0 - 1, y - cap, x0 + 1, y + cap], fill=(255, 255, 255))
    draw.rectangle([x1 - 1, y - cap, x1 + 1, y + cap], fill=(255, 255, 255))
    # label
    if bar_phys >= 1000:
        label = f'{bar_phys/1000:g} c{units[:-3]}Mpc/h' if units == 'ckpc/h' else f'{bar_phys/1000:g} Mpc'
    else:
        label = f'{bar_phys:g} {units}'
    try:
        font = ImageFont.truetype('/usr/share/fonts/dejavu/DejaVuSans.ttf',
                                  max(14, H // 40))
    except Exception:
        font = ImageFont.load_default()
    # anchor: baseline above bar
    bbox = draw.textbbox((0, 0), label, font=font)
    tw = bbox[2] - bbox[0]
    th = bbox[3] - bbox[1]
    tx = x0 + (bar_px - tw) // 2
    ty = y - cap - th - 6
    draw.text((tx, ty), label, fill=(255, 255, 255), font=font)
    return np.asarray(pil).astype(np.float64) / 255.0


def render_one(h5, png, field, cmap, norm, alpha_floor, alpha_gamma, bg,
               scale_bar=False, scale_units='ckpc/h', log_space=False):
    with h5py.File(h5, 'r') as f:
        data = f[field][:].astype(np.float64)
    if log_space:
        # Transform positive data to log10 before applying the norm (the
        # norm's vmin/vmax/vcenter are in log-space units).
        src = np.where(data > 0, np.log10(np.maximum(data, 1e-300)), norm.vmin)
    else:
        src = np.clip(data, norm.vmin, norm.vmax)
    x = np.asarray(norm(np.clip(src, norm.vmin, norm.vmax)), dtype=np.float64)
    rgba = cmap(x)
    alpha = alpha_floor + (1.0 - alpha_floor) * np.power(x, alpha_gamma)
    alpha[data <= 0] = 0.0
    rgba[..., 3] = alpha
    flat = rgba[..., :3] * rgba[..., 3:4] + bg * (1.0 - rgba[..., 3:4])
    flat = np.clip(flat, 0, 1)
    if scale_bar:
        flat = _draw_scale_bar(flat, h5, scale_units)
    mpimg.imsave(png, flat, origin='upper')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--frame-dir', required=True,
                    help='directory containing frame_XXXX/gas_column.h5')
    ap.add_argument('--png-dir', required=True)
    ap.add_argument('--frames', type=int, required=True)
    ap.add_argument('--field', default='gas_column_density')
    ap.add_argument('--cmap', default='matter_r_fade',
                    help='matter_r_fade, metal_gold, metal_cu, '
                         'metal_diverge_custom, or any matplotlib cmap name')
    ap.add_argument('--norm', default='log', choices=['log', 'divlog'],
                    help='log = LogNorm in linear data units (default); '
                         'divlog = TwoSlopeNorm on log10(data), requires '
                         '--log-pivot')
    ap.add_argument('--log-pivot', type=float, default=None,
                    help='log10(field) value at the neutral center of the '
                         'diverging cmap (required when --norm=divlog)')
    ap.add_argument('--h5-name', default='gas_column.h5',
                    help='basename of per-frame HDF5 file (gas_column.h5, '
                         'dm_column.h5, temperature.h5, metallicity.h5)')
    ap.add_argument('--vmin-pct', type=float, default=50.0)
    ap.add_argument('--vmax-pct', type=float, default=99.7)
    ap.add_argument('--vmin', type=float, default=None)
    ap.add_argument('--vmax', type=float, default=None)
    ap.add_argument('--sample-stride', type=int, default=8,
                    help='subsample frames when scanning for global vmin/vmax')
    ap.add_argument('--vmax-frame-q', type=float, default=0.5,
                    help='quantile across per-frame vmax values (smaller = '
                         'favor dimmer frames so early wide views are brighter)')
    ap.add_argument('--vmin-frame-q', type=float, default=0.5)
    ap.add_argument('--adaptive', action='store_true',
                    help='per-frame vmin/vmax from that frame alone, '
                         'Gaussian-smoothed across frames in log space')
    ap.add_argument('--smooth-sigma', type=float, default=30.0,
                    help='Gaussian sigma (in frames) for adaptive smoothing')
    ap.add_argument('--workers', type=int, default=16,
                    help='parallel render processes')
    ap.add_argument('--alpha-floor', type=float, default=0.0)
    ap.add_argument('--alpha-gamma', type=float, default=0.7)
    ap.add_argument('--bg', default='black')
    ap.add_argument('--scale-bar', action='store_true',
                    help='overlay a physical-scale bar in bottom-left using '
                         'camera metadata from each h5 header')
    ap.add_argument('--scale-units', default='ckpc/h')
    args = ap.parse_args()

    os.makedirs(args.png_dir, exist_ok=True)
    h5s = [f'{args.frame_dir}/frame_{i:04d}/{args.h5_name}' for i in range(args.frames)]
    h5s = [p for p in h5s if os.path.exists(p)]
    if not h5s:
        print(f'no frames found under {args.frame_dir}', file=sys.stderr)
        sys.exit(1)

    log_space = (args.norm == 'divlog')
    if args.norm == 'divlog' and args.log_pivot is None:
        print('--norm=divlog requires --log-pivot', file=sys.stderr)
        sys.exit(1)

    per_frame_range = None  # dict: frame_index -> (vmin, vmax)
    if args.adaptive and (args.vmin is None or args.vmax is None):
        print(f'[scan-adaptive] {len(h5s)} frames, sigma={args.smooth_sigma} ...',
              flush=True)
        per_frame_range = compute_adaptive_range(
            h5s, args.field, args.vmin_pct, args.vmax_pct,
            sigma=args.smooth_sigma, log_space=log_space,
            log_display=(not log_space))
        vmin = vmax = None  # unused globally
    elif args.vmin is None or args.vmax is None:
        print(f'[scan] {len(h5s)} frames, stride={args.sample_stride} ...', flush=True)
        vmin, vmax = collect_range(h5s, args.field,
                                   args.vmin_pct, args.vmax_pct,
                                   args.sample_stride,
                                   lo_frame_q=args.vmin_frame_q,
                                   hi_frame_q=args.vmax_frame_q,
                                   log_space=log_space)
        if args.vmin is not None: vmin = args.vmin
        if args.vmax is not None: vmax = args.vmax
    else:
        vmin, vmax = args.vmin, args.vmax
    if per_frame_range is not None:
        f0, fN = min(per_frame_range), max(per_frame_range)
        v0, v1 = per_frame_range[f0], per_frame_range[fN]
        print(f'[norm] adaptive: frame {f0} vmin={v0[0]:.3e} vmax={v0[1]:.3e} '
              f'-> frame {fN} vmin={v1[0]:.3e} vmax={v1[1]:.3e}', flush=True)
    elif log_space:
        print(f'[norm] divlog: vmin={vmin:.3f} pivot={args.log_pivot:.3f} '
              f'vmax={vmax:.3f} (all log10)', flush=True)
    else:
        print(f'[norm] log: vmin={vmin:.3e} vmax={vmax:.3e}', flush=True)

    cmap = resolve_cmap(args.cmap)
    bg = np.array(to_rgb(args.bg))

    tasks = []
    for i in range(args.frames):
        h5 = f'{args.frame_dir}/frame_{i:04d}/{args.h5_name}'
        png = f'{args.png_dir}/frame_{i:04d}.png'
        if not os.path.exists(h5):
            continue
        if per_frame_range is not None:
            vmin_i, vmax_i = per_frame_range[i]
        else:
            vmin_i, vmax_i = vmin, vmax
        tasks.append((i, h5, png, args.field, vmin_i, vmax_i, args.alpha_floor,
                      args.alpha_gamma, args.bg, args.scale_bar,
                      args.scale_units, args.cmap, args.norm, args.log_pivot))
    print(f'[render] {len(tasks)} frames with {args.workers} workers', flush=True)
    if args.workers > 1:
        with Pool(args.workers) as pool:
            for n, (i, _) in enumerate(pool.imap_unordered(_worker, tasks, chunksize=4)):
                if n % 50 == 0:
                    print(f'  [{n}/{len(tasks)}]', flush=True)
    else:
        for t in tasks:
            _worker(t)
    print('done')


def _worker(args):
    (i, h5, png, field, vmin, vmax, alpha_floor, alpha_gamma, bg_name,
     scale_bar, scale_units, cmap_name, norm_kind, log_pivot) = args
    cmap = resolve_cmap(cmap_name)
    if norm_kind == 'divlog':
        # vmin/vmax/log_pivot are all in log10 space
        pivot = log_pivot
        if not (vmin < pivot < vmax):
            pivot = 0.5 * (vmin + vmax)
        norm = TwoSlopeNorm(vmin=vmin, vcenter=pivot, vmax=vmax)
    else:
        norm = LogNorm(vmin=vmin, vmax=vmax, clip=True)
    bg = np.array(to_rgb(bg_name))
    render_one(h5, png, field, cmap, norm, alpha_floor, alpha_gamma, bg,
               scale_bar=scale_bar, scale_units=scale_units,
               log_space=(norm_kind == 'divlog'))
    return (i, png)


if __name__ == '__main__':
    main()
