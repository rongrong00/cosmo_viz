#!/usr/bin/env python3
"""
Stellar-light flyin: load PartType4 once, step the camera log-linearly from
r_outer -> r_inner along +x of the halo center, render each frame, and write
per-frame L_u/L_g/L_r (or L_g/L_r/L_i) maps to PNG via Lupton asinh.

Usage:
  python stellar_flyin.py --snapshot snap.h5 --fsps-grid grid.h5 \
      --center cx cy cz --r-outer 16915 --r-inner 40 --frames 240 \
      --out-dir output/flyin --clip-center cx cy cz --clip-radius 20 --clip-dim 0.1
"""
import argparse, os, time, numpy as np, h5py
from scipy.integrate import cumulative_trapezoid
from scipy.spatial import cKDTree
from PIL import Image


Z_SUN = 0.0127
MASS_UNIT_MSUN_PER_H = 1.0e10
H0_KM_S_MPC = 67.74
OM = 0.3089
ODE = 0.6911
HUBBLE_TIME_GYR = 977.79 / H0_KM_S_MPC


def age_table():
    a = np.logspace(-5, 0, 20000)
    integrand = 1.0 / (a * np.sqrt(OM / a ** 3 + ODE))
    age = np.concatenate(([0.0], cumulative_trapezoid(integrand, a))) * HUBBLE_TIME_GYR
    return a, age


def interp_fsps(log_age, logZsol, grid, band_keys):
    La = grid['log_age_gyr']; Lz = grid['logZsol']
    la = np.clip(log_age, La[0], La[-1]); lz = np.clip(logZsol, Lz[0], Lz[-1])
    ia = np.clip(np.searchsorted(La, la) - 1, 0, len(La) - 2)
    iz = np.clip(np.searchsorted(Lz, lz) - 1, 0, len(Lz) - 2)
    fa = (la - La[ia]) / (La[ia + 1] - La[ia])
    fz = (lz - Lz[iz]) / (Lz[iz + 1] - Lz[iz])
    out = {}
    for b in band_keys:
        G = grid[b]
        v00 = G[ia, iz]; v01 = G[ia, iz + 1]
        v10 = G[ia + 1, iz]; v11 = G[ia + 1, iz + 1]
        out[b] = ((1 - fa) * (1 - fz) * v00 + (1 - fa) * fz * v01 +
                  fa * (1 - fz) * v10 + fa * fz * v11)
    return out


def lupton_rgb(R_img, G_img, B_img, Q, stretch):
    r = R_img / stretch; g = G_img / stretch; b = B_img / stretch
    I = (r + g + b) / 3.0 + 1e-30
    f = np.arcsinh(Q * I) / (Q * I)
    R = np.clip(r * f, 0, 1); G = np.clip(g * f, 0, 1); B = np.clip(b * f, 0, 1)
    return (np.stack([R, G, B], axis=-1) * 255).astype(np.uint8)


def render_frame(coords, L_per_star, cam_pos, look_at, fov_deg, W, H,
                 sigma_pix, rad_mult, band_keys):
    """Splat each star as a fixed-pixel-size Gaussian stamp."""
    fwd = look_at - cam_pos; fwd /= np.linalg.norm(fwd)
    up_w = np.array([0.0, 0.0, 1.0])
    right = np.cross(fwd, up_w); right /= np.linalg.norm(right)
    up = np.cross(right, fwd)
    tan_x = np.tan(np.deg2rad(fov_deg) * 0.5)
    tan_y = tan_x * (H / W)
    d = coords - cam_pos
    xc = d @ right; yc = d @ up; zc = d @ fwd
    in_front = zc > 1.0
    ndc_x = xc / (zc * tan_x + 1e-30)
    ndc_y = yc / (zc * tan_y + 1e-30)
    px = (ndc_x * 0.5 + 0.5) * W
    py = (-ndc_y * 0.5 + 0.5) * H
    inf = in_front & (px >= -5) & (px < W + 5) & (py >= -5) & (py < H + 5)
    sel = np.where(inf)[0]
    imgs = {k: np.zeros((H, W), dtype=np.float32) for k in band_keys}
    if sel.size == 0:
        return imgs

    rad = max(1, int(np.ceil(rad_mult * sigma_pix)))
    xs = np.arange(-rad, rad + 1)
    XX, YY = np.meshgrid(xs, xs)
    stamp = np.exp(-(XX * XX + YY * YY) / (2.0 * sigma_pix * sigma_pix))
    stamp /= stamp.sum()
    px_i = np.round(px[sel]).astype(np.int32)
    py_i = np.round(py[sel]).astype(np.int32)
    L_sel = {k: L_per_star[k][sel].astype(np.float32) for k in band_keys}
    for dy in range(-rad, rad + 1):
        for dx in range(-rad, rad + 1):
            w = float(stamp[dy + rad, dx + rad])
            if w <= 0:
                continue
            ty = py_i + dy; tx = px_i + dx
            ok = (ty >= 0) & (ty < H) & (tx >= 0) & (tx < W)
            if not ok.any():
                continue
            ti = ty[ok] * W + tx[ok]
            for k in band_keys:
                np.add.at(imgs[k].ravel(), ti, w * L_sel[k][ok])
    return imgs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--snapshot', required=True)
    ap.add_argument('--fsps-grid', required=True)
    ap.add_argument('--center', type=float, nargs=3, required=True,
                    help='Halo center (ckpc/h) — also the look_at target.')
    ap.add_argument('--r-outer', type=float, default=16915.0)
    ap.add_argument('--r-inner', type=float, default=40.0)
    ap.add_argument('--frames', type=int, default=240)
    ap.add_argument('--fov', type=float, default=60.0)
    ap.add_argument('--image-width', type=int, default=1920)
    ap.add_argument('--image-height', type=int, default=1080)
    ap.add_argument('--out-dir', required=True)
    ap.add_argument('--channels', default='L_i,L_r,L_g',
                    help='R,G,B dataset keys (default is Illustris g,r,i -> B,G,R)')
    ap.add_argument('--Q', type=float, default=8.0)
    ap.add_argument('--stretch-pct', type=float, default=99.5)
    ap.add_argument('--splat-sigma-pix', type=float, default=0.8,
                    help='Fixed Gaussian sigma in pixels (applied to every star).')
    ap.add_argument('--splat-radius-mult', type=float, default=3.0,
                    help='Stamp footprint = radius_mult * sigma (pixels).')
    ap.add_argument('--brightness-exponent', type=float, default=0.0,
                    help='Scale per-star L by (d_ref/max(d_star,d_floor))^alpha '
                         'where d_ref=r_outer. alpha=2 is physical inverse-square '
                         '(stars brighten as camera approaches); 0 disables.')
    ap.add_argument('--brightness-d-floor', type=float, default=200.0,
                    help='Minimum star-to-camera distance (ckpc/h) used in '
                         'the brightness gain, prevents blow-up for stars '
                         'that pass near the camera.')
    ap.add_argument('--stretch-gain-weight', type=float, default=0.0,
                    help='In [0,1]. 0 = stretch from natural-L last frame '
                         '(far OK, close saturates). 1 = stretch from '
                         'boosted last frame (close OK, far crushed). '
                         'Intermediate values balance both: stretch = '
                         'stretch_natural * (stretch_boosted/stretch_natural)^w.')
    ap.add_argument('--zoom-end-frame', type=int, default=None,
                    help='Frame index at which the zoom reaches --r-inner. '
                         'Later frames dwell at r_inner with spiral frozen. '
                         'Default = frames-1.')
    ap.add_argument('--h', type=float, default=0.6774)
    ap.add_argument('--clip-center', type=float, nargs=3, default=None)
    ap.add_argument('--clip-radius', type=float, default=0.0,
                    help='Constant clip radius if --clip-radius-start/-end not given.')
    ap.add_argument('--clip-radius-start', type=float, default=None,
                    help='Clip radius at first frame (log-interp to --clip-radius-end).')
    ap.add_argument('--clip-radius-end', type=float, default=None,
                    help='Clip radius at last frame.')
    ap.add_argument('--clip-dim', type=float, default=0.0)
    ap.add_argument('--axis', type=str, default='x',
                    help='Approach axis from halo center (x, y, or z; linear mode only)')
    ap.add_argument('--mode', type=str, default='linear', choices=['linear', 'spiral'])
    ap.add_argument('--turns', type=float, default=2.0,
                    help='Number of full revolutions in spiral mode')
    ap.add_argument('--theta-start', type=float, default=0.0,
                    help='Starting azimuth in radians for spiral mode (default 0).')
    ap.add_argument('--spiral-plane', type=str, default='xy', choices=['xy','xz','yz'],
                    help='Plane in which the spiral orbits (up axis stays +z)')
    ap.add_argument('--world-rank', type=int,
                    default=int(os.environ.get('SLURM_PROCID', '0')))
    ap.add_argument('--world-size', type=int,
                    default=int(os.environ.get('SLURM_NTASKS', '1')))
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    # Load stars once
    t0 = time.time()
    with h5py.File(args.snapshot, 'r') as f:
        z_snap = float(f['Header'].attrs['Redshift'])
        a_snap = 1.0 / (1.0 + z_snap)
        pt4 = f['PartType4']
        coords = pt4['Coordinates'][:]
        a_form = pt4['GFM_StellarFormationTime'][:].astype(np.float64)
        Z = pt4['GFM_Metallicity'][:].astype(np.float64)
        m_init = pt4['GFM_InitialMass'][:].astype(np.float64)
    keep = (a_form > 0) & (a_form < a_snap + 1e-6) & (Z > 0) & (m_init > 0)
    coords = coords[keep]; a_form = a_form[keep]; Z = Z[keep]; m_init = m_init[keep]
    print(f'Loaded {coords.shape[0]} valid stars in {time.time()-t0:.1f}s. '
          f'z_snap={z_snap:.4f}', flush=True)

    # Soft clip mask: compute d^2 once. Per-frame radius is resolved later.
    d2_clip = None
    if args.clip_center is not None and (args.clip_radius > 0 or
                                         args.clip_radius_start is not None):
        c_clip = np.array(args.clip_center, dtype=np.float64)
        d2_clip = np.sum((coords - c_clip) ** 2, axis=1)

    # Ages
    A, AGE = age_table()
    age_form = np.interp(np.clip(a_form, A[0], A[-1]), A, AGE)
    age_snap = float(np.interp(a_snap, A, AGE))
    age_gyr = np.maximum(age_snap - age_form, 1e-5)
    log_age = np.log10(age_gyr)
    logZsol = np.log10(Z / Z_SUN)

    # FSPS lookup
    with h5py.File(args.fsps_grid, 'r') as f:
        grid = {'log_age_gyr': f['log_age_gyr'][:], 'logZsol': f['logZsol'][:]}
        band_keys = []
        for k in f.keys():
            if k.startswith('L_'):
                grid[k] = f[k][:]; band_keys.append(k)
    L = interp_fsps(log_age, logZsol, grid, band_keys)
    m_msun = m_init * MASS_UNIT_MSUN_PER_H / args.h
    # Keep a base (unmasked) L per band; mask is applied per frame below.
    L_per_star = {k: (L[k] * m_msun).astype(np.float32) for k in band_keys}
    print(f'FSPS lookup done in {time.time()-t0:.1f}s total', flush=True)

    # Camera path: log-linear zoom that reaches r_inner at frame `reach_frame`
    # and CONTINUES past it at the same rate for the remaining frames (no
    # dwell). Spiral angle follows the same schedule.
    center = np.array(args.center, dtype=np.float64)
    reach = (args.zoom_end_frame if args.zoom_end_frame is not None
             else args.frames - 1)
    reach = min(max(reach, 1), args.frames - 1)
    idx = np.arange(args.frames)
    rs = args.r_outer * (args.r_inner / args.r_outer) ** (idx / reach)
    thetas = args.theta_start + args.turns * 2.0 * np.pi * (idx / reach)
    if args.world_rank == 0:
        print(f'Reach r_inner={args.r_inner} at frame {reach}/{args.frames-1}; '
              f'final r={rs[-1]:.2f}  final theta={thetas[-1]:.3f} rad',
              flush=True)

    def cam_pos_for(i):
        r = rs[i]
        if args.mode == 'linear':
            axis_map = {'x': np.array([1.0, 0.0, 0.0]),
                        'y': np.array([0.0, 1.0, 0.0]),
                        'z': np.array([0.0, 0.0, 1.0])}
            return center + axis_map[args.axis.lower()] * r
        # spiral: angle comes from the precomputed schedule.
        theta = thetas[i]
        c, s = np.cos(theta), np.sin(theta)
        if args.spiral_plane == 'xy':
            off = np.array([c, s, 0.0])
        elif args.spiral_plane == 'xz':
            off = np.array([c, 0.0, s])
        else:
            off = np.array([0.0, c, s])
        return center + off * r

    # Per-frame clip radius schedule (log-interp, constant, or disabled).
    if args.clip_radius_start is not None and args.clip_radius_end is not None:
        clip_rs = args.clip_radius_start * (
            args.clip_radius_end / args.clip_radius_start) ** (idx / reach)
        if args.world_rank == 0:
            print(f'Clip radius: {clip_rs[0]:.1f} -> '
                  f'{clip_rs[min(reach, len(clip_rs)-1)]:.1f} ckpc/h at frame {reach}, '
                  f'final {clip_rs[-1]:.1f}', flush=True)
    elif args.clip_radius > 0:
        clip_rs = np.full(args.frames, args.clip_radius)
    else:
        clip_rs = None

    # Channel assignment and global stretch: compute stretch from the innermost
    # frame alone (it has highest peak) so later frames don't over-saturate.
    R_key, G_key, B_key = [s.strip() for s in args.channels.split(',')]

    def brightness_gain(i):
        """Per-star gain = (r_outer / max(d_star, d_floor))^alpha. Each
        star brightens by its own distance to the camera; the floor
        prevents blow-up when a star passes close to the camera."""
        if args.brightness_exponent == 0.0:
            return None
        cam = cam_pos_for(i)
        d2 = np.sum((coords - cam) ** 2, axis=1).astype(np.float32)
        d_floor2 = float(args.brightness_d_floor) ** 2
        d2_eff = np.maximum(d2, d_floor2)
        d_ref = float(args.r_outer)
        return np.power(d_ref * d_ref / d2_eff,
                        0.5 * args.brightness_exponent).astype(np.float32)

    def L_for_frame(i, apply_gain=True):
        gain = brightness_gain(i) if apply_gain else None
        if clip_rs is None or d2_clip is None:
            if gain is None:
                return L_per_star
            return {k: (L_per_star[k] * gain).astype(np.float32) for k in band_keys}
        r_clip = clip_rs[i]
        if args.clip_dim <= 0:
            mask = (d2_clip <= r_clip ** 2).astype(np.float32)
        else:
            mask = np.where(d2_clip <= r_clip ** 2, 1.0, args.clip_dim).astype(np.float32)
        if gain is not None:
            mask = mask * gain
        return {k: (L_per_star[k] * mask).astype(np.float32) for k in band_keys}

    # --- first pass: render the last frame twice (natural + boosted) and
    # interpolate the stretch between them. w=0 preserves far-frame
    # brightness but saturates close; w=1 preserves close detail but
    # crushes far; intermediate w is the aesthetic compromise. ---
    cam_last = cam_pos_for(len(rs) - 1)
    L_last_natural = L_for_frame(len(rs) - 1, apply_gain=False)
    imgs_last_natural = render_frame(coords, L_last_natural, cam_last, center, args.fov,
                                     args.image_width, args.image_height,
                                     args.splat_sigma_pix, args.splat_radius_mult,
                                     band_keys)
    stack_nat = np.stack([imgs_last_natural[R_key], imgs_last_natural[G_key], imgs_last_natural[B_key]])
    stretch_natural = float(np.percentile(stack_nat, args.stretch_pct) + 1e-30)

    w = float(np.clip(args.stretch_gain_weight, 0.0, 1.0))
    if args.brightness_exponent == 0.0 or w == 0.0:
        stretch = stretch_natural
        imgs_last = imgs_last_natural
        stretch_boosted = stretch_natural
    else:
        L_last_boosted = L_for_frame(len(rs) - 1, apply_gain=True)
        imgs_last = render_frame(coords, L_last_boosted, cam_last, center, args.fov,
                                 args.image_width, args.image_height,
                                 args.splat_sigma_pix, args.splat_radius_mult,
                                 band_keys)
        stack_boost = np.stack([imgs_last[R_key], imgs_last[G_key], imgs_last[B_key]])
        stretch_boosted = float(np.percentile(stack_boost, args.stretch_pct) + 1e-30)
        ratio = stretch_boosted / stretch_natural
        stretch = stretch_natural * (ratio ** w)
    print(f'Global stretch = {stretch:.3e}  (natural={stretch_natural:.3e}, '
          f'boosted={stretch_boosted:.3e}, w={w:.2f}, alpha='
          f'{args.brightness_exponent}, d_floor={args.brightness_d_floor})',
          flush=True)

    # --- render this rank's frames (round-robin assignment) ---
    my_frames = [i for i in range(len(rs)) if i % args.world_size == args.world_rank]
    print(f'[rank {args.world_rank}/{args.world_size}] rendering {len(my_frames)} '
          f'frames', flush=True)
    for i in my_frames:
        t0 = time.time()
        r = rs[i]
        cam_pos = cam_pos_for(i)
        if i == len(rs) - 1 and imgs_last is not None:
            imgs = imgs_last
        else:
            L_i = L_for_frame(i)
            imgs = render_frame(coords, L_i, cam_pos, center, args.fov,
                                args.image_width, args.image_height,
                                args.splat_sigma_pix, args.splat_radius_mult,
                                band_keys)
        rgb = lupton_rgb(imgs[R_key], imgs[G_key], imgs[B_key],
                         Q=args.Q, stretch=stretch)
        Image.fromarray(rgb).save(os.path.join(args.out_dir, f'frame_{i:04d}.png'))
        if (i // args.world_size) % 10 == 0 or i == my_frames[-1]:
            r_clip_str = f'  r_clip={clip_rs[i]:.1f}' if clip_rs is not None else ''
            print(f'[r{args.world_rank}] frame {i}/{len(rs)-1}  r={r:.1f}'
                  f'{r_clip_str}  ({time.time()-t0:.2f}s)', flush=True)
    print(f'DONE {len(rs)} frames in {args.out_dir}', flush=True)


if __name__ == '__main__':
    main()
