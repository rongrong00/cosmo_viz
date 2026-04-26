#!/usr/bin/env python3
"""
Python-side stellar light ray trace (dust-free).

For each PartType4 star:
  1. Compute age at snapshot redshift from GFM_StellarFormationTime (scale factor a_form).
  2. Bilinear-interp the FSPS SSP grid (log age, logZsol) -> L_u, L_g, L_r per M_sun.
  3. Scale by GFM_InitialMass (total formation-time mass, in internal units).
  4. Project star onto the camera (perspective pinhole) and splat into the image.
     Splat kernel is a small 2D Gaussian whose sigma comes from a minimum pixel
     size so individual particles don't disappear under subpixel sampling.

Outputs an HDF5 file with L_u, L_g, L_r in L_sun per pixel (total luminosity
through each pixel column — after projection each pixel tallies L from stars
whose splats landed on it).
"""
import argparse, time, sys
import numpy as np
import h5py
import yaml
from scipy.integrate import cumulative_trapezoid


# Flat LCDM (Planck15-ish). Age(a) computed via a precomputed table.
H0_KM_S_MPC = 67.74
OM = 0.3089
ODE = 0.6911
HUBBLE_TIME_GYR = 977.79 / H0_KM_S_MPC  # 1/H0 in Gyr


def _age_table():
    a = np.logspace(-5, 0, 20000)
    integrand = 1.0 / (a * np.sqrt(OM / a ** 3 + ODE))
    age = np.concatenate(([0.0], cumulative_trapezoid(integrand, a)))
    return a, age * HUBBLE_TIME_GYR


_A_TAB, _AGE_TAB = _age_table()


def age_gyr_from_a(a):
    a_clip = np.clip(a, _A_TAB[0], _A_TAB[-1])
    return np.interp(a_clip, _A_TAB, _AGE_TAB)


Z_SUN = 0.0127  # FSPS default total metal mass fraction for logzsol=0

# Internal-mass unit: GADGET-style 1e10 M_sun/h. Include h later in the mass
# scaling if needed; for relative RGB imaging the overall normalization is
# absorbed by the Lupton stretch anyway.
MASS_UNIT_MSUN_PER_H = 1.0e10


def load_camera(path):
    with open(path, 'r') as f:
        doc = yaml.safe_load(f)
    cam = doc['camera'] if 'camera' in doc else doc
    return cam


def build_camera_matrix(cam):
    pos = np.array(cam['position'], dtype=np.float64)
    look_at = np.array(cam['look_at'], dtype=np.float64)
    up = np.array(cam['up'], dtype=np.float64)
    fov = float(cam['fov'])
    W = int(cam['image_width'])
    H = int(cam['image_height'])

    f = look_at - pos
    f /= np.linalg.norm(f)
    r = np.cross(f, up); r /= np.linalg.norm(r)
    u_axis = np.cross(r, f)  # camera up (already normalized)
    # world -> camera: x_cam = (x - pos) . r ; y_cam = (x - pos) . u_axis ; z_cam = (x - pos) . f
    return {'pos': pos, 'right': r, 'up': u_axis, 'fwd': f,
            'fov': fov, 'W': W, 'H': H,
            'tan_half_fov_y': np.tan(np.deg2rad(fov) * 0.5) * (H / W) * (W / H)}
    # Note: matplotlib-style vertical FOV would use H/W; we use horizontal fov.


def interp_fsps(log_age, logZsol, grid):
    """Bilinear interp (Na x Nz) at a batch of (log_age, logZsol) points."""
    La = grid['log_age_gyr']; Lz = grid['logZsol']
    # Clamp to grid bounds
    la = np.clip(log_age, La[0], La[-1])
    lz = np.clip(logZsol, Lz[0], Lz[-1])
    ia = np.searchsorted(La, la) - 1
    ia = np.clip(ia, 0, len(La) - 2)
    iz = np.searchsorted(Lz, lz) - 1
    iz = np.clip(iz, 0, len(Lz) - 2)
    fa = (la - La[ia]) / (La[ia + 1] - La[ia])
    fz = (lz - Lz[iz]) / (Lz[iz + 1] - Lz[iz])
    out = {}
    band_keys = [k for k in grid.keys() if k.startswith('L_')]
    for band in band_keys:
        G = grid[band]
        v00 = G[ia, iz]; v01 = G[ia, iz + 1]
        v10 = G[ia + 1, iz]; v11 = G[ia + 1, iz + 1]
        out[band] = ((1 - fa) * (1 - fz) * v00 +
                     (1 - fa) * fz       * v01 +
                     fa       * (1 - fz) * v10 +
                     fa       * fz       * v11)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--snapshot', required=True)
    ap.add_argument('--camera', required=True)
    ap.add_argument('--fsps-grid', default='config/fsps_ugr_ssp.h5')
    ap.add_argument('--output', required=True, help='Output HDF5 with L_u,L_g,L_r maps')
    ap.add_argument('--h', type=float, default=0.6774, help='Little h for Planck15')
    ap.add_argument('--splat-sigma-pix', type=float, default=0.8,
                    help='Fixed splat Gaussian sigma in pixels (point-like for now)')
    ap.add_argument('--splat-radius-pix', type=float, default=3.0,
                    help='Splat footprint in pixels (3 sigma default)')
    ap.add_argument('--max-stars', type=int, default=0,
                    help='Optional cap on #stars loaded (0 = all). Random subsample.')
    ap.add_argument('--clip-center', type=float, nargs=3, default=None,
                    metavar=('CX', 'CY', 'CZ'),
                    help='Keep only stars within --clip-radius of this point (ckpc/h).')
    ap.add_argument('--clip-radius', type=float, default=0.0,
                    help='Spatial clip radius around --clip-center (ckpc/h). 0 = no clip.')
    ap.add_argument('--clip-dim', type=float, default=0.0,
                    help='Dimming factor applied to stars outside --clip-radius '
                         '(0 = remove completely, 1 = keep at full luminosity). '
                         'e.g. 0.2 keeps the environment visible but faded.')
    args = ap.parse_args()

    # Camera
    cam_doc = load_camera(args.camera)
    cam = build_camera_matrix(cam_doc)
    W, H = cam['W'], cam['H']
    aspect = W / H
    fov_x = np.deg2rad(cam['fov'])
    tan_half_fov_x = np.tan(fov_x * 0.5)
    tan_half_fov_y = tan_half_fov_x / aspect  # vertical half-fov
    print(f'Camera: pos={cam["pos"]} fov={cam["fov"]} deg image={W}x{H}', flush=True)

    # Load stars
    t0 = time.time()
    with h5py.File(args.snapshot, 'r') as f:
        z_snap = float(f['Header'].attrs['Redshift'])
        a_snap = 1.0 / (1.0 + z_snap)
        print(f'Snapshot z={z_snap:.4f} a={a_snap:.4f}', flush=True)
        pt4 = f['PartType4']
        coords = pt4['Coordinates'][:]  # ckpc/h
        a_form = pt4['GFM_StellarFormationTime'][:].astype(np.float64)
        Z = pt4['GFM_Metallicity'][:].astype(np.float64)
        m_init = pt4['GFM_InitialMass'][:].astype(np.float64)  # 1e10 Msun/h
    # Filter wind particles (negative a_form) and unphysical
    keep = (a_form > 0) & (a_form < a_snap + 1e-6) & (Z > 0) & (m_init > 0)
    dim_weight = None
    if args.clip_radius > 0 and args.clip_center is not None:
        c = np.array(args.clip_center, dtype=np.float64)
        d2 = np.sum((coords - c) ** 2, axis=1)
        inside = d2 <= args.clip_radius ** 2
        if args.clip_dim <= 0.0:
            keep &= inside
            print(f'Spatial clip (hard): center={c}, r={args.clip_radius} ckpc/h '
                  f'-> kept {int(keep.sum())}/{coords.shape[0]}', flush=True)
        else:
            dim_weight = np.where(inside, 1.0, args.clip_dim)
            print(f'Spatial clip (soft): center={c}, r={args.clip_radius} ckpc/h '
                  f'dim={args.clip_dim} inside={int(inside.sum())}/{inside.size}',
                  flush=True)
    coords = coords[keep]; a_form = a_form[keep]; Z = Z[keep]; m_init = m_init[keep]
    if dim_weight is not None:
        dim_weight = dim_weight[keep]
    if args.max_stars > 0 and coords.shape[0] > args.max_stars:
        rng = np.random.default_rng(0)
        idx = rng.choice(coords.shape[0], args.max_stars, replace=False)
        coords = coords[idx]; a_form = a_form[idx]; Z = Z[idx]; m_init = m_init[idx]
    N = coords.shape[0]
    print(f'Loaded {N} valid stars in {time.time()-t0:.1f}s', flush=True)

    # Convert comoving ckpc/h to physical kpc at snapshot: cube coords are in
    # ckpc/h. We render in a camera whose positions are specified in the SAME
    # ckpc/h frame as the snapshot (the cosmo_viz pipeline convention), so no
    # unit conversion is needed for projection.

    # Age: age(a_snap) - age(a_form), using Planck15.
    t0 = time.time()
    age_form = age_gyr_from_a(a_form)
    age_snap = float(age_gyr_from_a(np.array([a_snap]))[0])
    age_gyr = np.maximum(age_snap - age_form, 1.0e-5)  # floor at 10 kyr
    log_age = np.log10(age_gyr)
    print(f'age range [Gyr]: {age_gyr.min():.2e}..{age_gyr.max():.2e}  '
          f'median={np.median(age_gyr):.3f}  ({time.time()-t0:.1f}s)', flush=True)

    # Metallicity
    logZsol = np.log10(Z / Z_SUN)

    # FSPS lookup
    t0 = time.time()
    with h5py.File(args.fsps_grid, 'r') as f:
        grid = {'log_age_gyr': f['log_age_gyr'][:], 'logZsol': f['logZsol'][:]}
        for k in f.keys():
            if k.startswith('L_'):
                grid[k] = f[k][:]
    band_keys = [k for k in grid.keys() if k.startswith('L_')]
    L = interp_fsps(log_age, logZsol, grid)
    m_msun = m_init * MASS_UNIT_MSUN_PER_H / args.h
    if dim_weight is not None:
        m_msun = m_msun * dim_weight
    L_per_star = {k: L[k] * m_msun for k in band_keys}
    totals = {k: float(v.sum()) for k, v in L_per_star.items()}
    print(f'FSPS lookup done ({time.time()-t0:.1f}s). bands={band_keys}  '
          f'totals={totals}', flush=True)

    # Project: world -> camera frame
    t0 = time.time()
    d = coords - cam['pos']
    xc = d @ cam['right']
    yc = d @ cam['up']
    zc = d @ cam['fwd']
    # In front of camera
    in_front = zc > 1.0  # at least 1 ckpc/h in front
    # Normalized device coords
    ndc_x = xc / (zc * tan_half_fov_x + 1e-30)
    ndc_y = yc / (zc * tan_half_fov_y + 1e-30)
    # Pixel coords (image convention: y axis flipped for top-left origin)
    px = (ndc_x * 0.5 + 0.5) * W
    py = (-ndc_y * 0.5 + 0.5) * H
    in_frame = in_front & (px >= -5) & (px < W + 5) & (py >= -5) & (py < H + 5)
    sel = np.where(in_frame)[0]
    print(f'{sel.size} stars project into frame ({time.time()-t0:.1f}s)', flush=True)

    # Splat (small Gaussian). For speed, precompute a stamp and add to image.
    t0 = time.time()
    sigma = args.splat_sigma_pix
    rad = int(np.ceil(args.splat_radius_pix * sigma))
    xs = np.arange(-rad, rad + 1)
    ys = np.arange(-rad, rad + 1)
    XX, YY = np.meshgrid(xs, ys)
    stamp = np.exp(-(XX * XX + YY * YY) / (2.0 * sigma * sigma)).astype(np.float32)
    stamp /= stamp.sum()  # unit integral -> per-pixel share of L

    imgs = {k: np.zeros((H, W), dtype=np.float32) for k in band_keys}

    px_i = np.round(px[sel]).astype(np.int32)
    py_i = np.round(py[sel]).astype(np.int32)
    L_sel = {k: L_per_star[k][sel].astype(np.float32) for k in band_keys}

    H_img, W_img = H, W
    for dy in range(-rad, rad + 1):
        for dx in range(-rad, rad + 1):
            w = stamp[dy + rad, dx + rad]
            if w <= 0:
                continue
            ty = py_i + dy
            tx = px_i + dx
            ok = (ty >= 0) & (ty < H_img) & (tx >= 0) & (tx < W_img)
            if not ok.any():
                continue
            ti = ty[ok] * W_img + tx[ok]
            for k in band_keys:
                np.add.at(imgs[k].ravel(), ti, w * L_sel[k][ok])
    print(f'Splatted {sel.size} stars ({time.time()-t0:.1f}s)', flush=True)
    for k in band_keys:
        print(f'  {k} pixel max={imgs[k].max():.3e}  sum={imgs[k].sum():.3e}', flush=True)

    with h5py.File(args.output, 'w') as f:
        for k in band_keys:
            f.create_dataset(k, data=imgs[k], compression='gzip')
        hdr = f.create_group('Header')
        hdr.attrs['redshift'] = z_snap
        hdr.attrs['n_stars_total'] = int(N)
        hdr.attrs['n_stars_in_frame'] = int(sel.size)
        hdr.attrs['camera_fov'] = cam['fov']
    print(f'Wrote {args.output}', flush=True)


if __name__ == '__main__':
    main()
