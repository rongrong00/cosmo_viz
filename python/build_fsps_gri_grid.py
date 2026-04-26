#!/usr/bin/env python3
"""
Precompute a 2D (log age, log Z/Zsol) grid of SDSS g,r,i luminosities
per unit stellar mass for a single-burst SSP (Kroupa IMF, MILES + MIST).

Illustris/TNG mock-image convention: g,r,i -> B,G,R.
"""
import os, sys, numpy as np, h5py
import fsps

M_SUN_AB = {'sdss_g': 5.11, 'sdss_r': 4.65, 'sdss_i': 4.53}

def main():
    out = sys.argv[1] if len(sys.argv) > 1 else 'config/fsps_gri_ssp.h5'

    young = np.arange(-4.0, -1.0, 0.025)
    old   = np.arange(-1.0, 1.15 + 1e-9, 0.075)
    log_age_gyr = np.concatenate([young, old])
    logZsol = np.linspace(-2.3, 0.3, 14)

    sp = fsps.StellarPopulation(zcontinuous=1, sfh=0, imf_type=1)

    Na, Nz = len(log_age_gyr), len(logZsol)
    L_g = np.zeros((Na, Nz)); L_r = np.zeros((Na, Nz)); L_i = np.zeros((Na, Nz))

    for j, lz in enumerate(logZsol):
        sp.params['logzsol'] = float(lz)
        for i, la in enumerate(log_age_gyr):
            tage = float(10.0 ** la)
            mags = sp.get_mags(tage=tage, bands=['sdss_g', 'sdss_r', 'sdss_i'])
            L_g[i, j] = 10.0 ** ((M_SUN_AB['sdss_g'] - mags[0]) / 2.5)
            L_r[i, j] = 10.0 ** ((M_SUN_AB['sdss_r'] - mags[1]) / 2.5)
            L_i[i, j] = 10.0 ** ((M_SUN_AB['sdss_i'] - mags[2]) / 2.5)
        print(f'  logZsol={lz:+.2f}  done ({j+1}/{Nz})  '
              f'ages={Na}, L_r[0,{j}]={L_r[0,j]:.2e} L_r[-1,{j}]={L_r[-1,j]:.2e}',
              flush=True)

    os.makedirs(os.path.dirname(out) or '.', exist_ok=True)
    with h5py.File(out, 'w') as f:
        f.create_dataset('log_age_gyr', data=log_age_gyr)
        f.create_dataset('logZsol', data=logZsol)
        f.create_dataset('L_g', data=L_g)
        f.create_dataset('L_r', data=L_r)
        f.create_dataset('L_i', data=L_i)
        f.attrs['bands'] = ['sdss_g', 'sdss_r', 'sdss_i']
        f.attrs['units'] = 'L_sun_band / M_sun (SSP, Kroupa, MILES+MIST)'
        f.attrs['M_sun_AB'] = [M_SUN_AB['sdss_g'], M_SUN_AB['sdss_r'], M_SUN_AB['sdss_i']]
    print(f'wrote {out}  shape=({Na},{Nz})')

if __name__ == '__main__':
    main()
