"""sim_stress_shift calculations."""
import mc
import argparse
import os
import time
import numpy as np
import pandas as pd
import sim_stress_shared as SH
S = mc.S
RHO = 0.1
RATIOS = [0.0, 0.1, 0.25, 0.5]
MS = [np.inf, 4.0]
SDV = S.oneil_sd_curve()
NZ = {r: S.ParametricNoise.oneil(plate_shift_ratio=r) for r in RATIOS}
FSEED = int(mc.data_seed('figS1', 't3'))

def job(chunk):
    rows = []
    for n in chunk:
        rp = np.random.default_rng(S.seed_for(bytes([114, 54, 115]).decode('ascii'), 't3', 'pair', int(n)))
        cl = SH.CELLS[rp.integers(len(SH.CELLS))]
        (names, Acur) = SH.CUR[cl]
        (i, j) = rp.choice(Acur.shape[0], 2, replace=False)
        (A1, A2) = (Acur[i], Acur[j])
        (u1, u10, u2, u20) = (rp.standard_normal(4), rp.standard_normal(), rp.standard_normal(4), rp.standard_normal())
        rd = np.random.default_rng(S.seed_for(bytes([114, 54, 115]).decode('ascii'), 't3', 'data', int(n)))
        z = rd.standard_normal((4, 4))
        g = rd.standard_normal()
        mub = np.outer(A1, A2)
        sig_m = float(SDV(mub.mean()))
        for m in MS:
            if np.isfinite(m):
                e1 = A1 + SDV(A1) * (np.sqrt((1 - RHO) / m) * u1 + np.sqrt(RHO / m) * u10)
                e2 = A2 + SDV(A2) * (np.sqrt((1 - RHO) / m) * u2 + np.sqrt(RHO / m) * u20)
            else:
                (e1, e2) = (A1.copy(), A2.copy())
            for r in RATIOS:
                Y = np.empty((5, 5))
                Y[0, 0] = 1.0
                (Y[1:, 0], Y[0, 1:]) = (e1, e2)
                Y[1:, 1:] = mub + r * sig_m * g + SDV(mub) * z
                for ref in ('without', 'with'):
                    if r == 0.0 and ref == 'with':
                        continue
                    nz = NZ[r] if ref == 'with' else NZ[0.0]
                    res = mc.fit(Y, design='P', noise=nz, sa_replicates=m, sa_rho=RHO, seed=[FSEED, int(n)], two_stage=False, B=399, n_std=200, fragility='never')
                    rows.append(dict(n=int(n), cell=cl, m_s=float(m), ratio=r, ref=ref, p_t=res.p_tstd, p_s=res.p_scan, p_m=res.p_mean))
    return pd.DataFrame(rows)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--n', type=int, default=20000)
    ap.add_argument('--workers', type=int, default=mc.workers(3))
    ap.add_argument('--tag', default='')
    a = mc.smoke_args(ap.parse_args())
    chunks = [c.tolist() for c in np.array_split(np.arange(a.n), max(1, a.n // 100))]
    t0 = time.time()
    out = mc.pool_map(job, chunks, workers=a.workers, progress=20)
    d = pd.concat(out, ignore_index=True)
    d.to_parquet(os.path.join(mc.RESDIR, f'stress_shift{a.tag}.parquet'), index=False)
    print(f'done: {len(d)} rows, {time.time() - t0:.0f} s', flush=True)
if __name__ == '__main__':
    main()
