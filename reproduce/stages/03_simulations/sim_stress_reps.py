"""sim_stress_reps calculations."""
import mc
import argparse
import os
import time
import numpy as np
import pandas as pd
import sim_stress_shared as SH
S = mc.S
SDV = S.oneil_sd_curve()
NZ4 = S.ParametricNoise.oneil(tails='t4')
NZH = S.ParametricNoise.sd_table(S.SDCurve('table', kv=SDV.kv, ks=0.5 * SDV.ks), tails='t4')
FSEED = int(mc.data_seed('figS1', 'reps'))

def job(chunk):
    rows = []
    for n in chunk:
        rng = np.random.default_rng(mc.data_seed('figS1', 'reps', int(n)))
        cl = SH.CELLS[rng.integers(len(SH.CELLS))]
        (names, Acur) = SH.CUR[cl]
        (i, j) = rng.choice(Acur.shape[0], 2, replace=False)
        (A1, A2) = (Acur[i], Acur[j])
        mub = np.outer(A1, A2)
        e = rng.standard_t(4, (4, 4, 4)) / np.sqrt(2.0)
        Ys = np.empty((4, 5, 5))
        Ys[:, 0, 0] = 1.0
        Ys[:, 1:, 0] = A1
        Ys[:, 0, 1:] = A2
        Ys[:, 1:, 1:] = mub[None] + SDV(mub)[None] * e
        ra = mc.fit(Ys, design='P', noise=NZ4, seed=[FSEED, int(n)], two_stage=False, B=399, n_std=200, fragility='never')
        rb = mc.fit(Ys.mean(0), design='P', noise=NZH, sd=SDV, seed=[FSEED, int(n)], two_stage=False, B=399, n_std=200, fragility='never')
        rows.append(dict(n=int(n), cell=cl, p_matched=ra.p_tstd, p_single=rb.p_tstd, n_screened=len(ra.log.screened)))
    return pd.DataFrame(rows)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--n', type=int, default=20000)
    ap.add_argument('--workers', type=int, default=mc.workers(3))
    ap.add_argument('--tag', default='')
    a = mc.smoke_args(ap.parse_args())
    chunks = [c.tolist() for c in np.array_split(np.arange(a.n), max(1, a.n // 200))]
    t0 = time.time()
    d = pd.concat(mc.pool_map(job, chunks, workers=a.workers, progress=20), ignore_index=True)
    d.to_parquet(os.path.join(mc.RESDIR, f'stress_reps{a.tag}.parquet'), index=False)
    for c in ('p_matched', 'p_single'):
        print(c, [round(float((d[c] <= al).mean()), 4) for al in (0.05, 0.01, 0.005)])
    print(f'done: {len(d)} rows, {time.time() - t0:.0f} s', flush=True)
if __name__ == '__main__':
    main()
