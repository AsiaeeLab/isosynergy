"""Level of the O'Neil-configuration tests when the combination plates are offset from the reported curves.

Targets are the exact-Bliss O'Neil four-block stacks of the real-noise null check (reported curves as single
agents, real residuals from noise-blind donors in the other cell-line half). Each unit additionally receives an
offset delta shared by all its combination wells, delta ~ N(0, tau^2) (two-sided, so half of the units are
shifted toward antagonism), or a fixed delta (one-sided). Viability of the combination wells is lowered by
delta, which reads as uniform synergy relative to the reported curves. The reference is the production one
(noise bank, curves held fixed), single stage with B = 399.
"""
import argparse
import os
import sys
import time
import numpy as np
import pandas as pd
import mc
import common as C
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '02_realdata'))
import null_checks as NC
S = C.S
TAUS = [0.0, 0.005, 0.01, 0.02, 0.03, 0.05]
SHIFTS = [0.01, 0.02, 0.04, 0.08]


def job(args):
    (kind, size, tis) = args
    G = NC.G
    rows = []
    for ti in tis:
        s = G['SEL4'][ti]
        (eA, eB) = (G['EA'][s], G['EB'][s])
        mu = np.outer(eA[1:], eB[1:])
        (dd, dh) = NC.donors4(ti, mu)
        Y = np.empty((4, 5, 5))
        Y[:, :, 0] = eA[None]
        Y[:, 0, :] = eB[None]
        Y[:, 0, 0] = 1.0
        sdm = G['sd'](mu)
        for r in range(4):
            Y[r, 1:, 1:] = mu + G['RESID'][dd[r], r] * np.clip(sdm / G['SDY'][dd[r]], 0.5, 2.0)
        rng = np.random.default_rng(S.seed_for(C.SEED_TAG, 'offset', kind, size, int(ti)))
        delta = size * rng.standard_normal() if kind == 'random' else size
        Y[:, 1:, 1:] -= delta
        grp = str(G['CELL'][dd[0]])
        kw = {k: v for (k, v) in C.FIT_ONEIL.items() if k != 'design'}
        kw.update(two_stage=False, B=399)
        res = S.fit(Y, design='P', noise=G['bank'], plate_var=G['pvt'], group=grp,
                    seed=(C.SEED_TAG, 'offset', kind, size, int(ti)), **kw)
        rows.append(dict(kind=kind, size=size, target=int(ti), delta=float(delta), p_tstd=res.p_tstd,
                         p_scan=res.p_scan, p_mean=res.p_mean, mean_excess=res.mean_excess))
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--n', type=int, default=4000)
    ap.add_argument('--workers', type=int, default=mc.workers(16))
    a = mc.smoke_args(ap.parse_args())
    NC.init_oneil()
    N = NC.G['SEL4'].size
    idx = np.sort(np.random.default_rng(S.seed_for(C.SEED_TAG, 'offset-subset')).choice(N, min(a.n, N), replace=False))
    jobs = []
    for (kind, sizes) in (('random', TAUS), ('fixed', SHIFTS)):
        for size in sizes:
            for ch in np.array_split(idx, max(1, idx.size // 50)):
                jobs.append((kind, size, ch))
    t0 = time.time()
    res = pd.concat(mc.pool_map(job, jobs, workers=a.workers, progress=50), ignore_index=True)
    res.to_parquet(os.path.join(mc.RESDIR, 'offset_oneil.parquet'), index=False)
    rows = []
    for (kind, size), g in res.groupby(['kind', 'size']):
        n = len(g)
        for test in ('p_tstd', 'p_scan', 'p_mean'):
            for al in (0.05, 0.01):
                k = int((g[test] <= al).sum())
                (lo, hi) = mc.wilson(k, n)
                rows.append(dict(kind=kind, size=size, test=test, alpha=al, n=n, rate=k / n, lo=float(lo), hi=float(hi)))
    summ = pd.DataFrame(rows)
    summ.to_csv(os.path.join(mc.RESDIR, 'offset_oneil_summary.csv'), index=False)
    print(summ[summ.alpha == 0.05].to_string(), flush=True)
    print(f'{len(res)} fits, {time.time() - t0:.0f} s', flush=True)


if __name__ == '__main__':
    main()
