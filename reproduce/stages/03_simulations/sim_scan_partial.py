"""sim_scan_partial calculations."""
import mc
import argparse
import glob
import os
import time
import numpy as np
import pandas as pd
import sim_grid as SG
(S, sim) = (mc.S, mc.sim)
ITEM = 'scan_partial'
OUT = os.path.join(mc.RESDIR, 'scan_partial_chunks')
ALPHAS = (0.05, 0.01)

def block(I, i0):
    P = np.zeros((I, I))
    P[i0:i0 + 2, i0:i0 + 2] = 1.0
    return P

def ppattern(kind, I):
    m = I - 1
    if kind == 'top2x2':
        return sim.pattern('hdblock', I)
    if kind == 'low2x2':
        return block(I, 1)
    if kind == 'mid2x2':
        return block(I, m // 2)
    if kind == 'corner':
        return SG.pattern('corner', I)
    raise ValueError(kind)

def kinds_for(I):
    return ('top2x2', 'corner', 'low2x2') + (('mid2x2',) if I >= 5 else ())

def run_job(job):
    (design, I, surf, noise, n) = job
    out = os.path.join(OUT, f'{design}_{I}_{surf}_{noise}.csv')
    if os.path.exists(out):
        return out
    t0 = time.time()
    nz = SG.noise_law(noise)
    cseed = mc.data_seed(ITEM, design, I, surf, noise, 0)
    mu0 = sim.surface(surf, I)
    rng = np.random.default_rng(cseed)
    Y0 = sim.simulate(mu0, nz, design, n, rng)
    null_padj = []
    regs = None
    for t in range(n):
        r = mc.fit_sim(Y0[t], nz, design, [cseed, t])
        null_padj.append(np.array([q['p_adj'] for q in r.regions]))
        if regs is None:
            regs = [set(map(tuple, q['cells'])) for q in r.regions]
    null_padj = np.array(null_padj)
    recs = []
    for kind in kinds_for(I):
        P = ppattern(kind, I)
        syn = {(i, j) for i in range(1, I) for j in range(1, I) if P[i, j] > 0}
        free = np.array([len(c & syn) == 0 for c in regs])
        mu = mu0 * np.exp(-1.0 * P)
        rng = np.random.default_rng(cseed)
        Ys = sim.simulate(mu, nz, design, n, rng)
        padj = []
        for t in range(n):
            r = mc.fit_sim(Ys[t], nz, design, [cseed, t])
            padj.append(np.array([q['p_adj'] for q in r.regions]))
        padj = np.array(padj)
        rec = dict(design=design, I=I, surface=surf, noise=noise, kind=kind, n=n, n_regions=len(regs), n_free=int(free.sum()))
        for a in ALPHAS:
            thr = a * (1 + 1e-09)
            rec[f'k_fwer@{a}'] = int(np.any(padj[:, free] <= thr, 1).sum()) if free.any() else 0
            rec[f'k_true@{a}'] = int(np.any(padj[:, ~free] <= thr, 1).sum())
            rec[f'k_fwer_null@{a}'] = int(np.any(null_padj[:, free] <= thr, 1).sum()) if free.any() else 0
            rec[f'leak@{a}'] = int(((padj[:, free] <= thr) != (null_padj[:, free] <= thr)).sum())
        rec['leak_checks'] = int(n * free.sum())
        recs.append(rec)
    os.makedirs(OUT, exist_ok=True)
    pd.DataFrame(recs).to_csv(out, index=False)
    print(f'{os.path.basename(out)} {time.time() - t0:.0f} s', flush=True)
    return out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('what', choices=('run', 'collect'))
    ap.add_argument('--workers', type=int, default=mc.workers(6))
    ap.add_argument('--n', type=int, default=1000)
    ap.add_argument('--designs', nargs='+', default=['S'])
    a = mc.smoke_args(ap.parse_args())
    if a.what == 'collect':
        df = pd.concat([pd.read_csv(f) for f in sorted(glob.glob(os.path.join(OUT, '*.csv')))], ignore_index=True)
        for al in ALPHAS:
            (lo, hi) = mc.wilson(df[f'k_fwer@{al}'], df.n)
            (df[f'fwer@{al}'], df[f'fwer@{al}_lo'], df[f'fwer@{al}_hi']) = (df[f'k_fwer@{al}'] / df.n, lo, hi)
        df.to_csv(os.path.join(mc.RESDIR, 'scan_partial_fwer.csv'), index=False)
        for (d, g) in df.groupby('design'):
            for al in ALPHAS:
                j = g[f'fwer@{al}'].idxmax()
                print(d, al, 'max FWER', g.loc[j, [f'fwer@{al}', f'fwer@{al}_lo', f'fwer@{al}_hi', 'I', 'surface', 'noise', 'kind']].to_dict(), 'leaks', int(g[f'leak@{al}'].sum()), 'of', int(g.leak_checks.sum()))
        return
    J = [(d, I, s, nz, a.n) for d in a.designs for I in (10, 8, 6, 5, 4) for s in ('weak', 'moderate', 'strong') for nz in ('oneil', 'add')]
    t0 = time.time()
    mc.pool_map(run_job, J, workers=a.workers)
    print(f'done {len(J)} cells, {time.time() - t0:.0f} s', flush=True)
if __name__ == '__main__':
    main()
