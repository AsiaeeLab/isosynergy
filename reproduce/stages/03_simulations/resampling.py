"""Paired comparison of 399 and 1,999 reference draws, Supplement S8."""
import os
import sys
for _k in ['OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMBA_NUM_THREADS']:
    os.environ[_k] = '1'
import argparse
import time
from multiprocessing import Pool
import numpy as np
import pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import resampling_core as G
from resampling_core import H
from sirsyn import seed_for
PCOLS = ('tstd', 'scan', 'comb', 'std_raw')
PIDX = np.array([0, 6, 8, 1])

def run_cell(job):
    (part, design, I, surf, noise, N, Bref, Bs) = job
    t_start = time.time()
    dp = design == 'P'
    (pa, pb, w1) = H.geom(I, I)
    reg = H.regions(I)
    (tk, tp, gk, gp, sk, sp) = G.law(noise)
    rows = [('null', 0.0)]
    if not part.startswith('confirm'):
        rows += [(k, a) for k in G.PATTERNS_FOR[surf] for a in G.amps_for(k)]
    mus = [G.truth(k, a, surf, I) for (k, a) in rows]
    rng_d = np.random.default_rng(seed_for(bytes([114, 55, 103]).decode('ascii'), part, 'data', design, I, surf, noise))
    Ed = rng_d.standard_normal((N, I, I))
    R = len(rows)
    Ys = np.empty((R, N, I, I))
    for (r, mu) in enumerate(mus):
        for t in range(N):
            Ys[r, t] = H.gen_design(mu, tk, tp, Ed[t], H.KV, H.KS, dp)
    for r in range(1, R):
        assert np.array_equal(Ys[r][:, :, 0], Ys[0][:, :, 0]) and np.array_equal(Ys[r][:, 0, :], Ys[0][:, 0, :])
    rng_b = np.random.default_rng(seed_for(bytes([114, 55, 103]).decode('ascii'), part, 'boot', design, I, surf, noise))
    rng_s = np.random.default_rng(seed_for(bytes([114, 55, 103]).decode('ascii'), part, 'std', design, I, surf, noise))
    nB = len(Bs)
    P = np.empty((nB, R, N, len(PCOLS)))
    obs = np.empty(H.NS + 2 * reg.nR)
    t_boot = 0.0
    for t in range(N):
        E = np.concatenate([rng_b.standard_normal((Bref, I, I)), rng_s.standard_normal((G.NSTD, I, I))])
        t1 = time.perf_counter()
        T = H.boot_table(Ys[0, t], gk, gp, E, H.KV, H.KS, dp, pa, pb, w1, sk, sp, H.KV, H.KS, reg.cells, reg.cptr)
        t_boot += time.perf_counter() - t1
        tabs = [T if B == Bref else np.ascontiguousarray(np.concatenate([T[:B + 1], T[Bref + 1:]])) for B in Bs]
        for r in range(R):
            if r > 0:
                H.plate_stats(Ys[r, t], pa, pb, w1, sk, sp, H.KV, H.KS, reg.cells, reg.cptr, obs)
                for tab in tabs:
                    tab[0] = obs
            for ib in range(nB):
                P[ib, r, t] = G.pvals_core(tabs[ib], reg.nR, Bs[ib])[PIDX]
    PM = np.stack([G.mean_test(Ys[r], design) for r in range(R)])
    recs = []
    for (ib, B) in enumerate(Bs):
        for (r, (kind, amp)) in enumerate(rows):
            rec = dict(part=part, design=design, I=I, surface=surf, noise=noise, B=B, pattern=kind, amp=amp, n=N)
            rec.update(G.effects(mus[r], noise))
            for a in G.ALPHAS:
                thr = a * (1 + 1e-09)
                for (c, name) in enumerate(PCOLS):
                    rec[f'k_{name}@{a}'] = int(np.sum(P[ib, r, :, c] <= thr))
                rec[f'k_mean@{a}'] = int(np.sum(PM[r] <= thr))
            recs.append(rec)
    nul = pd.DataFrame({'dataset': np.arange(N)})
    for (ib, B) in enumerate(Bs):
        for (c, name) in enumerate(PCOLS):
            nul[f'p_{name}_B{B}'] = P[ib, 0, :, c]
    nul['p_mean'] = PM[0]
    (nul['part'], nul['design'], nul['I'], nul['surface'], nul['noise']) = (part, design, I, surf, noise)
    secs = time.time() - t_start
    for rec in recs:
        rec['cell_secs'] = secs
        rec['boot_secs'] = t_boot
    return (pd.DataFrame(recs), nul, (part, design, I, surf, noise, N, secs, t_boot))

def jobs_for(part, n=None):
    J = []
    if part == 'main':
        N = n or 2000
        for design in ('S', 'P'):
            for I in G.GRIDS:
                for surf in G.SURFS:
                    for noise in ('add', 'oneil'):
                        J.append((part, design, I, surf, noise, N, 399, (399,)))
    elif part == 'bcheck':
        N = n or 1000
        for design in ('S', 'P'):
            for I in G.GRIDS:
                J.append((part, design, I, 'moderate', 'oneil', N, 1999, (399, 1999)))
    elif part == 'confirm':
        N = n or 2000
        for I in G.GRIDS:
            for noise in ('add', 'oneil'):
                for k in range(3):
                    J.append((f'confirm{k}', 'S', I, 'weak', noise, N, 399, (399,)))
    else:
        raise ValueError(part)
    return J

def cost(j):
    per = {4: 0.01, 5: 0.02, 6: 0.035, 8: 0.08, 10: 0.2}[j[2]]
    return j[5] * per * (j[6] + G.NSTD) / 600 * (1 + len(j[7]))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('part')
    ap.add_argument('--workers', type=int, default=6)
    ap.add_argument('--n', type=int, default=None)
    ap.add_argument('--tag', default='')
    a = ap.parse_args()
    a.workers=G.paths.WORKERS
    if G.paths.SMOKE:a.n=4
    J = jobs_for(a.part, a.n)
    J.sort(key=lambda j: -cost(j))
    print(f'{len(J)} jobs on {a.workers} workers; rough estimate {sum((cost(j) for j in J)) / a.workers:.0f} s', flush=True)
    t0 = time.time()
    (outs, nuls, logs) = ([], [], [])
    with Pool(a.workers) as pool:
        for (i, (df, nul, lg)) in enumerate(pool.imap_unordered(run_cell, J)):
            outs.append(df)
            nuls.append(nul)
            logs.append(lg)
            print(f'{i + 1}/{len(J)} {lg[:6]} {lg[6]:.0f} s (bootstrap tables {lg[7]:.0f} s); elapsed {time.time() - t0:.0f} s', flush=True)
    res = pd.concat(outs, ignore_index=True)
    res.to_csv(os.path.join(G.RES, f'resampling_counts_{a.part}{a.tag}.csv'), index=False)
    pd.concat(nuls, ignore_index=True).to_parquet(os.path.join(G.RES, f'resampling_null_pvalues_{a.part}{a.tag}.parquet'), index=False)
    print(f'done {a.part}: {len(J)} cells, {len(res)} rows, {time.time() - t0:.0f} s', flush=True)
if __name__ == '__main__':
    main()
