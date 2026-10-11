"""Search and independent validation over antagonistic surfaces, Supplement S4."""
import os
import sys
for _k in ['OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMBA_NUM_THREADS']:
    os.environ[_k] = '1'
import argparse
import json
import time
from multiprocessing import Pool
import numpy as np
import pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import null_search_core as H
from sirsyn import seed_for
B = 399
ALPHAS = (0.05, 0.01)
STATS = {'std_iso': 0, 'std_raw': 1, 'std_rawref': 2, 'scan': -1}
SEARCHED = ('std_iso', 'std_raw', 'scan')
CAP = 1.2

def kth_thr(Tb, alpha):
    k = int(np.floor(alpha * (Tb.shape[0] + 1) + 1e-09))
    return -np.sort(-Tb, axis=0)[k - 1]

class Setup:

    def __init__(self, design, I, surf, tname, bname, n, rng, rng_s):
        (self.design, self.I) = (design, I)
        (self.pa, self.pb, self.w1) = H.geom(I, I)
        self.reg = H.regions(I)
        (self.tk, self.tp, self.tc) = H.TRUTH[tname]
        (self.gk, self.gp, self.gc, self.sk, self.sp) = H.BOOT[bname]
        self.dp = design == 'P'
        self.mu0 = H.surface(surf, I)
        self.n = n
        self.Ed = H.std_noise(rng, self.tc, (n, I, I))
        nR = self.reg.nR
        self.thr = {}
        self.h = {}
        thr_s = {a: np.empty((n, 3)) for a in ALPHAS}
        thr_z = {a: np.empty(n) for a in ALPHAS}
        sd_s = np.empty((n, 3))
        self.mU = np.empty((n, nR))
        self.sU = np.empty((n, nR))
        for t in range(n):
            Y = H.gen_design(self.mu0, self.tk, self.tp, self.Ed[t], H.KV, H.KS, self.dp)
            E = np.concatenate([H.std_noise(rng, self.gc, (B, I, I)), H.std_noise(rng_s, self.gc, (H.NSTD, I, I))])
            T = H.boot_table(Y, self.gk, self.gp, E, H.KV, H.KS, self.dp, self.pa, self.pb, self.w1, self.sk, self.sp, H.KV, H.KS, self.reg.cells, self.reg.cptr)
            (m, sd) = H.std_moments(T, 6, nR, B)
            (self.mU[t], self.sU[t]) = (m, sd)
            Tb = T[1:B + 1]
            Mb = ((Tb[:, 6:6 + nR] - m) / sd).max(1)
            for a in ALPHAS:
                thr_s[a][t] = kth_thr(Tb[:, :3], a)
                thr_z[a][t] = kth_thr(Mb[:, None], a)[0]
            sd_s[t] = Tb[:, :3].std(0, ddof=1)
        for a in ALPHAS:
            for (nm, k) in STATS.items():
                self.thr[nm, a] = thr_z[a] if k < 0 else thr_s[a][:, k]
        for (nm, k) in STATS.items():
            self.h[nm] = np.full(n, 0.1) if k < 0 else np.maximum(0.05 * sd_s[:, k], 1e-09)

    def observed(self, delta):
        mu = self.mu0.copy()
        mu[1:, 1:] += delta
        Ys = np.empty(self.Ed.shape)
        for t in range(self.Ed.shape[0]):
            Ys[t] = H.gen_design(mu, self.tk, self.tp, self.Ed[t], H.KV, H.KS, self.dp)
        return H.stats_many(Ys, self.pa, self.pb, self.w1, self.sk, self.sp, H.KV, H.KS, self.reg.cells, self.reg.cptr)

    def value_thr(self, S, nm, alpha):
        if STATS[nm] >= 0:
            return (S[:, STATS[nm]], self.thr[nm, alpha])
        nR = self.reg.nR
        return (((S[:, 6:6 + nR] - self.mU) / self.sU).max(1), self.thr[nm, alpha])

def validate(design, I, surf, tname, bname, cands, nval, rng, rng_s):
    (pa, pb, w1) = H.geom(I, I)
    reg = H.regions(I)
    (tk, tp, tc) = H.TRUTH[tname]
    (gk, gp, gc, sk, sp) = H.BOOT[bname]
    dp = design == 'P'
    mu0 = H.surface(surf, I)
    mus = []
    for (_, _, _, d) in cands:
        mu = mu0.copy()
        mu[1:, 1:] += d
        mus.append(mu)
    out = np.empty((len(cands), nval, H.NS + 3))
    for t in range(nval):
        e = H.std_noise(rng, tc, (I, I))
        E = np.concatenate([H.std_noise(rng, gc, (B, I, I)), H.std_noise(rng_s, gc, (H.NSTD, I, I))])
        Y = H.gen_design(mu0, tk, tp, e, H.KV, H.KS, dp)
        T = H.boot_table(Y, gk, gp, E, H.KV, H.KS, dp, pa, pb, w1, sk, sp, H.KV, H.KS, reg.cells, reg.cptr)
        for (c, mu) in enumerate(mus):
            Yc = H.gen_design(mu, tk, tp, e, H.KV, H.KS, dp)
            H.plate_stats(Yc, pa, pb, w1, sk, sp, H.KV, H.KS, reg.cells, reg.cptr, T[0])
            (p, _, _) = H.pvals_from_table(T, reg.nR, 0.05, B)
            out[c, t] = p
    return out

def starts(I, rng):
    m = I - 1
    out = [('bliss', np.zeros((m, m)))]
    for p in ('broad', 'highdose', 'lowdose', 'lshape', 'corner'):
        P = H.pattern(p, I)[1:, 1:]
        for A in (0.02, 0.1):
            out.append((f'{p}_{A}', A * P))
    d = np.zeros((m, m))
    d[0, 0] = np.nan
    out.append(('cell11_to_1.10', d))
    for k in range(2):
        out.append((f'rand{k}', 0.15 * rng.random((m, m)) * (rng.random((m, m)) < 0.4)))
    return out

def search_one(S, nm, alpha, rng, steps):
    I = S.I
    m = I - 1
    capd = CAP - S.mu0[1:, 1:]
    h = S.h[nm]

    def obj(delta):
        (v, thr) = S.value_thr(S.observed(delta), nm, alpha)
        hard = float(np.mean(v > thr + 1e-12 * np.maximum(1.0, np.abs(thr))))
        soft = float(np.mean(1.0 / (1.0 + np.exp(-np.clip((v - thr) / h, -50, 50)))))
        return (soft, hard)
    best = []
    for (sname, d0) in starts(I, rng):
        d = d0.copy()
        if np.isnan(d).any():
            d = np.zeros((m, m))
            d[0, 0] = max(0.0, 1.1 - S.mu0[1, 1])
        d = np.clip(d, 0.0, capd)
        (fs, fh) = obj(d)
        f0h = fh
        sc = 0.05
        for it in range(steps):
            prop = d.copy()
            if rng.random() < 0.5:
                (i, j) = (rng.integers(m), rng.integers(m))
                prop[i, j] += sc * rng.standard_normal()
            else:
                prop += sc / np.sqrt(m * m) * 2.0 * rng.standard_normal((m, m))
            prop = np.clip(prop, 0.0, capd)
            (ps, ph) = obj(prop)
            if ps > fs:
                (d, fs, fh) = (prop, ps, ph)
                sc = min(0.3, sc * 1.3)
            else:
                sc = max(0.002, sc * 0.93)
        best.append(dict(start=sname, soft=fs, hard=fh, hard_start=f0h, delta=d.copy()))
    best.sort(key=lambda z: -z['soft'])
    return best

def job(cfg):
    (design, I, surf, tname, bname, ncrn, nval, steps) = cfg
    t0 = time.time()
    rng = np.random.default_rng(seed_for(bytes([114, 54, 104]).decode('ascii'), 'search', *cfg[:5]))
    rng_s = np.random.default_rng(seed_for(bytes([114, 54, 104]).decode('ascii'), 'search_std', *cfg[:5]))
    S = Setup(design, I, surf, tname, bname, ncrn, rng, rng_s)
    t_setup = time.time() - t0
    paths = []
    cands = [('bliss', '-', 0.0, np.zeros((I - 1, I - 1)))]
    S0 = S.observed(np.zeros((I - 1, I - 1)))
    for nm in SEARCHED:
        for a in ALPHAS:
            best = search_one(S, nm, a, rng, steps)
            (v0, t0_) = S.value_thr(S0, nm, a)
            crn0 = float(np.mean(v0 > t0_ + 1e-12 * np.maximum(1, np.abs(t0_))))
            for (r, b) in enumerate(best):
                paths.append(dict(design=design, I=I, surface=surf, truth=tname, boot=bname, stat=nm, alpha=a, rank=r, start=b['start'], crn_soft=b['soft'], crn_hard=b['hard'], crn_hard_start=b['hard_start'], crn_hard_bliss=crn0, max_delta=float(b['delta'].max()), mean_delta=float(b['delta'].mean()), delta=json.dumps(np.round(b['delta'], 4).tolist())))
            for r in range(2):
                cands.append((f'{nm}@{a}#{r}', nm, a, best[r]['delta']))
            nt = [b for b in best if b['delta'].max() >= 0.02]
            if nt:
                cands.append((f'{nm}@{a}#nt', nm, a, nt[0]['delta']))
    rv = np.random.default_rng(seed_for(bytes([114, 54, 104]).decode('ascii'), 'search_val', *cfg[:5]))
    rvs = np.random.default_rng(seed_for(bytes([114, 54, 104]).decode('ascii'), 'search_val_std', *cfg[:5]))
    PV = validate(design, I, surf, tname, bname, cands, nval, rv, rvs)
    col = {'std_iso': 0, 'std_raw': 1, 'std_rawref': 2, 'scan': 6, 'comb': 8}
    rej = {}
    for (c, (cname, nm, a, d)) in enumerate(cands):
        for (st, q) in col.items():
            for al in ALPHAS:
                rej[cname, st, al] = PV[c, :, q] <= al + 1e-12
    rows = []
    for (cname, nm, a, d) in cands:
        mu = S.mu0.copy()
        mu[1:, 1:] += d
        eta = H.excess(mu)
        for st in col:
            for al in ALPHAS:
                x = rej[cname, st, al]
                x0 = rej['bliss', st, al]
                k = int(x.sum())
                (lo, hi) = H.wilson(k, nval)
                dif = x.astype(float) - x0.astype(float)
                rows.append(dict(design=design, I=I, surface=surf, truth=tname, boot=bname, cand=cname, searched_stat=nm, searched_alpha=a, stat=st, alpha=al, n=nval, rej=k / nval, lo=float(lo), hi=float(hi), diff_vs_bliss=float(dif.mean()), se_diff=float(dif.std(ddof=1) / np.sqrt(nval)), n_new_rej=int((x & ~x0).sum()), n_lost_rej=int((~x & x0).sum()), min_eta=float(eta.min()), max_mu=float(mu[1:, 1:].max()), delta=json.dumps(np.round(d, 4).tolist())))
    print(f'search {cfg[:5]}: setup {t_setup:.0f} s, total {time.time() - t0:.0f} s', flush=True)
    return (rows, paths)

def configs(ncrn, nval, steps):
    C = []
    for design in ('S', 'P'):
        for I in (5, 4):
            for surf in ('weak', 'moderate', 'strong', 'flat'):
                C.append((design, I, surf, 'C_hetero', 'hetero', ncrn, nval, steps))
    for (tname, bname) in (('A_gauss', 'gauss_add'), ('B_lnorm', 'lnorm'), ('D_t4', 't4_add')):
        C.append(('S', 5, 'moderate', tname, bname, ncrn, nval, steps))
        C.append(('S', 4, 'weak', tname, bname, ncrn, nval, steps))
    return C

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--workers', type=int, default=10)
    ap.add_argument('--ncrn', type=int, default=1500)
    ap.add_argument('--nval', type=int, default=10000)
    ap.add_argument('--steps', type=int, default=60)
    ap.add_argument('--limit', type=int, default=0)
    ap.add_argument('--tag', default='')
    a = ap.parse_args()
    a.workers = H.paths.WORKERS
    if H.paths.SMOKE:
        a.ncrn=4; a.nval=4; a.steps=2
        global B
        B=39;H.NSTD=9
    C = configs(a.ncrn, a.nval, a.steps)
    if a.limit:
        C = C[:a.limit]
    C.sort(key=lambda c: -(3 if c[0] == 'S' and c[1] == 5 else 1))
    print(f'{len(C)} configurations', flush=True)
    t0 = time.time()
    (rows, paths) = ([], [])
    with Pool(a.workers) as p:
        for (r, pth) in p.imap_unordered(job, C):
            rows += r
            paths += pth
    pd.DataFrame(rows).to_csv(os.path.join(H.RES, f'null_search{a.tag}.csv'), index=False)
    pd.DataFrame(paths).to_csv(os.path.join(H.RES, f'null_search_paths{a.tag}.csv'), index=False)
    print(f'done: {len(C)} configurations in {time.time() - t0:.0f} s', flush=True)
if __name__ == '__main__':
    main()
