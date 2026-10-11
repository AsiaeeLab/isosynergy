"""Extended vertex and coordinate search for the Berger–Boos supremum."""
import os
import sys
for _k in ['OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMBA_NUM_THREADS']:
    os.environ[_k] = '1'
import argparse
import itertools
import time
from multiprocessing import Pool
import numpy as np
import pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import bb_core as K
import bb_operating as O
G = K.G5

def vertices(L, U):
    n = len(L)
    out = []
    for cuts in itertools.product([0, 1], repeat=n - 1):
        blocks = []
        s = 0
        for (i, c) in enumerate(cuts):
            if c:
                blocks.append((s, i))
                s = i + 1
        blocks.append((s, n - 1))
        for choice in itertools.product([0, 1], repeat=len(blocks)):
            v = np.empty(n)
            ok = True
            for ((i, k), ch) in zip(blocks, choice):
                (lo, hi) = (L[i], U[k])
                if lo > hi + 1e-12:
                    ok = False
                    break
                v[i:k + 1] = lo if ch == 0 else hi
            if ok and np.all(np.diff(v) <= 1e-12):
                out.append(v)
    V = np.unique(np.round(np.array(out), 12), axis=0)
    return np.ascontiguousarray(V[np.argsort(-V.sum(1))])

def ascent(a, b, best, t, LA, UA, LB, UB, args, nval=9, sweeps=5):
    n_eval = 0
    for sweep in range(sweeps):
        improved = False
        for drug in (0, 1):
            cur = a if drug == 0 else b
            (L, U) = (LA, UA) if drug == 0 else (LB, UB)
            for i in range(4):
                lo_i = max(L[i], cur[i + 1] if i < 3 else L[i])
                hi_i = min(U[i], cur[i - 1] if i > 0 else U[i])
                if hi_i - lo_i < 1e-09:
                    continue
                for v in np.linspace(lo_i, hi_i, nval):
                    cand = cur.copy()
                    cand[i] = v
                    (aa, bb) = (cand, b) if drug == 0 else (a, cand)
                    c = K.count_one_es(np.ascontiguousarray(aa), np.ascontiguousarray(bb), t, best, *args)
                    n_eval += 1
                    if c > best:
                        best = c
                        cur[i] = v
                        improved = True
        if not improved:
            break
    return (best, a, b, n_eval)

def job(args_):
    (noise, base, ms, cond, amp, n, kind_row, ks, c_def) = args_
    t0 = time.time()
    Y = O.gen_row(noise, base, ms, cond, amp, n)
    spec2 = K.BandSpec(noise, ms, O.BETA, 8, 'exact')
    kind = K.KIND[noise]
    out = []
    for (k, cd) in zip(ks, c_def):
        (ZA, ZB, ZI) = O.ref_noise(noise, base, ms, cond, k)
        args = (5, 5, ms, kind, ZA, ZB, ZI, G.pa, G.pb, G.w, K.KV, K.KS)
        y = np.ascontiguousarray(Y[k])
        t = K.t_std(y, 5, 5, G.pa, G.pb, G.w, kind, K.KV, K.KS)
        (LA, UA, _) = spec2.band(y[G.EA])
        (LB, UB, _) = spec2.band(y[G.EB])
        rng = np.random.default_rng(K.seed('cands', noise, base, ms, cond, k))
        CA = np.vstack([K.candidates(y[G.EA], LA, UA, rng, n_flat=6, n_rand=5), vertices(LA, UA)])
        CB = np.vstack([K.candidates(y[G.EB], LB, UB, rng, n_flat=6, n_rand=5), vertices(LB, UB)])
        pairs = np.array([(i, j) for i in range(len(CA)) for j in range(len(CB))], dtype=np.int64)
        cnt = K.count_pairs(CA, CB, pairs, t, *args)
        best = int(cnt.max())
        top = np.argsort(-cnt)[:5]
        n_eval = len(pairs)
        for q in top:
            (a, b) = (CA[pairs[q, 0]].copy(), CB[pairs[q, 1]].copy())
            (bst, a, b, ne) = ascent(a, b, int(cnt[q]), t, LA, UA, LB, UB, args)
            n_eval += ne
            best = max(best, bst)
        out.append(dict(noise=noise, base=base, ms=ms, cond=cond, k=int(k), c_default=int(cd), c_ext=best, n_vert_A=len(vertices(LA, UA)), n_vert_B=len(vertices(LB, UB)), c_prod_ext=int(cnt.max()), n_eval=n_eval))
    print(noise, base, ms, cond, len(ks), f'{time.time() - t0:.0f}s', flush=True)
    return pd.DataFrame(out)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--workers', type=int, default=10)
    a = ap.parse_args()
    a.workers=K.paths.WORKERS
    if K.paths.SMOKE: O.B_REF=39;O.N_SIZE=4;O.N_POWER=4
    count = 2 if K.paths.SMOKE else 150
    sr = pd.read_parquet(os.path.join(K.RES, 'bb_search.parquet'))
    sr['p_BB_search'] = np.minimum(1, (1 + sr.c_search) / (O.B_REF + 1) + O.BETA)
    rng = np.random.default_rng(K.seed('searchcheck'))
    rel = sr[sr.p_BB_search <= 0.1]
    pick = pd.concat([rel.iloc[rng.choice(len(rel), min(count, len(rel)), replace=False)], sr.iloc[rng.choice(len(sr), min(count,len(sr)), replace=False)]]).drop_duplicates(['noise', 'base', 'ms', 'cond', 'k'])
    amps = O.load_amps()
    jobs = []
    for ((noise, base, ms, cond), s) in pick.groupby(['noise', 'base', 'ms', 'cond']):
        amp = O.SIZE_CONDS[cond] if cond in O.SIZE_CONDS else amps[noise, base, cond]
        n = O.N_SIZE if cond in O.SIZE_CONDS else O.N_POWER
        s = s.sort_values('k')
        jobs.append((noise, base, ms, cond, amp, n, '', s.k.to_numpy(), s.c_search.to_numpy()))
    print('datasets', len(pick), flush=True)
    with Pool(a.workers) as p:
        res = p.map(job, jobs, chunksize=1)
    df = pd.concat(res, ignore_index=True)
    df = df.merge(pick[['noise', 'base', 'ms', 'cond', 'k', 'p_BB_search']], on=['noise', 'base', 'ms', 'cond', 'k'])
    df['B'] = O.B_REF
    df.to_parquet(os.path.join(K.RES, 'bb_search_check.parquet'), index=False)
    df['gain'] = df.c_ext - df.c_default
    print(df.gain.describe())
    print('frac ext > default', (df.gain > 0).mean(), 'max gain (counts of 999)', df.gain.max())
    rel = df[df.p_BB_search <= 0.1]
    print('decision-relevant: n', len(rel), 'frac gain>0', (rel.gain > 0).mean(), 'flip at 0.05', ((rel.p_BB_search <= 0.05) & ((1 + rel.c_ext) / (O.B_REF + 1) + O.BETA > 0.05)).sum())
if __name__ == '__main__':
    main()
