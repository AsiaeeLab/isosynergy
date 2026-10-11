"""Tail comparisons for ordered single-agent curves in the band analysis."""
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
import bb_core as K
G = K.G5
VALIDATION_DRAWS = 100 if K.paths.SMOKE else 200000
QS = [0.9, 0.95, 0.99]
N_FAM = dict(random=150, shift=60, single=60, hill=45, band=90)
ZCRIT = 4.4

def rand_curve(rng, n=4):
    lo = rng.uniform(0.0, 0.8)
    hi = rng.uniform(lo + 0.05, 1.05)
    v = np.sort(rng.uniform(lo, hi, n))[::-1]
    if rng.uniform() < 0.3:
        k = rng.integers(0, n - 1)
        v[k + 1] = v[k]
        v = np.minimum.accumulate(v)
    return v

def make_pairs(noise, ms, rng):
    P = []
    for _ in range(N_FAM['random']):
        (a2, b2) = (rand_curve(rng), rand_curve(rng))
        a = np.minimum(a2, rand_curve(rng))
        b = np.minimum(b2, rand_curve(rng))
        P.append(('random', a, b, a2, b2))
    for q in range(N_FAM['shift']):
        (a, b) = (rand_curve(rng), rand_curve(rng))
        d = [0.02, 0.05, 0.1][q % 3]
        P.append((f'shift', a, b, np.minimum(a + d, 1.2), np.minimum(b + d, 1.2)))
    for _ in range(N_FAM['single']):
        (a, b) = (rand_curve(rng), rand_curve(rng))
        (a2, b2) = (a.copy(), b.copy())
        tgt = a2 if rng.uniform() < 0.5 else b2
        i = rng.integers(0, 4)
        cap = 1.2 if i == 0 else tgt[i - 1]
        tgt[i] = tgt[i] + rng.uniform(0.2, 1.0) * (cap - tgt[i])
        P.append(('single', a, b, a2, b2))
    Es = [0.2, 0.3, 0.45, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99]
    hp = [(E, E2) for E in Es for E2 in Es if E2 < E]
    for k in rng.permutation(len(hp))[:N_FAM['hill']]:
        (E, E2) = hp[k]
        a = K.hill_curve(E)[1:]
        a2 = K.hill_curve(E2)[1:]
        P.append(('hill', a, a, a2, a2))
    bs = K.BandSpec(noise, ms, 0.005, 8, 'exact')
    q = 0
    while q < N_FAM['band']:
        base = ['weak', 'moderate', 'strong'][q % 3]
        (a0, b0) = K.base_ab(base)
        Y = K.gen_data(rng, 1, a0, b0, np.outer(a0, b0).ravel(), 5, 5, ms, noise)[0]
        (LA, UA, okA) = bs.band(Y[G.EA])
        (LB, UB, okB) = bs.band(Y[G.EB])
        if not (okA and okB):
            continue
        CA = K.candidates(Y[G.EA], LA, UA, rng)
        CB = K.candidates(Y[G.EB], LB, UB, rng)
        (i, j) = (rng.integers(1, len(CA)), rng.integers(1, len(CB)))
        P.append((f'band_{base}', CA[i], CB[j], UA, UB))
        q += 1
    return P

def law(a, b, noise, ms, Z):
    (ZA, ZB, ZI) = Z
    return K.ref_T_bliss(np.ascontiguousarray(a, float), np.ascontiguousarray(b, float), 5, 5, ms, K.KIND[noise], ZA, ZB, ZI, G.pa, G.pb, G.w, K.KV, K.KS)

def compare(T, T2):
    rec = {}
    for q in QS:
        t = np.quantile(T, q)
        d = (T2 >= t - 1e-12).astype(float) - (T >= t - 1e-12)
        rec[f'd_{q}'] = d.mean()
        rec[f'se_{q}'] = d.std(ddof=1) / np.sqrt(len(d))
        rec[f'p_{q}'] = (T >= t - 1e-12).mean()
    rec['frac_path_down'] = float((T2 < T - 1e-09).mean())
    return rec

def job(args):
    (noise, ms, B) = args
    t0 = time.time()
    rng = np.random.default_rng(K.seed('mono', 'pairs', noise, ms))
    P = make_pairs(noise, ms, rng)
    Z = K.draw_noise(np.random.default_rng(K.seed('mono', 'crn', noise, ms)), B, 5, 5, ms, noise)
    rows = []
    for (k, (fam, a, b, a2, b2)) in enumerate(P):
        assert np.all(a <= a2 + 1e-12) and np.all(b <= b2 + 1e-12)
        (T, T2) = (law(a, b, noise, ms, Z), law(a2, b2, noise, ms, Z))
        rec = dict(noise=noise, ms=ms, pair=k, family=fam, a=np.round(a, 4).tolist(), b=np.round(b, 4).tolist(), a2=np.round(a2, 4).tolist(), b2=np.round(b2, 4).tolist(), max_gap=float(max((a2 - a).max(), (b2 - b).max())))
        rec.update(compare(T, T2))
        rows.append(rec)
    df = pd.DataFrame(rows)
    z = np.min(np.stack([df[f'd_{q}'] / df[f'se_{q}'].clip(lower=1e-12) for q in QS]), axis=0)
    df['zmin'] = z
    val = []
    Zv = K.draw_noise(np.random.default_rng(K.seed('mono', 'val', noise, ms)), VALIDATION_DRAWS, 5, 5, ms, noise)
    for k in np.argsort(z)[:5]:
        (_, a, b, a2, b2) = P[k]
        (T, T2) = (law(a, b, noise, ms, Zv), law(a2, b2, noise, ms, Zv))
        rec = dict(noise=noise, ms=ms, pair=int(k), family=df.family[k], zmin_screen=float(z[k]), a=np.round(a, 4).tolist(), b=np.round(b, 4).tolist(), a2=np.round(a2, 4).tolist(), b2=np.round(b2, 4).tolist())
        rec.update(compare(T, T2))
        val.append(rec)
    print(noise, ms, f'{time.time() - t0:.0f}s', flush=True)
    return (df, pd.DataFrame(val))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--workers', type=int, default=9)
    ap.add_argument('--B', type=int, default=20000)
    a = ap.parse_args()
    a.workers=K.paths.WORKERS
    if K.paths.SMOKE:
        a.B=39
        for key in N_FAM:N_FAM[key]=2
    jobs = [(n, m, a.B) for n in K.NOISES for m in (1, 4, 16)]
    with Pool(a.workers) as p:
        out = p.map(job, jobs)
    df = pd.concat([o[0] for o in out], ignore_index=True)
    val = pd.concat([o[1] for o in out], ignore_index=True)
    df['B'] = a.B
    val['B'] = VALIDATION_DRAWS
    df.to_csv(os.path.join(K.RES, 'band_order_pairs.csv.gz'), index=False)
    val.to_csv(os.path.join(K.RES, 'band_order_validate.csv'), index=False)
    fam = df.family.str.replace('_(weak|moderate|strong)$', '', regex=True)
    df['fam'] = fam
    rows = []
    for ((noise, ms, f), s) in df.groupby(['noise', 'ms', 'fam']):
        rec = dict(noise=noise, ms=ms, family=f, n_pairs=len(s))
        for q in QS:
            zz = s[f'd_{q}'] / s[f'se_{q}'].clip(lower=1e-12)
            rec[f'n_viol_{q}'] = int((zz < -ZCRIT).sum())
            rec[f'min_d_{q}'] = float(s[f'd_{q}'].min())
        rows.append(rec)
    summ = pd.DataFrame(rows)
    summ.to_csv(os.path.join(K.RES, 'band_order_summary.csv'), index=False)
    pd.set_option('display.width', 250)
    print(summ.to_string(index=False))
    print(val[['noise', 'ms', 'family', 'zmin_screen'] + [f'd_{q}' for q in QS] + [f'se_{q}' for q in QS]].round(4).to_string(index=False))
if __name__ == '__main__':
    main()
