"""Size of V2Sp at exact Bliss and at the boundary of its mean-excess null."""
import os
import sys
for _k in ['OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMBA_NUM_THREADS']:
    os.environ[_k] = '1'
import argparse
import time
from multiprocessing import Pool
import numpy as np
import pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import companion_core as E
import companion_noise as C
from sirsyn import seed_for
GRIDS = [4, 5, 8]
SURFS = list(C.SURFACES)
NULL_NOISE = [('G', 0.05), ('G', 0.1), ('G', 0.2), ('G', 0.3), ('L', 0.5), ('L', 1.0), ('L', 2.0), ('V', 0.5), ('V', 1.0), ('V', 2.0), ('H', 1.0), ('B', 0.5), ('B', 1.0), ('B', 2.0), ('HB', 1.0)]
LOGNULL_NOISE = [('G', 0.05), ('G', 0.1), ('G', 0.2), ('G', 0.3), ('L', 0.5), ('L', 1.0), ('L', 2.0)]
S1_NOISE = [('G', 0.2), ('L', 1.0), ('V', 1.0), ('H', 1.0), ('B', 1.0)]
COMP_NOISE = [('G', 0.1), ('G', 0.2), ('L', 1.0), ('V', 1.0), ('H', 1.0), ('B', 1.0), ('HB', 1.0)]
POWER_NOISE = [('G', 0.1, 'v'), ('G', 0.2, 'v'), ('G', 0.2, 'log'), ('L', 1.0, 'v'), ('L', 1.0, 'log'), ('V', 1.0, 'v'), ('V', 2.0, 'v'), ('H', 1.0, 'v'), ('B', 1.0, 'v'), ('HB', 1.0, 'v')]
ANT_AMPS = [0.02, 0.05, 0.1]
SYN_AMPS = [0.02, 0.03, 0.04, 0.06, 0.08, 0.1, 0.15]
ALPHAS = [0.05, 0.01, 0.005]
TESTS = ['V2Sp']
CHUNK = 5000

def truth(c):
    (n, s) = (c['n'], c['surface'])
    k = c['kind']
    if k == 'null':
        return C.bliss_ref(n, s)
    if k.startswith('ant_'):
        return C.antagonism_truth(n, s, k[4:], c['amp'])[0]
    if k.startswith('mixv_'):
        return E.mixed_truth_v(n, s, k[5:], c['amp'])[0]
    if k.startswith('mixlog_'):
        return C.mixed_truth(n, s, k[7:], c['amp'])[0]
    if k.startswith('syn_'):
        return C.synergy_truth(n, s, k[4:], c['amp'])[0]
    raise ValueError(k)

def run_tests(Y, vfun):
    return {'V2Sp': E.v_kc(Y, '2sp')}

def run_condition(c):
    y = truth(c)
    if y is None:
        return []
    cn = E.CellNoise(y, c['family'], c['level'], c['null'])
    vfun = E.var_function(c['family'], c['level'], c['null'])
    rng = np.random.default_rng(seed_for(bytes([114, 52, 101]).decode('ascii'), c['stage'], c['design'], c['family'], c['level'], c['null'], c['n'], c['surface'], c['kind'], c['amp']))
    icc = 0.1 if c['design'] == 'S1' else 0.0
    acc = {t: np.zeros((2, len(ALPHAS))) for t in TESTS}
    est_sum = {t: 0.0 for t in TESTS}
    nd = 0
    while nd < c['nsim']:
        m = min(CHUNK, c['nsim'] - nd)
        Y = cn.draw('S', m, rng, icc=icc)[:, 0]
        o = run_tests(Y, vfun)
        for t in TESTS:
            (ps, pa) = (np.nan_to_num(o[t]['p_syn'], nan=1.0), np.nan_to_num(o[t]['p_ant'], nan=1.0))
            for (k, a) in enumerate(ALPHAS):
                acc[t][0, k] += (ps < a).sum()
                acc[t][1, k] += (pa < a).sum()
            est_sum[t] += float(np.sum(o[t]['est']))
        nd += m
    yref = C.bliss_ref(c['n'], c['surface'])
    ym = y if c['null'] == 'v' else np.exp(cn.loc + cn.sd ** 2 / 2)
    ym = np.array(ym, float)
    ym[0, 0] = 1.0
    B_true = float((np.outer(ym[1:, 0], ym[0, 1:]) - ym[1:, 1:]).mean())
    logex = float((np.log(np.outer(y[1:, 0], y[0, 1:])) - np.log(y[1:, 1:])).mean())
    rows = []
    for t in TESTS:
        r = dict(c)
        r.update(test=t, nd=nd, B_true=B_true, logexc_true=logex, mean_est=est_sum[t] / nd, dev_from_ref=float(np.abs(y - yref).max()))
        for (k, a) in enumerate(ALPHAS):
            r[f'syn_{a}'] = acc[t][0, k] / nd
            r[f'ant_{a}'] = acc[t][1, k] / nd
        rows.append(r)
    return rows

def conditions(stage):
    out = []
    base = dict(stage=stage, amp=0.0)
    if stage == 'null':
        for n in GRIDS:
            for s in SURFS:
                for (f, l) in NULL_NOISE:
                    out.append(dict(base, design='S', family=f, level=l, null='v', n=n, surface=s, kind='null', nsim=20000))
                for (f, l) in LOGNULL_NOISE:
                    out.append(dict(base, design='S', family=f, level=l, null='log', n=n, surface=s, kind='null', nsim=20000))
                for (f, l) in S1_NOISE:
                    out.append(dict(base, design='S1', family=f, level=l, null='v', n=n, surface=s, kind='null', nsim=20000))
    elif stage == 'comp':
        for n in GRIDS:
            for s in SURFS:
                for (f, l) in COMP_NOISE:
                    kinds = [(f'ant_{p}', a) for p in ('broad', 'highdose') for a in ANT_AMPS]
                    kinds += [(f'mixv_{p}', a) for p in ('hilo', 'checker') for a in (0.3, 1.0)]
                    kinds += [(f'mixlog_{p}', a) for p in ('hilo', 'checker') for a in (0.3, 1.0)]
                    for (k, a) in kinds:
                        out.append(dict(base, design='S', family=f, level=l, null='v', n=n, surface=s, kind=k, amp=a, nsim=4000))
    elif stage == 'power':
        for n in GRIDS:
            for s in SURFS:
                for (f, l, nu) in POWER_NOISE:
                    for p in ('broad', 'highdose', 'corner', 'lowdose'):
                        for a in SYN_AMPS:
                            out.append(dict(base, design='S', family=f, level=l, null=nu, n=n, surface=s, kind=f'syn_{p}', amp=a, nsim=2000))
    out.sort(key=lambda c: -c['n'] * c['n'] * c['nsim'])
    return out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('stage', choices=['null', 'comp', 'power'])
    ap.add_argument('--workers', type=int, default=10)
    a = ap.parse_args()
    conds = conditions(a.stage)
    conds = [c for c in conds if c['null']=='v' and c['design']=='S' and not c['kind'].startswith('mixlog')]
    if E.paths.SMOKE:
        for c in conds:c['nsim']=4
    print(a.stage, len(conds), 'conditions', flush=True)
    t0 = time.time()
    rows = []
    with Pool(E.paths.WORKERS) as pool:
        for (i, rr) in enumerate(pool.imap_unordered(run_condition, conds, chunksize=1)):
            rows.extend(rr)
            if (i + 1) % 50 == 0:
                print(f'{i + 1}/{len(conds)} {time.time() - t0:.0f}s', flush=True)
    df = pd.DataFrame(rows)
    df.to_parquet(os.path.join(E.RES, f'companion_{a.stage}.parquet'))
    print('done', len(df), f'{time.time() - t0:.0f}s', flush=True)
if __name__ == '__main__':
    main()
