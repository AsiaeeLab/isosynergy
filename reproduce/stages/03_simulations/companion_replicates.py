"""Replicated and external-reference mean tests under exact viability nulls."""
import os
import sys
for _k in ['OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMBA_NUM_THREADS']:
    os.environ[_k] = '1'
from multiprocessing import Pool
import numpy as np
import pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import companion_core as E
import companion_noise as C
from sirsyn import seed_for
NOISE = [('G', 0.05), ('G', 0.1), ('G', 0.2), ('G', 0.3), ('L', 0.5), ('L', 1.0), ('L', 2.0), ('V', 0.5), ('V', 1.0), ('V', 2.0), ('H', 1.0), ('B', 0.5), ('B', 1.0), ('B', 2.0), ('HB', 1.0)]
ALPHAS = [0.05, 0.005]

def task(c):
    (n, s, f, l, des) = c
    y = C.bliss_ref(n, s)
    cn = E.CellNoise(y, f, l, 'v')
    rng = np.random.default_rng(seed_for(bytes([114, 52, 101]).decode('ascii'), 'designPR', n, s, f, l, des))
    (m, icc) = {'P': (1, 0.0), 'R0': (4, 0.0), 'R1': (4, 0.1)}[des]
    rows = []
    acc = {}
    nd = 0
    for _ in range(4):
        Yr = cn.draw('P', 4 if E.paths.SMOKE else 5000, rng, icc=icc, m=m)
        out = {}
        if des == 'P':
            Y = Yr[:, 0]
            out['t_viab_fe'] = C.t_viability_fixed_edges(None, Y)[:2]
            out['t_log_fe'] = C.t_fixed_edges(C.to_log(Y, 0.02))[:2]
        else:
            Zr = np.log(np.maximum(Yr, 0.02))
            Zr[..., 0, 0] = 0.0
            o = C.replicate_tests(Zr, Yr)
            out['t_viab_block'] = o['t_viability_block_level'][:2]
            out['t_viab_pe'] = o['t_viability_pure_error'][:2]
            out['t_log_block'] = o['t_block_level'][:2]
        for (t, (ps, pa)) in out.items():
            a = acc.setdefault(t, np.zeros(4))
            for (k, al) in enumerate(ALPHAS):
                a[k] += (ps < al).sum()
                a[2 + k] += (pa < al).sum()
        nd += Yr.shape[0]
    for (t, a) in acc.items():
        rows.append(dict(n=n, surface=s, family=f, level=l, design=des, test=t, nd=nd, **{f'syn_{al}': a[k] / nd for (k, al) in enumerate(ALPHAS)}, **{f'ant_{al}': a[2 + k] / nd for (k, al) in enumerate(ALPHAS)}))
    return rows

def main():
    conds = [(n, s, f, l, d) for d in ('P', 'R0', 'R1') for n in (8, 5, 4) for s in C.SURFACES for (f, l) in NOISE]
    rows = []
    with Pool(E.paths.WORKERS) as pool:
        for rr in pool.imap_unordered(task, conds):
            rows.extend(rr)
    df = pd.DataFrame(rows)
    df.to_parquet(os.path.join(E.RES, 'companion_replicates_null.parquet'))
    df['fam'] = df.family + df.level.astype(str)
    print(df.pivot_table(index='fam', columns=['design', 'test'], values='syn_0.05', aggfunc='max').round(3).to_string())
if __name__ == '__main__':
    main()
