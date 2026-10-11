"""sim_fragility calculations."""
import argparse
import os
import time
import mc
import numpy as np
import pandas as pd
(S, sim) = (mc.S, mc.sim)
LAWS = {'A': lambda : S.ParametricNoise.additive(0.08), 'B': lambda : S.ParametricNoise.lognormal(0.12), 'C': lambda : S.ParametricNoise.oneil()}
NULL_LEVELS = (0.0, 0.004, 0.01, 0.02)
CONDS = [('weak', 'null', 2000), ('moderate', 'null', 2000)] + [('weak', f'syn_corner_{a}', 1000) for a in ('1.00', '2.00')] + [('moderate', f'dip_{a}', 1000) for a in ('0.50', '1.00')] + [('moderate', f'syn_broad_{a}', 1000) for a in ('0.20', '0.30')] + [('moderate', 'syn_hdblock_0.80', 1000), ('strong', 'sat_0.60', 1000)]
CHUNK = 250
NSTD = 2
MAGS = np.load(os.path.join(mc.INPUTS, 'gross_errors.npz'))['absdev']

def truth(cond, surf, I):
    if cond.startswith('dip'):
        mu = sim.surface(surf, I).copy()
        mu[2, 2] *= np.exp(-float(cond.split('_')[1]))
        return mu
    return sim.truth(cond, surf, I)

def run_job(job):
    (I, surf, cond, law, start, n) = job
    t0 = time.time()
    nz = LAWS[law]()
    mu = truth(cond, surf, I)
    syn = np.zeros((I, I), bool)
    syn[1:, 1:] = sim.bliss_excess(mu) > 1e-09
    ds = mc.data_seed('fig6_frag', I, surf, cond, law, start)
    cs = mc.data_seed('fig6_frag', 'cont', I, surf, cond, law, start)
    Ys = sim.simulate(mu, nz, 'S', n, np.random.default_rng(ds))
    rows = []
    for p in NULL_LEVELS if cond == 'null' else (0.0,):
        (Rc, cm) = sim.contaminate(Ys[:, None], np.random.default_rng(cs), p, MAGS)
        for t in range(n):
            r = mc.fit_sim(Rc[t, 0], nz, 'S', [ds, t], fragility='auto', n_std=NSTD)
            c = cm[t]
            row = dict(t=t, start=start, level=p, p=r.p_tstd, n_cont=int(c.sum()), n_cont_edge=int(c[0, :].sum() + c[:, 0].sum()), n_cont_int=int(c[1:, 1:].sum()), computed=bool(r.fragility.get('computed', False)), loco_p=np.nan, loco_flag=False, loco_cell='', loco_cell_cont=False, loco_cell_syn=False, sa_p=np.nan, sa_flag=False, sa_dose='', sa_cont=False)
            f = r.fragility
            if row['computed']:
                (i, j) = f['loco_cell']
                row.update(loco_p=f['loco_p'], loco_flag=bool(f['loco_flag']), loco_cell=f'{i}{j}', loco_cell_cont=bool(c[i, j]), loco_cell_syn=bool(syn[i, j]))
                if f.get('sa_dose') is not None:
                    (sd, k) = f['sa_dose']
                    row.update(sa_p=f['sa_p'], sa_flag=bool(f['sa_flag']), sa_dose=f'{sd}{k}', sa_cont=bool(c[k, 0] if sd == 'A' else c[0, k]))
            rows.append(row)
    df = pd.DataFrame(rows)
    df = df.astype({'p': 'float32', 'loco_p': 'float32', 'sa_p': 'float32'})
    (df['I'], df['surface'], df['cond'], df['law']) = (I, surf, cond, law)
    df['secs'] = time.time() - t0
    return df

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--n', type=int, default=0)
    ap.add_argument('--workers', type=int, default=3)
    a = mc.smoke_args(ap.parse_args())
    J = []
    for I in (5, 4):
        for law in 'ABC':
            for (surf, cond, N) in CONDS:
                N = a.n or N
                for s in range(0, N, CHUNK):
                    J.append((I, surf, cond, law, s, min(CHUNK, N - s)))
    J.sort(key=lambda j: -(j[0] * (4 if j[2] == 'null' else 1)))
    print(f'fragility: {len(J)} jobs', flush=True)
    t0 = time.time()
    out = mc.pool_map(run_job, J, workers=mc.workers(a.workers), progress=20)
    res = pd.concat(out, ignore_index=True)
    tag = f'_n{a.n}' if a.n and not mc.paths.SMOKE else ''
    path = os.path.join(mc.RESDIR, f'fig6_fragility{tag}.parquet')
    res.to_parquet(path, index=False)
    print(f'done: {len(res)} rows, {time.time() - t0:.0f} s -> {path}', flush=True)
if __name__ == '__main__':
    main()
