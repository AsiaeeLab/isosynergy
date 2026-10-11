"""sim_missing calculations."""
import argparse
import os
import time
import mc
import numpy as np
import pandas as pd
from sirsyn import _kernels as K
(S, sim) = (mc.S, mc.sim)
LAWS = {'A': lambda : S.ParametricNoise.additive(0.08), 'B': lambda : S.ParametricNoise.lognormal(0.12), 'C': lambda : S.ParametricNoise.oneil()}
SURFS = ('weak', 'moderate', 'strong')
POWER_CONDS = [('moderate', f'syn_broad_{a}') for a in ('0.10', '0.20', '0.30')] + [('moderate', f'syn_hdblock_{a}') for a in ('0.40', '0.80', '1.20')] + [('weak', f'syn_corner_{a}') for a in ('0.50', '1.00', '2.00')] + [('strong', f'sat_{g}') for g in ('0.30', '0.60', '1.00')]
CHUNK = 250
NSTD = 2

def neighbor_fill(Y, side, k):
    edge = (Y[1:, 0] if side == 'A' else Y[0, 1:]).copy()
    n = edge.size
    obs = [q for q in range(n) if q != k - 1]
    fo = K.pava_dec(np.ascontiguousarray(edge[obs]))
    fit = dict(zip(obs, fo))
    lo = fit[k - 2] if k - 2 >= 0 else 1.0
    fill = 0.5 * (lo + fit[k]) if k <= n - 1 else lo
    Yf = Y.copy()
    if side == 'A':
        Yf[k, 0] = fill
    else:
        Yf[0, k] = fill
    return Yf

def drop_sa(Y, side, k):
    Ya = Y.copy()
    if side == 'A':
        Ya[k, 0] = np.nan
    else:
        Ya[0, k] = np.nan
    return Ya

def run_job(job):
    (part, design, I, surf, cond, law, start, n) = job
    t0 = time.time()
    nz = LAWS[law]()
    mu = sim.truth(cond, surf, I)
    eta = sim.bliss_excess(mu)
    syn = eta > 1e-09
    ds = mc.data_seed('fig6_missing', part, design, I, surf, cond, law, start)
    rng = np.random.default_rng(ds)
    rng_m = np.random.default_rng(mc.data_seed('fig6_missing', 'miss', part, design, I, surf, cond, law, start))
    Ys = sim.simulate(mu, nz, design, n, rng)
    cells = [(i, j) for i in range(1, I) for j in range(1, I)]
    rows = []
    for t in range(n):
        Y = Ys[t]
        perm = rng_m.permutation(len(cells))[:3]
        miss = [cells[q] for q in perm]
        side = 'A' if rng_m.random() < 0.5 else 'B'
        kd = int(rng_m.integers(1, I))
        seed = [ds, t]
        row = dict(t=t, start=start, miss=';'.join((f'{i}{j}' for (i, j) in miss)), sa_side=side, sa_dose=kd)
        tests = [('full', Y)]
        for q in (1, 2, 3):
            Ym = Y.copy()
            for c in miss[:q]:
                Ym[c] = np.nan
            tests.append((f'int{q}', Ym))
            row[f'nsyn_miss{q}'] = int(sum((syn[i - 1, j - 1] for (i, j) in miss[:q])))
        if design == 'S':
            if part == 'size':
                tests.append(('nfill', neighbor_fill(Y, side, kd)))
            tests.append(('rowdrop', drop_sa(Y, side, kd)))
        for (lab, Yt) in tests:
            r = mc.fit_sim(Yt, nz, design, seed, n_std=NSTD)
            row[f'p_{lab}'] = r.p_tstd
            if lab == 'rowdrop':
                assert r.log.analyzed_shape == ((I - 1, I) if side == 'A' else (I, I - 1))
        rows.append(row)
    df = pd.DataFrame(rows)
    df = df.astype({c: 'float32' for c in df.columns if c.startswith('p_')})
    (df['part'], df['design'], df['I'], df['surface'], df['cond'], df['law']) = (part, design, I, surf, cond, law)
    df['n_syn_cells'] = int(syn.sum())
    df['secs'] = time.time() - t0
    return df

def jobs_for(part, n_override=0, cell=None):
    J = []
    if part == 'confirm':
        (I, law, surf) = cell
        for s in range(0, 20000, CHUNK):
            J.append((part, 'S', I, surf, 'null', law, s, CHUNK))
        return J
    for I in (5, 4):
        for law in 'ABC':
            if part in ('size', 'sizeP'):
                design = 'S' if part == 'size' else 'P'
                for surf in SURFS:
                    N = n_override or 2000
                    for s in range(0, N, CHUNK):
                        J.append((part, design, I, surf, 'null', law, s, min(CHUNK, N - s)))
            elif part == 'power':
                for (surf, cond) in POWER_CONDS:
                    N = n_override or 1000
                    for s in range(0, N, CHUNK):
                        J.append((part, 'S', I, surf, cond, law, s, min(CHUNK, N - s)))
    return J

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('part', choices=['size', 'sizeP', 'power', 'confirm'])
    ap.add_argument('--n', type=int, default=0)
    ap.add_argument('--workers', type=int, default=3)
    ap.add_argument('--cell', default='4,B,moderate')
    a = mc.smoke_args(ap.parse_args())
    c = a.cell.split(',')
    J = jobs_for(a.part, a.n, (int(c[0]), c[1], c[2]))
    if a.n and a.part == 'confirm':
        J = [j[:6] + (j[6], a.n) for j in J[:1]]
    J.sort(key=lambda j: -j[2])
    print(f'{a.part}: {len(J)} jobs, {sum((j[7] for j in J))} datasets', flush=True)
    t0 = time.time()
    out = mc.pool_map(run_job, J, workers=mc.workers(a.workers), progress=20)
    res = pd.concat(out, ignore_index=True)
    tag = f'_n{a.n}' if a.n and not mc.paths.SMOKE else ''
    if a.part == 'confirm':
        tag = f'_{c[0]}{c[1]}{c[2]}' + tag
    os.makedirs(mc.RESDIR, exist_ok=True)
    path = os.path.join(mc.RESDIR, f'fig6_missing_{a.part}{tag}.parquet')
    res.to_parquet(path, index=False)
    print(f'done {a.part}: {len(res)} rows, {time.time() - t0:.0f} s -> {path}', flush=True)
if __name__ == '__main__':
    main()
