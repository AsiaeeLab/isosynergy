"""sim_monotone calculations."""
import argparse
import os
import time
import mc
import numpy as np
import pandas as pd
(S, sim) = (mc.S, mc.sim)
LAWS = {'A': lambda : S.ParametricNoise.additive(0.08), 'B': lambda : S.ParametricNoise.lognormal(0.12), 'C': lambda : S.ParametricNoise.oneil()}
SURF_NULL = ['base', 'horm_0.03', 'horm_0.07', 'horm_0.15', 'hormA_0.07', 'flat_0.03', 'flat_0.07', 'revtop_0.01', 'revtop_0.05', 'revtop_0.15', 'revmid_0.05', 'revmid_0.15']
BUMPS = ['bumpC_0.15', 'bumpC_0.25', 'bumpK_0.05', 'bumpK_0.10', 'bumpR_0.10', 'bumpR_0.20']
SURF_POW = ['base', 'horm_0.07', 'horm_0.15', 'revmid_0.05']
PATS = ['syn_hdblock_0.80', 'syn_corner_2.00', 'dip_0.50']
CHUNK = 250
NSTD = 2

def nm_edges(kind, I):
    base = sim.surface('moderate', I)[:, 0]
    (a, b) = (base.copy(), base.copy())
    if kind == 'base':
        return (a, b)
    (name, v) = kind.rsplit('_', 1)
    v = float(v)
    if name == 'horm':
        a[1] = 1 + v
        b[1] = 1 + v
    elif name == 'hormA':
        a[1] = 1 + v
    elif name == 'flat':
        b[1:] = 1 + v
    elif name == 'revtop':
        a[-1] = a[-2] + v
        b[-1] = b[-2] + v
    elif name == 'revmid':
        a[3] = a[2] + v
        b[3] = b[2] + v
    else:
        raise ValueError(kind)
    return (a, b)

def nm_truth(surf, cond, I):
    (a, b) = nm_edges(surf, I)
    mu = np.outer(a, b)
    if cond == 'null':
        return mu
    parts = cond.split('_')
    out = mu.copy()
    if parts[0] in ('bumpC', 'bumpK', 'bumpR'):
        v = float(parts[1])
        if parts[0] == 'bumpC':
            out[2, 2] += v
        elif parts[0] == 'bumpK':
            out[I - 1, I - 1] += v
        else:
            out[2, 1:] += v
        return out
    if parts[0] == 'dip':
        out[2, 2] *= np.exp(-float(parts[1]))
        return out
    if parts[0] == 'sat':
        out[1:, 1:] = np.maximum(mu[1:, 1:], 0.0) ** (1.0 + float(parts[1]))
        return out
    (_, pat, amp) = parts
    return mu * np.exp(-float(amp) * sim.pattern(pat, I))

def is_nonmono(mu):
    return bool((np.diff(mu, axis=0) > 1e-12).any() or (np.diff(mu, axis=1) > 1e-12).any())

def run_job(job):
    (part, design, I, surf, cond, law, start, n) = job
    t0 = time.time()
    nz = LAWS[law]()
    mu = nm_truth(surf, cond, I)
    ds = mc.data_seed('fig6_mono', part, design, I, surf, cond, law, start)
    Ys = sim.simulate(mu, nz, design, n, np.random.default_rng(ds))
    P = np.empty((n, 2))
    for t in range(n):
        r = mc.fit_sim(Ys[t], nz, design, [ds, t], n_std=NSTD)
        P[t] = (r.p_tstd, r.p_mean)
    df = pd.DataFrame(P.astype('float32'), columns=['p', 'p_mean'])
    (df['t'], df['start']) = (np.arange(n), start)
    (df['part'], df['design'], df['I'], df['surface'], df['cond'], df['law']) = (part, design, I, surf, cond, law)
    df['nonmono_truth'] = is_nonmono(mu)
    df['secs'] = time.time() - t0
    return df

def jobs_for(part, n_override=0, cell=None):
    J = []

    def add(design, I, surf, cond, law, N):
        for s in range(0, N, CHUNK):
            J.append((part, design, I, surf, cond, law, s, min(CHUNK, N - s)))
    if part == 'confirm':
        (design, I, law, surf) = cell
        add(design, I, surf, 'null', law, n_override or 20000)
        return J
    for design in ('S', 'P'):
        if part == 'size':
            for law in 'ABC':
                for I in (5, 4):
                    for surf in SURF_NULL:
                        if I == 4 and surf.startswith('revmid'):
                            continue
                        add(design, I, surf, 'null', law, n_override or 2000)
                    for cond in BUMPS:
                        add(design, I, 'base', cond, law, n_override or 2000)
        else:
            for law in 'AC':
                for surf in SURF_POW:
                    for cond in PATS:
                        add(design, 5, surf, cond, law, n_override or 1000)
    return J

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('part', choices=['size', 'power', 'confirm'])
    ap.add_argument('--n', type=int, default=0)
    ap.add_argument('--workers', type=int, default=3)
    ap.add_argument('--cell', default='S,5,B,horm_0.07')
    a = mc.smoke_args(ap.parse_args())
    c = a.cell.split(',')
    J = jobs_for(a.part, a.n, (c[0], int(c[1]), c[2], c[3]))
    J.sort(key=lambda j: -j[2])
    print(f'{a.part}: {len(J)} jobs, {sum((j[7] for j in J))} datasets', flush=True)
    t0 = time.time()
    out = mc.pool_map(run_job, J, workers=mc.workers(a.workers), progress=50)
    res = pd.concat(out, ignore_index=True)
    tag = f'_n{a.n}' if a.n and not mc.paths.SMOKE else ''
    if a.part == 'confirm':
        tag = f'_{c[0]}{c[1]}{c[2]}{c[3]}' + tag
    path = os.path.join(mc.RESDIR, f'fig6_monotone_{a.part}{tag}.parquet')
    res.to_parquet(path, index=False)
    print(f'done {a.part}: {len(res)} rows, {time.time() - t0:.0f} s -> {path}', flush=True)
if __name__ == '__main__':
    main()
