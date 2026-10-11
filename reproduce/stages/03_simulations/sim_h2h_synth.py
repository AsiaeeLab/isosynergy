"""sim_h2h_synth calculations."""
import mc
import argparse
import json
import os
import time
import numpy as np
import pandas as pd
(S, sim) = (mc.S, mc.sim)
CV_LN = float(np.sqrt(np.exp(0.12 ** 2) - 1.0))

def law(name):
    return {'A': lambda : S.ParametricNoise.additive(0.08), 'B': lambda : S.ParametricNoise.lognormal(0.12), 'C': lambda : S.ParametricNoise.oneil(), 'D': lambda : S.ParametricNoise.additive(0.08, tails='t4'), 'Bmis': lambda : S.ParametricNoise.constant_cv(CV_LN), 'Cmis': lambda : S.ParametricNoise.additive(0.08), 'Dmis': lambda : S.ParametricNoise.additive(0.08)}[name]()
MISREF = {'B': 'Bmis', 'C': 'Cmis', 'D': 'Dmis'}
ANT_MAIN = ['ant_broad_0.05', 'ant_broad_0.25', 'ant_highdose_0.05', 'ant_highdose_0.25', 'ant_lowdose_0.10', 'ant_lowdose_0.25', 'ant_lshape_0.05', 'ant_lshape_0.25']
ANT_NUIS = ['ant_broad_0.05', 'ant_highdose_0.05', 'ant_lowdose_0.10', 'ant_lshape_0.05']
POWER = [('moderate', 'syn_broad_0.20'), ('moderate', 'syn_hdblock_0.80'), ('weak', 'syn_corner_1.00'), ('moderate', 'syn_bump_0.30'), ('moderate', 'syn_lowdose_0.10'), ('strong', 'sat_0.60'), ('weak', 'mixed_0.60')]
CONFIRM_JSON = os.path.join(mc.RESDIR, 'h2h_confirm_cells.json')

def truth(cond, surf, I):
    if cond.startswith('mixed_'):
        amp = float(cond.split('_')[1])
        mu = sim.surface(surf, I)
        blk = sim.pattern('hdblock', I) > 0
        out = mu.copy()
        out[blk] = mu[blk] * np.exp(-amp)
        other = np.zeros((I, I), bool)
        other[1:, 1:] = True
        other &= ~blk
        out[other] = mu[other] + (mu[blk] - out[blk]).sum() / other.sum()
        return out
    return sim.truth(cond, surf, I)

def job(args):
    (part, design, I, surf, cond, tl, start, n) = args
    ds = mc.data_seed(f'h2h_{part}', design, I, surf, cond, tl, start)
    rng = np.random.default_rng(ds)
    mu = truth(cond, surf, I)
    nz = law(tl)
    Ys = sim.simulate(mu, nz, design, n, rng)
    refs = [('matched', nz)]
    if part == 'size' and cond == 'null' and (tl in MISREF):
        refs.append(('misspecified', law(MISREF[tl])))
    out = []
    for (rname, rn) in refs:
        P = np.empty((n, 3), np.float32)
        for t in range(n):
            r = mc.fit_sim(Ys[t], rn, design, [ds, t])
            P[t] = (r.p_tstd, r.p_scan, r.p_mean)
        df = pd.DataFrame(P, columns=['p_tstd', 'p_scan', 'p_mean'])
        df['ref'] = rname
        out.append(df)
    df = pd.concat(out, ignore_index=True)
    eta = sim.bliss_excess(mu)
    (df['design'], df['I'], df['surface'], df['cond'], df['law']) = (design, I, surf, cond, tl)
    (df['eff_mean'], df['eff_max']) = (float(eta.mean()), float(eta.max()))
    return df

def cells(part):
    C = []
    if part == 'size':
        for design in ('S', 'P'):
            for I in (5, 4):
                for tl in ('A', 'B', 'C', 'D'):
                    for surf in ('weak', 'moderate', 'strong'):
                        for cond in ['null'] + ANT_MAIN:
                            C.append((design, I, surf, cond, tl))
                    for surf in ('flat', 'half'):
                        for cond in ['null'] + ANT_NUIS:
                            C.append((design, I, surf, cond, tl))
    elif part == 'power':
        for design in ('S', 'P'):
            for (surf, cond) in POWER:
                C.append((design, 5, surf, cond, 'C'))
    elif part == 'confirm':
        with open(CONFIRM_JSON) as f:
            C = [tuple(c) for c in json.load(f)['cells']]
    return C

def pick_confirm_cells():
    df = pd.read_parquet(os.path.join(mc.RESDIR, 'h2h_size.parquet'))
    g = df[(df.ref == 'matched') & (df.cond == 'null')]
    r = g.groupby(['design', 'I', 'surface', 'law']).p_tstd.apply(lambda p: float(np.mean(p <= 0.05)))
    out = []
    for design in ('S', 'P'):
        rr = r.xs(design, level='design')
        ((I, surf, tl), v) = (rr.idxmax(), float(rr.max()))
        out.append(dict(design=design, I=int(I), surface=surf, law=tl, rate=v))
    cl = [[o['design'], o['I'], o['surface'], 'null', o['law']] for o in out]
    with open(CONFIRM_JSON, 'w') as f:
        json.dump(dict(cells=cl, selection=out), f, indent=1)
    print('confirm cells:', out, flush=True)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('part', choices=['size', 'power', 'confirm'])
    ap.add_argument('--n', type=int, default=0, help='datasets per cell (default 2,000 / 1,000 / 20,000)')
    ap.add_argument('--chunk', type=int, default=0)
    ap.add_argument('--only', nargs='*', default=None, help='restrict to design+grid tags, e.g. S5 P4')
    ap.add_argument('--workers', type=int, default=mc.workers(3))
    ap.add_argument('--tag', default='')
    a = mc.smoke_args(ap.parse_args())
    n = a.n or {'size': 2000, 'power': 1000, 'confirm': 20000}[a.part]
    chunk = a.chunk or {'size': 500, 'power': 250, 'confirm': 1000}[a.part]
    if a.part == 'confirm' and (not os.path.exists(CONFIRM_JSON)):
        pick_confirm_cells()
    C = cells(a.part)
    if a.only:
        C = [c for c in C if f'{c[0]}{c[1]}' in a.only]
    jobs = [(a.part,) + c + (s, min(chunk, n - s)) for c in C for s in range(0, n, chunk)]
    cost = {('S', 5): 5.7, ('P', 5): 4.3, ('S', 4): 3.4, ('P', 4): 2.6}
    est = sum((j[7] * cost[j[1], j[2]] * (2 if j[4] == 'null' and j[5] in MISREF and (a.part == 'size') else 1) for j in jobs)) / 1000 / a.workers
    jobs.sort(key=lambda j: -j[7] * cost[j[1], j[2]])
    print(f'{a.part}: {len(C)} cells x {n} datasets, {len(jobs)} jobs, about {est / 60:.0f} min on {a.workers} workers', flush=True)
    t0 = time.time()
    out = mc.pool_map(job, jobs, workers=a.workers, progress=100)
    res = pd.concat(out, ignore_index=True)
    for c in ('ref', 'design', 'surface', 'cond', 'law'):
        res[c] = res[c].astype('category')
    os.makedirs(mc.RESDIR, exist_ok=True)
    f = os.path.join(mc.RESDIR, f'h2h_{a.part}{a.tag}.parquet')
    res.to_parquet(f, index=False)
    print(f'wrote {f}: {len(res)} rows, {time.time() - t0:.0f} s', flush=True)
if __name__ == '__main__':
    main()
