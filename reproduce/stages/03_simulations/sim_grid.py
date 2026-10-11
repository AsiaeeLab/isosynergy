"""sim_grid calculations."""
import mc
import argparse
import glob
import os
import time
import numpy as np
import pandas as pd
(S, sim) = (mc.S, mc.sim)
ITEM = 'fig4_grid'
GRIDS = (4, 5, 6, 8, 10)
ALPHAS = (0.05, 0.01, 0.005)
TESTS = ('tstd', 'scan', 'mean')
CANON = {'broad': 'moderate', 'highdose': 'moderate', 'corner': 'weak', 'sat': 'strong'}
OUTDIR = os.path.join(mc.RESDIR, 'grid_chunks')
AMPS = {('S', 'broad'): (0.02, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.37, 0.45, 0.55, 0.65, 0.8, 1.0, 1.25, 1.6, 2.2, 3.0), ('S', 'highdose'): (0.1, 0.25, 0.45, 0.65, 0.85, 1.0, 1.2, 1.4, 1.6, 1.8, 2.0, 2.3, 2.6, 3.0), ('S', 'corner'): (0.15, 0.35, 0.55, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0, 2.25, 2.5, 2.75, 3.0), ('S', 'sat'): (0.02, 0.05, 0.08, 0.12, 0.16, 0.2, 0.24, 0.28, 0.33, 0.4, 0.5, 0.65, 0.85, 1.2), ('P', 'broad'): (0.02, 0.04, 0.06, 0.08, 0.1, 0.12, 0.15, 0.18, 0.22, 0.27, 0.33, 0.45), ('P', 'highdose'): (0.1, 0.2, 0.28, 0.36, 0.45, 0.55, 0.65, 0.78, 0.92, 1.08, 1.3, 1.6), ('P', 'corner'): (0.2, 0.35, 0.45, 0.55, 0.65, 0.75, 0.85, 0.95, 1.1, 1.3, 1.6, 2.0), ('P', 'sat'): (0.02, 0.035, 0.05, 0.065, 0.08, 0.1, 0.125, 0.15, 0.18, 0.22, 0.27, 0.35)}

def upos(I):
    u = np.full(I, np.nan)
    u[1:] = np.linspace(0.0, 1.0, I - 1)
    return u

def pattern(kind, I):
    u = upos(I)
    P = np.zeros((I, I))
    if kind == 'broad':
        P[1:, 1:] = 1.0
    elif kind == 'highdose':
        P[1:, 1:] = np.outer(u[1:], u[1:])
    elif kind == 'corner':
        r = np.maximum(0.0, 3.0 * (u[1:] - 2.0 / 3.0))
        P[1:, 1:] = np.outer(r, r)
    else:
        raise ValueError(kind)
    return P

def truth(kind, amp, surf, I):
    mu = sim.surface(surf, I)
    if kind == 'null' or amp == 0:
        return mu
    if kind == 'sat':
        out = mu.copy()
        out[1:, 1:] = mu[1:, 1:] ** (1.0 + amp)
        return out
    return mu * np.exp(-amp * pattern(kind, I))

def noise_law(noise):
    return S.ParametricNoise.oneil() if noise == 'oneil' else S.ParametricNoise.additive(0.08)

def effects(mu, noise):
    bl = np.outer(mu[1:, 0], mu[0, 1:])
    eta = bl - mu[1:, 1:]
    sd = noise_law(noise).sd(bl)
    z = eta / sd
    return dict(mean_eta=float(eta.mean()), max_eta=float(eta.max()), mean_z=float(z.mean()), max_z=float(z.max()))

def cells():
    out = []
    for design in ('S', 'P'):
        for I in GRIDS:
            for surf in ('weak', 'moderate', 'strong'):
                for noise in ('oneil', 'add'):
                    truths = [('null', 0.0)]
                    if noise == 'oneil':
                        for (pat, s) in CANON.items():
                            if s == surf:
                                truths += [(pat, float(a)) for a in AMPS[design, pat]]
                    n_null = 4000 if design == 'S' and surf == 'weak' else 2000
                    out.append((design, I, surf, noise, truths, n_null))
    return out
COST_MS = {4: 4.5, 5: 7.0, 6: 13.0, 8: 27.0, 10: 48.0}

def jobs_for(n_power, chunk, grids):
    J = []
    for (design, I, surf, noise, truths, n_null) in cells():
        if I not in grids:
            continue
        n_null_eff = n_null if n_power >= 2000 else n_power * n_null // 2000
        for start in range(0, n_null_eff, chunk):
            n = min(chunk, n_null_eff - start)
            tr = truths if start < n_power else [('null', 0.0)]
            J.append((design, I, surf, noise, tuple(tr), start, n))
    J.sort(key=lambda j: -len(j[4]) * j[6] * COST_MS[j[1]])
    return J

def run_job(job):
    (design, I, surf, noise, truths, start, n) = job
    tag = os.environ.get('GRID_TAG', 'main')
    out = os.path.join(OUTDIR, tag, f'{design}_{I}_{surf}_{noise}_{start:05d}.parquet')
    if os.path.exists(out):
        return (out, 0.0, 0)
    t0 = time.time()
    nz = noise_law(noise)
    cseed = mc.data_seed(ITEM, design, I, surf, noise, start)
    recs = []
    for (kind, amp) in truths:
        mu = truth(kind, amp, surf, I)
        rng = np.random.default_rng(cseed)
        Ys = sim.simulate(mu, nz, design, n, rng)
        P = np.empty((n, 3))
        for t in range(n):
            r = mc.fit_sim(Ys[t], nz, design, [cseed, t])
            P[t] = (r.p_tstd, r.p_scan, r.p_mean)
        recs.append(pd.DataFrame(dict(design=design, I=I, surface=surf, noise=noise, pattern=kind, amp=amp, dataset=np.arange(start, start + n), p_tstd=P[:, 0].astype(np.float32), p_scan=P[:, 1].astype(np.float32), p_mean=P[:, 2])))
    df = pd.concat(recs, ignore_index=True)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    df.to_parquet(out + '.tmp', index=False)
    os.replace(out + '.tmp', out)
    secs = time.time() - t0
    print(f'{os.path.basename(out)}: {len(truths)} truths x {n} datasets, {secs:.0f} s ({1000 * secs / (len(truths) * n):.1f} ms/fit)', flush=True)
    return (out, secs, len(truths) * n)

def pava_inc(y):
    (vals, cnts) = ([], [])
    for v in map(float, y):
        vals.append(v)
        cnts.append(1)
        while len(vals) > 1 and vals[-2] > vals[-1]:
            (v2, c2) = (vals.pop(), cnts.pop())
            (v1, c1) = (vals.pop(), cnts.pop())
            vals.append((v1 * c1 + v2 * c2) / (c1 + c2))
            cnts.append(c1 + c2)
    out = []
    for (v, c) in zip(vals, cnts):
        out += [v] * c
    return np.array(out)

def cross(amps, y, target=0.8):
    y = pava_inc(y)
    if y[0] >= target:
        return amps[0]
    for i in range(1, len(amps)):
        if y[i] >= target > y[i - 1]:
            return amps[i - 1] + (target - y[i - 1]) * (amps[i] - amps[i - 1]) / (y[i] - y[i - 1])
    return np.nan

def counts(pv):
    keys = ['design', 'I', 'surface', 'noise', 'pattern', 'amp']
    recs = []
    for (key, g) in pv.groupby(keys, sort=True):
        rec = dict(zip(keys, key))
        if rec['pattern'] == 'null':
            rec['n_all'] = len(g)
            g = g[g.dataset < 2000]
        rec['n'] = len(g)
        rec.update(effects(truth(rec['pattern'], rec['amp'], rec['surface'], int(rec['I'])), rec['noise']))
        for t in TESTS:
            p = g[f'p_{t}'].to_numpy(float)
            for a in ALPHAS:
                k = int(np.sum(p <= a * (1 + 1e-09)))
                (lo, hi) = mc.wilson(k, len(p))
                rec[f'k_{t}@{a}'] = k
                rec[f'{t}@{a}'] = k / len(p)
                rec[f'{t}@{a}_lo'] = float(lo)
                rec[f'{t}@{a}_hi'] = float(hi)
        recs.append(rec)
    return pd.DataFrame(recs)

def effect80(cnt):
    recs = []
    base = ['design', 'I', 'surface', 'noise']
    for (key, g) in cnt[cnt.noise == 'oneil'].groupby(base):
        kv = dict(zip(base, key))
        null = g[g.pattern == 'null']
        for pat in sorted(set(g.pattern) - {'null'}):
            gg = pd.concat([null, g[g.pattern == pat]]).sort_values('amp')
            amps = gg.amp.to_numpy(float)
            for t in TESTS:
                for a in ALPHAS:
                    y = gg[f'{t}@{a}'].to_numpy()
                    a80 = cross(amps, y)
                    alo = cross(amps, gg[f'{t}@{a}_hi'].to_numpy())
                    ahi = cross(amps, gg[f'{t}@{a}_lo'].to_numpy())
                    rec = dict(kv, pattern=pat, test=t, alpha=a, a80=a80, a80_lo=alo, a80_hi=ahi, reached=bool(np.isfinite(a80)), power_at_max=float(y[-1]), amp_max=float(amps[-1]))
                    for (lab, aa) in (('', a80), ('_lo', alo), ('_hi', ahi)):
                        e = effects(truth(pat, aa, kv['surface'], int(kv['I'])), kv['noise']) if np.isfinite(aa) else dict(mean_eta=np.nan, max_eta=np.nan, mean_z=np.nan, max_z=np.nan)
                        for k in ('mean_eta', 'max_eta', 'mean_z', 'max_z'):
                            rec[f'{k}80{lab}'] = e[k]
                    emax = effects(truth(pat, amps[-1], kv['surface'], int(kv['I'])), kv['noise'])
                    (rec['mean_eta_at_max'], rec['max_eta_at_max']) = (emax['mean_eta'], emax['max_eta'])
                    recs.append(rec)
    return pd.DataFrame(recs)

def size_table(pv):
    nul = pv[pv.pattern == 'null']
    recs = []
    for (key, g) in nul.groupby(['design', 'I', 'surface', 'noise']):
        rec = dict(zip(['design', 'I', 'surface', 'noise'], key), n=len(g))
        for t in TESTS:
            p = g[f'p_{t}'].to_numpy(float)
            for a in ALPHAS:
                k = int(np.sum(p <= a * (1 + 1e-09)))
                (lo, hi) = mc.wilson(k, len(p))
                (rec[f'k_{t}@{a}'], rec[f'{t}@{a}'], rec[f'{t}@{a}_lo'], rec[f'{t}@{a}_hi']) = (k, k / len(p), lo, hi)
        recs.append(rec)
    return pd.DataFrame(recs)

def collect(tag):
    files = sorted(glob.glob(os.path.join(OUTDIR, tag, '*.parquet')))
    pv = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    sfx = '' if tag == 'main' else f'_{tag}'
    pv.to_parquet(os.path.join(mc.RESDIR, f'grid_pvalues{sfx}.parquet'), index=False)
    cnt = counts(pv)
    cnt.to_csv(os.path.join(mc.RESDIR, f'grid_counts{sfx}.csv'), index=False)
    e80 = effect80(cnt)
    e80.to_csv(os.path.join(mc.RESDIR, f'grid_effect80{sfx}.csv'), index=False)
    sz = size_table(pv)
    sz.to_csv(os.path.join(mc.RESDIR, f'grid_size{sfx}.csv'), index=False)
    print(f'collected {len(files)} chunk files, {len(pv)} analyses; {len(cnt)} truth rows; {len(e80)} effect80 rows; {len(sz)} null cells')

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('what', choices=('run', 'collect', 'plan'))
    ap.add_argument('--workers', type=int, default=mc.workers(6))
    ap.add_argument('--n', type=int, default=2000)
    ap.add_argument('--chunk', type=int, default=250)
    ap.add_argument('--grids', type=int, nargs='+', default=[10, 8, 6, 5, 4])
    ap.add_argument('--tag', default='main')
    a = mc.smoke_args(ap.parse_args())
    os.environ['GRID_TAG'] = a.tag
    if a.what == 'collect':
        collect(a.tag)
        return
    J = jobs_for(a.n, a.chunk, set(a.grids))
    nfit = sum((len(j[4]) * j[6] for j in J))
    est = sum((len(j[4]) * j[6] * COST_MS[j[1]] for j in J)) / 1000
    print(f'{len(J)} jobs, {nfit:,} fits; estimated {est / 3600:.2f} CPU h, {est / a.workers / 60:.0f} min on {a.workers} workers', flush=True)
    if a.what == 'plan':
        return
    t0 = time.time()
    res = mc.pool_map(run_job, J, workers=a.workers)
    tot = sum((r[1] for r in res))
    print(f'done: {len(J)} jobs, {time.time() - t0:.0f} s wall, {tot:.0f} s summed', flush=True)
if __name__ == '__main__':
    main()
