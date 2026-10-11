"""sim_h2h_realpower calculations."""
import mc
import argparse
import os
import sys
import time
import numpy as np
import pandas as pd
sys.path.insert(0, mc.FINAL)
import common as C
import null_checks as N
(S, sim) = (mc.S, mc.sim)
G = N.G
CONDS = {'oneil': ['null', 'syn_broad_0.05', 'syn_broad_0.10', 'syn_highdose_0.10', 'syn_highdose_0.20', 'sat_0.5', 'sat_1.0'], 'almanac': ['null', 'syn_broad_0.05', 'syn_broad_0.10', 'syn_highdose_0.10', 'syn_highdose_0.20', 'sat_0.5', 'sat_1.0']}

def apply_cond(mu, cond):
    if cond == 'null':
        return mu.copy()
    if cond.startswith('sat'):
        g = float(cond.split('_')[1])
        return np.where(mu < 1.0, np.maximum(mu, 0.0) ** (1.0 + g), mu)
    (s, kind, a) = cond.split('_')
    a = float(a)
    I = mu.shape[0] + 1
    P = sim.interaction_pattern(I, I, kind)[1:, 1:]
    if s == 'syn':
        return mu - np.minimum(a * P, mu / 2.0)
    return mu + a * P

def row_of(res):
    return dict(p_tstd=res.p_tstd, p_tstd_stage1=res.p_tstd_stage1, p_scan=res.p_scan, p_companion=res.p_mean, p_mean_boot=res.p_mean_boot, stage=res.stage, n_claims=len(res.claims), n_screened=len(res.log.screened))

def job_oneil(args):
    (conds, tis) = args
    if isinstance(conds, str):
        conds = (conds,)
    kw = {k: v for (k, v) in C.FIT_ONEIL.items() if k != 'design'}
    rows = []
    for ti in tis:
        s = G['SEL4'][ti]
        (eA, eB) = (G['EA'][s], G['EB'][s])
        mu0 = np.outer(eA[1:], eB[1:])
        (dd, dh) = N.donors4(ti, mu0)
        for cond in conds:
            rows.append(_fit_oneil(ti, cond, eA, eB, mu0, dd, kw))
    return pd.DataFrame(rows)

def _fit_oneil(ti, cond, eA, eB, mu0, dd, kw):
    fs = mc.data_seed('fig3_realpower', 'oneil', cond)
    mu = apply_cond(mu0, cond)
    Y = np.empty((4, 5, 5))
    Y[:, :, 0] = eA[None]
    Y[:, 0, :] = eB[None]
    Y[:, 0, 0] = 1.0
    sdm = G['sd'](mu)
    for r in range(4):
        Y[r, 1:, 1:] = mu + G['RESID'][dd[r], r] * np.clip(sdm / G['SDY'][dd[r]], 0.5, 2.0)
    grp = str(G['CELL'][dd[0]])
    res = mc.fit(Y, design='P', noise=G['bank'], plate_var=G['pvt'], group=grp, seed=[fs, int(ti)], fragility='never', **kw)
    eta = mu0 - mu
    row = dict(study='oneil', cond=cond, target=int(ti), mean_bliss=float(mu0.mean()), eff_mean=float(eta.mean()), eff_max=float(eta.max()))
    row.update(row_of(res))
    return row

def job_almanac(args):
    (cond, ts) = args
    fs = mc.data_seed('fig3_realpower', 'almanac', cond)
    kw = {k: v for (k, v) in C.FIT_ALMANAC.items() if k != 'design'}
    NP = G['NP']
    rows = []
    for t in ts:
        p = t % NP
        sg = 1.0 if t < NP else -1.0
        mu0 = np.outer(G['EA'][p][1:], G['EB'][p][1:])
        mu = apply_cond(mu0, cond)
        T = np.empty((4, 4))
        T[:, 0] = G['EA'][p]
        T[0, :] = G['EB'][p]
        T[1:, 1:] = mu
        Y = T + sg * G['N'][p]
        Y[0, 0] = 1.0
        res = mc.fit(Y, design='S', noise=G['bank'], group=str(G['CELL'][p]), seed=[fs, int(t)], fragility='never', **kw)
        eta = mu0 - mu
        row = dict(study='almanac', cond=cond, target=int(t), mean_bliss=float(mu0.mean()), eff_mean=float(eta.mean()), eff_max=float(eta.max()))
        row.update(row_of(res))
        rows.append(row)
    return pd.DataFrame(rows)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--study', required=True, choices=['oneil', 'almanac'])
    ap.add_argument('--n', type=int, default=0, help='targets per condition (0 = all)')
    ap.add_argument('--conds', nargs='*', default=None)
    ap.add_argument('--workers', type=int, default=mc.workers(3))
    ap.add_argument('--chunk', type=int, default=50)
    ap.add_argument('--tag', default='')
    ap.add_argument('--interleave', action='store_true')
    ap.add_argument('--resume', action='store_true', help='interleaved runs: continue from the checkpoint file')
    ap.add_argument('--batch', type=int, default=45, help='jobs per checkpoint (interleaved runs)')
    a = mc.smoke_args(ap.parse_args())
    if a.study == 'oneil':
        N.init_oneil()
        (NT, job) = (G['SEL4'].size, job_oneil)
    else:
        N.init_almanac()
        (NT, job) = (2 * G['NP'], job_almanac)
    idx = np.arange(NT)
    if a.n and a.n < NT:
        rng = np.random.default_rng(mc.data_seed('fig3_realpower', a.study, 'subset'))
        idx = np.sort(rng.choice(idx, a.n, replace=False))
    conds = a.conds or CONDS[a.study]
    os.makedirs(mc.RESDIR, exist_ok=True)
    f = os.path.join(mc.RESDIR, f'h2h_realpower_{a.study}{a.tag}.parquet')
    if a.interleave:
        rng = np.random.default_rng(mc.data_seed('fig3_realpower', a.study, 'order'))
        idx = rng.permutation(idx)
        jobs = [(tuple(conds), idx[s:s + a.chunk]) for s in range(0, idx.size, a.chunk)]
        out = []
        if a.resume and os.path.exists(f):
            prev = pd.read_parquet(f)
            done = set(prev.target[prev.groupby('target').cond.transform('nunique') == len(conds)])
            keep = []
            for j in jobs:
                d = [int(t) in done for t in j[1]]
                assert all(d) or not any(d), 'checkpoint ends inside a job'
                if not any(d):
                    keep.append(j)
            out = [prev[prev.target.isin(done)]]
            print(f'resume: {len(done)} targets done, {len(keep)} of {len(jobs)} jobs left', flush=True)
            jobs = keep
        print(f'{a.study}: {idx.size} targets x {len(conds)} conditions interleaved, {len(jobs)} jobs, {a.workers} workers, checkpoint every {a.batch} jobs', flush=True)
        t0 = time.time()
        res = pd.concat(out, ignore_index=True) if out else None
        for b in range(0, len(jobs), a.batch):
            out += mc.pool_map(job, jobs[b:b + a.batch], workers=a.workers)
            res = pd.concat(out, ignore_index=True)
            res.to_parquet(f + '.tmp', index=False)
            os.replace(f + '.tmp', f)
            print(f'  checkpoint: {res.target.nunique()} targets, {len(res)} rows, {time.time() - t0:.0f} s', flush=True)
        print(f'wrote {f}: {len(res)} rows, {time.time() - t0:.0f} s', flush=True)
        return
    jobs = [(c, idx[s:s + a.chunk]) for c in conds for s in range(0, idx.size, a.chunk)]
    print(f'{a.study}: {idx.size} targets x {len(conds)} conditions, {len(jobs)} jobs, {a.workers} workers', flush=True)
    t0 = time.time()
    out = mc.pool_map(job, jobs, workers=a.workers, progress=20)
    res = pd.concat(out, ignore_index=True)
    res.to_parquet(f, index=False)
    print(f'wrote {f}: {len(res)} rows, {time.time() - t0:.0f} s', flush=True)
    for (c, g) in res.groupby('cond', sort=False):
        print(c, f'n={len(g)} eff={g.eff_mean.mean():.3f}', ' '.join((f'{k}={np.mean(g[k] <= 0.05):.3f}' for k in ('p_tstd', 'p_scan', 'p_companion'))), f'stage2={np.mean(g.stage == 2):.2f}', flush=True)
if __name__ == '__main__':
    main()
