"""sim_screen_bank calculations."""
import argparse
import os
import sys
import time
import mc
import numpy as np
import pandas as pd
sys.path.insert(0, mc.FINAL)
import common as C
import null_checks as N1
(S, sim) = (mc.S, mc.sim)
MAGS = np.load(os.path.join(mc.INPUTS, 'gross_errors.npz'))['absdev']
LEVELS = {'natural': 0.0, 'add0.01': 0.01, 'add0.02': 0.02}
G = N1.G

def build(ti):
    s = G['SEL4'][ti]
    (eA, eB) = (G['EA'][s], G['EB'][s])
    mu = np.outer(eA[1:], eB[1:])
    (dd, dh) = N1.donors4(ti, mu)
    Y = np.empty((4, 5, 5))
    Y[:, :, 0] = eA[None]
    Y[:, 0, :] = eB[None]
    Y[:, 0, 0] = 1.0
    sdm = G['sd'](mu)
    for r in range(4):
        Y[r, 1:, 1:] = mu + G['RESID'][dd[r], r] * np.clip(sdm / G['SDY'][dd[r]], 0.5, 2.0)
    return (Y, str(G['CELL'][dd[0]]))

def fit(Y, grp, ti, screen):
    kw = {k: v for (k, v) in C.FIT_ONEIL.items() if k not in ('design', 'screen_replicates')}
    return mc.fit(Y, design='P', noise=G['bank'], plate_var=G['pvt'], group=grp, seed=(C.SEED_TAG, 'null', 'oneil', int(ti)), screen_replicates=screen, **kw)

def job(args):
    (tis, arms) = args
    rows = []
    for ti in tis:
        (Y, grp) = build(ti)
        cs = mc.data_seed('fig6_bank', 'cont', int(ti))
        for (lev, p) in LEVELS.items():
            (Rc, cm) = sim.contaminate(Y[None], np.random.default_rng(cs), p, MAGS, edges=False)
            for arm in ('noscreen', 'screen'):
                if (lev, arm) not in arms:
                    continue
                r = fit(Rc[0], grp, ti, arm == 'screen')
                rows.append(dict(target=int(ti), level=lev, arm=arm, p=r.p_tstd, p_stage1=r.p_tstd_stage1, stage=r.stage, n_cont=int(cm[0].sum()), n_screened=len(r.log.screened), ref_flags=float(r.log.reference_flags_per_plate or 0.0)))
    return pd.DataFrame(rows)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--n', type=int, default=0)
    ap.add_argument('--workers', type=int, default=3)
    ap.add_argument('--verify', type=int, default=0, help='reproduce the final null check on N targets and exit')
    a = mc.smoke_args(ap.parse_args())
    N1.init_oneil()
    null = pd.read_parquet(os.path.join(C.RES, 'null_oneil.parquet')).set_index('target')
    N = G['SEL4'].size
    if a.verify:
        idx = np.sort(np.random.default_rng(1).choice(N, a.verify, replace=False))
        d = job((idx, {('natural', 'screen')}))
        diff = np.abs(d.p.values - null.loc[d.target, 'p_tstd'].values).max()
        print(f'verify: {len(d)} targets, max |p - final null check p| = {diff:.3g}')
        return
    idx = np.arange(N)
    arms = {(lv, arm) for lv in LEVELS for arm in ('noscreen', 'screen')}
    if a.n:
        idx = np.sort(np.random.default_rng(mc.data_seed('fig6_bank', 'subset')).choice(idx, a.n, replace=False))
    else:
        arms.discard(('natural', 'screen'))
    chunks = [(c, arms) for c in np.array_split(idx, max(1, idx.size // 40))]
    print(f'bank: {idx.size} targets, {len(arms)} arms, {len(chunks)} chunks', flush=True)
    t0 = time.time()
    out = mc.pool_map(job, chunks, workers=mc.workers(a.workers), progress=50)
    res = pd.concat(out, ignore_index=True)
    if not a.n:
        nat = null.reset_index()
        res = pd.concat([res, pd.DataFrame(dict(target=nat.target, level='natural', arm='screen', p=nat.p_tstd, p_stage1=nat.p_tstd_stage1, stage=nat.stage, n_cont=0, n_screened=nat.n_screened, ref_flags=nat.reference_flags_per_plate))], ignore_index=True)
    res['p'] = res['p'].astype('float64')
    tag = f'_n{a.n}' if a.n and not mc.paths.SMOKE else ''
    path = os.path.join(mc.RESDIR, f'fig6_screen_bank{tag}.parquet')
    res.to_parquet(path, index=False)
    print(f'done: {len(res)} rows, {time.time() - t0:.0f} s -> {path}', flush=True)
if __name__ == '__main__':
    main()
