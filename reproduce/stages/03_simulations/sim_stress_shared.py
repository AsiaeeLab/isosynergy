"""sim_stress_shared calculations."""
import mc
import argparse
import os
import time
import numpy as np
import pandas as pd
from sirsyn import _kernels as K
S = mc.S
D = 20
RHO = 0.1
Q = 0.05
SYN = ['syn_broad_0.10', 'syn_highdose_0.20', 'sat_0.5']
ANT = ['ant_broad_0.05', 'ant_highdose_0.10']
INF = np.inf
CONFIGS = {'null': [(INF, 'shared', bytes([82, 48]).decode('ascii'))] + [(m, sh, ref) for m in (1.0, 4.0, 8.0) for sh in ('shared', 'indep') for ref in (bytes([82, 48]).decode('ascii'), bytes([82, 49]).decode('ascii'))], 'mixed': [(INF, 'shared', bytes([82, 48]).decode('ascii')), (1.0, 'shared', bytes([82, 49]).decode('ascii')), (4.0, 'shared', bytes([82, 49]).decode('ascii')), (8.0, 'shared', bytes([82, 49]).decode('ascii'))]}
NOISE = S.ParametricNoise.oneil()
SDV = S.oneil_sd_curve()

def oneil_curves():
    B = np.load(os.path.join(mc.INPUTS, 'noise_bank_oneil.npz'))
    sel = np.where(B['group_size'] == 4)[0]
    (dA, dB, cl) = (B['drugA'], B['drugB'], B['cell_line'])
    EA = B['edge_drugA_raw'] / B['edge_drugA_raw'][:, :1]
    EB = B['edge_drugB_raw'] / B['edge_drugB_raw'][:, :1]
    cur = {}
    for s in sel:
        for (d, e) in ((dA[s], EA[s]), (dB[s], EB[s])):
            k = (str(d), str(cl[s]))
            if k not in cur:
                cur[k] = np.maximum(K.pava_dec(np.ascontiguousarray(e[1:], dtype=float)), 0.0)
    out = {}
    for c in sorted({k[1] for k in cur}):
        ds = sorted((k[0] for k in cur if k[1] == c))
        out[c] = (np.array(ds), np.array([cur[d, c] for d in ds]))
    return out
CUR = oneil_curves()
CELLS = sorted(CUR)

def apply_cond(mu, cond):
    if cond == 'null':
        return mu.copy()
    if cond.startswith('sat'):
        g = float(cond.split('_')[1])
        return np.where(mu < 1.0, np.maximum(mu, 0.0) ** (1.0 + g), mu)
    (s, kind, a) = cond.split('_')
    a = float(a)
    P = mc.sim.interaction_pattern(5, 5, kind)[1:, 1:]
    return mu - np.minimum(a * P, mu / 2.0) if s == 'syn' else mu + a * P

def rep_curve(a, u, u0, m):
    if not np.isfinite(m):
        return a.copy()
    return a + SDV(a) * (np.sqrt((1 - RHO) / m) * u + np.sqrt(RHO / m) * u0)

def setup(scen, screen):
    rd = np.random.default_rng(S.seed_for(bytes([114, 54, 115]).decode('ascii'), 't1', 'design', scen, screen))
    cl = CELLS[rd.integers(len(CELLS))]
    (names, Acur) = CUR[cl]
    di = np.sort(rd.choice(Acur.shape[0], D, replace=False))
    A = Acur[di]
    pairs = [(i, j) for i in range(D) for j in range(i + 1, D)]
    npair = len(pairs)
    cls = np.array(['null'] * npair, dtype=object)
    cond = np.array(['null'] * npair, dtype=object)
    if scen == 'mixed':
        perm = rd.permutation(npair)
        (ks, ka) = (perm[:npair // 10], perm[npair // 10:npair // 10 + npair // 5])
        (cls[ks], cls[ka]) = ('syn', 'ant')
        cond[ks] = np.array(SYN, dtype=object)[rd.integers(0, len(SYN), ks.size)]
        cond[ka] = np.array(ANT, dtype=object)[rd.integers(0, len(ANT), ka.size)]
    re = np.random.default_rng(S.seed_for(bytes([114, 54, 115]).decode('ascii'), 't1', 'edge', scen, screen))
    (U, U0) = (re.standard_normal((D, 4)), re.standard_normal(D))
    ri = np.random.default_rng(S.seed_for(bytes([114, 54, 115]).decode('ascii'), 't1', 'edge_indep', scen, screen))
    (Ui, Ui0) = (ri.standard_normal((npair, 2, 4)), ri.standard_normal((npair, 2)))
    rz = np.random.default_rng(S.seed_for(bytes([114, 54, 115]).decode('ascii'), 't1', 'data', scen, screen))
    Z = rz.standard_normal((npair, 4, 4))
    return (cl, A, pairs, cls, cond, U, U0, Ui, Ui0, Z)

def reported(A, U, U0, Ui, Ui0, i, j, k, m, share):
    if share == 'shared':
        return (rep_curve(A[i], U[i], U0[i], m), rep_curve(A[j], U[j], U0[j], m))
    return (rep_curve(A[i], Ui[k, 0], Ui0[k, 0], m), rep_curve(A[j], Ui[k, 1], Ui0[k, 1], m))

def fit(Y, m, ref, seed, two_stage):
    sa = m if ref == bytes([82, 49]).decode('ascii') and np.isfinite(m) else INF
    return mc.fit(Y, design='P', noise=NOISE, sa_replicates=sa, sa_rho=RHO, seed=seed, two_stage=two_stage, B=399, B2=9999, n_std=200, fragility='never')

def bh_count(p, q=Q):
    p = np.sort(np.asarray(p, float))
    ok = p <= q * np.arange(1, p.size + 1) / p.size
    return int(np.max(np.where(ok)[0]) + 1) if ok.any() else 0

def job(args):
    (scen, screen) = args
    t0 = time.time()
    (cl, A, pairs, cls, cond, U, U0, Ui, Ui0, Z) = setup(scen, screen)
    npair = len(pairs)
    sseed = int(mc.data_seed('figS1', 't1', scen, screen))
    Ys = {}
    rows = []
    for (k, (i, j)) in enumerate(pairs):
        mub = np.outer(A[i], A[j])
        Yint = apply_cond(mub, cond[k]) + SDV(apply_cond(mub, cond[k])) * Z[k]
        for (c, (m, share, ref)) in enumerate(CONFIGS[scen]):
            (e1, e2) = reported(A, U, U0, Ui, Ui0, i, j, k, m, share)
            Y = np.empty((5, 5))
            Y[0, 0] = 1.0
            (Y[1:, 0], Y[0, 1:]) = (e1, e2)
            Y[1:, 1:] = Yint
            Ys[k, c] = Y
            r = fit(Y, m, ref, [sseed, k], False)
            rows.append(dict(scen=scen, screen=screen, cell=cl, pair=k, i=i, j=j, cls=cls[k], cond=cond[k], m_s=float(m), share=share, ref=ref, cfg=c, p_t1=r.p_tstd, p_s1=r.p_scan, p_m=r.p_mean, p_t=np.nan, stage2=False, perr=float(np.mean((np.outer(e1, e2) - mub) / SDV(mub)))))
    df = pd.DataFrame(rows)
    fam = []
    for (c, (m, share, ref)) in enumerate(CONFIGS[scen]):
        sub = df[df.cfg == c]
        idx = sub.index.to_numpy()
        cand = sub[np.minimum(sub.p_t1, sub.p_s1) <= 0.05].sort_values(['p_t1', 'p_s1'], kind='stable')
        rec = dict(scen=scen, screen=screen, cell=cl, m_s=float(m), share=share, ref=ref, n_cand=len(cand))
        if scen == 'mixed' and (not np.isfinite(m)):
            rec.update(n_eval=0)
            fam.append(rec)
            continue
        df.loc[idx, 'p_t'] = sub.p_t1.to_numpy()
        stopped = False
        n_eval = 0
        for (q, row) in cand.iterrows():
            r = fit(Ys[int(row.pair), c], m, ref, [sseed, int(row.pair)], True)
            df.at[q, 'p_t'] = r.p_tstd
            df.at[q, 'stage2'] = r.stage == 2
            n_eval += 1
            if scen == 'null' and r.p_tstd <= Q / npair:
                stopped = True
                break
        rec['n_eval'] = n_eval
        if stopped:
            rec.update(any_disc=1, V=np.nan, lazy_stop=True)
        else:
            p = df.loc[idx, 'p_t'].to_numpy()
            V = bh_count(p)
            rec.update(any_disc=int(V > 0), V=V, lazy_stop=False)
            if scen == 'mixed':
                o = np.argsort(p, kind='stable')
                rej = np.zeros(npair, bool)
                rej[o[:V]] = True
                isn = sub.cls.to_numpy() != 'syn'
                rec.update(n_disc=int(rej.sum()), n_false=int((rej & isn).sum()), n_true=int((rej & ~isn).sum()), n_syn=int((~isn).sum()))
        fam.append(rec)
    print(f'{scen} screen {screen}: {time.time() - t0:.1f} s', flush=True)
    return (df, pd.DataFrame(fam))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--scen', default='null')
    ap.add_argument('--screens', type=int, default=400)
    ap.add_argument('--start', type=int, default=0)
    ap.add_argument('--workers', type=int, default=mc.workers(3))
    ap.add_argument('--tag', default='')
    a = mc.smoke_args(ap.parse_args())
    jobs = [(a.scen, s) for s in range(a.start, a.start + a.screens)]
    t0 = time.time()
    out = mc.pool_map(job, jobs, workers=a.workers)
    d = pd.concat([o[0] for o in out], ignore_index=True)
    f = pd.concat([o[1] for o in out], ignore_index=True)
    os.makedirs(mc.RESDIR, exist_ok=True)
    d.to_parquet(os.path.join(mc.RESDIR, f'stress_shared_{a.scen}{a.tag}.parquet'), index=False)
    f.to_csv(os.path.join(mc.RESDIR, f'stress_shared_{a.scen}_family{a.tag}.csv'), index=False)
    print(f'done {a.scen}: {len(d)} rows, {len(f)} family rows, {time.time() - t0:.0f} s', flush=True)
if __name__ == '__main__':
    main()
