"""sim_bigl calculations."""
import mc
import argparse
import os
import sys
import time
import numpy as np
import pandas as pd
S = mc.S
ITEM = 'fig5_bigl'
DATA = os.path.join(mc.INPUTS, 'bigl_comparison_data')
OUT = os.path.join(mc.RESDIR, 'bigl_ours.parquet')
SURFS = ['weak', 'moderate', 'strong']
NOISES = ['gauss', 'oneil']
ALTS = ['ant_broad_0.10', 'syn_broad_0.10', 'syn_highdose_0.40', 'syn_corner_0.60', 'sat_0.25']
CONDS = ['null'] + ALTS
(N_NULL, N_ALT) = (1000, 500)
SD_GAUSS = 0.08
CHUNK = 100
MGRID = np.linspace(-0.2, 1.3, 31)

def setting_name(r, surf, noise):
    return f'r{r}_{surf}_{noise}'

def load(r, surf, noise, cond):
    n = N_NULL if cond == 'null' else N_ALT
    Y = np.fromfile(os.path.join(DATA, f'{setting_name(r, surf, noise)}__{cond}.f64'), dtype='<f8')
    return Y.reshape(n, r, 5, 5)

def true_noise(noise):
    return S.ParametricNoise.additive(SD_GAUSS) if noise == 'gauss' else S.ParametricNoise.oneil()

def est_noise(Yrep):
    m = Yrep.mean(0).ravel()[1:]
    v = Yrep.var(0, ddof=1).ravel()[1:]
    A = np.column_stack([np.ones_like(m), m])
    (c, *_) = np.linalg.lstsq(A, v, rcond=None)
    vp = v[v > 0]
    (lo, hi) = (vp.min(), v.max()) if vp.size else (1e-08, 1e-08)
    s = np.sqrt(np.clip(c[0] + c[1] * MGRID, lo, hi))
    return (S.ParametricNoise.sd_table(S.SDCurve('table', kv=MGRID, ks=s)), (float(c[0]), float(c[1])))

def job(args):
    (r, surf, noise, cond, start, n) = args
    st = setting_name(r, surf, noise)
    Y = load(r, surf, noise, cond)[start:start + n]
    nz = true_noise(noise)
    design = 'R' if r > 1 else 'S'
    seed_o = mc.data_seed(ITEM, st, cond, 'true', start)
    seed_e = mc.data_seed(ITEM, st, cond, 'est', start)
    rows = []
    for t in range(Y.shape[0]):
        idx = start + t
        X = Y[t] if r > 1 else Y[t, 0]
        t0 = time.perf_counter()
        res = mc.fit_sim(X, nz, design, [seed_o, idx], normalize='ignore')
        dt = time.perf_counter() - t0
        row = dict(setting=st, r=r, surf=surf, noise=noise, cond=cond, idx=idx, p_tstd=res.p_tstd, p_scan=res.p_scan, p_mean=res.p_mean, companion=res.mean_test.get('name', ''), n_screened=len(res.log.screened), t_fit=dt)
        if r > 1:
            (ne, coef) = est_noise(Y[t])
            t0 = time.perf_counter()
            re = mc.fit_sim(X, ne, design, [seed_e, idx], normalize='ignore')
            row.update(p_tstd_est=re.p_tstd, p_scan_est=re.p_scan, t_fit_est=time.perf_counter() - t0, est_c0=coef[0], est_c1=coef[1])
        rows.append(row)
    return pd.DataFrame(rows)

def jobs(nmax=None):
    out = []
    for r in (4, 1):
        for surf in SURFS:
            for noise in NOISES:
                for cond in CONDS if r == 4 else ['null']:
                    n = N_NULL if cond == 'null' else N_ALT
                    if nmax:
                        n = min(n, nmax)
                    for s in range(0, n, CHUNK):
                        out.append((r, surf, noise, cond, s, min(CHUNK, n - s)))
    return out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--workers', type=int, default=mc.workers(2))
    ap.add_argument('--n', type=int, default=None)
    a = mc.smoke_args(ap.parse_args())
    J = jobs(a.n)
    job((4, 'weak', 'gauss', 'null', 0, 1))
    job((1, 'weak', 'gauss', 'null', 0, 1))
    t0 = time.time()
    parts = mc.pool_map(job, J, workers=a.workers, progress=10)
    df = pd.concat(parts, ignore_index=True)
    out = OUT if a.n is None or mc.paths.SMOKE else OUT.replace('.parquet', f'_smoke{a.n}.parquet')
    os.makedirs(os.path.dirname(out), exist_ok=True)
    df.to_parquet(out, index=False)
    print(f'done {len(df)} datasets in {time.time() - t0:.0f} s -> {out}', flush=True)
    g = df.groupby(['r', 'setting', 'cond'])
    print(g[['p_tstd', 'p_scan', 'p_mean']].agg(lambda x: np.mean(np.asarray(x) <= 0.05)).round(3).to_string())
EXP = str(mc.paths.LIB)
PROD_Q = os.path.join(str(mc.paths.REAL), 'screen_oneil_q.parquet')
REAL_META = os.path.join(mc.INPUTS, 'bigl_real_meta.csv')

def load_ours(path=OUT):
    return pd.read_parquet(path)

def bigl_decisions():
    if EXP not in sys.path:
        sys.path.insert(0, EXP)
    import bigl_summary as T
    b = T.load_bigl()
    d = pd.concat([b[['config', 'setting', 'cond', 'idx', 'file', 'ok']], T.bigl_decisions(b)], axis=1)
    return d

def rate(x, alpha=0.05):
    x = np.asarray(x)
    if x.dtype != bool:
        x = x <= alpha * (1 + 1e-09)
    (k, n) = (int(x.sum()), int(x.size))
    (lo, hi) = mc.wilson(k, n)
    return dict(k=k, n=n, p=k / n if n else np.nan, lo=float(lo), hi=float(hi))

def real_production():
    meta = pd.read_csv(REAL_META)
    q = pd.read_parquet(PROD_Q)
    q = q.assign(bkey=q.block_ids.astype(str))
    meta = meta.assign(bkey=meta.block_ids.astype(str).str.replace(';', ','))
    m = meta.merge(q, on='bkey', how='left', suffixes=('', '_prod'), validate='one_to_one')
    assert m.unit_id.notna().all() and (m.group_key == m.group_key_prod).all()
    assert (m.cell_line == m.cell_line_prod).all()
    return m

def kappa(x, y):
    (x, y) = (np.asarray(x, bool), np.asarray(y, bool))
    n = x.size
    po = np.mean(x == y)
    pe = x.mean() * y.mean() + (1 - x.mean()) * (1 - y.mean())
    return dict(n=n, both=int((x & y).sum()), only1=int((x & ~y).sum()), only2=int((~x & y).sum()), neither=int((~x & ~y).sum()), agree=float(po), kappa=float((po - pe) / (1 - pe)))
if __name__ == '__main__':
    main()
