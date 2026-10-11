"""Collect Berger–Boos size and power counts for Supplementary Figure S2."""
import os
import numpy as np
import pandas as pd
import bb_core as K
RES = K.RES
ALPHA = 0.05

def w(k, n, d=3):
    if n == 0:
        return 'n/a'
    (lo, hi) = K.wilson(k, n)
    return f'{k / n:.{d}f} [{lo:.{d}f}, {hi:.{d}f}]'

def md(df):
    cols = list(df.columns)
    out = ['| ' + ' | '.join(map(str, cols)) + ' |', '|' + '---|' * len(cols)]
    for r in df.itertuples(index=False):
        out.append('| ' + ' | '.join(map(str, r)) + ' |')
    return '\n'.join(out)

def oc_tables(sr):
    oc = pd.read_parquet(os.path.join(RES, 'bb_operating.parquet'))
    B = 39 if K.paths.SMOKE else 999
    sr = sr.copy()
    sr['p_BB_search'] = np.minimum(1.0, (1 + sr.c_search) / (B + 1) + 0.005)
    rows = []
    for ((noise, base, ms, cond), s) in oc.groupby(['noise', 'base', 'ms', 'cond'], sort=False):
        n = len(s)
        rec = dict(noise=noise, base=base, ms=ms, cond=cond, amp=s.amp.iloc[0], n=n, row=s.row.iloc[0], band_empty=int((~s.band_ok).sum()))
        for a in (0.05, 0.01):
            for (m, col) in [('oracle', 'p_oracle'), ('plugin', 'p_plugin'), ('BB_env', 'p_BB_env'), ('cond', 'p_cond')]:
                rec[f'k_{m}_{a}'] = int((s[col] <= a + 1e-12).sum())
            env_rej = s[s.p_BB_env <= a + 1e-12].k.to_numpy()
            ss = sr[(sr.noise == noise) & (sr.base == base) & (sr.ms == ms) & (sr.cond == cond) & sr.k.isin(env_rej)]
            n_s = len(ss)
            kept = int((ss.p_BB_search <= a + 1e-12).sum())
            rec[f'n_env_rej_{a}'] = len(env_rej)
            rec[f'n_env_rej_searched_{a}'] = n_s
            rec[f'n_env_rej_kept_{a}'] = kept
            if n_s == len(env_rej):
                rec[f'k_BB_search_{a}'] = kept
                rec[f'BB_search_exact_{a}'] = True
            else:
                rec[f'k_BB_search_{a}'] = len(env_rej) * kept / max(n_s, 1)
                rec[f'BB_search_exact_{a}'] = False
        rec['median_width'] = float(np.nanmedian(0.5 * (s.width_A + s.width_B)))
        rows.append(rec)
    t = pd.DataFrame(rows)
    t.to_csv(os.path.join(RES, 'bb_table_rows.csv'), index=False)

    def fmt(t, a=0.05):
        out = t[['noise', 'base', 'ms', 'cond', 'amp', 'n']].copy()
        for m in ['oracle', 'plugin', 'BB_env', 'BB_search', 'cond']:
            vals = []
            for r in t.itertuples():
                k = t.loc[r.Index, f'k_{m}_{a}']
                if m == 'BB_search' and (not t.loc[r.Index, f'BB_search_exact_{a}']):
                    ne = t.loc[r.Index, f'n_env_rej_{a}']
                    ns = t.loc[r.Index, f'n_env_rej_searched_{a}']
                    kp = t.loc[r.Index, f'n_env_rej_kept_{a}']
                    (lo, hi) = K.wilson(kp, ns)
                    vals.append(f'{k / r.n:.3f} [{ne * lo / r.n:.3f}, {ne * hi / r.n:.3f}]*')
                else:
                    vals.append(w(int(round(k)), r.n))
            out['conditional' if m == 'cond' else m] = vals
        return out
    size = fmt(t[t.row == 'size'])
    power = fmt(t[t.row == 'power'])
    size.to_csv(os.path.join(RES, 'bb_table_size.csv'), index=False)
    power.to_csv(os.path.join(RES, 'bb_table_power.csv'), index=False)
    print(md(size))
    print()
    print(md(power))
    tt = t.copy()
    M = ['oracle', 'plugin', 'BB_env', 'BB_search', 'conditional']
    for m in M:
        tt[m] = tt[f"k_{('cond' if m == 'conditional' else m)}_0.05"] / tt.n
    print(tt[tt.row == 'size'].groupby(['cond', 'noise'])[M].agg(['min', 'max']).round(4))
    print(tt[tt.row == 'power'].groupby(['ms', 'noise'])[M].mean().round(3))
    print(tt[tt.row == 'power'].groupby(['ms'])[M].mean().round(3))
    pw = tt[tt.row == 'power']
    print('power ratio to oracle (mean over rows with oracle >= 0.3):')
    q = pw[pw.oracle >= 0.3]
    print(q.groupby('ms').apply(lambda g: pd.Series({m: (g[m] / g.oracle).mean() for m in M})).round(3))
    return t

def ext_tables():
    e = pd.read_parquet(os.path.join(RES, 'bb_operating_ext.parquet'))
    se = pd.read_parquet(os.path.join(RES, 'bb_search_ext.parquet')) if os.path.exists(os.path.join(RES, 'bb_search_ext.parquet')) else None
    rows = []
    for ((noise, base, ms, cond), s) in e.groupby(['noise', 'base', 'ms', 'cond'], sort=False):
        n = len(s)
        rec = dict(noise=noise, base=base, ms=ms, cond=cond, amp=s.amp.iloc[0], n=n, row=s.row.iloc[0])
        for (m, col) in [('oracle', 'p_oracle'), ('plugin', 'p_plugin'), ('BB_env', 'p_BB_env'), ('cond', 'p_cond')]:
            rec[f'k_{m}_0.05'] = int((s[col] <= ALPHA + 1e-12).sum())
        rec['k_BB_search_0.05'] = np.nan
        if se is not None and ms == 64:
            env_rej = s[s.p_BB_env <= ALPHA + 1e-12].k.to_numpy()
            ss = se[(se.noise == noise) & (se.base == base) & (se.ms == ms) & (se.cond == cond) & se.k.isin(env_rej)]
            kept = int((np.minimum(1.0, (1 + ss.c_search) / (40 if K.paths.SMOKE else 1000) + 0.005) <= ALPHA + 1e-12).sum())
            (rec['n_env_rej_0.05'], rec['n_env_rej_searched_0.05'], rec['n_env_rej_kept_0.05']) = (len(env_rej), len(ss), kept)
            rec['k_BB_search_0.05'] = len(env_rej) * kept / max(len(ss), 1)
            rec['BB_search_exact_0.05'] = len(ss) == len(env_rej)
        rows.append(rec)
    t = pd.DataFrame(rows)
    t.to_csv(os.path.join(RES, 'bb_table_rows_ext.csv'), index=False)
    out = t[['noise', 'base', 'ms', 'cond', 'amp', 'n']].copy()
    for m in ['oracle', 'plugin', 'BB_env', 'BB_search', 'cond']:
        vals = []
        for r in t.itertuples():
            k = t.loc[r.Index, f'k_{m}_0.05']
            if m == 'BB_search':
                if np.isnan(k):
                    vals.append('not run')
                elif t.loc[r.Index, 'BB_search_exact_0.05']:
                    vals.append(w(int(round(k)), r.n))
                else:
                    (ne, ns, kp) = (t.loc[r.Index, c] for c in ['n_env_rej_0.05', 'n_env_rej_searched_0.05', 'n_env_rej_kept_0.05'])
                    (lo, hi) = K.wilson(kp, ns)
                    vals.append(f'{k / r.n:.3f} [{ne * lo / r.n:.3f}, {ne * hi / r.n:.3f}]*')
            else:
                vals.append(w(int(k), r.n))
        out['conditional' if m == 'cond' else m] = vals
    out.to_csv(os.path.join(RES, 'bb_table_ext.csv'), index=False)
    print(md(out))
    M = ['oracle', 'plugin', 'BB_env', 'BB_search', 'conditional']
    tt = t.copy()
    for m in M:
        tt[m] = tt[f"k_{('cond' if m == 'conditional' else m)}_0.05"] / tt.n
    print(tt[tt.cond == 'bliss'].groupby('ms')[M].agg(['min', 'max']).round(4))
    pw = tt[tt.row == 'power']
    print(pw.groupby('ms')[M].mean().round(3))
    q = pw[pw.oracle >= 0.3]
    print(q.groupby('ms').apply(lambda g: pd.Series({m: (g[m] / g.oracle).mean() for m in M})).round(3))
if __name__ == '__main__':
    sr = pd.read_parquet(os.path.join(RES, 'bb_search.parquet'))
    oc_tables(sr)
    ext_tables()
