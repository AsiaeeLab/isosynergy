"""Paired effect thresholds for the two reference sizes in Supplement S8."""
import paths
import numpy as np
import pandas as pd
import resampling_core as G
TESTS3 = ('tstd','scan','mean')
def pava_inc(y):
    y = list(map(float, y))
    (vals, cnts) = ([], [])
    for v in y:
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
def rates(df, tests=('tstd', 'scan', 'mean', 'comb')):
    out = df.copy()
    for t in tests:
        for a in G.ALPHAS:
            k = out[f'k_{t}@{a}'].to_numpy()
            n = out['n'].to_numpy()
            (lo, hi) = G.wilson(k, n)
            out[f'{t}@{a}'] = k / n
            out[f'{t}@{a}_lo'] = lo
            out[f'{t}@{a}_hi'] = hi
    return out
def effect80(cur, keys):
    recs = []
    for (key, g) in cur.groupby(keys):
        kv = dict(zip(keys, key))
        pat = kv['pattern']
        if pat == 'null':
            continue
        null = cur
        for k in keys:
            if k != 'pattern':
                null = null[null[k] == kv[k]]
        null = null[null.pattern == 'null']
        gg = pd.concat([null, g]).sort_values('amp')
        amps = gg.amp.to_numpy()
        for t in TESTS3 + ('comb',):
            for a in G.ALPHAS:
                y = gg[f'{t}@{a}'].to_numpy()
                a80 = cross(amps, y)
                alo = cross(amps, gg[f'{t}@{a}_hi'].to_numpy())
                ahi = cross(amps, gg[f'{t}@{a}_lo'].to_numpy())
                rec = dict(kv, test=t, alpha=a, a80=a80, a80_lo=alo, a80_hi=ahi, reached=bool(np.isfinite(a80)), power_at_max=float(y[-1]), amp_max=float(amps[-1]))
                for (lab, aa) in (('', a80), ('_lo', alo), ('_hi', ahi)):
                    if np.isfinite(aa):
                        e = G.effects(G.truth(pat, aa, kv['surface'], int(kv['I'])), kv['noise'])
                    else:
                        e = {k: np.nan for k in ('mean_eta', 'max_eta', 'mean_z', 'max_z', 'n_syn', 'min_mu')}
                    for k in ('mean_eta', 'max_eta', 'mean_z', 'max_z'):
                        rec[k + '80' + lab] = e[k]
                    if lab == '':
                        rec['min_mu80'] = e['min_mu']
                recs.append(rec)
    return pd.DataFrame(recs)
counts=rates(pd.read_csv(paths.SIMS/'resampling_counts_bcheck.csv',keep_default_na=False))
e=effect80(counts,['design','I','surface','noise','pattern','B'])
e.to_csv(paths.SIMS/'resampling_effect80.csv',index=False)
