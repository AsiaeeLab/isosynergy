"""h2h_lib calculations."""
import os
import numpy as np
import pandas as pd
import mc
TESTS = ('tstd', 'scan', 'mean')
LAW_NAME = {'A': 'additive Gaussian', 'B': 'log-normal', 'C': "O'Neil SD curve", 'D': 't4'}
NULL_CSV = os.path.join(str(mc.paths.REAL), 'null_summary.csv')

def load(part):
    df = pd.read_parquet(os.path.join(mc.RESDIR, f'h2h_{part}.parquet'))
    for c in ('ref', 'design', 'surface', 'cond', 'law'):
        if c in df:
            df[c] = df[c].astype(str)
    return df

def rates(df, keys, alpha=0.05):
    rows = []
    by = keys if len(keys) > 1 else keys[0]
    for (key, g) in df.groupby(by, sort=False, observed=True):
        r = dict(zip(keys, key if isinstance(key, tuple) else (key,)))
        r['n'] = len(g)
        for t in TESTS:
            k = int(np.sum(np.nan_to_num(g[f'p_{t}'].to_numpy(float), nan=1.0) <= alpha * (1 + 1e-06)))
            r[f'k_{t}'] = k
            r[t] = k / len(g)
        if 'eff_mean' in g:
            r['eff_mean'] = float(g.eff_mean.iloc[0])
            r['eff_max'] = float(g.eff_max.iloc[0])
        rows.append(r)
    return pd.DataFrame(rows)

def worst(R, test, mask=None):
    g = R if mask is None else R[mask]
    i = g[test].idxmax()
    r = g.loc[i]
    (lo, hi) = mc.wilson(r[f'k_{test}'], r['n'])
    return dict(rate=float(r[test]), lo=float(lo), hi=float(hi), k=int(r[f'k_{test}']), n=int(r['n']), cell=' '.join((f'{k}={r[k]}' for k in ('design', 'I', 'surface', 'law', 'cond') if k in r)))

def null_checks():
    d = pd.read_csv(NULL_CSV)
    return d

def fr(x, nd=3):
    from decimal import Decimal, ROUND_HALF_UP
    return str(Decimal(repr(round(float(x), 12))).quantize(Decimal(1).scaleb(-nd), rounding=ROUND_HALF_UP))

def fci(w, nd=3):
    return f"{fr(w['rate'], nd)} [{fr(w['lo'], nd)}, {fr(w['hi'], nd)}]"
