"""DrugCombDB missing calculations."""
import json
import os
import time
from multiprocessing import Pool
import numpy as np
import pandas as pd
import common as C
from sirsyn import _kernels as K
from sirsyn.geometry import order_pairs
G = {}

def init():
    (meta, Y, DA, DB) = C.load_oneil()
    Ym = Y.mean(1)
    Ym = Ym / Ym[:, :1, :1]
    G['Ym'] = Ym
    G['meta'] = meta
    I = J = 5
    G['pairs'] = {}
    for i in range(1, I):
        for j in range(1, J):
            obs = np.ones(I * J, bool)
            obs[i * J + j] = False
            G['pairs'][i, j] = (obs,) + order_pairs(I, J, obs)

def job(idx):
    rows = []
    I = J = 5
    for s in idx:
        Y = np.ascontiguousarray(G['Ym'][s])
        e = {'iso': [], 'bliss': [], 'iso_clip': [], 'corner': []}
        for ((i, j), (obs, pa, pb)) in G['pairs'].items():
            (a, b, th) = K.iso_fit(Y, obs, pa, pb)
            T = th.reshape(I, J)
            lowdose = [T[p, q] for p in range(0, i + 1) for q in range(0, j + 1) if (p, q) != (i, j)]
            highdose = [T[p, q] for p in range(i, I) for q in range(j, J) if (p, q) != (i, j)]
            U = min(lowdose)
            bl = a[i - 1] * b[j - 1]
            if highdose:
                L = max(highdose)
                iso = 0.5 * (L + U)
                ic = min(max(bl, L), U)
            else:
                iso = U
                ic = min(bl, U)
            y = Y[i, j]
            e['iso'].append(iso - y)
            e['bliss'].append(bl - y)
            e['iso_clip'].append(ic - y)
            e['corner'].append(not highdose)
        r = dict(set_id=int(s))
        cn = np.array(e['corner'])
        for k in ('iso', 'bliss', 'iso_clip'):
            v = np.array(e[k])
            r[f'rmse_{k}'] = float(np.sqrt(np.mean(v ** 2)))
            r[f'rmse_{k}_noncorner'] = float(np.sqrt(np.mean(v[~cn] ** 2)))
            r[f'mae_{k}'] = float(np.mean(np.abs(v)))
        rows.append(r)
    return pd.DataFrame(rows)

def main():
    t0 = time.time()
    init()
    n = G['Ym'].shape[0]
    chunks = np.array_split(np.arange(n), 400)
    with Pool(C.MAX_WORKERS) as pool:
        out = pd.concat(pool.map(job, chunks), ignore_index=True)
    out = out.merge(G['meta'][['set_id', 'unit_id', 'cell_line', 'group_size', 'batch']], on='set_id')
    out.to_parquet(os.path.join(C.RES, 'missing_well_sets.parquet'), index=False)
    summ = dict(sets=int(len(out)), wells_held_out=int(16 * len(out)))
    for k in ('iso', 'bliss', 'iso_clip'):
        for suf in ('', '_noncorner'):
            v = out[f'rmse_{k}{suf}']
            summ[f'median_rmse_{k}{suf}'] = float(v.median())
            summ[f'iqr_rmse_{k}{suf}'] = [float(v.quantile(0.25)), float(v.quantile(0.75))]
    summ['share_sets_iso_better_than_bliss'] = float((out.rmse_iso < out.rmse_bliss).mean())
    summ['median_paired_difference_iso_minus_bliss'] = float((out.rmse_iso - out.rmse_bliss).median())
    summ['seconds'] = round(time.time() - t0, 1)
    with open(os.path.join(C.RES, 'missing_well_summary.json'), 'w') as f:
        json.dump(summ, f, indent=1)
    print(json.dumps(summ, indent=1))
if __name__ == '__main__':
    main()
