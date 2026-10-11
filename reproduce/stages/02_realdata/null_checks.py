"""DrugCombDB null_checks calculations."""
import argparse
import os
import time
from multiprocessing import Pool
import numpy as np
import pandas as pd
import common as C
S = C.S
G = {}

def init_oneil():
    B = np.load(C.BANK_ONEIL)
    G['SEL4'] = np.where(B['group_size'] == 4)[0]
    G['CELL'] = B['cell_line']
    G['YBAR'] = B['ybar_raw']
    G['RESID'] = B['resid_raw']
    G['EA'] = B['edge_drugA_raw'] / B['edge_drugA_raw'][:, :1]
    G['EB'] = B['edge_drugB_raw'] / B['edge_drugB_raw'][:, :1]
    D = np.load(C.DONORS_FILE)
    assert np.array_equal(D['targets'], G['SEL4'])
    G['DONORS'] = D['donors']
    uc = np.unique(G['CELL'])
    hm = {c: S.seed_for(bytes([114, 50, 98]).decode('ascii'), 'half', c) % 2 for c in uc}
    G['HALF'] = np.array([hm[c] for c in G['CELL']], dtype=np.int8)
    G['YBM'] = G['YBAR'][:, 1:, 1:].mean((1, 2))
    G['sd'] = S.oneil_sd_curve()
    G['SDY'] = G['sd'](G['YBAR'][:, 1:, 1:])
    (G['bank'], G['pvt']) = C.oneil_noise()

def donors4(ti, mu_t):
    SEL4 = G['SEL4']
    s = SEL4[ti]
    dh = int(G['HALF'][G['DONORS'][ti, 0]])
    rng = np.random.default_rng(S.seed_for(bytes([114, 54, 115]).decode('ascii'), 't2', 'donors', int(ti)))
    cand = SEL4[(G['HALF'][SEL4] == dh) & (G['CELL'][SEL4] != G['CELL'][s])]
    cand = rng.choice(cand, 600, replace=False)
    o = np.argsort(np.abs(G['YBM'][cand] - mu_t.mean()), kind='stable')[:4]
    return (cand[o], dh)

def job_oneil(tis):
    rows = []
    for ti in tis:
        s = G['SEL4'][ti]
        (eA, eB) = (G['EA'][s], G['EB'][s])
        mu = np.outer(eA[1:], eB[1:])
        (dd, dh) = donors4(ti, mu)
        Y = np.empty((4, 5, 5))
        Y[:, :, 0] = eA[None]
        Y[:, 0, :] = eB[None]
        Y[:, 0, 0] = 1.0
        sdm = G['sd'](mu)
        for r in range(4):
            Y[r, 1:, 1:] = mu + G['RESID'][dd[r], r] * np.clip(sdm / G['SDY'][dd[r]], 0.5, 2.0)
        grp = str(G['CELL'][dd[0]])
        assert C.fold_of(grp) == dh
        res = S.fit(Y, design='P', noise=G['bank'], plate_var=G['pvt'], group=grp, seed=(C.SEED_TAG, 'null', 'oneil', int(ti)), **{k: v for (k, v) in C.FIT_ONEIL.items() if k != 'design'})
        row = dict(target=int(ti), set_id=int(s), cell_line=str(G['CELL'][s]), donor_half=dh, donors=','.join(map(str, dd)), mean_bliss=float(mu.mean()))
        row.update(C.flatten(res))
        rows.append(row)
    return pd.DataFrame(rows)

def init_almanac():
    A = np.load(C.ALM_PAIRS)
    (Y1, Y2) = (A['Y1'].astype(float), A['Y2T'].astype(float))
    N = (Y1 - Y2) / np.sqrt(2.0)
    N[:, 0, 0] = 0.0
    EA = (Y1[:, :, 0] + Y2[:, :, 0]) / 2.0
    EB = (Y1[:, 0, :] + Y2[:, 0, :]) / 2.0
    EA[:, 0] = 1.0
    EB[:, 0] = 1.0
    G['NP'] = Y1.shape[0]
    (G['N'], G['EA'], G['EB']) = (N, EA, EB)
    G['CELL'] = A['cell_line']
    G['bank'] = C.almanac_bank()

def job_almanac(ts):
    rows = []
    NP = G['NP']
    for t in ts:
        p = t % NP
        sg = 1.0 if t < NP else -1.0
        T = np.empty((4, 4))
        T[:, 0] = G['EA'][p]
        T[0, :] = G['EB'][p]
        T[1:, 1:] = np.outer(G['EA'][p][1:], G['EB'][p][1:])
        Y = T + sg * G['N'][p]
        Y[0, 0] = 1.0
        res = S.fit(Y, design='S', noise=G['bank'], group=str(G['CELL'][p]), seed=(C.SEED_TAG, 'null', 'almanac', int(t)), **{k: v for (k, v) in C.FIT_ALMANAC.items() if k != 'design'})
        row = dict(target=int(t), pair=int(p), sign=int(sg), cell_line=str(G['CELL'][p]), mean_bliss=float(T[1:, 1:].mean()))
        row.update(C.flatten(res))
        rows.append(row)
    return pd.DataFrame(rows)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--study', required=True, choices=['oneil', 'almanac'])
    ap.add_argument('--n', type=int, default=0)
    ap.add_argument('--workers', type=int, default=16)
    a = ap.parse_args()
    if a.study == 'oneil':
        init_oneil()
        N = G['SEL4'].size
        job = job_oneil
    else:
        init_almanac()
        N = 2 * G['NP']
        job = job_almanac
    idx = np.arange(N)
    if a.n:
        idx = np.sort(np.random.default_rng(S.seed_for(C.SEED_TAG, 'null-subset', a.study)).choice(idx, a.n, replace=False))
    chunks = np.array_split(idx, max(1, idx.size // 25))
    t0 = time.time()
    out = []
    nw = min(a.workers, C.MAX_WORKERS)
    with Pool(nw) as pool:
        for (i, df) in enumerate(pool.imap_unordered(job, chunks)):
            out.append(df)
            if (i + 1) % 50 == 0:
                print(f'{i + 1}/{len(chunks)} chunks, {time.time() - t0:.0f}s', flush=True)
    res = pd.concat(out, ignore_index=True).sort_values('target')
    tag = '' if not a.n else f'_n{a.n}'
    res.to_parquet(os.path.join(C.RES, f'null_{a.study}{tag}.parquet'), index=False)
    print(f'done {a.study}: {len(res)} targets, {time.time() - t0:.0f}s', flush=True)
    for col in ('p_tstd', 'p_scan', 'p_companion', 'p_fixed_edge'):
        if col in res:
            print(col, [round(C.rate_row(res[col], lv)['rate'], 4) for lv in (0.05, 0.01, 0.005)], flush=True)
    print('stage 2 share', float((res.stage == 2).mean()), 'screened per target', float(res.n_screened.mean()))
if __name__ == '__main__':
    main()
