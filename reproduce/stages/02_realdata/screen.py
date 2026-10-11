"""DrugCombDB screen calculations."""
import argparse
import os
import time
from multiprocessing import Pool
import numpy as np
import pandas as pd
import common as C
import runner as Rn

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--study', required=True, choices=['oneil', 'almanac'])
    ap.add_argument('--n', type=int, default=0)
    ap.add_argument('--workers', type=int, default=16)
    a = ap.parse_args()
    Rn.init(a.study)
    N = len(Rn.G['meta'])
    idx = np.arange(N)
    if a.n:
        idx = np.sort(np.random.default_rng(C.S.seed_for(C.SEED_TAG, 'screen-subset', a.study)).choice(idx, a.n, replace=False))
    rng = np.random.default_rng(1)
    perm = rng.permutation(idx)
    chunks = np.array_split(perm, max(1, idx.size // 40))
    t0 = time.time()
    out = []
    with Pool(min(a.workers, C.MAX_WORKERS)) as pool:
        for (i, df) in enumerate(pool.imap_unordered(Rn.job, chunks)):
            out.append(df)
            if (i + 1) % 100 == 0:
                el = time.time() - t0
                print(f'{i + 1}/{len(chunks)} chunks, {el:.0f}s, eta {el / (i + 1) * (len(chunks) - i - 1):.0f}s', flush=True)
    res = pd.concat(out, ignore_index=True).sort_values('idx').reset_index(drop=True)
    meta = Rn.G['meta'].iloc[res.idx.values].reset_index(drop=True)
    res = pd.concat([meta, res], axis=1)
    res.insert(0, 'study', 'ONEIL' if a.study == 'oneil' else 'ALMANAC')
    tag = '' if not a.n else f'_n{a.n}'
    res.to_parquet(os.path.join(C.RES, f'screen_{a.study}{tag}.parquet'), index=False)
    nerr = int(res['error'].notna().sum()) if 'error' in res else 0
    print(f'done {a.study}: {len(res)} units, {nerr} errors, {time.time() - t0:.0f}s', flush=True)
    print('p_tstd<=0.05', float((res.p_tstd <= 0.05).mean()), 'stage2', float((res.stage == 2).mean()))
if __name__ == '__main__':
    main()
