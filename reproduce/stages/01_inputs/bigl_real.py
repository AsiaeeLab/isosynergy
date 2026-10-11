"""Select 300 O'Neil groups for the BIGL comparison and verify their raw wells."""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import pandas as pd
import bigl_design as C
from bigl_design import X
CSV = str(C.paths.raw_file('drugcombdb/drugcombs_response.csv'))
N = 300

def main():
    B = np.load(os.path.join(C.RES, 'noise_bank_oneil.npz'))
    sel4 = np.where(B['group_size'] == 4)[0]
    rng = np.random.default_rng(X.seed_for(bytes([114, 55, 98]).decode('ascii'), 'real'))
    pick = np.sort(rng.choice(sel4, N, replace=False))
    Y = B['Y_raw'][pick].astype('<f8')
    doses = np.stack([B['doseA'][pick], B['doseB'][pick]], 1).astype('<f8')
    blocks = B['block_id'][pick]
    want = set((int(b) for b in blocks.ravel()))
    parts = []
    for ch in pd.read_csv(CSV, usecols=['BlockID', 'Row', 'Col', 'ConcRow', 'ConcCol', 'Response', 'source'], chunksize=2000000):
        parts.append(ch[ch.BlockID.isin(want)])
    raw = pd.concat(parts)
    assert (raw.source == 'ONEIL').all()
    maxdiff = 0.0
    for g in range(N):
        for k in range(4):
            b = raw[raw.BlockID == blocks[g, k]]
            assert len(b) == 25, (g, k, len(b))
            ri = {c: i for (i, c) in enumerate(np.sort(b.ConcRow.unique()))}
            ci = {c: i for (i, c) in enumerate(np.sort(b.ConcCol.unique()))}
            M = np.full((5, 5), np.nan)
            for (_, w) in b.iterrows():
                M[ri[w.ConcRow], ci[w.ConcCol]] = w.Response / 100.0
            maxdiff = max(maxdiff, float(np.nanmax(np.abs(M - Y[g, k]))))
            assert np.allclose(np.sort(b.ConcRow.unique()), doses[g, 0]) and np.allclose(np.sort(b.ConcCol.unique()), doses[g, 1])
    print(f'max |bank Y_raw - raw viability| over {N * 4 * 25} cells: {maxdiff:.3e}')
    base = os.path.join(C.DATA, 'real_oneil300')
    Y.tofile(base + '.f64')
    doses.tofile(base + '.doses.f64')
    np.array([X.seed_for(bytes([114, 55, 98]).decode('ascii'), 'bigl', 'real', k) % 2147483647 for k in range(N)], dtype='<i4').tofile(base + '.seeds.i32')
    meta = pd.DataFrame(dict(k=np.arange(N), bank_index=pick, group_key=B['group_key'][pick], drugA=B['drugA'][pick], drugB=B['drugB'][pick], cell_line=B['cell_line'][pick], block_ids=[';'.join(map(str, b)) for b in blocks], doseA=[';'.join((f'{d:g}' for d in x)) for x in doses[:, 0]], doseB=[';'.join((f'{d:g}' for d in x)) for x in doses[:, 1]], edges_identical=[bool(np.all(Y[g, :, 0, :] == Y[g, :1, 0, :]) and np.all(Y[g, :, :, 0] == Y[g, :1, :, 0])) for g in range(N)]))
    meta.to_csv(os.path.join(C.RES, 'bigl_real_meta.csv'), index=False)
    print(meta.head(), meta.edges_identical.mean(), meta.cell_line.nunique(), 'cell lines')
if __name__ == '__main__':
    main()
