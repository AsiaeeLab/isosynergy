"""DrugCombDB prepare calculations."""
import json
import os
import time
import numpy as np
import pandas as pd
import common as C

def main():
    t0 = time.time()
    raw = pd.read_csv(C.RAW_CSV, usecols=['BlockID', 'Row', 'Col', 'DrugRow', 'DrugCol', 'ConcRow', 'ConcCol', 'Response', 'source'])
    chk = {}
    chk['raw_rows_by_source'] = raw.groupby('source').size().to_dict()
    chk['raw_blocks_by_source'] = raw.groupby('source').BlockID.nunique().to_dict()
    on = raw[raw.source == 'ONEIL'].sort_values(['BlockID', 'Row', 'Col'], kind='mergesort')
    cnt = on.groupby('BlockID').size()
    assert (cnt == 25).all(), "O'Neil blocks must be 5 x 5"
    nb = len(cnt)
    assert np.all(on.Row.values.reshape(nb, 25) == np.repeat(np.arange(1, 6), 5)[None])
    assert np.all(on.Col.values.reshape(nb, 25) == np.tile(np.arange(1, 6), 5)[None])
    bid = on.BlockID.values[::25]
    Yb = on.Response.values.reshape(nb, 5, 5) / 100.0
    dA = on.ConcRow.values.reshape(nb, 5, 5)[:, :, 0]
    dB = on.ConcCol.values.reshape(nb, 5, 5)[:, 0, :]
    pos = {b: k for (k, b) in enumerate(bid)}
    Bk = np.load(C.BANK_ONEIL)
    S = Bk['block_id'].shape[0]
    Y = np.empty((S, 4, 5, 5))
    DA = np.empty((S, 5))
    DB = np.empty((S, 5))
    for s in range(S):
        idx = [pos[int(b)] for b in Bk['block_id'][s]]
        Y[s] = Yb[idx]
        DA[s] = dA[idx[0]]
        DB[s] = dB[idx[0]]
        assert np.allclose(dA[idx], dA[idx[0]][None]) and np.allclose(dB[idx], dB[idx[0]][None])
    chk['oneil_blocks'] = int(nb)
    chk['oneil_sets'] = int(S)
    chk['oneil_blocks_in_sets'] = int(np.unique(Bk['block_id']).size)
    chk['oneil_max_abs_diff_vs_bank_Y_raw'] = float(np.abs(Y - Bk['Y_raw']).max())
    chk['oneil_max_abs_diff_doses_vs_bank'] = float(max(np.abs(DA - Bk['doseA']).max(), np.abs(DB - Bk['doseB']).max()))
    chk['oneil_doses_increasing'] = bool(np.all(np.diff(DA, axis=1) > 0) and np.all(np.diff(DB, axis=1) > 0))
    chk['oneil_control_range'] = [float(Y[:, :, 0, 0].min()), float(Y[:, :, 0, 0].max())]
    chk['oneil_nan'] = int(np.isnan(Y).sum())
    E = np.concatenate([Y[:, :, 1:, 0], Y[:, :, 0, 1:]], 2)
    chk['oneil_sets_edges_identical_across_blocks'] = int(np.sum(np.abs(E - E[:, :1]).max((1, 2)) == 0))
    meta = pd.DataFrame(dict(set_id=np.arange(S), group_key=Bk['group_key'], drugA=Bk['drugA'], drugB=Bk['drugB'], cell_line=Bk['cell_line'], group_size=Bk['group_size'].astype(int), batch=Bk['batch'].astype(int), experiment_ids=[','.join(x) for x in Bk['experiment_id']], block_ids=[','.join(map(str, x)) for x in Bk['block_id']]))
    meta['unit_id'] = meta.group_key + '|batch' + meta.batch.astype(str)
    first = on.drop_duplicates('BlockID').set_index('BlockID')
    b0 = Bk['block_id'][:, 0]
    chk['oneil_drug_names_agree'] = bool(np.all(first.loc[b0, 'DrugRow'].values == meta.drugA.values) and np.all(first.loc[b0, 'DrugCol'].values == meta.drugB.values))
    np.savez_compressed(os.path.join(C.RES, 'inputs_oneil.npz'), Y=Y, doseA=DA, doseB=DB)
    meta.to_parquet(os.path.join(C.RES, 'meta_oneil.parquet'), index=False)
    print(f"O'Neil: {S} sets, max |Y - bank Y_raw| = {chk['oneil_max_abs_diff_vs_bank_Y_raw']:.3g}, {time.time() - t0:.0f}s", flush=True)
    al = raw[raw.source == 'ALMANAC'].sort_values(['BlockID', 'Row', 'Col'], kind='mergesort')
    cnt = al.groupby('BlockID').size()
    incomplete = cnt[cnt != 16]
    chk['almanac_blocks_raw'] = int(len(cnt))
    chk['almanac_blocks_incomplete'] = {int(k): int(v) for (k, v) in incomplete.items()}
    al = al[al.BlockID.isin(cnt[cnt == 16].index)]
    nb = al.BlockID.nunique()
    assert np.all(al.Row.values.reshape(nb, 16) == np.repeat(np.arange(1, 5), 4)[None])
    assert np.all(al.Col.values.reshape(nb, 16) == np.tile(np.arange(1, 5), 4)[None])
    Ya = al.Response.values.reshape(nb, 4, 4) / 100.0
    aDA = al.ConcRow.values.reshape(nb, 4, 4)[:, :, 0]
    aDB = al.ConcCol.values.reshape(nb, 4, 4)[:, 0, :]
    abid = al.BlockID.values[::16]
    pr = pd.read_parquet(C.PROCESSED, columns=['experiment_id', 'block_id', 'drugA', 'drugB', 'cell_line', 'study_source', 'row_index', 'col_index'], filters=[('study_source', '==', 'ALMANAC')])
    pr = pr[(pr.row_index == 1) & (pr.col_index == 1)].set_index('block_id')
    missing = np.setdiff1d(abid, pr.index.values)
    chk['almanac_blocks_without_metadata'] = int(missing.size)
    keep = np.isin(abid, pr.index.values)
    (abid, Ya, aDA, aDB) = (abid[keep], Ya[keep], aDA[keep], aDB[keep])
    first = al.drop_duplicates('BlockID').set_index('BlockID').loc[abid]
    m = pr.loc[abid]
    ameta = pd.DataFrame(dict(block_id=abid, experiment_id=m.experiment_id.values, drugA=first.DrugRow.values, drugB=first.DrugCol.values, cell_line=m.cell_line.values))
    chk['almanac_drug_names_agree_processed'] = bool(np.all(m.drugA.values == ameta.drugA.values) and np.all(m.drugB.values == ameta.drugB.values))
    ameta['unit_id'] = ameta.experiment_id
    chk['almanac_plates'] = int(len(ameta))
    chk['almanac_doses_increasing'] = int(np.sum(np.all(np.diff(aDA, axis=1) > 0, 1) & np.all(np.diff(aDB, axis=1) > 0, 1)))
    chk['almanac_control_range'] = [float(Ya[:, 0, 0].min()), float(Ya[:, 0, 0].max())]
    chk['almanac_nan'] = int(np.isnan(Ya).sum())
    chk['almanac_cell_lines'] = int(ameta.cell_line.nunique())
    P = np.load(C.ALM_PAIRS)
    bpos = {b: k for (k, b) in enumerate(abid)}
    i1 = np.array([bpos[int(b)] for b in P['block1']])
    i2 = np.array([bpos[int(b)] for b in P['block2']])
    Y1 = Ya[i1].copy()
    Y2T = np.swapaxes(Ya[i2], 1, 2).copy()
    Y1[:, 0, 0] = 1.0
    Y2T[:, 0, 0] = 1.0
    chk['almanac_pairs'] = int(P['block1'].size)
    chk['almanac_pairs_max_abs_diff_Y1'] = float(np.abs(Y1 - P['Y1']).max())
    chk['almanac_pairs_max_abs_diff_Y2T'] = float(np.abs(Y2T - P['Y2T']).max())
    chk['almanac_pairs_dose_match'] = bool(np.allclose(aDA[i1], aDB[i2]) and np.allclose(aDB[i1], aDA[i2]))
    np.savez_compressed(os.path.join(C.RES, 'inputs_almanac.npz'), Y=Ya, doseA=aDA, doseB=aDB)
    ameta.to_parquet(os.path.join(C.RES, 'meta_almanac.parquet'), index=False)
    chk['seconds'] = round(time.time() - t0, 1)
    with open(os.path.join(C.RES, 'prepare_checks.json'), 'w') as f:
        json.dump(chk, f, indent=1, default=str)
    print(json.dumps(chk, indent=1, default=str))
if __name__ == '__main__':
    main()
