"""DrugCombDB reproducibility calculations."""
import json
import os
import numpy as np
import pandas as pd
from scipy import stats
import common as C
SCORES = [('activity', 'activity'), ('mean Bliss excess', 'mean_excess'), ('T_std statistic', 't_std'), ('-log10 p(T_std)', 'mlp_tstd')]
NB = 30 if C.paths.SMOKE else 2000

def kappa(a, b):
    (a, b) = (np.asarray(a, bool), np.asarray(b, bool))
    po = np.mean(a == b)
    pe = a.mean() * b.mean() + (1 - a.mean()) * (1 - b.mean())
    return (po - pe) / (1 - pe) if pe < 1 else np.nan

def call_rows(x, y, comp, label):
    rows = []
    for (nm, col, thr) in (('T_std BH (q <= 0.05)', 'q_tstd', 0.05), ('T_std nominal p <= 0.05', 'p_tstd', 0.05), ('companion BH (q <= 0.05)', 'q_companion', 0.05), ('companion nominal p <= 0.05', 'p_companion', 0.05), ('scan: region claimed (FWER 0.05)', 'n_regions_claimed', None)):
        if col not in x:
            continue
        if thr is None:
            (a, b) = (x[col].values > 0, y[col].values > 0)
        else:
            (a, b) = (x[col].values <= thr, y[col].values <= thr)
        N = len(a)
        (n1, n2, both) = (int(a.sum()), int(b.sum()), int((a & b).sum()))
        tab = [[both, n1 - both], [n2 - both, N - n1 - n2 + both]]
        fp = stats.fisher_exact(tab, alternative='greater')[1] if N else np.nan
        rows.append(dict(comparison=comp, subset=label, call=nm, N=N, n_first=n1, n_second=n2, n_both=both, expected_both=n1 * n2 / N if N else np.nan, kappa=kappa(a, b), fisher_p_greater=fp))
    return rows

def cont_rows(x, y, comp, label, seed):
    rows = []
    for (nm, col) in SCORES:
        r = C.spearman_boot(x[col].values, y[col].values, n_boot=NB, seed=seed)
        r.update(comparison=comp, subset=label, score=nm)
        rows.append(r)
    z = (x.mean_excess.values - y.mean_excess.values) / np.sqrt(x.mean_excess_se.values ** 2 + y.mean_excess_se.values ** 2)
    z = z[np.isfinite(z)]
    rows.append(dict(comparison=comp, subset=label, score='z-difference of mean excess: SD (robust SD)', n=int(z.size), spearman=float(np.std(z, ddof=1)), lo=float(1.4826 * np.median(np.abs(z - np.median(z)))), hi=np.nan))
    return rows

def prep(d):
    d = d.copy()
    d['mlp_tstd'] = -np.log10(d.p_tstd)
    return d

def main():
    dO = prep(pd.read_parquet(os.path.join(C.RES, 'screen_oneil_q.parquet')))
    dA = prep(pd.read_parquet(os.path.join(C.RES, 'screen_almanac_q.parquet')))
    (cont, calls) = ([], [])
    g8 = dO[dO.group_size == 8]
    b0 = g8[g8.batch == 0].set_index('group_key').sort_index()
    b1 = g8[g8.batch == 1].set_index('group_key').loc[b0.index]
    Yo = np.load(os.path.join(C.RES, 'inputs_oneil.npz'))['Y']
    (e0, e1) = (Yo[b0.set_id.values, 0], Yo[b1.set_id.values, 0])
    same = (np.abs(e0[:, 1:, 0] - e1[:, 1:, 0]).max(1) < 1e-12) & (np.abs(e0[:, 0, 1:] - e1[:, 0, 1:]).max(1) < 1e-12)
    pairs = pd.DataFrame({'group_key': b0.index, 'cell_line': b0.cell_line.values, 'edges_identical': same})
    for c in ['activity', 'mean_excess', 'mean_excess_se', 't_std', 'p_tstd', 'q_tstd', 'p_companion', 'q_companion', 'mlp_tstd', 'n_regions_claimed']:
        pairs[c + '_b0'] = b0[c].values
        pairs[c + '_b1'] = b1[c].values
    pairs.to_parquet(os.path.join(C.RES, 'repro_pairs_oneil.parquet'), index=False)
    Zo = np.load(os.path.join(C.RES, 'inputs_oneil.npz'))
    def max_dose_ratio(D0, D1):
        (u, v) = (D0[:, 1:], D1[:, 1:])
        return np.maximum(u / v, v / u).max(1)
    ratio = np.maximum(max_dose_ratio(Zo['doseA'][b0.set_id.values], Zo['doseA'][b1.set_id.values]), max_dose_ratio(Zo['doseB'][b0.set_id.values], Zo['doseB'][b1.set_id.values]))
    matched = ratio <= 1.01
    pairs['max_dose_ratio'] = ratio
    pairs.to_parquet(os.path.join(C.RES, 'repro_pairs_oneil.parquet'), index=False)
    window = dict(groups=int(len(ratio)), dose_matched=int(matched.sum()), differ_over_1_5=int((ratio > 1.5).sum()), between=int(((ratio > 1.01) & (ratio <= 1.5)).sum()), max_ratio=float(ratio.max()))
    with open(os.path.join(C.RES, 'repro_dose_window.json'), 'w') as f:
        json.dump(window, f, indent=1)
    for (lab, m) in (('all', np.ones(len(b0), bool)), ('edges identical across batches', same)):
        cont += cont_rows(b0[m], b1[m], "O'Neil batch 0 vs batch 1", lab, seed=11)
        calls += call_rows(b0[m], b1[m], "O'Neil batch 0 vs batch 1", lab)
    ukey = dA.cell_line + '|' + np.where(dA.drugA <= dA.drugB, dA.drugA, dA.drugB) + '|' + np.where(dA.drugA <= dA.drugB, dA.drugB, dA.drugA)
    cnt = ukey.value_counts()
    P = dA.assign(ukey=ukey)[ukey.isin(cnt[cnt == 2].index)].sort_values(['ukey', 'block_id'])
    P1 = P.iloc[::2].set_index('ukey')
    P2 = P.iloc[1::2].set_index('ukey').loc[P1.index]
    Ya = np.load(os.path.join(C.RES, 'inputs_almanac.npz'))['Y']
    pos = {b: k for (k, b) in enumerate(dA.block_id.values)}
    y1 = Ya[[pos[b] for b in P1.block_id]]
    y2 = np.swapaxes(Ya[[pos[b] for b in P2.block_id]], 1, 2)
    ident = np.abs(y1 - y2).reshape(len(y1), -1).max(1) < 1e-12
    keep = ~ident
    ap = pd.DataFrame({'ukey': P1.index, 'cell_line': P1.cell_line.values, 'identical_copies': ident})
    for c in ['activity', 'mean_excess', 'mean_excess_se', 't_std', 'p_tstd', 'q_tstd', 'p_companion', 'q_companion', 'mlp_tstd', 'n_regions_claimed']:
        ap[c + '_1'] = P1[c].values
        ap[c + '_2'] = P2[c].values
    ap.to_parquet(os.path.join(C.RES, 'repro_pairs_almanac.parquet'), index=False)
    cont += cont_rows(P1[keep], P2[keep], 'ALMANAC orientation pairs', 'all (identical copies dropped)', seed=12)
    calls += call_rows(P1[keep], P2[keep], 'ALMANAC orientation pairs', 'all (identical copies dropped)')

    def ckey(d):
        (c, a, b) = (C.norm_name(d.cell_line), C.norm_name(d.drugA), C.norm_name(d.drugB))
        return pd.Series(c).str.cat([np.where(a <= b, a, b), np.where(a <= b, b, a)], sep='|').values
    dO['ckey'] = ckey(dO)
    dA['ckey'] = ckey(dA)
    Ob = dO[dO.batch == 0].drop_duplicates('ckey').set_index('ckey')
    Af = dA.sort_values('block_id').drop_duplicates('ckey').set_index('ckey')
    common = Ob.index.intersection(Af.index)
    (Om, Am) = (Ob.loc[common], Af.loc[common])
    cp = pd.DataFrame({'ckey': common, 'cell_line': Om.cell_line.values})
    for c in ['activity', 'mean_excess', 't_std', 'p_tstd', 'q_tstd', 'p_companion', 'q_companion']:
        cp[c + '_oneil'] = Om[c].values
        cp[c + '_almanac'] = Am[c].values
    cp.to_parquet(os.path.join(C.RES, 'repro_pairs_cross.parquet'), index=False)
    cont += cont_rows(Om, Am, "O'Neil vs ALMANAC (cross-study)", 'all matches', seed=13)
    calls += call_rows(Om, Am, "O'Neil vs ALMANAC (cross-study)", 'all matches')
    # O'Neil cross-batch comparison split by dose window (appended so earlier rows keep their positions)
    for (lab, m) in (('dose grids within 1%', matched), ('dose grids differ', ~matched)):
        cont += cont_rows(b0[m], b1[m], "O'Neil batch 0 vs batch 1", lab, seed=11)
        calls += call_rows(b0[m], b1[m], "O'Neil batch 0 vs batch 1", lab)
    pd.DataFrame(cont).to_csv(os.path.join(C.RES, 'repro_continuous.csv'), index=False)
    pd.DataFrame(calls).to_csv(os.path.join(C.RES, 'repro_calls.csv'), index=False)
    sc = []
    for (st, d) in (('ONEIL', dO), ('ALMANAC', dA)):
        for ((n1, c1), (n2, c2)) in ((SCORES[0], SCORES[1]), (SCORES[0], SCORES[2]), (SCORES[1], SCORES[2])):
            r = C.spearman_boot(d[c1].values, d[c2].values, n_boot=500, seed=14, cluster=d.cell_line.values)
            r.update(study=st, x=n1, y=n2)
            sc.append(r)
    pd.DataFrame(sc).to_csv(os.path.join(C.RES, 'score_correlations.csv'), index=False)
    pd.set_option('display.width', 250)
    print(pd.DataFrame(cont).to_string())
    print(pd.DataFrame(calls).to_string())
    print(pd.DataFrame(sc).to_string())
if __name__ == '__main__':
    main()
