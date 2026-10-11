"""DrugCombDB controls calculations."""
import json
import os
import numpy as np
import pandas as pd
from scipy import stats
import common as C
SRC = str(C.paths.DATA / 'control_membership.csv')
SAME_LINE = ['ON-A ', 'ON-A2 ', 'ON-B ']

def pct_within(d, col):
    r = d.groupby('cell_line')[col].rank(method='average')
    n = d.groupby('cell_line')[col].transform(lambda s: s.notna().sum())
    return (r - 0.5) / n

def drug_matched(d, sel, col='mean_excess'):
    out = np.full(len(sel), np.nan)
    for (k, (_, r)) in enumerate(sel.iterrows()):
        m = (d.cell_line == r.cell_line) & (d.pair != r.pair) & (d.drugA.isin([r.drugA, r.drugB]) | d.drugB.isin([r.drugA, r.drugB]))
        v = d.loc[m, col].dropna().values
        if v.size and np.isfinite(r[col]):
            out[k] = float(np.mean(v < r[col]) + 0.5 * np.mean(v == r[col]))
    return out

def summarize(u, bg, label, role, study):
    u = u.copy()
    u['cluster'] = u.pair + '|' + u.cell_line
    det = dict(tstd_p05=u.p_tstd <= 0.05, tstd_bh=u.q_tstd <= 0.05, comp_p05=u.p_companion <= 0.05, comp_bh=u.q_companion <= 0.05, scan_claim=u.n_regions_claimed > 0)
    for (k, v) in det.items():
        u['det_' + k] = v.astype(float)
    agg = {c: 'mean' for c in ['pct_mean_excess', 'pct_activity', 'pct_t_std', 'mean_excess', 'activity', 'pct_drugmatched'] + ['det_' + k for k in det]}
    cl = u.groupby('cluster').agg(agg)
    row = dict(study=study, group=label, role=role, n_units=len(u), n_clusters=len(cl), n_cell_lines=u.cell_line.nunique(), n_pairs=u.pair.nunique(), median_mean_excess=float(cl.mean_excess.median()), bg_median_mean_excess=float(bg.mean_excess.median()))
    for c in ('pct_mean_excess', 'pct_activity', 'pct_t_std', 'pct_drugmatched'):
        v = cl[c].dropna().values
        row['mean_' + c] = float(v.mean()) if v.size else np.nan
        row['median_' + c] = float(np.median(v)) if v.size else np.nan
        row['n_above_0.5_' + c] = int((v > 0.5).sum())
        row['wilcoxon_p_' + c] = float(stats.wilcoxon(v - 0.5, alternative='greater').pvalue) if v.size >= 5 else np.nan
    for k in det:
        v = cl['det_' + k].values
        kk = float(v.sum())
        row['rate_' + k] = float(v.mean())
        (lo, hi) = C.wilson(kk, len(v))
        (row['rate_' + k + '_lo'], row['rate_' + k + '_hi']) = (lo, hi)
        bgv = {'tstd_p05': bg.p_tstd <= 0.05, 'tstd_bh': bg.q_tstd <= 0.05, 'comp_p05': bg.p_companion <= 0.05, 'comp_bh': bg.q_companion <= 0.05, 'scan_claim': bg.n_regions_claimed > 0}[k]
        row['bg_rate_' + k] = float(bgv.mean())
        row['binom_p_' + k] = float(stats.binomtest(int(round(kk)), len(v), max(row['bg_rate_' + k], 1e-09), alternative='greater').pvalue) if len(v) else np.nan
    return row

def main():
    src = pd.read_csv(SRC, low_memory=False)
    dO = pd.read_parquet(os.path.join(C.RES, 'screen_oneil_q.parquet'))
    dA = pd.read_parquet(os.path.join(C.RES, 'screen_almanac_q.parquet'))
    for d in (dO, dA):
        d['pair'] = np.where(d.drugA < d.drugB, d.drugA + ' + ' + d.drugB, d.drugB + ' + ' + d.drugA)
        for c in ('mean_excess', 'activity', 't_std'):
            d['pct_' + c] = pct_within(d, c)
    so = src[src.study == 'ONEIL'].copy()
    so['unit_id'] = so.drugA + '|' + so.drugB + '|' + so.cell_line + '|batch' + so.batch.astype(int).astype(str)
    so = so.merge(dO[['unit_id'] + [c for c in dO.columns if c not in so.columns]], on='unit_id', how='left')
    n_unmatched_o = int(so.p_tstd.isna().sum())
    sa = src[src.study == 'ALMANAC'].copy()
    keep_cols = [c for c in dA.columns if c not in sa.columns or c == 'experiment_id']
    sa = sa.merge(dA[keep_cols], on='experiment_id', how='left')
    n_unmatched_a = int(sa.p_tstd.isna().sum())
    units = pd.concat([so, sa], ignore_index=True)
    units['pct_drugmatched'] = np.nan
    for (st, d) in (('ONEIL', dO), ('ALMANAC', dA)):
        k = units.study == st
        units.loc[k, 'pct_drugmatched'] = drug_matched(d, units[k])
    rows = []
    for (st, d) in (('ONEIL', dO), ('ALMANAC', dA)):
        U = units[units.study == st]
        ctrl_ids = set(U.unit_id)
        groups = list(U.groupby('group'))
        if st == 'ONEIL':
            same = U[U.group.str.startswith(tuple(SAME_LINE))]
            groups.append(('ON-A + A2 + B pooled: independent same-line evidence', same))
        else:
            allpos = U[~U.group.str.startswith('AL-N1')]
            groups.append(('AL-ALL every source-reported in vitro positive (union)', allpos))
        for (lab, u) in groups:
            bg = d[d.cell_line.isin(u.cell_line.unique()) & ~d.unit_id.isin(ctrl_ids)]
            role = 'negative' if '-N' in lab.split(' ')[0] else 'positive'
            rows.append(summarize(u, bg, lab, role, st))
        weak = d[(d.edgeA_top >= 0.9) & (d.edgeB_top >= 0.9) & ~d.unit_id.isin(ctrl_ids)].copy()
        weak['pct_drugmatched'] = np.nan
        bg = d[~d.unit_id.isin(ctrl_ids) & ~d.unit_id.isin(weak.unit_id)]
        rows.append(summarize(weak, bg, f"{('ON-N3' if st == 'ONEIL' else 'AL-N2')} inactive pairs (both top-dose single agents >= 0.9)", 'negative', st))
    summ = pd.DataFrame(rows)
    g = dO.groupby('pair')
    agg = pd.DataFrame(dict(n_units=g.size(), mean_pct_mean_excess=g.pct_mean_excess.mean(), mean_pct_activity=g.pct_activity.mean(), mean_pct_t_std=g.pct_t_std.mean(), frac_tstd_p05=g.p_tstd.apply(lambda s: float(np.mean(s <= 0.05)))))
    for c in ['mean_pct_mean_excess', 'mean_pct_t_std']:
        agg['rank_' + c] = agg[c].rank(pct=True)
    pl = []
    po = units[(units.study == 'ONEIL') & units.group.str.match('ON-(P|N1|S) ')]
    for ((pr, grp), h) in po.groupby(['pair', 'group']):
        if pr not in agg.index:
            continue
        (a, b) = pr.split(' + ')
        share = [q for q in agg.index if q != pr and (a in q.split(' + ') or b in q.split(' + '))]
        v = agg.loc[share, 'mean_pct_mean_excess'].values
        x = agg.loc[pr, 'mean_pct_mean_excess']
        r = agg.loc[pr].to_dict()
        r.update(pair=pr, group=grp, n_pairs_total=len(agg), rank_among_drug_sharing_pairs=float(np.mean(v < x) + 0.5 * np.mean(v == x)))
        pl.append(r)
    pl = pd.DataFrame(pl)
    units.to_csv(os.path.join(C.RES, 'controls_units.csv'), index=False)
    summ.to_csv(os.path.join(C.RES, 'controls_summary.csv'), index=False)
    pl.to_csv(os.path.join(C.RES, 'controls_pair_level.csv'), index=False)
    sl = units[(units.study == 'ONEIL') & units.group.str.startswith(tuple(SAME_LINE))]
    sl_tab = sl[['group', 'pair', 'cell_line', 'batch', 'mean_excess', 'mean_excess_se', 'pct_mean_excess', 'p_tstd', 'q_tstd', 'p_companion', 'q_companion', 'p_fixed_edge']].sort_values(['pair', 'cell_line', 'batch'])
    sl_tab.to_csv(os.path.join(C.RES, 'controls_same_line_units.csv'), index=False)
    info = dict(oneil_control_units_unmatched=n_unmatched_o, almanac_control_units_unmatched=n_unmatched_a, oneil_pairs_total=int(len(agg)))
    with open(os.path.join(C.RES, 'controls.json'), 'w') as f:
        json.dump(info, f, indent=1)
    pd.set_option('display.width', 300)
    pd.set_option('display.max_columns', 60)
    print(info)
    print(summ[['study', 'group', 'n_clusters', 'mean_pct_mean_excess', 'wilcoxon_p_mean_excess' if False else 'wilcoxon_p_pct_mean_excess', 'median_pct_mean_excess', 'mean_pct_t_std', 'rate_tstd_p05', 'bg_rate_tstd_p05', 'rate_tstd_bh', 'bg_rate_tstd_bh', 'rate_comp_p05', 'bg_rate_comp_p05']].to_string())
    print(pl[['pair', 'group', 'n_units', 'mean_pct_mean_excess', 'rank_mean_pct_mean_excess', 'rank_among_drug_sharing_pairs']].to_string())
    print(sl_tab.to_string())
if __name__ == '__main__':
    main()
