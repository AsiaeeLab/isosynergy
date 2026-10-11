"""DrugCombDB families calculations."""
import json
import os
import numpy as np
import pandas as pd
import common as C
S = C.S
Q = 0.05

def add_q(d, fam):
    d = d.copy()
    d['family'] = fam
    for (col, out) in (('p_tstd', 'q_tstd'), ('p_companion', 'q_companion')):
        d[out] = np.nan
        for (f, idx) in d.groupby('family').groups.items():
            d.loc[idx, out] = S.bh_qvalues(d.loc[idx, col].values, 'one')
    d['p_companion_relnull'] = np.nan
    d['q_companion_relnull'] = np.nan
    for (f, idx) in d.groupby('family').groups.items():
        (pr, mu0, s0) = S.empirical_null(d.loc[idx, 'p_companion'].values)
        d.loc[idx, 'p_companion_relnull'] = pr
        d.loc[idx, 'q_companion_relnull'] = S.bh_qvalues(pr, 'one')
    return d

def pool_size(d):
    return d.log_notes.str.extract('(\\d+) eligible sets')[0].astype(float)

def counts(d, study):
    c = {}
    pair = np.where(d.drugA < d.drugB, d.drugA + ' + ' + d.drugB, d.drugB + ' + ' + d.drugA)
    d = d.assign(pair=pair)
    c['units'] = int(len(d))
    c['families'] = int(d.family.nunique())
    c['cell_lines'] = int(d.cell_line.nunique())
    c['drug_pairs'] = int(pd.Series(pair).nunique())
    for (lab, qcol) in (('tstd', 'q_tstd'), ('companion', 'q_companion'), ('companion_relnull', 'q_companion_relnull')):
        call = d[qcol] <= Q
        c[f'bh_calls_{lab}'] = int(call.sum())
        c[f'bh_calls_{lab}_cell_lines'] = int(d.cell_line[call].nunique())
        c[f'bh_calls_{lab}_families'] = int(d.family[call].nunique())
        c[f'bh_calls_{lab}_drug_pairs'] = int(d.pair[call].nunique())
    (tc, mc) = (d.q_tstd <= Q, d.q_companion <= Q)
    c['bh_calls_tstd_and_companion'] = int((tc & mc).sum())
    for lv in (0.05, 0.01, 0.005):
        c[f'nominal_tstd_p_le_{lv}'] = int((d.p_tstd <= lv).sum())
        c[f'nominal_companion_p_le_{lv}'] = int((d.p_companion <= lv).sum())
    c['nominal_tstd_share_p_le_0.05'] = float((d.p_tstd <= 0.05).mean())
    c['nominal_companion_share_p_le_0.05'] = float((d.p_companion <= 0.05).mean())
    rc = d.n_regions_claimed > 0
    hd = d.region_types_claimed.fillna('').str.contains('rectangle')
    c['scan_any_region_claimed_0.05'] = int(rc.sum())
    c['scan_high_dose_rectangle_claimed_0.05'] = int(hd.sum())
    c['scan_only_2x2_blocks_claimed'] = int((rc & ~hd).sum())
    c['scan_claim_among_tstd_bh_calls'] = int((rc & tc).sum())
    c['stage2_units'] = int((d.stage == 2).sum())
    c['companion_degenerate'] = int(d.companion_degenerate.astype(bool).sum())
    c['units_with_screened_replicate'] = int((d.n_screened > 0).sum())
    c['screened_replicates_total'] = int(d.n_screened.sum())
    c['units_with_dropped_dose'] = int((d.n_dropped_doses > 0).sum())
    c['units_with_missing_cell'] = int((d.n_missing_cells > 0).sum())
    if study == 'almanac':
        q2 = d.q_tstd_floored <= Q
        c['bh_calls_tstd_resolution_floored'] = int(q2.sum())
        fr = d[d.fragility_computed.astype(bool)]
        c['fragility_computed_units (p_tstd<=0.05)'] = int(len(fr))
        c['fragility_loco_flag_share_nominal_calls'] = float(fr.loco_flag.astype(bool).mean())
        c['fragility_sa_flag_share_nominal_calls'] = float(fr.sa_flag.astype(bool).mean())
        c['fragility_either_flag_share_nominal_calls'] = float((fr.loco_flag.astype(bool) | fr.sa_flag.astype(bool)).mean())
        frb = fr[fr.q_tstd <= Q]
        c['fragility_bh_calls'] = int(len(frb))
        c['fragility_loco_flag_share_bh_calls'] = float(frb.loco_flag.astype(bool).mean()) if len(frb) else np.nan
        c['fragility_sa_flag_share_bh_calls'] = float(frb.sa_flag.astype(bool).mean()) if len(frb) else np.nan
        c['tstd_p_at_minimum_1e-4'] = int((d.p_tstd <= 0.00010001).sum())
    return c

def drug_clusters(d):
    rows = []
    call = d.q_tstd <= Q
    for (f, g) in d.groupby('family'):
        gc = g[call.loc[g.index]]
        drugs = pd.concat([g.drugA, g.drugB]).unique()
        for dr in drugs:
            tested = int(((g.drugA == dr) | (g.drugB == dr)).sum())
            k = int(((gc.drugA == dr) | (gc.drugB == dr)).sum())
            if k:
                rows.append(dict(family=f, study=g.study.iloc[0], drug=dr, n_calls=k, n_tested=tested, family_calls=int(len(gc)), family_units=int(len(g))))
    out = pd.DataFrame(rows)
    if len(out):
        out['expected_if_random'] = out.family_calls * out.n_tested / out.family_units
        out = out.sort_values(['n_calls'], ascending=False)
    return out

def panels(dO, dA):
    P = pd.read_csv(C.PANELS)
    P['study'] = P.panel_id.str.split(':').str[0]
    P['design'] = P.panel_id.str.split(':').str[2]
    b2u = {}
    for (u, ex) in zip(dO.unit_id, dO.experiment_ids):
        for e in ex.split(','):
            b2u[e] = u
    a2u = dict(zip(dA.experiment_id, dA.unit_id))
    P['unit_id'] = [b2u.get(m) if s == 'ONEIL' else a2u.get(m) for (s, m) in zip(P.study, P.matrix_id)]
    nmiss = int(P.unit_id.isna().sum())
    dd = pd.concat([dO[['unit_id', 'p_tstd', 'p_companion']].assign(study='ONEIL'), dA[['unit_id', 'p_tstd', 'p_companion']].assign(study='ALMANAC')])
    P = P.merge(dd, on=['study', 'unit_id'], how='left')
    n0 = len(P)
    P = P.drop_duplicates(['panel_id', 'unit_id'])
    ndup = n0 - len(P)
    P['q_tstd_panel'] = np.nan
    P['q_companion_panel'] = np.nan
    for (pid, idx) in P.groupby('panel_id').groups.items():
        P.loc[idx, 'q_tstd_panel'] = S.bh_qvalues(P.loc[idx, 'p_tstd'].values, 'one')
        P.loc[idx, 'q_companion_panel'] = S.bh_qvalues(P.loc[idx, 'p_companion'].values, 'one')
    g = P.groupby('panel_id')
    per = pd.DataFrame(dict(study=g.study.first(), design=g.design.first(), n_tests=g.size(), calls_tstd=g.q_tstd_panel.apply(lambda s: int((s <= Q).sum())), calls_companion=g.q_companion_panel.apply(lambda s: int((s <= Q).sum()))))
    rows = []
    for ((st, de), h) in per.groupby(['study', 'design']):
        rows.append(dict(study=st, design=de, panels=int(len(h)), tests_per_panel_median=float(h.n_tests.median()), tests_per_panel_min=int(h.n_tests.min()), calls_tstd_total=int(h.calls_tstd.sum()), calls_tstd_per_panel=float(h.calls_tstd.mean()), share_panels_tstd_call=float((h.calls_tstd > 0).mean()), calls_companion_total=int(h.calls_companion.sum()), calls_companion_per_panel=float(h.calls_companion.mean()), share_panels_companion_call=float((h.calls_companion > 0).mean()), distinct_units_tstd_called=int(P[(P.study == st) & (P.design == de) & (P.q_tstd_panel <= Q)].unit_id.nunique())))
    summ = pd.DataFrame(rows)
    return (P, summ, dict(members_unmapped=nmiss, duplicate_members_dropped=ndup))

def main():
    import sys
    tag = sys.argv[1] if len(sys.argv) > 1 else ''
    dO = pd.read_parquet(os.path.join(C.RES, f'screen_oneil{tag}.parquet'))
    dA = pd.read_parquet(os.path.join(C.RES, f'screen_almanac{tag}.parquet'))
    for d in (dO, dA):
        if 'error' in d and d['error'].notna().any():
            raise SystemExit(f"errors in screen: {d['error'].notna().sum()}")
    dO = add_q(dO, 'ONEIL|' + dO.cell_line + '|batch' + dO.batch.astype(str))
    dA = add_q(dA, 'ALMANAC|' + dA.cell_line)
    pool = pool_size(dA)
    floor = 1.0 / (2.0 * pool + 1.0)
    dA['p_tstd_floored'] = np.maximum(dA.p_tstd, floor)
    dA['bank_pool_sets'] = pool
    dA['q_tstd_floored'] = np.nan
    for (f, idx) in dA.groupby('family').groups.items():
        dA.loc[idx, 'q_tstd_floored'] = S.bh_qvalues(dA.loc[idx, 'p_tstd_floored'].values, 'one')
    dO['bank_pool_sets'] = pool_size(dO)
    dO.to_parquet(os.path.join(C.RES, f'screen_oneil_q{tag}.parquet'), index=False)
    dA.to_parquet(os.path.join(C.RES, f'screen_almanac_q{tag}.parquet'), index=False)
    out = {'null_tested': C.NULL_NOTE, 'ONEIL': counts(dO, 'oneil'), 'ALMANAC': counts(dA, 'almanac')}
    fam = []
    for d in (dO, dA):
        for (f, g) in d.groupby('family'):
            fam.append(dict(family=f, study=g.study.iloc[0], cell_line=g.cell_line.iloc[0], units=len(g), bh_tstd=int((g.q_tstd <= Q).sum()), bh_companion=int((g.q_companion <= Q).sum()), bh_companion_relnull=int((g.q_companion_relnull <= Q).sum()), nominal_tstd=int((g.p_tstd <= 0.05).sum()), median_mean_excess=float(g.mean_excess.median()), median_activity=float(g.activity.median())))
    pd.DataFrame(fam).to_csv(os.path.join(C.RES, f'family_calls{tag}.csv'), index=False)
    cl = pd.concat([drug_clusters(dO), drug_clusters(dA)], ignore_index=True)
    cl.to_csv(os.path.join(C.RES, f'drug_clusters{tag}.csv'), index=False)
    for st in ('ONEIL', 'ALMANAC'):
        c = cl[cl.study == st] if len(cl) else cl
        if len(c):
            top = c.iloc[0]
            out[st]['drug_clusters_ge3_calls'] = int((c.n_calls >= 3).sum())
            out[st]['drug_clusters_ge5_calls'] = int((c.n_calls >= 5).sum())
            out[st]['largest_drug_cluster'] = dict(family=top.family, drug=top.drug, n_calls=int(top.n_calls), n_tested=int(top.n_tested), family_calls=int(top.family_calls), expected_if_random=float(top.expected_if_random))
            sh = []
            for (f, h) in c.groupby('family'):
                if h.family_calls.iloc[0] >= 2:
                    sh.append(h.n_calls.max() / h.family_calls.iloc[0])
            out[st]['median_share_calls_on_top_drug'] = float(np.median(sh)) if sh else np.nan
    # Same O'Neil SIR p-values: BH within the declared families against BH pooled over all O'Neil units.
    fam_call = (dO.q_tstd <= Q).values
    pooled_call = S.bh_qvalues(dO.p_tstd.values, 'one') <= Q
    pooled = dict(q=Q, units=int(len(dO)), families=int(dO.family.nunique()), within_families=int(fam_call.sum()),
                  pooled=int(pooled_call.sum()), both=int((fam_call & pooled_call).sum()),
                  families_only=int((fam_call & ~pooled_call).sum()), pooled_only=int((~fam_call & pooled_call).sum()))
    with open(os.path.join(C.RES, f'bh_pooled_vs_families{tag}.json'), 'w') as f:
        json.dump(pooled, f, indent=1)
    (P, ps, pinfo) = panels(dO, dA)
    P.to_parquet(os.path.join(C.RES, f'panel_calls{tag}.parquet'), index=False)
    ps.to_csv(os.path.join(C.RES, f'panel_summary{tag}.csv'), index=False)
    out['panels'] = pinfo
    with open(os.path.join(C.RES, f'family_counts{tag}.json'), 'w') as f:
        json.dump(out, f, indent=1, default=float)
    print(json.dumps(out, indent=1, default=float))
    print(ps.to_string())
if __name__ == '__main__':
    main()
