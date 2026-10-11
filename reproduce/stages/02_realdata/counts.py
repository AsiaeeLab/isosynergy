"""DrugCombDB counts calculations."""
import json
import os
import numpy as np
import pandas as pd
import common as C

def J(name):
    with open(os.path.join(C.RES, name)) as f:
        return json.load(f)

def main():
    rows = []

    def add(desc, val, src):
        rows.append(dict(description=desc, value=val, source=src))
    pc = J('prepare_checks.json')
    dO = pd.read_parquet(os.path.join(C.RES, 'screen_oneil_q.parquet'))
    dA = pd.read_parquet(os.path.join(C.RES, 'screen_almanac_q.parquet'))
    add('raw DrugCombDB blocks, ONEIL source', pc['raw_blocks_by_source']['ONEIL'], 'prepare_checks.json')
    add('raw DrugCombDB blocks, ALMANAC source', pc['raw_blocks_by_source']['ALMANAC'], 'prepare_checks.json')
    add('raw DrugCombDB blocks, other sources (CLOUD, nih; not analyzed)', pc['raw_blocks_by_source']['CLOUD'] + pc['raw_blocks_by_source']['nih'], 'prepare_checks.json')
    add('ALMANAC blocks excluded as incomplete (fewer than 16 wells)', len(pc['almanac_blocks_incomplete']), 'prepare_checks.json (block 359695, 2 wells)')
    add("O'Neil drug pair x cell line groups", int(dO.group_key.nunique()), 'screen_oneil_q.parquet')
    add("O'Neil four-block groups (one batch)", int((dO.group_size == 4).sum()), 'screen_oneil_q.parquet')
    add("O'Neil eight-block groups (two batches, analyzed per batch)", int(dO[dO.group_size == 8].group_key.nunique()), 'screen_oneil_q.parquet')
    add("O'Neil analyzed units (group x batch, 4 blocks each)", int(len(dO)), 'screen_oneil_q.parquet')
    add("O'Neil blocks analyzed", 4 * int(len(dO)), 'screen_oneil_q.parquet')
    add("O'Neil cell lines", int(dO.cell_line.nunique()), 'screen_oneil_q.parquet')
    add("O'Neil drugs", int(pd.concat([dO.drugA, dO.drugB]).nunique()), 'screen_oneil_q.parquet')
    pO = np.where(dO.drugA < dO.drugB, dO.drugA + '+' + dO.drugB, dO.drugB + '+' + dO.drugA)
    add("O'Neil unordered drug pairs", int(pd.Series(pO).nunique()), 'screen_oneil_q.parquet')
    add("O'Neil testing families (cell line x batch)", int(dO.family.nunique()), 'screen_oneil_q.parquet')
    add('ALMANAC analyzed plates', int(len(dA)), 'screen_almanac_q.parquet')
    add('ALMANAC cell lines (= families)', int(dA.cell_line.nunique()), 'screen_almanac_q.parquet')
    add('ALMANAC drug names (DrugCombDB spellings)', int(pd.concat([dA.drugA, dA.drugB]).nunique()), 'screen_almanac_q.parquet')
    pA = np.where(dA.drugA < dA.drugB, dA.drugA + '+' + dA.drugB, dA.drugB + '+' + dA.drugA)
    add('ALMANAC unordered drug pairs', int(pd.Series(pA).nunique()), 'screen_almanac_q.parquet')
    for (st, d) in (("O'Neil", dO), ('ALMANAC', dA)):
        add(f'{st} units with a fitting error (untestable)', int(d['error'].notna().sum()) if 'error' in d else 0, 'screen parquet')
        add(f'{st} units with a degenerate companion SE (companion untestable)', int(d.companion_degenerate.astype(bool).sum()), 'screen parquet')
        add(f'{st} units with a dropped single-agent dose (rule R)', int((d.n_dropped_doses > 0).sum()), 'screen parquet')
        add(f'{st} units with a missing combination well', int((d.n_missing_cells > 0).sum()), 'screen parquet')
        add(f'{st} units sent to the second Monte Carlo stage', int((d.stage == 2).sum()), 'screen parquet')
    add("O'Neil units with at least one screened replicate value", int((dO.n_screened > 0).sum()), 'screen_oneil_q.parquet')
    add("O'Neil screened replicate values (of 23,052 x 64 combination-well values)", int(dO.n_screened.sum()), 'screen_oneil_q.parquet')
    nO = pd.read_parquet(os.path.join(C.RES, 'null_oneil.parquet'))
    nA = pd.read_parquet(os.path.join(C.RES, 'null_almanac.parquet'))
    add("null check targets, O'Neil (four-block sets, indep4 construction)", int(len(nO)), 'null_oneil.parquet')
    add('null check targets, ALMANAC (orientation pairs x 2 signs)', int(len(nA)), 'null_almanac.parquet')
    rp = pd.read_parquet(os.path.join(C.RES, 'repro_pairs_oneil.parquet'))
    add("O'Neil groups measured in both batches", int(len(rp)), 'repro_pairs_oneil.parquet')
    add('  of which single-agent edges identical across batches', int(rp.edges_identical.sum()), 'repro_pairs_oneil.parquet')
    ra = pd.read_parquet(os.path.join(C.RES, 'repro_pairs_almanac.parquet'))
    add('ALMANAC orientation pairs (two plates)', int(len(ra)), 'repro_pairs_almanac.parquet')
    add('  of which identical copies (dropped)', int(ra.identical_copies.sum()), 'repro_pairs_almanac.parquet')
    rc = pd.read_parquet(os.path.join(C.RES, 'repro_pairs_cross.parquet'))
    add('cross-study matches (pair x cell line in both studies)', int(len(rc)), 'repro_pairs_cross.parquet')
    add('  cell lines of the cross-study matches', int(rc.cell_line.nunique()), 'repro_pairs_cross.parquet')
    fc = J('family_counts.json')
    for st in ('ONEIL', 'ALMANAC'):
        for k in ('bh_calls_tstd', 'bh_calls_tstd_cell_lines', 'bh_calls_tstd_drug_pairs', 'bh_calls_companion', 'bh_calls_tstd_and_companion', 'bh_calls_companion_relnull', 'nominal_tstd_p_le_0.05', 'scan_any_region_claimed_0.05', 'scan_high_dose_rectangle_claimed_0.05'):
            add(f'{st} {k}', fc[st][k], 'family_counts.json')
    add('ALMANAC bh_calls_tstd_resolution_floored', fc['ALMANAC']['bh_calls_tstd_resolution_floored'], 'family_counts.json')
    add('ALMANAC T_std p-values at the Monte Carlo minimum 1e-4', fc['ALMANAC']['tstd_p_at_minimum_1e-4'], 'family_counts.json')
    ps = pd.read_csv(os.path.join(C.RES, 'panel_summary.csv'))
    for r in ps.itertuples():
        add(f'planned panels, {r.study} {r.design}', int(r.panels), 'panel_summary.csv')
    w = J('window_summary.json')
    for st in ('oneil', 'almanac'):
        for k in ('drugs_with_cmax_present', 'units_both_drugs_with_cmax', 'units_untestable_after_restriction', 'units_testable', 'units_window_keeps_all_doses', 'tstd_bh_calls_full', 'tstd_bh_calls_window'):
            add(f'window {st} {k}', w[st][k], 'window_summary.json')
    cs = pd.read_csv(os.path.join(C.RES, 'controls_summary.csv'))
    for r in cs.itertuples():
        add(f'control clusters: {r.study} {r.group}', int(r.n_clusters), 'controls_summary.csv')
    mw = J('missing_well_summary.json')
    add('missing-well holdout: sets', mw['sets'], 'missing_well_summary.json')
    add('missing-well holdout: wells held out', mw['wells_held_out'], 'missing_well_summary.json')
    out = pd.DataFrame(rows)
    out.to_csv(os.path.join(C.RES, 'counts_reconciliation.csv'), index=False)
    pd.set_option('display.width', 250)
    pd.set_option('display.max_colwidth', 110)
    print(out.to_string())
if __name__ == '__main__':
    main()
