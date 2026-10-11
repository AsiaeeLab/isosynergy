"""DrugCombDB null_summary calculations."""
import json
import os
import numpy as np
import pandas as pd
import common as C
LEVELS = [0.05, 0.01, 0.005, 0.001, 0.0001]
TESTS = {'oneil': [('SIR statistic', 'p_tstd'), ('region scan (global, = any region claim)', 'p_scan'), ('companion: block-level t (primary)', 'p_companion'), ('companion: fixed-edge t + block variance / 4', 'p_fixed_edge'), ('bootstrap mean excess (same reference)', 'p_mean_boot')], 'almanac': [('SIR statistic', 'p_tstd'), ('region scan (global, = any region claim)', 'p_scan'), ('companion: V2Sp', 'p_companion'), ('bootstrap mean excess (same reference)', 'p_mean_boot')]}

def main():
    rows = []
    extra = {}
    for study in ('oneil', 'almanac'):
        d = pd.read_parquet(os.path.join(C.RES, f'null_{study}.parquet'))
        for (lab, col) in TESTS[study]:
            for lv in LEVELS:
                r = C.rate_row(d[col], lv)
                r.update(study=study, test=lab, column=col, ratio_to_nominal=r['rate'] / lv)
                rows.append(r)
        extra[study] = dict(n=int(len(d)), stage2_share=float((d.stage == 2).mean()), screened_per_target=float(d.n_screened.mean()), reference_flags_per_plate=float(d.reference_flags_per_plate.mean()))
        if study == 'almanac':
            f = d[d.fragility_computed.astype(bool)]
            extra[study]['null_calls_with_fragility'] = int(len(f))
            extra[study]['null_calls_loco_flag_share'] = float(f.loco_flag.astype(bool).mean()) if len(f) else np.nan
            extra[study]['null_calls_sa_flag_share'] = float(f.sa_flag.astype(bool).mean()) if len(f) else np.nan
            pool = d.log_notes.str.extract('(\\d+) eligible sets')[0].astype(float)
            extra[study]['bank_pool_sets_median'] = float(pool.median())
            extra[study]['bank_pool_sets_range'] = [float(pool.min()), float(pool.max())]
            extra[study]['distinct_reference_noise_matrices_median'] = float(2 * pool.median())
            extra[study]['note'] = 'single plates draw one residual of one donor pair (2 positions = 2 signs): at most 2 x pool distinct reference noise matrices, so p-values below about 1 / (2 x pool) are not resolved by the bank even with B2 = 9,999'
        else:
            pool = d.log_notes.str.extract('(\\d+) eligible sets')[0].astype(float)
            extra[study]['bank_pool_sets_median'] = float(pool.median())
            extra[study]['bank_pool_sets_range'] = [float(pool.min()), float(pool.max())]
    out = pd.DataFrame(rows)[['study', 'test', 'column', 'level', 'k', 'n', 'rate', 'lo', 'hi', 'ratio_to_nominal', 'n_nan']]
    out.to_csv(os.path.join(C.RES, 'null_summary.csv'), index=False)
    with open(os.path.join(C.RES, 'null_bank_resolution.json'), 'w') as f:
        json.dump(extra, f, indent=1)
    for study in ('oneil', 'almanac'):
        print(f'\n{study}: {extra[study]}')
        s = out[out.study == study]
        for lab in s.test.unique():
            t = s[s.test == lab]
            print(f'  {lab}: ' + '; '.join((f'{r.level:g}: {r.rate:.4f} [{r.lo:.4f}, {r.hi:.4f}]' for r in t.itertuples())))
if __name__ == '__main__':
    main()
