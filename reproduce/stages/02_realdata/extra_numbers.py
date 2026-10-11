"""DrugCombDB extra_numbers calculations."""
import json
import os
import numpy as np
import pandas as pd
import common as C
S = C.S

def main():
    dO = pd.read_parquet(os.path.join(C.RES, 'screen_oneil_q.parquet'))
    dA = pd.read_parquet(os.path.join(C.RES, 'screen_almanac_q.parquet'))
    out = {}
    out['oneil_p_tstd_at_minimum_1e-4'] = int((dO.p_tstd <= 0.00010001).sum())
    out['oneil_bh_tstd_calls_with_p_at_minimum'] = int(((dO.p_tstd <= 0.00010001) & (dO.q_tstd <= 0.05)).sum())
    out['oneil_share_units_bh_tstd'] = float((dO.q_tstd <= 0.05).mean())
    for b in (0, 1):
        k = dO.batch == b
        out[f'oneil_batch{b}_units'] = int(k.sum())
        out[f'oneil_batch{b}_share_bh_tstd'] = float((dO.q_tstd[k] <= 0.05).mean())
        out[f'oneil_batch{b}_share_p_tstd_le_0.05'] = float((dO.p_tstd[k] <= 0.05).mean())
    out['oneil_median_mean_excess'] = float(dO.mean_excess.median())
    out['oneil_median_activity'] = float(dO.activity.median())
    out['almanac_median_mean_excess'] = float(dA.mean_excess.median())
    out['almanac_median_activity'] = float(dA.activity.median())
    out['almanac_share_p_tstd_le_0.05'] = float((dA.p_tstd <= 0.05).mean())
    ns=pd.read_csv(os.path.join(C.RES,'null_summary.csv'))
    null_rate=ns[(ns.study=='almanac') & (ns.column=='p_tstd') & (ns.level==.05)].rate.iloc[0]
    out['almanac_expected_p_tstd_le_0.05_under_null_check_rate_0.0395'] = float(round(null_rate,4) * len(dA))
    null_data=pd.read_parquet(os.path.join(C.RES,'null_almanac.parquet'))
    minimum_rate=float((null_data.p_tstd<=.00010001).mean())
    out['almanac_expected_p_at_minimum_under_null_check_rate_0.0096'] = float(round(minimum_rate,4) * len(dA))
    W = pd.read_parquet(os.path.join(C.RES, 'window_units.parquet'))
    wa = W[(W.study == 'ALMANAC') & W.testable].copy()
    pool = dA.set_index('unit_id').loc[wa.unit_id, 'bank_pool_sets'].values
    fl = 1.0 / (2 * pool + 1)
    for pre in ('full', 'win'):
        p = np.maximum(wa[pre + '_p_tstd'].astype(float).values, fl)
        q = np.full(len(wa), np.nan)
        for (f, idx) in wa.reset_index(drop=True).groupby('family').groups.items():
            q[idx] = S.bh_qvalues(p[idx])
        out[f'almanac_window_{pre}_bh_tstd_calls_floored'] = int(np.sum(q <= 0.05))

    def wall(log):
        for line in open(os.path.join(C.LOGS, log)):
            if 'Elapsed (wall clock)' in line:
                t = line.strip().split(': ')[-1].split(':')
                return float(t[-2]) * 60 + float(t[-1]) + (float(t[0]) * 3600 if len(t) == 3 else 0)
        return np.nan
    for (st, n) in (('oneil', len(dO)), ('almanac', len(dA))):
        w = wall(f'screen_{st}.log')
        out[f'runtime_screen_{st}_wall_s'] = w
        out[f'runtime_screen_{st}_core_ms_per_unit'] = 1000 * C.MAX_WORKERS * w / n
    for st in ('oneil', 'almanac'):
        out[f'runtime_null_{st}_wall_s'] = wall(f'null_{st}.log')
    with open(os.path.join(C.RES, 'extra_numbers.json'), 'w') as f:
        json.dump(out, f, indent=1)
    print(json.dumps(out, indent=1))
if __name__ == '__main__':
    main()
