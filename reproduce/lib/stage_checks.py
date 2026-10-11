"""Consistency checks derived from per-dataset stage outputs."""
import paths
import numpy as np
import pandas as pd
import sirsyn


def stage_metrics(release):
    result={}
    for study in ['oneil','almanac']:
        screen=pd.read_parquet(paths.REAL/f'screen_{study}_q.parquet').set_index('unit_id')
        r=release[release.study==study.upper()].set_index('unit_id').loc[screen.index]
        for col in ['p_tstd','q_tstd','p_scan','p_companion','q_companion','t_std','mean_excess','mean_excess_se','activity']:
            diff=np.abs(screen[col].to_numpy(float)-r[col].to_numpy(float))
            result[f'{study}.release_difference.{col}']=float(np.nanmax(diff))
        n=pd.read_parquet(paths.REAL/f'null_{study}.parquet')
        summary=pd.read_csv(paths.REAL/'null_summary.csv')
        errors=[]
        for row in summary[summary.study==study].itertuples():
            v=n[row.column].to_numpy(float)
            k=int((v<=row.level).sum())
            errors += [abs(k-row.k),abs(len(v)-row.n)]
        result[f'{study}.null_count_difference']=max(errors)
    w=pd.read_parquet(paths.REAL/'window_units.parquet')
    # Within-family BH is recomputed from the fitted p-values in the window table.
    for prefix in ['full','win']:
        for test in ['tstd','companion']:
            pcol=f'{prefix}_p_{test}';qcol=f'{prefix}_q_{test}'
            differences=[]
            for _,g in w[w.testable].groupby(['study','family']):
                q=sirsyn.bh_qvalues(g[pcol].to_numpy(float))
                differences.append(np.nanmax(np.abs(q-g[qcol].to_numpy(float))))
            result[f'window.{prefix}.{test}.q_difference']=float(max(differences))
    return result
