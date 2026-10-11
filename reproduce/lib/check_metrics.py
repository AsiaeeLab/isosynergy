"""Headline quantities derived independently from the released per-unit table."""
import numpy as np
import pandas as pd
from scipy import stats
import sirsyn as S
import common as C


def release_metrics(d):
    out={'rows':len(d),'columns':len(d.columns)}
    for name,study in [('oneil','ONEIL'),('almanac','ALMANAC')]:
        g=d[d.study==study].copy()
        hit=g.q_tstd<=.05; comp=g.q_companion<=.05
        pairs=np.where(g.drug_row<g.drug_col,g.drug_row+'|'+g.drug_col,g.drug_col+'|'+g.drug_row)
        values=dict(units=len(g),families=g.family.nunique(),cell_lines=g.cell_line.nunique(),drugs=pd.concat([g.drug_row,g.drug_col]).nunique(),pairs=len(set(pairs)),bh_calls=int(hit.sum()),bh_share=float(hit.mean()),bh_lines=g.loc[hit,'cell_line'].nunique(),bh_pairs=len(set(pairs[hit])),companion_bh=int(comp.sum()),bh_both=int((hit&comp).sum()),nominal=int((g.p_tstd<=.05).sum()),nominal_share=float((g.p_tstd<=.05).mean()),p_floor=int((g.p_tstd<=.00010001).sum()),bh_at_floor=int((hit&(g.p_tstd<=.00010001)).sum()),scan_calls=int((g.p_scan<=.05).sum()),floored_calls=int((g.q_tstd_floored<=.05).sum()),screened=int(g.n_screened.sum()),units_screened=int((g.n_screened>0).sum()),missing=int((g.n_missing_cells>0).sum()),dropped=int((g.n_dropped_doses>0).sum()),stage2=int((g.mc_stage==2).sum()))
        sel=g.p_tstd<=.05
        values['cell_fragility']=float((g.loc[sel,'fragility_cell_p']>.05).mean())
        values['sa_fragility']=float((g.loc[sel,'fragility_sa_p']>.05).mean())
        q=np.full(len(g),np.nan)
        g=g.reset_index(drop=True)
        for _,idx in g.groupby('family').groups.items():q[idx]=S.bh_qvalues(g.loc[idx,'p_tstd'].to_numpy(),'one')
        values['q_max_difference']=float(np.max(np.abs(q-g.q_tstd)))
        out.update({name+'.'+k:v for k,v in values.items()})
    o=d[d.study=='ONEIL'].copy()
    o['key']=o.unit_id.str.replace(r'\|batch[01]$','',regex=True)
    p=o.pivot(index='key',columns='batch',values=['activity','mean_excess','mean_excess_se','t_std','q_tstd']).dropna()
    out['oneil.batch_pairs']=len(p)
    c0=p['q_tstd'][0]<=.05;c1=p['q_tstd'][1]<=.05
    out['oneil.batch_first']=int(c0.sum());out['oneil.batch_second']=int(c1.sum());out['oneil.batch_both']=int((c0&c1).sum());out['oneil.batch_expected']=float(c0.sum()*c1.sum()/len(p))
    po=np.mean(c0==c1);pe=c0.mean()*c1.mean()+(1-c0.mean())*(1-c1.mean())
    out['oneil.batch_kappa']=float((po-pe)/(1-pe))
    for col in ['activity','mean_excess','t_std']:
        r=C.spearman_boot(p[col][0].to_numpy(),p[col][1].to_numpy(),seed=11)
        for k in ['spearman','lo','hi']:out['oneil.repro.'+col+'.'+k]=r[k]
    z=(p.mean_excess[0]-p.mean_excess[1])/np.sqrt(p.mean_excess_se[0]**2+p.mean_excess_se[1]**2)
    out['oneil.batch_z_sd']=float(z.std(ddof=1));out['oneil.batch_z_robust']=float(1.4826*np.median(np.abs(z-np.median(z))))
    out['oneil.batch_mean0']=float(p.mean_excess[0].mean());out['oneil.batch_mean1']=float(p.mean_excess[1].mean())
    a=d[d.study=='ALMANAC'].copy();a['key']=a.cell_line+'|'+np.where(a.drug_row<a.drug_col,a.drug_row+'|'+a.drug_col,a.drug_col+'|'+a.drug_row)
    cnt=a.key.value_counts();a=a[a.key.isin(cnt[cnt==2].index)].sort_values(['key','unit_id'])
    first=a.iloc[::2];second=a.iloc[1::2];out['almanac.orientation_pairs']=len(first)
    for col in ['activity','mean_excess','t_std']:
        r=C.spearman_boot(first[col].to_numpy(),second[col].to_numpy(),seed=12)
        for k in ['spearman','lo','hi']:out['almanac.repro.'+col+'.'+k]=r[k]
    ex=d[d.unit_id=='TEMOZOLOMIDE|MK-4827|EFM192B|batch0'].iloc[0]
    for col in ['t_std','p_tstd','mean_excess','mean_excess_se','activity']:out['example.'+col]=float(ex[col])
    half=stats.t.ppf(.975,int(ex.n_replicates)-1)*ex.mean_excess_se
    out['example.mean_ci_lo']=float(ex.mean_excess-half)
    out['example.mean_ci_hi']=float(ex.mean_excess+half)
    f=d[d.family==ex.family];out['example.family_units']=len(f);out['example.family_bh']=int((f.q_tstd<=.05).sum())
    return out
