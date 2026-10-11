"""Summarize measured BIGL runtime by configuration and replication."""
import paths
import bigl_summary as B
import pandas as pd
b=B.load_bigl()
b['r']=b.setting.str.extract(r'^r(\d+)_')[0].astype(float)
rows=[]
for (config,r),g in b[b.cond!='real'].groupby(['config','r']):
    v=g.t_total
    rows.append(dict(method='BIGL '+config+' (B.B = 100, B.CP = 50)',r=int(r),n=len(v),median=v.median(),mean=v.mean(),p95=v.quantile(.95)))
pd.DataFrame(rows).to_csv(paths.INPUTS/'bigl_runtime.csv',index=False)
