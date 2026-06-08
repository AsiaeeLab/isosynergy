import pyarrow.parquet as pq, pandas as pd, numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
def bh(p):
    p=np.asarray(p); n=len(p); o=np.argsort(p); q=np.empty(n); r=p[o]*n/np.arange(1,n+1); q[o]=np.clip(np.minimum.accumulate(r[::-1])[::-1],0,1); return q
rel=pq.read_table('dataset_release/sir_drugcombdb_synergy_calls.parquet',
    columns=['matrix_id','cell_line','bliss_score','hsa_score','loewe_score','zip_score']).to_pandas()
adp=pq.read_table('results/drugcombdb_sir_adaptive_final.parquet').to_pandas()
m=adp.merge(rel,left_on='experiment_id',right_on='matrix_id',how='left')
m=m[m.cell_line_x!='not_available'].copy()
m['hit']=False
for cl,idx in m.groupby('cell_line_x').groups.items():
    m.loc[idx,'hit']=bh(m.loc[idx,'p_final'].values)<=0.05

bases=['bliss_score','zip_score','loewe_score','hsa_score']; labs=['Bliss','ZIP','Loewe','HSA']
fig,ax=plt.subplots(1,2,figsize=(10,4.0))
# Panel A: enrichment (hits vs non-hits) boxplots
data=[]; pos=[]; cols=[]
for i,b in enumerate(bases):
    h=m.loc[m.hit,b].dropna(); nh=m.loc[~m.hit,b].dropna()
    data+=[nh.values,h.values]; pos+=[i*2-0.35,i*2+0.35]; cols+=['#bbbbbb','#c0392b']
bp=ax[0].boxplot(data,positions=pos,widths=0.6,showfliers=False,patch_artist=True)
for patch,c in zip(bp['boxes'],cols): patch.set_facecolor(c)
ax[0].set_xticks([i*2 for i in range(4)]); ax[0].set_xticklabels(labs)
ax[0].axhline(0,color='k',lw=0.6,ls=':')
ax[0].set_ylabel('Baseline synergy score'); ax[0].set_title('A  SIR hits are enriched for baseline synergy')
ax[0].set_ylim(-25,30)
from matplotlib.patches import Patch
ax[0].legend(handles=[Patch(facecolor='#bbbbbb',label='non-hit'),Patch(facecolor='#c0392b',label='SIR hit')],loc='upper left',fontsize=8,frameon=False)
# Panel B: per-cell-line Jaccard with top-5% of each baseline (signal-rich cell lines)
cls=['SKMEL30','VCAP','ES2','MDAMB436','A2058','RPMI7951','KPL1','CAOV3']
J={b:[] for b in bases[:3]}
for cl in cls:
    s=m[m.cell_line_x==cl]; H=set(s.loc[s.hit,'experiment_id'])
    for b in bases[:3]:
        thr=s[b].quantile(0.95); T=set(s.loc[s[b]>=thr,'experiment_id'])
        J[b].append(len(H&T)/len(H|T) if (H|T) else 0)
x=np.arange(len(cls)); w=0.25
for k,(b,c) in enumerate(zip(bases[:3],['#2980b9','#27ae60','#8e44ad'])):
    ax[1].bar(x+(k-1)*w,J[b],w,label=b.split('_')[0],color=c)
ax[1].set_xticks(x); ax[1].set_xticklabels(cls,rotation=45,ha='right',fontsize=7)
ax[1].set_ylabel('Jaccard overlap'); ax[1].set_ylim(0,0.5)
ax[1].set_title('B  SIR hits vs each baseline top-5% (per cell line)')
ax[1].legend(fontsize=8,frameon=False)
ax[1].axhline(0.05,color='k',lw=0.6,ls=':')
plt.tight_layout(); plt.savefig('doc/figures/baseline_concordance.pdf',bbox_inches='tight')
print('saved doc/figures/baseline_concordance.pdf')
print('Panel A medians: ', {b:(round(m.loc[m.hit,b].median(),2),round(m.loc[~m.hit,b].median(),2)) for b in bases})
print('Panel B mean Jaccard:', {b:round(np.mean(J[b]),3) for b in bases[:3]})
