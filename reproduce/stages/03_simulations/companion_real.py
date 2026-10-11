"""Mean-test null rates on O'Neil single blocks and ALMANAC orientation differences."""
import paths
import json
import numpy as np
import pandas as pd
from scipy import stats
from sirsyn import seed_for
from companion_core import v_kc


def main():
    b=np.load(paths.INPUTS/'noise_bank_oneil.npz');don=np.load(paths.INPUTS/'oneil_donors.npz')
    sel=np.where(b['group_size']==4)[0];raw=b['Y_raw'];yb=b['ybar_raw']
    half=np.array([seed_for(bytes([114,50,98]).decode('ascii'),'half',c)%2 for c in b['cell_line']])
    Z=-raw[sel]
    E=(Z[:,:,1:,:1]+Z[:,:,:1,1:]-Z[:,:,:1,:1]-Z[:,:,1:,1:]).reshape(len(sel),4,-1)
    V=E.mean(2).var(1,ddof=1);pe=E.var(1,ddof=1).mean(1)/E.shape[2]
    m=yb[sel,1:,1:].mean((1,2));bins=np.array([-1.,.1,.3,.5,.7,.9,10.]);bi=np.digitize(m,bins)-1
    centers=np.array([m[bi==j].mean() for j in range(len(bins)-1)])
    table=np.zeros((2,len(centers)))
    for h in (0,1):
        for j in range(len(centers)):
            k=(bi==j)&(half[sel]==h);table[h,j]=max(0.,V[k].mean()-pe[k].mean())
    ids=np.arange(4 if paths.SMOKE else len(sel));s=sel[ids];d=don['donors'][ids,0];k=don['k'][ids,0]
    ea=b['edge_drugA_raw'][s]/b['edge_drugA_raw'][s,:1];eb=b['edge_drugB_raw'][s]/b['edge_drugB_raw'][s,:1]
    mu=ea[:,1:,None]*eb[:,None,1:]
    sd=pd.read_csv(paths.INPUTS/'sd_by_mean.csv').sort_values('mean_viability_raw')
    sdv=lambda x:np.interp(x,sd.mean_viability_raw,sd.pooled_sd)
    interior=mu+b['resid_raw'][d,k]*np.clip(sdv(mu)/sdv(yb[d,1:,1:]),.5,2.)
    observed_mean=np.clip(interior.mean((1,2)),0.,1.5);variance=np.empty(len(ids))
    for h in (0,1):
        use=(1-half[d])==h;variance[use]=np.interp(observed_mean[use],centers,table[h])
    excess=(mu-interior).reshape(len(ids),-1)
    p=stats.t.sf(excess.mean(1)/np.sqrt(excess.var(1,ddof=1)/16+variance),15)
    pd.DataFrame(dict(target=s,donor=d,p_companion=p)).to_parquet(paths.SIMS/'companion_real_oneil.parquet',index=False)
    out={'oneil_single':{'n':len(ids),**{f'rate_{a}':float((p<a).mean()) for a in [.05,.01,.005,.001]}}}
    a=np.load(paths.INPUTS/'almanac_orientation_pairs.npz');y1,y2=a['Y1'].astype(float),a['Y2T'].astype(float)
    if paths.SMOKE:y1,y2=y1[:4],y2[:4]
    ea=(y1[:,:,0]+y2[:,:,0])/2.;eb=(y1[:,0,:]+y2[:,0,:])/2.;ea[:,0]=eb[:,0]=1.
    truth=np.empty_like(y1);truth[:,:,0]=ea;truth[:,0,:]=eb;truth[:,1:,1:]=ea[:,1:,None]*eb[:,None,1:]
    noise=(y1-y2)/np.sqrt(2.);Y=np.concatenate([truth+noise,truth-noise]);Y[:,0,0]=1.
    p=np.nan_to_num(v_kc(Y,'2sp')['p_syn'],nan=1.)
    pd.DataFrame(dict(target=np.arange(len(p)),p_companion=p)).to_parquet(paths.SIMS/'companion_real_almanac.parquet',index=False)
    out['almanac']={'n':len(p),**{f'rate_{a}':float((p<a).mean()) for a in [.05,.01,.005,.001]}}
    (paths.SIMS/'companion_real_numbers.json').write_text(json.dumps(out,indent=2)+'\n');print(out)

if __name__=='__main__':main()
