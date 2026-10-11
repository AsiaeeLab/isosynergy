"""Construct dose grids, O'Neil replicate sets, orientation pairs and noise summaries."""
import paths
import json
import numpy as np
import pandas as pd


def main():
    raw = pd.read_csv(paths.raw_file('drugcombdb/drugcombs_response.csv'))
    scored = pd.read_csv(paths.raw_file('drugcombdb/drugcombs_scored.csv'))
    scores = scored.rename(columns={'ID':'BlockID','Drug1':'drug1','Drug2':'drug2','Cell line':'cell_line'})
    scores.insert(1, 'experiment_id', scores.BlockID.map(lambda n: f'drugcombdb__block{n:07d}'))
    scores.to_parquet(paths.INPUTS / 'drugcombdb_block_scores.parquet', index=False)
    d = raw[raw.source.isin(['ONEIL','ALMANAC'])].copy()
    d = d.rename(columns={'BlockID':'block_id','Row':'row_index','Col':'col_index','DrugRow':'drugA','DrugCol':'drugB','ConcRow':'doseA','ConcCol':'doseB','source':'study_source'})
    d['response_raw'] = d.Response / 100.0
    d['response'] = np.clip(d.response_raw, 0.0, 1.0)
    d['cell_line'] = d.block_id.map(scores.set_index('BlockID').cell_line)
    d['experiment_id'] = d.block_id.map(lambda n: f'drugcombdb__block{n:07d}')
    for key, unit in [('doseA','ConcRowUnit'),('doseB','ConcColUnit')]:
        u = d[unit].str.strip().str.lower()
        m = u.map({'m':1e6,'mol':1e6,'molar':1e6,'mm':1e3,'mmol':1e3,'um':1.,'µm':1.,'nm':1e-3,'pm':1e-6})
        sel = u.notna() & (u != '')
        d.loc[sel,key] = d.loc[sel,key] * m[sel]
    counts = d.groupby('block_id').agg(ni=('doseA','nunique'), nj=('doseB','nunique'))
    keep = counts[(counts.ni >= 3) & (counts.nj >= 3)].index
    d = d[d.block_id.isin(keep)].sort_values(['block_id','row_index','col_index'], kind='mergesort')
    cols = ['experiment_id','drugA','drugB','cell_line','doseA','doseB','response','block_id','row_index','col_index','study_source']
    d[cols].to_parquet(paths.INPUTS / 'drugcomb_matrices.parquet', index=False)
    d[['block_id','row_index','col_index','response_raw']].to_parquet(paths.INPUTS / 'raw_response.parquet', index=False)
    on = d[d.study_source == 'ONEIL']
    meta = on.iloc[::25][['experiment_id','drugA','drugB','cell_line','block_id']].reset_index(drop=True)
    YR = on.response_raw.to_numpy().reshape(-1,5,5)
    DA = on.doseA.to_numpy().reshape(-1,5,5); DB = on.doseB.to_numpy().reshape(-1,5,5)
    gid, uniq = pd.factorize(meta.drugA + '|' + meta.drugB + '|' + meta.cell_line)
    cnt = np.bincount(gid); order = np.argsort(gid,kind='mergesort'); starts=np.r_[0,np.cumsum(cnt)]
    rows=[]; idx=[]
    for g in range(len(uniq)):
        members=order[starts[g]:starts[g+1]]
        members=members[np.argsort(meta.block_id.values[members])]
        if cnt[g] == 4:
            batches=[members]
        elif cnt[g] == 8:
            batches=[members[:4],members[4:]]
        else:
            raise ValueError(f'Unexpected replicate count {cnt[g]}')
        for bi, mem in enumerate(batches):
            rows.append(dict(group_key=uniq[g], group_size=int(cnt[g]), batch=bi, drugA=meta.drugA.values[mem[0]],drugB=meta.drugB.values[mem[0]],cell_line=meta.cell_line.values[mem[0]]))
            idx.append(mem)
    sets=pd.DataFrame(rows); idx=np.array(idx); Y=YR[idx]; ybar=Y.mean(1); resid=(Y[:,:,1:,1:]-ybar[:,None,1:,1:])*np.sqrt(4./3.)
    bank={k:sets[k].values.astype(str) for k in ['group_key','drugA','drugB','cell_line']}
    bank.update(group_size=sets.group_size.to_numpy(np.int8),batch=sets.batch.to_numpy(np.int8),experiment_id=meta.experiment_id.values[idx].astype(str),block_id=meta.block_id.values[idx].astype(np.int32),Y_raw=Y,ybar_raw=ybar,resid_raw=resid,edge_drugA_raw=ybar[:,:,0],edge_drugB_raw=ybar[:,0,:],doseA=DA[idx[:,0]][:,:,0],doseB=DB[idx[:,0]][:,0,:])
    np.savez_compressed(paths.INPUTS/'noise_bank_oneil.npz',**bank)
    # The SD curve uses all replicate sets and viability-bin boundaries.
    yi=Y[:,:,1:,1:].reshape(len(Y),4,16); vm=yi.mean(1); e=(yi-vm[:,None])*np.sqrt(4./3.); var=(e**2).mean(1)
    bins=np.array([-np.inf,.05,.1,.2,.3,.4,.5,.6,.7,.8,.9,.95,1.,1.05,1.1,np.inf]); labels=np.digitize(vm,bins[1:-1]); sd=[]
    for b in range(len(bins)-1):
        sel=labels==b
        sd.append(dict(scale='identity_raw',bin_lo=bins[b],bin_hi=bins[b+1],n_wells=int(sel.sum()),mean_viability_raw=float(vm[sel].mean()),pooled_sd=float(np.sqrt(var[sel].mean())),robust_sd=float(1.4826*np.median(np.abs(e.transpose(0,2,1)[sel].ravel())))))
    pd.DataFrame(sd).to_csv(paths.INPUTS/'sd_by_mean.csv',index=False)
    al=d[d.study_source=='ALMANAC']; am=al.iloc[::16][['block_id','drugA','drugB','cell_line']].reset_index(drop=True)
    ya=al.response_raw.to_numpy().reshape(-1,4,4); da=al.doseA.to_numpy().reshape(-1,4,4)[:,:,0]; db=al.doseB.to_numpy().reshape(-1,4,4)[:,0,:]
    am['key']=[f'{min(a,b)}|{max(a,b)}|{c}' for a,b,c in zip(am.drugA,am.drugB,am.cell_line)]
    c=am.key.value_counts(); two=am[am.key.isin(c[c==2].index)].sort_values('block_id')
    result={k:[] for k in ['Y1','Y2T','cell_line','drugA','drugB','block1','block2','doseA','doseB']}
    for _, pair in two.groupby('key',sort=True):
        i,j=pair.index; r=pair.iloc[0]; a=ya[i].copy(); b=ya[j].T.copy();a[0,0]=b[0,0]=1.
        assert np.allclose(da[i],db[j],rtol=1e-6) and np.allclose(db[i],da[j],rtol=1e-6)
        for k,v in dict(Y1=a,Y2T=b,cell_line=r.cell_line,drugA=r.drugA,drugB=r.drugB,block1=am.block_id[i],block2=am.block_id[j],doseA=da[i],doseB=db[i]).items():result[k].append(v)
    np.savez_compressed(paths.INPUTS/'almanac_orientation_pairs.npz',**{k:np.array(v) for k,v in result.items()})
    sel=np.where(bank['group_size']==4)[0]; yi=Y[sel,:,1:,1:];devs=[];means=[]
    for k in range(4):
        others=np.delete(yi,k,axis=1);m3=others.mean(1);s3=others.std(1,ddof=1);diff=yi[:,k]-m3;hit=(np.abs(diff)>.5)&(s3<.05);devs.append(diff[hit]);means.append(m3[hit])
    devs=np.concatenate(devs)
    np.savez(paths.INPUTS/'gross_errors.npz',absdev=np.abs(devs),dev=devs,mean_others=np.concatenate(means))
    print(f'Built {len(sets)} replicate sets and {len(result["Y1"])} orientation pairs.',flush=True)

if __name__ == '__main__':
    main()
