"""Observed curve shapes, replicate errors, and orientation-pair tail probabilities."""
import paths
import json
import numpy as np
from sirsyn import seed_for
from companion_noise import sd_v_pooled


def main():
    bank=np.load(paths.INPUTS/'noise_bank_oneil.npz')
    sel=np.where(bank['group_size']==4)[0]
    Y=bank['Y_raw'][sel,:,1:,1:]
    errors=np.load(paths.INPUTS/'gross_errors.npz')
    devs=errors['dev'];mo=errors['mean_others']
    out=dict(n_sets=int(sel.size),n_interior_wells=int(sel.size*16),gross_n=int(devs.size),gross_rate_per_well=float(devs.size/(sel.size*16)),gross_share_negative=float((devs<0).mean()))
    out['gross_absdev_quantiles_0_5_25_50_75_95_100']=np.quantile(np.abs(devs),[0,.05,.25,.5,.75,.95,1]).round(4).tolist()
    out['gross_mean_others_quantiles']=np.quantile(mo,[0,.05,.25,.5,.75,.95,1]).round(4).tolist()
    z=np.stack([np.abs(Y[:,k]-np.median(np.delete(Y,k,axis=1),1))/sd_v_pooled(np.median(np.delete(Y,k,axis=1),1)) for k in range(4)],1)
    out['screen_real']={str(c):dict(per_replicate=float((z>c).mean()),per_well_any=float((z>c).any(1).mean()),per_well_ge2=float(((z>c).sum(1)>=2).mean())) for c in (3,4,5,6,8)}
    EA=bank['edge_drugA_raw'][sel]/bank['edge_drugA_raw'][sel][:,:1]
    EB=bank['edge_drugB_raw'][sel]/bank['edge_drugB_raw'][sel][:,:1]
    E=np.concatenate([EA,EB])[:,1:];Eu=np.unique(E.round(8),axis=0)
    above1=(Eu>1).any(1);nonmono=(np.diff(Eu,axis=1)>0).any(1);inc=np.diff(Eu,axis=1).max(1)
    out['edges']=dict(unique_curves=int(len(Eu)),share_curves_above1=float(above1.mean()),share_curves_nonmonotone_nonzero_doses=float(nonmono.mean()),max_value_quantiles_among_above1_25_50_75_90_99=np.quantile(Eu.max(1)[above1],[.25,.5,.75,.9,.99]).round(4).tolist(),argmax_dose_among_above1=np.bincount(Eu[above1].argmax(1),minlength=4).tolist(),largest_increase_quantiles_among_nonmono_25_50_75_90_99=np.quantile(inc[nonmono],[.25,.5,.75,.9,.99]).round(4).tolist(),violation_step_counts_d1d2_d2d3_d3d4=(np.diff(Eu,axis=1)>0).sum(0).tolist(),share_sets_nonmonotone_edge_nonzero_doses=float(((np.diff(EA[:,1:],axis=1)>0).any(1)|(np.diff(EB[:,1:],axis=1)>0).any(1)).mean()),share_sets_some_edge_above1=float(((EA[:,1:]>1).any(1)|(EB[:,1:]>1).any(1)).mean()))
    (paths.INPUTS/'oneil_noise_summary.json').write_text(json.dumps(out,indent=2)+'\n')
    al=np.load(paths.INPUTS/'almanac_orientation_pairs.npz')
    y1,y2=al['Y1'].astype(float),al['Y2T'].astype(float)
    half=np.array([seed_for(bytes([114,50,98]).decode('ascii'),'half',c)%2 for c in al['cell_line']])
    noise=(y1-y2)/np.sqrt(2.);mean=(y1+y2)/2.
    nc=np.ones((4,4),bool);nc[0,0]=False
    bins=np.array([-1.,.05,.1,.2,.3,.4,.5,.6,.7,.8,.9,.95,1.,1.05,1.1,1.2,10.])
    Z=np.empty_like(noise)
    for h in (0,1):
        k=half==h;d,m=noise[k][:,nc],mean[k][:,nc];centers=[];variances=[]
        for lo,hi in zip(bins[:-1],bins[1:]):
            keep=(m>=lo)&(m<hi)
            if keep.sum()<50:continue
            centers.append(float(m[keep].mean()));variances.append(float(np.mean(d[keep]**2)))
        Z[k]=noise[k]/np.sqrt(np.interp(mean[k],centers,variances))
    zz=Z[:,nc]
    out=dict(n_pairs=len(noise),P_abs_z_gt3=float(np.mean(np.abs(zz)>3)),P_abs_z_gt4=float(np.mean(np.abs(zz)>4)),gauss_P_gt3=.0027,gauss_P_gt4=6.3e-5,mean_z2=float(np.mean(zz**2)),kurtosis=float(np.mean(zz**4)/np.mean(zz**2)**2))
    out['tail_ratio_at4']=out['P_abs_z_gt4']/out['gauss_P_gt4']
    (paths.INPUTS/'almanac_noise_summary.json').write_text(json.dumps(out,indent=2)+'\n')
    print(json.dumps(out))

if __name__=='__main__':main()
