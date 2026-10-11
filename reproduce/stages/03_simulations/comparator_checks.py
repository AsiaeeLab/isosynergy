"""Pooling and complete-pairwise contrast calculations for Supplementary Section S5."""
import paths
import json
from multiprocessing import Pool
import numpy as np
import pandas as pd
from scipy import stats
from sirsyn import seed_for
import companion_noise as C


def pooling_job(job):
    n,s,f,l,design=job
    y=C.bliss_ref(n,s)
    total=4 if paths.SMOKE else 20000
    counts=np.zeros(2,int)
    for start in range(0,total,2500):
        m=min(2500,total-start)
        rng=np.random.default_rng(seed_for(bytes([114,50,97]).decode('ascii'),'null',design,f,l,s,n,'null','none',0.0,start))
        yr,_=C.simulate(y,'P',f,l,m,rng,m=1 if design=='P' else 4,icc=0.0 if design=='P' else .1)
        if design=='P':
            p=C.sfp_pooled(None,yr[:,0])[0]
        else:
            p=C.scores_block_level(C._pscores(yr,1.0).reshape(m,4,-1))[0]
        counts += [(p<=a).sum() for a in [.05,.005]]
    return dict(design=design,grid=n,surface=s,noise=f,level=l,n=total,rate05=counts[0]/total,rate005=counts[1]/total)


def main():
    rows=[]
    I=5;nA=nB=4;sigma=.1;alpha=.05;nd=4 if paths.SMOKE else 20000
    df=nA+nB+nA*nB-3;critical=stats.t.ppf(1-alpha,df)
    for spread in [0.,.25,.5,1.,2.]:
        for control_noise in [True,False]:
            rng=np.random.default_rng(seed_for(bytes([101,54]).decode('ascii'),'dm',spread,control_noise))
            u=np.r_[0.,-spread*np.arange(1,I)/(I-1)]
            theta=(u[:,None]+u[None,:]).ravel();k=0
            for _ in range(nd):
                z=theta+sigma*rng.standard_normal(I*I)
                if not control_noise:z[0]=0.
                Z=z.reshape(I,I);x,y,zz=Z[1:,0],Z[0,1:],Z[1:,1:]
                variance=(((x-x.mean())**2).sum()+((y-y.mean())**2).sum()+((zz-zz.mean())**2).sum())/df
                t=(x.mean()+y.mean()-zz.mean())/np.sqrt(variance*(1/nA+1/nB+1/(nA*nB)))
                k+=t>critical
            rows.append(dict(single_agent_log_spread=spread,control_noise=control_noise,n=nd,size_demidenko_miller=k/nd))
    pd.DataFrame(rows).to_csv(paths.SIMS/'complete_pairwise_size.csv',index=False)
    jobs=[(n,s,f,l,design) for design in ['P','R1'] for n in [4,5,6,8] for s in C.SURFACES for f,l in [('G',.05),('G',.1),('G',.2),('G',.3),('L',.5),('L',1.),('L',2.),('V',.5),('V',1.),('V',2.),('H',1.),('B',.5),('B',1.),('B',2.),('HB',1.)]]
    with Pool(paths.WORKERS) as pool:rows=pool.map(pooling_job,jobs)
    d=pd.DataFrame(rows);d.to_csv(paths.SIMS/'pooling_size.csv',index=False)
    out=dict(normal_reference_size_df15=float(stats.t.sf(stats.norm.ppf(.95),15)),pooling_max05=float(d[d.design=='P'].rate05.max()),pooling_max005=float(d[d.design=='P'].rate005.max()),block_max05=float(d[d.design=='R1'].rate05.max()),block_max005=float(d[d.design=='R1'].rate005.max()))
    # The displayed analytic example uses the three-decimal normal critical value.
    out['normal_reference_size_df15_rounded_critical'] = float(stats.t.sf(1.644,15))
    empirical = d[(d.design == 'P') & (d.noise == 'V') & (d.level == 1.)]
    out['pooling_empirical_scale_max05'] = float(empirical.rate05.max())
    (paths.SIMS/'comparator_numbers.json').write_text(json.dumps(out,indent=2)+'\n')


if __name__=='__main__':main()
