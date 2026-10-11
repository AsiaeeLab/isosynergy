"""Derive the rounded SD curves used by the synthetic noise families."""
import paths
import json
import numpy as np
import pandas as pd
from sirsyn.noise import oneil_sd_curve


def main():
    tab=pd.read_csv(paths.INPUTS/'sd_by_mean.csv')
    tab=tab[tab.scale=='identity_raw'].sort_values('mean_viability_raw').iloc[:13]
    bank=np.load(paths.INPUTS/'noise_bank_oneil.npz')
    y=bank['Y_raw'][:,:,1:,1:]
    mean=y.mean(1)
    transformed=np.log(np.maximum(y,.01))
    residual=(transformed-transformed.mean(1)[:,None])*np.sqrt(4./3.)
    variance=(residual**2).mean(1)
    bounds=np.array([-np.inf,.05,.1,.2,.3,.4,.5,.6,.7,.8,.9,.95,1.,1.05,1.1,np.inf])
    label=np.digitize(mean,bounds[1:-1])
    bins=[0,1,2,4,6,8,10,11,12]
    log_sd=[round(float(np.sqrt(variance[label==b].mean())),3) for b in bins]
    out=dict(viability_mean=tab.mean_viability_raw.round(6).tolist(),viability_sd=tab.pooled_sd.round(6).tolist(),viability_robust_sd=tab.robust_sd.round(6).tolist(),positive_noise_mean=[.025,.075,.15,.35,.55,.75,.925,.975,1.025],positive_noise_log_sd=log_sd)
    curve=oneil_sd_curve()
    assert np.array_equal(out['viability_mean'],curve.kv)
    assert np.array_equal(out['viability_sd'],curve.ks)
    (paths.INPUTS/'noise_model_curves.json').write_text(json.dumps(out,indent=2)+'\n')
    print(json.dumps(out))

if __name__=='__main__':main()
