"""Shared dose-response designs for the BIGL comparison."""
import paths
import os, json
import numpy as np
import sirsyn as X
from sirsyn import simulate as H
from types import SimpleNamespace
RES = str(paths.INPUTS)
DATA = str(paths.INPUTS / 'bigl_comparison_data')
DOSES = H.dose_grid(4, 1.0, 2.0)
SURFS = ['weak', 'moderate', 'strong']
NOISES = ['gauss', 'oneil']
REPS = [4, 1]
SD_GAUSS = 0.08
N_NULL = 1000
N_ALT = 500
ALTS = ['ant_broad_0.10', 'syn_broad_0.10', 'syn_highdose_0.40', 'syn_corner_0.60', 'sat_0.25']
CONDS = ['null'] + ALTS
sd_v = X.oneil_sd_curve()
H.KV = sd_v.kv
H.KS = sd_v.ks

def setting_name(r, surf, noise):
    return f'r{r}_{surf}_{noise}'

def all_settings():
    return [(r, s, n) for r in REPS for s in SURFS for n in NOISES]

def n_for(cond):
    return N_NULL if cond == 'null' else N_ALT

def truth(cond, surf):
    return H.truth(cond, surf, 5)

def well_sd(mu, noise):
    if noise == 'gauss':
        return np.full_like(mu, SD_GAUSS)
    return sd_v(mu)

def generate(r, surf, noise, cond, n):
    mu = truth(cond, surf)
    sd = well_sd(mu, noise)
    rng = np.random.default_rng(X.seed_for(bytes([114, 55, 98]).decode('ascii'), 'data', setting_name(r, surf, noise), cond))
    e = rng.standard_normal((n, r, 5, 5))
    return mu[None, None] + sd[None, None] * e

def data_path(r, surf, noise, cond):
    return os.path.join(DATA, f'{setting_name(r, surf, noise)}__{cond}.f64')

def write_meta():
    meta = dict(doses=list(map(float, DOSES)), settings=[setting_name(*s) for s in all_settings()], conds=CONDS, n_null=N_NULL, n_alt=N_ALT, sd_gauss=SD_GAUSS, oneil_kv=list(map(float, H.KV)), oneil_ks=list(map(float, H.KS)), layout='float64 little endian, shape (n, r, 5, 5): dataset, replicate, i (drug 1 dose), j (drug 2 dose)')
    with open(os.path.join(DATA, 'meta.json'), 'w') as f:
        json.dump(meta, f, indent=1)
