"""Read per-dataset BIGL decisions and convergence status."""
import paths
import glob, os
import numpy as np
import pandas as pd
RES = str(paths.INPUTS)
ALPHAS = (0.05, 0.01)
TAG = {0.1: '010', 0.05: '005', 0.01: '001'}

def load_bigl():
    fs = sorted(glob.glob(os.path.join(RES, 'bigl_raw', '*', '*.csv')))
    d = pd.concat([pd.read_csv(f, low_memory=False) for f in fs], ignore_index=True)
    base = d.file.str.replace('.f64', '', regex=False)
    d['setting'] = np.where(base.str.contains('__'), base.str.split('__').str[0], base)
    d['cond'] = np.where(base.str.contains('__'), base.str.split('__').str[1], 'real')
    d['ok'] = (d.fit_err == 0) & (d.surf_err == 0)
    return d

def bigl_decisions(d):
    ok = d.ok.values
    out = pd.DataFrame(index=d.index)
    B = d.maxR_B.fillna(100).values
    nge_syn = np.where(d.maxR_nge_syn.isna(), B, d.maxR_nge_syn.values)
    nge_abs = np.where(d.maxR_nge_abs.isna(), B, d.maxR_nge_abs.values)
    syn_dir = (d.dir == 'dec') & (d.meanR_rawmean < 0) | (d.dir == 'inc') & (d.meanR_rawmean > 0)
    for a in ALPHAS:
        t = TAG[a]
        out[f'maxR_syn_q_{t}'] = ok & (d[f'maxR_rej_syn_{t}'].fillna(0).values == 1)
        out[f'maxR_syn_x_{t}'] = ok & ((1 + nge_syn) / (B + 1) <= a + 1e-12)
        out[f'maxR_any_q_{t}'] = ok & (d[f'maxR_rej_any_{t}'].fillna(0).values == 1)
        out[f'maxR_any_x_{t}'] = ok & ((1 + nge_abs) / (B + 1) <= a + 1e-12)
        out[f'maxR_ant_q_{t}'] = ok & (d[f'maxR_rej_ant_{t}'].fillna(0).values == 1)
        out[f'maxR_asym_{t}'] = ok & (d.maxR_pasym_syn.fillna(1).values <= a)
        out[f'meanR_{t}'] = ok & (d.meanR_p.fillna(1).values < a - 1e-12)
        out[f'meanR_sign_{t}'] = out[f'meanR_{t}'] & syn_dir.values
        out[f'meanR_asym_{t}'] = ok & (d.meanR_pasym.fillna(1).values <= a)
    out['maxR_syn_q2a_005'] = ok & (d.maxR_rej_syn_010.fillna(0).values == 1)
    out['maxR_syn_x2a_005'] = ok & ((1 + nge_syn) / (B + 1) <= 0.1 + 1e-12)
    return out

def degenerate(d, lo, hi):
    bad = np.zeros(len(d), bool)
    for k in ('h1', 'h2'):
        bad |= d[f'c_{k}'].abs().values > 20
    for k in (bytes([101, 49]).decode('ascii'), bytes([101, 50]).decode('ascii')):
        bad |= (d[f'c_{k}'].values < lo - 3) | (d[f'c_{k}'].values > hi + 3)
    for k in ('m1', 'm2'):
        bad |= (d[f'c_{k}'].values < -0.5) | (d[f'c_{k}'].values > 1.5)
    bad |= (d.c_b.values < 0.5) | (d.c_b.values > 1.5)
    return bad & d.fit_err.eq(0).values
