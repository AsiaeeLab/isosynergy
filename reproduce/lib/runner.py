"""DrugCombDB runner calculations."""
import warnings
import numpy as np
import pandas as pd
import common as C
S = C.S
G = {}

def init(study):
    G.clear()
    G['study'] = study
    if study == 'oneil':
        (meta, Y, DA, DB) = C.load_oneil()
        (G['bank'], G['pvt']) = C.oneil_noise()
    else:
        (meta, Y, DA, DB) = C.load_almanac()
        G['bank'] = C.almanac_bank()
    (G['meta'], G['Y'], G['DA'], G['DB']) = (meta, Y, DA, DB)
    G['cell'] = meta.cell_line.values.astype(str)
    G['unit'] = meta.unit_id.values.astype(str)

def fit_one(i, keep_a=None, keep_b=None, tag='screen'):
    warnings.filterwarnings('ignore', message='All-NaN slice encountered')
    study = G['study']
    Y = np.array(G['Y'][i], float, copy=True)
    ctrl = Y[..., 0, 0].copy()
    Y = Y / ctrl[..., None, None]
    if keep_a is not None:
        ka = np.concatenate([[True], np.asarray(keep_a, bool)])
        kb = np.concatenate([[True], np.asarray(keep_b, bool)])
        Y[..., ~ka, :] = np.nan
        Y[..., :, ~kb] = np.nan
    doses = (G['DA'][i], G['DB'][i])
    seed = (C.SEED_TAG, 'screen', study, G['unit'][i])
    if study == 'oneil':
        kw = dict(C.FIT_ONEIL)
        res = S.fit(Y, doses, noise=G['bank'], plate_var=G['pvt'], group=G['cell'][i], seed=seed, **kw)
    else:
        kw = dict(C.FIT_ALMANAC)
        res = S.fit(Y, doses, noise=G['bank'], group=G['cell'][i], seed=seed, **kw)
    row = dict(idx=int(i), control_raw=float(np.mean(ctrl)))
    row.update(C.flatten(res))
    row['doses_a_analyzed'] = ';'.join((f'{x:.6g}' for x in res.doses_a))
    row['doses_b_analyzed'] = ';'.join((f'{x:.6g}' for x in res.doses_b))
    row['seed_entropy'] = str(res.settings['seed_entropy'])
    return row

def job(idx):
    rows = []
    for i in idx:
        try:
            rows.append(fit_one(int(i)))
        except Exception as e:
            rows.append(dict(idx=int(i), error=f'{type(e).__name__}: {e}'))
    return pd.DataFrame(rows)

def job_window(items):
    rows = []
    for (i, ka, kb) in items:
        try:
            r = fit_one(int(i), ka, kb, tag='window')
        except Exception as e:
            r = dict(idx=int(i), error=f'{type(e).__name__}: {e}')
        rows.append(r)
    return pd.DataFrame(rows)
