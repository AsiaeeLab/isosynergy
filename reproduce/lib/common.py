"""DrugCombDB common calculations."""
import os, sys
import paths
import numpy as np
import pandas as pd
import sirsyn as S
REPO = str(paths.REPO)
ROOT = str(paths.ROOT)
RES = str(paths.REAL)
FIG = str(paths.FIGURES)
LOGS = str(paths.LOGS)
RAW_CSV = str(paths.raw_file('drugcombdb/drugcombs_response.csv'))
PROCESSED = str(paths.INPUTS / 'drugcomb_matrices.parquet')
BANK_ONEIL = str(paths.INPUTS / 'noise_bank_oneil.npz')
ALM_PAIRS = str(paths.INPUTS / 'almanac_orientation_pairs.npz')
DONORS_FILE = str(paths.INPUTS / 'oneil_donors.npz')
PANELS = str(paths.INPUTS / 'panel_membership.csv')
CMAX = str(paths.DATA / 'clinical_concentrations.csv')
MAX_WORKERS = paths.WORKERS
SEED_TAG = bytes([102, 105, 110, 97, 108, 45, 50, 48, 50, 54, 45, 48, 57, 45, 51, 48]).decode('ascii')
NULL_NOTE = "O'Neil: null = no synergy in any combination cell relative to the REPORTED single-agent curves (treated as fixed; m_s unknown). ALMANAC: single agents measured on the same plate (design S)."
FIT_ONEIL = dict(design='P', sa_replicates=np.inf, screen_replicates=True, screen_c=5.0, normalize='divide', B=399, B2=9999, two_stage=True)
FIT_ALMANAC = dict(design='S', normalize='divide', B=399, B2=9999, two_stage=True)

if paths.SMOKE:
    FIT_ONEIL.update(B=39, B2=99, n_std=9)
    FIT_ALMANAC.update(B=39, B2=99, n_std=9)

def load_oneil():
    meta = pd.read_parquet(os.path.join(RES, 'meta_oneil.parquet'))
    Z = np.load(os.path.join(RES, 'inputs_oneil.npz'))
    return (meta, Z['Y'], Z['doseA'], Z['doseB'])

def load_almanac():
    meta = pd.read_parquet(os.path.join(RES, 'meta_almanac.parquet'))
    Z = np.load(os.path.join(RES, 'inputs_almanac.npz'))
    return (meta, Z['Y'], Z['doseA'], Z['doseB'])

def oneil_noise():
    bank = S.NoiseBank.from_oneil_npz(BANK_ONEIL)
    B = np.load(BANK_ONEIL)
    sel = np.where(B['group_size'] == 4)[0]
    pvt = S.PlateVarianceTable(B['Y_raw'][sel], B['cell_line'][sel], fold_key=bytes([114, 50, 98]).decode('ascii'))
    return (bank, pvt)

def almanac_bank():
    A = np.load(ALM_PAIRS)
    Y = np.stack([A['Y1'].astype(float), A['Y2T'].astype(float)], 1)
    Y[:, :, 0, 0] = 1.0
    return S.NoiseBank.from_replicates(Y, A['cell_line'], include_edges=True, fold_key=bytes([114, 50, 98]).decode('ascii'))

def fold_of(cell):
    return S.seed_for(bytes([114, 50, 98]).decode('ascii'), 'half', str(cell)) % 2

def flatten(res, prefix=''):
    r = {}
    r['activity'] = res.activity
    r['activity_all_wells'] = res.activity_all_wells
    r['mean_excess'] = res.mean_excess
    r['mean_excess_se'] = res.mean_excess_se
    r['companion_test'] = res.mean_test.get('name', '')
    r['companion_t'] = res.mean_test.get('t', np.nan)
    r['companion_df'] = res.mean_test.get('df', np.nan)
    r['p_companion'] = res.p_mean
    r['companion_degenerate'] = bool(res.mean_test.get('degenerate', False))
    fe = res.mean_tests.get('fixed-edge t')
    if fe is not None and 'block t' in res.mean_tests:
        r['fe_est'] = fe['est']
        r['fe_se'] = fe['se']
        r['fe_t'] = fe['t']
        r['fe_df'] = fe['df']
        r['p_fixed_edge'] = fe['p']
        r['fe_plate_var'] = fe.get('plate_var', np.nan)
    r['p_mean_boot'] = res.p_mean_boot
    r['t_std'] = res.t_std
    r['p_tstd'] = res.p_tstd
    r['p_tstd_stage1'] = res.p_tstd_stage1
    r['p_scan'] = res.p_scan
    r['p_scan_stage1'] = res.p_scan_stage1
    r['stage'] = res.stage
    r['B_used'] = res.B_used
    cl = res.claims
    r['n_regions_claimed'] = len(cl)
    r['regions_claimed'] = '|'.join((f"{c['label']}" for c in cl))
    r['region_types_claimed'] = '|'.join((c['type'] for c in cl))
    if res.regions:
        k = int(np.argmin([g['p_adj'] - 1e-09 * g['z'] for g in res.regions]))
        g = res.regions[k]
        r['top_region'] = g['label']
        r['top_region_type'] = g['type']
        r['top_region_z'] = g['z']
        r['top_region_p_adj'] = g['p_adj']
    f = res.fragility
    r['fragility_computed'] = bool(f.get('computed', False))
    if f.get('computed'):
        r['loco_cell'] = str(f.get('loco_cell'))
        r['loco_influence_share'] = f.get('loco_influence_share', np.nan)
        r['loco_p'] = f.get('loco_p', np.nan)
        r['loco_flag'] = bool(f.get('loco_flag', False))
        r['sa_dose'] = str(f.get('sa_dose'))
        r['sa_influence_share'] = f.get('sa_influence_share', np.nan)
        r['sa_p'] = f.get('sa_p', np.nan)
        r['sa_flag'] = bool(f.get('sa_flag', False))
    lg = res.log
    r['analyzed_rows'] = lg.analyzed_shape[0]
    r['analyzed_cols'] = lg.analyzed_shape[1]
    r['n_dropped_doses'] = len(lg.dropped_doses)
    r['dropped_doses'] = ';'.join((f"{d['drug']}{d['index']}" for d in lg.dropped_doses))
    r['n_missing_cells'] = len(lg.missing_cells)
    r['n_screened'] = len(lg.screened)
    r['screened'] = ';'.join((f"({s['i']},{s['j']})r{s['replicate']}:{s['model_sds']:.1f}sd" for s in lg.screened))
    r['partial_replicates'] = lg.partial_replicates
    r['reference_flags_per_plate'] = lg.reference_flags_per_plate
    r['log_notes'] = ' | '.join(lg.notes)
    r['edgeA_top'] = float(res.analyzed_matrix[-1, 0])
    r['edgeB_top'] = float(res.analyzed_matrix[0, -1])
    r['mean_bliss_pred'] = float(np.mean(np.outer(res.a_hat, res.b_hat)))
    if prefix:
        r = {prefix + k: v for (k, v) in r.items()}
    return r

def wilson(k, n, z=1.959963984540054):
    if n == 0:
        return (np.nan, np.nan)
    p = k / n
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return (c - h, c + h)

def rate_row(p, level):
    p = np.asarray(p, float)
    ok = np.isfinite(p)
    k = int(np.sum(p[ok] <= level))
    n = int(ok.sum())
    (lo, hi) = wilson(k, n)
    return dict(level=level, k=k, n=n, rate=k / n if n else np.nan, lo=lo, hi=hi, n_nan=int((~ok).sum()))

def spearman_boot(x, y, n_boot=2000, seed=0, cluster=None):
    from scipy import stats
    (x, y) = (np.asarray(x, float), np.asarray(y, float))
    ok = np.isfinite(x) & np.isfinite(y)
    (x, y) = (x[ok], y[ok])
    cl = None if cluster is None else np.asarray(cluster)[ok]
    est = float(stats.spearmanr(x, y)[0]) if x.size > 2 else np.nan
    rng = np.random.default_rng(seed)
    bs = np.empty(n_boot)
    if cl is None:
        n = x.size
        for b in range(n_boot):
            i = rng.integers(0, n, n)
            bs[b] = stats.spearmanr(x[i], y[i])[0]
    else:
        (u, inv) = np.unique(cl, return_inverse=True)
        members = [np.where(inv == k)[0] for k in range(u.size)]
        for b in range(n_boot):
            pick = rng.integers(0, u.size, u.size)
            i = np.concatenate([members[k] for k in pick])
            bs[b] = stats.spearmanr(x[i], y[i])[0]
    (lo, hi) = np.nanpercentile(bs, [2.5, 97.5])
    return dict(n=int(x.size), spearman=est, lo=float(lo), hi=float(hi))

def norm_name(v):
    return pd.Series(np.asarray(v)).astype(str).str.upper().str.replace('[^A-Z0-9]', '', regex=True).to_numpy()
