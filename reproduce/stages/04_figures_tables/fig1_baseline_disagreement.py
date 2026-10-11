"""fig1_baseline_disagreement calculations."""
import mc
import os
import numpy as np
import pandas as pd
import style as ST
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm, Normalize
from matplotlib.patches import Rectangle
METHODS = ['Bliss', 'HSA', 'Loewe', 'ZIP']
TOP_Q = 0.05
SCORES = os.path.join(str(mc.paths.INPUTS), 'drugcombdb_block_scores.parquet')
SUMMARY_R = os.path.join(str(mc.paths.INPUTS), 'drugcombdb_baseline_disagreement.parquet')
OUT_CSV = os.path.join(ST.RESDIR, 'baseline_disagreement_drugcombdb.csv')
N_EXPECTED = 391652

def q_type8(x, p):
    x = np.sort(np.asarray(x, float))
    n = x.size
    m = (p + 1.0) / 3.0
    h = n * p + m
    j = int(np.floor(h))
    g = h - j
    lo = x[min(max(j, 1), n) - 1]
    hi = x[min(max(j + 1, 1), n) - 1]
    return (1.0 - g) * lo + g * hi

def complete(df):
    X = df[METHODS].astype(float)
    return X[np.isfinite(X.values).all(axis=1)].reset_index(drop=True)

def metrics(X, top_q=TOP_Q):
    V = X[METHODS].values
    k = len(METHODS)
    cor = np.corrcoef(V.T)
    pos = V > 0
    dis = np.array([[np.mean(pos[:, i] != pos[:, j]) for j in range(k)] for i in range(k)])
    thr = np.array([q_type8(V[:, i], 1.0 - top_q) for i in range(k)])
    top = V >= thr
    jac = np.array([[(top[:, i] & top[:, j]).sum() / (top[:, i] | top[:, j]).sum() for j in range(k)] for i in range(k)])
    mk = lambda a: pd.DataFrame(a, index=METHODS, columns=METHODS)
    return ({'pearson_corr': mk(cor), 'pos_disagree_rate': mk(dis), 'top_jaccard': mk(jac)}, thr)

def check_against_r(res, summary_path, tol=1e-12):
    s = pd.read_parquet(summary_path)
    worst = 0.0
    for (name, M) in res.items():
        p = s[s.metric == name].pivot(index='method1', columns='method2', values='value').loc[METHODS, METHODS]
        worst = max(worst, float(np.abs(p.values - M.values).max()))
    if worst > tol:
        raise AssertionError(f'recomputed metrics differ from {summary_path}: max |diff| = {worst:.3g}')
    return worst

def long_table(res, n, dataset):
    rows = []
    for (name, M) in res.items():
        for a in METHODS:
            for b in METHODS:
                rows.append(dict(dataset=dataset, metric=name, method1=a, method2=b, value=float(M.loc[a, b]), n=n))
    return pd.DataFrame(rows)
(ROWS, COLS) = (METHODS[1:], METHODS[:-1])
CBAR_LAB = {'pearson_corr': 'Pearson correlation $r$', 'pos_disagree_rate': 'Sign disagreement rate', 'top_jaccard': 'Jaccard overlap of top-5% sets'}

def heat(ax, M, metric, cax):
    if metric == 'pearson_corr':
        (cmap, norm) = (plt.get_cmap('RdBu_r'), TwoSlopeNorm(vmin=-1.0, vcenter=0.0, vmax=1.0))
        ticks = [-1, -0.5, 0, 0.5, 1]
    else:
        (cmap, norm) = (plt.get_cmap('YlGnBu'), Normalize(0.0, 1.0))
        ticks = [0, 0.25, 0.5, 0.75, 1]
    A = np.full((len(ROWS), len(COLS)), np.nan)
    for (i, r) in enumerate(ROWS):
        for (j, c) in enumerate(COLS):
            if METHODS.index(c) < METHODS.index(r):
                A[i, j] = M.loc[r, c]
    im = ax.imshow(np.ma.masked_invalid(A), cmap=cmap, norm=norm, aspect='equal')
    for i in range(len(ROWS)):
        for j in range(len(COLS)):
            if np.isfinite(A[i, j]):
                rgba = cmap(norm(A[i, j]))
                lum = 0.299 * rgba[0] + 0.587 * rgba[1] + 0.114 * rgba[2]
                ax.text(j, i, f'{A[i, j]:.2f}'.replace('-', '−'), ha='center', va='center', fontsize=8, color='white' if lum < 0.5 else 'black')
    ax.set_xticks(range(len(COLS)))
    ax.set_xticklabels(COLS)
    ax.set_yticks(range(len(ROWS)))
    ax.set_yticklabels(ROWS)
    ax.tick_params(which='both', length=0)
    for sp in ax.spines.values():
        sp.set_visible(False)
    for i in range(len(ROWS)):
        for j in range(len(COLS)):
            if np.isfinite(A[i, j]):
                ax.add_patch(Rectangle((j - 0.5, i - 0.5), 1, 1, fill=False, ec='white', lw=1.2, clip_on=False))
    cb = plt.colorbar(im, cax=cax, orientation='horizontal', ticks=ticks)
    cb.set_label(CBAR_LAB[metric], fontsize=7.5)
    cb.ax.tick_params(labelsize=7, length=2, width=0.5)
    cb.outline.set_linewidth(0.5)
    return im

def figure(res, name, width):
    fig = plt.figure(figsize=(width, 2.45))
    order = ['pearson_corr', 'pos_disagree_rate', 'top_jaccard']
    gap = 0.045
    (left0, wpan) = (0.075, (0.985 - 0.075 - 2 * gap) / 3)
    for (k, metric) in enumerate(order):
        x0 = left0 + k * (wpan + gap)
        ax = fig.add_axes([x0 + 0.03, 0.3, wpan - 0.03, 0.62])
        cax = fig.add_axes([x0 + 0.05, 0.13, wpan - 0.07, 0.04])
        heat(ax, res[metric], metric, cax)
        ST.panel_label(ax, 'ABC'[k], x=-0.42, y=1.0)
    return ST.save(fig, name, pad=0.02)

def main():
    df = pd.read_parquet(SCORES)
    X = complete(df)
    n = len(X)
    print(f'rows {len(df):,}; matrices with all four scores finite: {n:,}')
    assert n == N_EXPECTED, n
    (res, thr) = metrics(X)
    worst = check_against_r(res, SUMMARY_R)
    print(f'reproduces {os.path.relpath(SUMMARY_R, ST.REPO)} (max |diff| {worst:.2g}); top-5% thresholds', dict(zip(METHODS, np.round(thr, 3))))
    os.makedirs(ST.RESDIR, exist_ok=True)
    long_table(res, n, 'DrugCombDB').to_csv(OUT_CSV, index=False)
    for (name, M) in res.items():
        print(name)
        print(M.round(4).to_string())
    off = [res['pos_disagree_rate'].loc[a, b] for (i, a) in enumerate(METHODS) for b in METHODS[i + 1:]]
    print(f'sign disagreement range over pairs: {min(off):.4f} to {max(off):.4f}')
    figure(res, 'fig1_baseline_disagreement', width=ST.TEXTWIDTH)
if __name__ == '__main__':
    main()
