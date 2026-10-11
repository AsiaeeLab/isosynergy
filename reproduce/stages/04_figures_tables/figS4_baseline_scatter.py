"""Supplementary S4_baseline_scatter calculations."""
import mc
import os
import numpy as np
import pandas as pd
import style as ST
import matplotlib.pyplot as plt
import fig1_baseline_disagreement as F1
N_SUB = 5000
SEED = mc.data_seed('figS4_baseline_scatter', 'subsample')
OUT_CSV = os.path.join(ST.RESDIR, 'baseline_scatter_subsample.csv')
SUPP_WIDTH = 6.0

def main():
    df = pd.read_parquet(F1.SCORES)
    X = F1.complete(df)
    assert len(X) == F1.N_EXPECTED, len(X)
    idx = np.sort(np.random.default_rng(SEED).choice(len(X), N_SUB, replace=False))
    Xs = X.iloc[idx].reset_index(drop=True)
    os.makedirs(ST.RESDIR, exist_ok=True)
    Xs.assign(row_in_complete=idx).to_csv(OUT_CSV, index=False)
    print(f'seed {SEED}; subsample {len(Xs):,} of {len(X):,}')
    print('subsample Pearson r:\n', Xs.corr().round(3).to_string())
    pad = lambda lo, hi: (lo - 0.04 * (hi - lo), hi + 0.04 * (hi - lo))
    lim = {m: pad(Xs[m].min(), Xs[m].max()) for m in F1.METHODS}
    pairs = [('Bliss', 'HSA'), ('Bliss', 'ZIP'), ('HSA', 'ZIP'), ('Bliss', 'Loewe'), ('HSA', 'Loewe'), ('ZIP', 'Loewe')]
    (fig, axs) = plt.subplots(2, 3, figsize=(SUPP_WIDTH, 4.0))
    for (ax, (c, r)) in zip(axs.ravel(), pairs):
        ax.scatter(Xs[c], Xs[r], s=1.2, color=ST.COL['tstd'], alpha=0.25, edgecolors='none', rasterized=True)
        lo = min(lim[c][0], lim[r][0])
        hi = max(lim[c][1], lim[r][1])
        ax.plot([lo, hi], [lo, hi], color='0.3', lw=0.8, ls='--')
        ax.axhline(0, color='0.75', lw=0.5, zorder=0)
        ax.axvline(0, color='0.75', lw=0.5, zorder=0)
        ax.set_xlim(lim[c])
        ax.set_ylim(lim[r])
        ax.set_xlabel(f'{c} score')
        ax.set_ylabel(f'{r} score')
    fig.tight_layout(w_pad=1.0, h_pad=1.0)
    ST.save(fig, 'figS4_baseline_scatter', pad=0.02)
if __name__ == '__main__':
    main()
