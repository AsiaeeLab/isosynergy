"""Supplementary S6_null_checks calculations."""
import mc
import os
import numpy as np
import pandas as pd
import style as ST
import matplotlib.pyplot as plt
RES = str(mc.paths.REAL)
LEVELS = [0.05, 0.01, 0.005]
TESTS = {'oneil': [('p_tstd', ST.TSTD, ST.COL['tstd'], 'o'), ('p_scan', 'Region scan', ST.COL['scan'], 's'), ('p_companion', 'Block-level $t$', ST.COL['mean'], 'D'), ('p_fixed_edge', 'Fixed-edge $t$', ST.COL['gray'], '^')], 'almanac': [('p_tstd', ST.TSTD, ST.COL['tstd'], 'o'), ('p_scan', 'Region scan', ST.COL['scan'], 's'), ('p_companion', 'V2Sp', ST.COL['mean'], 'D')]}
N_EXPECT = {'oneil': 22422, 'almanac': 6124}

def main():
    N = pd.read_csv(os.path.join(RES, 'null_summary.csv'))
    (fig, axs) = plt.subplots(1, 2, figsize=(6.45, 2.45), sharey=True)
    for (k, st) in enumerate(('oneil', 'almanac')):
        ax = axs[k]
        nt = len(TESTS[st])
        for (j, (c, lab, colr, mk)) in enumerate(TESTS[st]):
            t = N[(N.study == st) & (N.column == c) & N.level.isin(LEVELS)].set_index('level').loc[LEVELS]
            assert (t.n == N_EXPECT[st]).all()
            x = np.arange(len(LEVELS)) + (j - (nt - 1) / 2) * 0.17
            r = t.rate / t.index
            ax.errorbar(x, r, yerr=np.maximum(0, [r - t.lo / t.index, t.hi / t.index - r]), fmt=mk, ms=3.3, color=colr, lw=0.9, capsize=1.5, label=lab)
            print(st, c, ' '.join((f'{lv}: {rt:.4f} [{lo:.4f}, {hi:.4f}]' for (lv, rt, lo, hi) in zip(LEVELS, t.rate, t.lo, t.hi))))
        ax.axhline(1, color='0.45', lw=0.7, ls='--')
        ax.set_xticks(range(len(LEVELS)))
        ax.set_xticklabels([str(v) for v in LEVELS])
        ax.set_xlim(-0.5, len(LEVELS) - 0.5)
        ax.set_xlabel('Nominal level $\\alpha$')
        if k == 0:
            ax.set_ylabel('Rejection rate / $\\alpha$')
        ax.legend(loc='upper left', handletextpad=0.3, borderaxespad=0.2)
        ST.panel_label(ax, 'AB'[k], x=-0.14 if k == 0 else -0.06, y=1.03)
    ax.set_ylim(0, None)
    fig.tight_layout(w_pad=1.5)
    ST.save(fig, 'figS6_null_checks', pad=0.02)
if __name__ == '__main__':
    main()
