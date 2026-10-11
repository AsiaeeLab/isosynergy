"""fig4_grid_power calculations."""
import mc
import style as ST
import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
GRIDS = (4, 5, 6, 8, 10)
GRID_COL = {4: '#9ecae1', 5: '#6baed6', 6: '#3182bd', 8: '#08519c', 10: '#08306b'}
TESTS = ('tstd', 'scan', 'mean')
MK = {'tstd': 'o', 'scan': 's', 'mean': '^'}
OFF = {'tstd': -0.12, 'scan': 0.0, 'mean': 0.12}
GRAY = '0.55'
XLIM = {'broad': (0.005, 0.5), 'highdose': (0.003, 0.2), 'sat': (0.0025, 0.2)}

def glab(I):
    return f'${I}\\times{I}$'

def main():
    cnt = pd.read_csv(os.path.join(ST.RESDIR, 'grid_counts.csv'), keep_default_na=False)
    e80 = pd.read_csv(os.path.join(ST.RESDIR, 'grid_effect80.csv'))
    design = 'S'
    (fig, axs) = plt.subplots(2, 3, figsize=(ST.TEXTWIDTH, 4.55))
    panels = [('broad', 'moderate'), ('highdose', 'moderate'), ('corner', 'weak'), ('sat', 'strong')]
    for (ax, (pat, surf)) in zip([axs[0, 0], axs[0, 1], axs[0, 2], axs[1, 0]], panels):
        xcol = 'max_eta' if pat == 'corner' else 'mean_eta'
        for I in GRIDS:
            g = cnt[(cnt.design == design) & (cnt.noise == 'oneil') & (cnt.surface == surf) & (cnt.I == I) & cnt.pattern.isin([pat, 'null'])].sort_values('amp')
            if pat != 'corner':
                g = g[g[xcol] > 0]
            ax.plot(g[xcol], g['tstd@0.05'], color=GRID_COL[I], lw=1.2, marker='o', ms=1.8, label=glab(I))
        ax.axhline(0.8, color=GRAY, lw=0.6, ls=':')
        ax.axhline(0.05, color=GRAY, lw=0.6, ls=':')
        ax.set_ylim(0, 1.0)
        if pat == 'corner':
            ax.set_xlim(0, 0.5)
            ax.set_xlabel('Top-cell Bliss excess')
        else:
            ax.set_xscale('log')
            ax.set_xticks([0.003, 0.01, 0.03, 0.1, 0.3])
            ax.set_xticklabels(['0.003', '0.01', '0.03', '0.1', '0.3'])
            ax.set_xlim(*XLIM[pat])
            ax.minorticks_off()
            ax.set_xlabel('Mean Bliss excess')
        ax.set_ylabel(f'Power of {ST.TSTD} at level 0.05')
    axs[0, 0].legend(loc='upper left', handlelength=1.4, borderaxespad=0.2)
    handles = []
    for (ax, pat, which) in ((axs[1, 1], 'broad', 'mean'), (axs[1, 2], 'corner', 'max')):
        for t in TESTS:
            for (a, fill, ls) in ((0.05, True, '-'), (0.005, False, '--')):
                g = e80[(e80.design == design) & (e80.noise == 'oneil') & (e80.pattern == pat) & (e80.test == t) & np.isclose(e80.alpha, a)].sort_values('I')
                y = g[f'{which}_eta80'].to_numpy(float)
                lo = g[f'{which}_eta80_lo'].to_numpy(float)
                hi = g[f'{which}_eta80_hi'].to_numpy(float)
                x = g.I.to_numpy(float) + OFF[t]
                ok = np.isfinite(y)
                col = ST.COL[t]
                (h,) = ax.plot(x[ok], y[ok], color=col, ls=ls, lw=1.0, marker=MK[t], ms=3.6, mfc=col if fill else 'white', mec=col, mew=0.9, label=f'{ST.LAB[t]}, {a:g}')
                if pat == 'broad':
                    handles.append(h)
                okb = ok & np.isfinite(lo) & np.isfinite(hi)
                ax.vlines(x[okb], lo[okb], hi[okb], color=col, lw=0.6)
        ax.set_xticks(list(GRIDS))
        ax.set_xticklabels([str(I) for I in GRIDS])
        ax.set_xlim(3.4, 10.6)
        ax.set_xlabel('Doses per agent, including zero')
        if which == 'mean':
            ax.set_ylabel('Mean excess for 80% power')
            ax.set_ylim(0, 0.42)
        else:
            ax.set_ylabel('Top-cell excess\nfor 80% power')
            ax.set_ylim(0, 0.5)
    axs[1, 2].legend(handles=handles, loc='lower left', ncol=1, handlelength=2.2, borderaxespad=0.2, labelspacing=0.3)
    for (ax, lab) in zip(axs.ravel(), 'ABCDEF'):
        ST.panel_label(ax, lab, x=-0.27)
    fig.tight_layout(w_pad=1.0, h_pad=1.2)
    ST.save(fig, 'fig4_grid_power')
if __name__ == '__main__':
    main()
