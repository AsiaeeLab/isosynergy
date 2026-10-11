"""fig5_bigl calculations."""
import mc
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import style as ST
import sim_bigl as SB
SET_ORDER = [(s, n) for s in SB.SURFS for n in SB.NOISES]
SET_LAB = [f"{ {'weak': 'Weak', 'moderate': 'Mod.', 'strong': 'Strong'}[s]}\n{('Gauss.' if n == 'gauss' else 'O' + chr(39) + 'Neil')}" for (s, n) in SET_ORDER]
ALT_LAB = {'ant_broad_0.10': 'Antag-\nonism', 'syn_broad_0.10': 'Broad', 'syn_highdose_0.40': 'High\ndose', 'syn_corner_0.60': 'Top\ncorner', 'sat_0.25': 'Satu-\nrating'}
(CB, CB2) = (ST.COL['bigl'], ST.COL['bigl2'])
OURS = [('tstd', ST.TSTD, 'ours', 'p_tstd', ST.COL['tstd'], 'o', True), ('scan', 'Region scan', 'ours', 'p_scan', ST.COL['scan'], 's', True), ('mean', 'Mean test', 'ours', 'p_mean', ST.COL['mean'], 'D', True)]
BIGL = [('ship_max', 'BIGL maxR, distributed reference', 'ship_eq', 'maxR_syn_q_005', CB2, '^', True), ('ship_mean', 'BIGL meanR, distributed reference', 'ship_eq', 'meanR_005', CB2, 'v', True), ('vb_max', 'BIGL maxR, viability reference', 'vb_eq', 'maxR_syn_q_005', CB, '^', False), ('vb_mean', 'BIGL meanR, viability reference', 'vb_eq', 'meanR_005', CB, 'v', False)]
VB10 = ('vb_max10', 'BIGL maxR, viability reference, $\\alpha = 0.10$', 'vb_eq', 'maxR_syn_q2a_005', CB, '^', True)

def handle(color, marker, filled):
    return Line2D([], [], ls='none', marker=marker, ms=4, color=color, mfc=color if filled else 'white', mew=0.9)

def rates(ours, bd, r, cond, series):
    out = {}
    for (key, lab, src, col, *_) in series:
        v = []
        for (s, n) in SET_ORDER:
            st = SB.setting_name(r, s, n)
            if src == 'ours':
                x = ours[(ours.setting == st) & (ours.cond == cond)][col].values
            else:
                x = bd[(bd.config == src) & (bd.setting == st) & (bd.cond == cond)][col].values.astype(bool)
            v.append(SB.rate(x) if len(x) else None)
        out[key] = v
    return out

def panel_size(ax, ours, bd, r):
    ser = OURS + BIGL
    R = rates(ours, bd, r, 'null', ser)
    xs = np.arange(len(SET_ORDER))
    w = 0.115
    for (q, (key, lab, src, col, c, mk, fill)) in enumerate(ser):
        x = xs + (q - (len(ser) - 1) / 2) * w
        for (xi, t) in zip(x, R[key]):
            if t is None:
                continue
            (p, lo, hi) = (max(t['p'], 0.001), max(t['lo'], 0.001), max(t['hi'], 0.001))
            ax.errorbar([xi], [p], yerr=np.maximum(0, [[p - lo], [hi - p]]), fmt=mk, ms=3.3, color=c, mfc=c if fill else 'white', mew=0.8, elinewidth=0.7, capsize=0, lw=0)
    ax.axhline(0.05, color='0.4', lw=0.7, ls='--', zorder=0)
    ax.set_yscale('log')
    ax.set_ylim(0.001, 1.0)
    ax.set_xlim(-0.55, len(SET_ORDER) - 0.45)
    ax.set_xticks(xs)
    ax.set_xticklabels(SET_LAB)
    for b in xs[:-1] + 0.5:
        ax.axvline(b, color='0.9', lw=0.5, zorder=0)
    ax.set_ylabel('Rejection rate at $\\alpha = 0.05$')
    return R

def panel_power(ax, ours, bd):
    ser = OURS + BIGL[2:] + [VB10]
    xs = np.arange(len(SB.ALTS))
    w = 0.13
    summ = {}
    for (q, (key, lab, src, col, c, mk, fill)) in enumerate(ser):
        (m, lo, hi) = ([], [], [])
        for cond in SB.ALTS:
            v = [t['p'] for t in rates(ours, bd, 4, cond, [(key, lab, src, col)])[key]]
            v = np.array(v, float)
            m.append(v.mean())
            lo.append(v.min())
            hi.append(v.max())
        (m, lo, hi) = map(np.array, (m, lo, hi))
        summ[key] = dict(zip(SB.ALTS, m))
        x = xs + (q - (len(ser) - 1) / 2) * w
        ax.errorbar(x, m, yerr=np.maximum(0, [m - lo, hi - m]), fmt=mk, ms=3.6, color=c, mfc=c if fill else 'white', mew=0.8, elinewidth=0.7, capsize=0, lw=0)
    ax.axhline(0.05, color='0.4', lw=0.7, ls='--', zorder=0)
    for b in xs[:-1] + 0.5:
        ax.axvline(b, color='0.9', lw=0.5, zorder=0)
    ax.set_xticks(xs)
    ax.set_xticklabels([ALT_LAB[a] for a in SB.ALTS])
    ax.set_xlim(-0.55, len(SB.ALTS) - 0.45)
    ax.set_ylim(-0.02, 1.02)
    ax.set_ylabel('Rejection rate at $\\alpha = 0.05$')
    ax.set_xlabel('Truth, 4 replicates')
    return summ

def real_rows(bd):
    m = SB.real_production()
    rows = []
    for (lab, col, c, mk) in ((ST.TSTD, 'p_tstd', ST.COL['tstd'], 'o'), ('Region scan', 'p_scan', ST.COL['scan'], 's'), ('Block-level $t$', 'p_companion', ST.COL['mean'], 'D')):
        t = SB.rate(m[col].values)
        rows.append((lab, t['k'], t['n'], c, mk, True))
    real = bd[bd.cond == 'real']
    for (cfg, var, c, fill) in (('ship_eq', 'distributed, equal', CB2, True), ('ship_model', 'distributed, model', CB2, False), ('vb_eq', 'viability, equal', CB, False)):
        x = real[real.config == cfg]
        ok = x.ok.values
        for (stat, col, mk) in (('maxR', 'maxR_syn_q_005', '^'), ('meanR', 'meanR_005', 'v')):
            k = int(x[col].values[ok].sum())
            rows.append((f'BIGL {stat}, {var}', k, int(ok.sum()), c, mk, fill))
    return rows

def panel_real(ax, bd):
    rows = real_rows(bd)
    ys = np.arange(len(rows))[::-1]
    for (y, (lab, k, n, c, mk, fill)) in zip(ys, rows):
        (lo, hi) = mc.wilson(k, n)
        p = k / n
        ax.errorbar([p], [y], xerr=np.maximum(0, [[p - lo], [hi - p]]), fmt=mk, ms=4, color=c, mfc=c if fill else 'white', mew=0.9, elinewidth=0.8, capsize=0, lw=0)
        ax.text(1.03, y, f'{n}', va='center', ha='left', fontsize=7, transform=ax.get_yaxis_transform())
    ax.text(1.03, len(rows) - 0.35, '$n$', va='bottom', ha='left', fontsize=7, transform=ax.get_yaxis_transform())
    ax.axhline(len(rows) - 3.5, color='0.85', lw=0.6)
    ax.set_yticks(ys)
    ax.set_yticklabels([r[0] for r in rows])
    ax.set_ylim(-0.6, len(rows) - 0.4)
    ax.set_xlim(-0.02, 1.02)
    ax.set_xlabel('Call rate at $\\alpha = 0.05$')
    return rows

def main():
    ours = SB.load_ours()
    bd = SB.bigl_decisions()
    fig = plt.figure(figsize=(6.55, 5.7))
    gs = fig.add_gridspec(2, 2, width_ratios=[1.0, 1.0], height_ratios=[1, 1.05], hspace=0.5, wspace=0.28, left=0.09, right=0.955, top=0.855, bottom=0.08)
    axA = fig.add_subplot(gs[0, 0])
    axB = fig.add_subplot(gs[0, 1])
    axC = fig.add_subplot(gs[1, 0])
    gsd = gs[1, 1].subgridspec(1, 2, width_ratios=[0.62, 1.0], wspace=0.0)
    axD = fig.add_subplot(gsd[0, 1])
    panel_size(axA, ours, bd, 4)
    axA.set_xlabel('Single agents and noise, 4 replicates')
    panel_size(axB, ours, bd, 1)
    axB.set_xlabel('Single agents and noise, 1 replicate')
    axB.set_ylabel('')
    panel_power(axC, ours, bd)
    panel_real(axD, bd)
    for (ax, L) in ((axA, 'A'), (axB, 'B'), (axC, 'C')):
        ST.panel_label(ax, L, x=-0.17)
    ST.panel_label(axD, 'D', x=-1.02)
    hs = [handle(c, mk, f) for (_, _, _, _, c, mk, f) in OURS + BIGL + [VB10]]
    ls = [t[1] for t in OURS + BIGL + [VB10]]
    blank = Line2D([], [], ls='none', alpha=0)
    (hs, ls) = (hs[:5] + [blank] + hs[5:], ls[:5] + [''] + ls[5:])
    fig.legend(hs, ls, loc='upper center', ncol=3, bbox_to_anchor=(0.53, 1.0), handletextpad=0.3, columnspacing=1.2, borderaxespad=0.2)
    ST.save(fig, 'fig5_bigl', pad=0.02)
if __name__ == '__main__':
    main()
