"""fig6_missing_outliers calculations."""
import json
import os
import mc
import style as ST
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
R = mc.RESDIR
A05 = 0.05
NUM = {}

def rd(name):
    return pd.read_parquet(os.path.join(R, name))

def wil(k, n):
    (k, n) = (np.asarray(k, float), np.asarray(n, float))
    (lo, hi) = mc.wilson(k, n)
    return (k / n, lo, hi)

def hbar_points(ax, y, k, n, color, marker, label, mfc=None, off=0.0):
    (p, lo, hi) = wil(k, n)
    ax.errorbar(p, np.asarray(y) + off, xerr=np.maximum(0, [p - lo, hi - p]), fmt=marker, color=color, mfc=mfc or color, ms=3.6, lw=0.8, capsize=1.5, label=label)
    return p

def logx(ax, ticks):
    ax.set_xscale('log')
    ax.set_xticks(ticks)
    ax.set_xticklabels([f'{t:g}' for t in ticks])
    ax.xaxis.set_minor_formatter(mticker.NullFormatter())

def logy(ax, ticks):
    ax.set_yscale('log')
    ax.set_yticks(ticks)
    ax.set_yticklabels([f'{t:g}' for t in ticks])
    ax.yaxis.set_minor_formatter(mticker.NullFormatter())
MISS_TESTS = [('full', 'None'), ('int1', '1 combination well missing'), ('int2', '2 combination wells missing'), ('int3', '3 combination wells missing'), ('rowdrop', 'Single-agent dose row dropped')]

def panel_a(ax):
    d = rd('fig6_missing_size.parquet')
    y = np.arange(len(MISS_TESTS))[::-1]
    for (I, mk, off, mfc) in ((5, 'o', 0.15, None), (4, 's', -0.15, 'white')):
        z = d[d.I == I]
        k = [(z[f'p_{t}'] <= A05).sum() for (t, _) in MISS_TESTS]
        p = hbar_points(ax, y, k, [len(z)] * len(k), ST.COL['tstd'], mk, f'${I}\\times{I}$', mfc=mfc, off=off)
        NUM[f'A_{I}x{I}'] = dict(zip([t for (t, _) in MISS_TESTS], map(float, p)), n=len(z))
    ax.axvline(A05, ls='--', lw=0.7, color=ST.COL['gray'])
    ax.set_yticks(y)
    ax.set_yticklabels([lab for (_, lab) in MISS_TESTS])
    lo = min(0.025, min((min((v for (kk, v) in NUM[f'A_{I}x{I}'].items() if kk != 'n')) for I in (4, 5))) * 0.85)
    hi = max((max((v for (kk, v) in NUM[f'A_{I}x{I}'].items() if kk != 'n')) for I in (4, 5)))
    hi = max(hi, 0.075)
    ticks = [t for t in (0.02, 0.03, 0.05, 0.08) if lo <= t <= hi * 1.2]
    logx(ax, ticks)
    ax.set_xlim(lo, hi * 1.2)
    ax.set_xlabel('Rejection rate ($\\alpha = 0.05$)')
    ax.set_ylim(-0.6, len(MISS_TESTS) - 0.4)
    ax.legend(loc='upper right', handletextpad=0.3, borderaxespad=0.2)

def bars(ax, conds, names, series, getter, legend_kw):
    w = 0.8 / len(series)
    x = np.arange(len(conds))
    for (q, (key, lab, col)) in enumerate(series):
        v = [getter(cd, key) for cd in conds]
        ax.bar(x + (q - (len(series) - 1) / 2) * w, v, w * 0.95, color=col, label=lab, lw=0)
    ax.set_xticks(x)
    ax.set_xticklabels(names)
    ax.set_ylim(0, 1)
    ax.set_ylabel('Power ($\\alpha = 0.05$)')
    ax.legend(**legend_kw)

def panel_b(ax):
    d = rd('fig6_missing_power.parquet')
    d = d[(d.I == 5) & (d.law == 'C')]
    conds = ['syn_broad_0.20', 'syn_hdblock_1.20', 'syn_corner_2.00', 'sat_0.30']
    names = ['Broad\n0.20', 'High-dose\nblock 1.20', 'Top corner\n2.00', 'Saturating\n0.30']
    series = [('full', 'No missing well', '#0072B2'), ('int1', '1 combination well missing', '#56A0D3'), ('int3', '3 combination wells missing', '#A9CCE8'), ('rowdrop', 'Single-agent dose row dropped', '0.35')]

    def get(cd, key):
        return float((d[d.cond == cd][f'p_{key}'] <= A05).mean())
    NUM['B'] = {cd: {k: get(cd, k) for (k, _, _) in series} for cd in conds}
    bars(ax, conds, names, series, get, dict(loc='upper left', ncol=1, handlelength=1.0, handletextpad=0.4, borderaxespad=0.1, labelspacing=0.25))
    ax.set_ylim(0, 1.34)
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.spines['left'].set_bounds(0, 1)

def panel_c(ax):
    s = rd('fig6_screen_size.parquet')
    s = s[s.I == 5]
    lev_x = {0.0: 0.0, 0.004: 0.4, 0.01: 1.0, 0.02: 2.0}
    styles = [('C', '#0072B2', 'o', "Synthetic, O'Neil SD law"), ('A', '#56B4E9', '^', 'Synthetic, additive')]
    NUM['C_synth'] = {}
    for (law, col, mk, lab) in styles:
        y = s[s.law == law]
        for (arm, ls) in (('noscreen', '--'), ('screen', '-')):
            z = y[y.arm == arm].groupby('level').p.agg([lambda v: int((v <= A05).sum()), 'size'])
            (p, lo, hi) = wil(z.iloc[:, 0].values, z.iloc[:, 1].values)
            xx = np.array([lev_x[v] for v in z.index]) + {'C': -0.09, 'A': -0.03}[law]
            ax.errorbar(xx, p, yerr=np.maximum(0, [p - lo, hi - p]), fmt=mk, ls=ls, color=col, ms=3.6, lw=0.8, capsize=1.5, mfc=col if arm == 'screen' else 'white')
            NUM['C_synth'][f'{law}_{arm}'] = dict(zip(map(float, z.index), map(float, p)), n=int(z.iloc[0, 1]))
    bank_file = os.path.join(R, 'fig6_screen_bank.parquet')
    b = pd.read_parquet(bank_file) if os.path.exists(bank_file) else None
    if b is None:
        print("WARNING: O'Neil bank results missing; panel C drawn without them (development only)")
    bx = {'natural': 0.0, 'add0.01': 1.0, 'add0.02': 2.0}
    NUM['C_bank'] = {}
    for (arm, ls) in (('noscreen', '--'), ('screen', '-')) if b is not None else ():
        z = b[b.arm == arm].groupby('level').p.agg([lambda v: int((v <= A05).sum()), 'size']).loc[list(bx)]
        (p, lo, hi) = wil(z.iloc[:, 0].values, z.iloc[:, 1].values)
        ax.errorbar(np.array([bx[v] for v in z.index]) + {'noscreen': 0.05, 'screen': 0.11}[arm], p, yerr=np.maximum(0, [p - lo, hi - p]), fmt='s', ls=ls, color='0.2', ms=3.4, lw=0.8, capsize=1.5, mfc='0.2' if arm == 'screen' else 'white')
        NUM['C_bank'][arm] = dict(zip(z.index, map(float, p)), n=int(z.iloc[0, 1]))
    ax.axhline(A05, ls=':', lw=0.8, color=ST.COL['gray'])
    ax.set_xticks([0, 0.4, 1, 2])
    ax.set_xticklabels(['0', '0.4', '1', '2'])
    ax.set_xlabel('Added gross-error rate per well (%)')
    hi = max((max((v for (kk, v) in dd.items() if kk != 'n')) for dd in NUM['C_synth'].values()))
    top = 0.2 if hi < 0.15 else 0.3
    logy(ax, [t for t in (0.03, 0.05, 0.1, 0.2, 0.3) if t <= top])
    ax.set_ylim(0.025, top)
    ax.set_ylabel('Rejection rate ($\\alpha = 0.05$)')
    from matplotlib.lines import Line2D
    h1 = [Line2D([], [], color='#0072B2', marker='o', ls='', ms=3.6, label="Synthetic, O'Neil SD law"), Line2D([], [], color='#56B4E9', marker='^', ls='', ms=3.6, label='Synthetic, additive'), Line2D([], [], color='0.2', marker='s', ls='', ms=3.4, label="O'Neil residual bank")]
    h2 = [Line2D([], [], color='0.4', marker='o', mfc='white', ls='--', ms=3.4, lw=0.8, label='No screen'), Line2D([], [], color='0.4', marker='o', ls='-', ms=3.4, lw=0.8, label='Replicate screen')]
    leg = ax.legend(handles=h1, loc='upper left', handletextpad=0.4, borderaxespad=0.1, labelspacing=0.3)
    ax.add_artist(leg)
    ax.legend(handles=h2, loc='lower right', handletextpad=0.4, borderaxespad=0.1, labelspacing=0.3)

def panel_d(ax):
    w = rd('fig6_screen_power.parquet')
    w = w[(w.I == 5) & (w.law == 'C')]
    conds = ['syn_broad_0.10', 'syn_hdblock_0.40', 'syn_corner_1.00', 'sat_0.15']
    names = ['Broad\n0.10', 'High-dose\nblock 0.40', 'Top corner\n1.00', 'Saturating\n0.15']
    series = [((0.0, 'noscreen'), 'Clean, no screen', '#A9CCE8'), ((0.0, 'screen'), 'Clean, screen', '#0072B2'), ((0.02, 'noscreen'), '2% errors, no screen', '#EBB0A6'), ((0.02, 'screen'), '2% errors, screen', '#B2182B')]

    def get(cd, key):
        (lev, arm) = key
        z = w[(w.cond == cd) & (w.level == lev) & (w.arm == arm)]
        return float((z.p <= A05).mean())
    NUM['D'] = {cd: {f'{k[0]}_{k[1]}': get(cd, k) for (k, _, _) in series} for cd in conds}
    bars(ax, conds, names, series, get, dict(loc='upper left', ncol=1, handlelength=1.0, handletextpad=0.4, borderaxespad=0.1, labelspacing=0.25))
    ax.set_ylim(0, 1.34)
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.spines['left'].set_bounds(0, 1)
FRAG_ROWS = [('null', 0.0, 'moderate', 'Exact Bliss, clean'), ('null', 0.004, 'moderate', 'Exact Bliss, 0.4% errors'), ('null', 0.01, 'moderate', 'Exact Bliss, 1% errors'), ('null', 0.02, 'moderate', 'Exact Bliss, 2% errors'), ('syn_corner_2.00', 0.0, 'weak', 'Top corner cell 2.0'), ('dip_1.00', 0.0, 'moderate', 'Middle cell 1.0'), ('syn_hdblock_0.80', 0.0, 'moderate', 'High-dose block 0.8'), ('syn_broad_0.20', 0.0, 'moderate', 'Broad 0.2'), ('syn_broad_0.30', 0.0, 'moderate', 'Broad 0.3'), ('sat_0.60', 0.0, 'strong', 'Saturating 0.6')]

def panel_e(ax):
    d = rd('fig6_fragility.parquet')
    d = d[(d.I == 5) & (d.law == 'C')]
    (kc, ks, ns, labs) = ([], [], [], [])
    for (cond, lev, surf, nm) in FRAG_ROWS:
        y = d[(d.cond == cond) & (d.level == lev) & (d.surface == surf)]
        c = y.p <= A05
        kc.append(int((c & y.loco_flag).sum()))
        ks.append(int((c & y.sa_flag).sum()))
        ns.append(int(c.sum()))
        labs.append(f'{nm} ({int(c.sum())})')
    yy = np.arange(len(FRAG_ROWS))[::-1]
    pc = hbar_points(ax, yy, kc, ns, '#0072B2', 'o', 'Combination cell set to missing', off=0.15)
    ps = hbar_points(ax, yy, ks, ns, '0.3', 's', 'Single-agent dose row dropped', mfc='white', off=-0.15)
    NUM['E'] = {r[3]: dict(calls=n, cell=float(a), row=float(b), call_rate=None) for (r, n, a, b) in zip(FRAG_ROWS, ns, pc, ps)}
    ax.set_yticks(yy)
    ax.set_yticklabels(labs)
    ax.set_xlim(0, 1)
    ax.set_ylim(-0.6, len(FRAG_ROWS) - 0.4)
    ax.set_xlabel('Share of calls no longer at $p \\leq 0.05$')
    ax.legend(loc='lower right', handletextpad=0.3, borderaxespad=0.0, bbox_to_anchor=(1.0, 1.01), ncol=1)
MONO_ROWS = [('base', 'null', 'Monotone'), ('horm_0.03', 'null', 'Hormesis 1.03'), ('horm_0.07', 'null', 'Hormesis 1.07'), ('horm_0.15', 'null', 'Hormesis 1.15'), ('flat_0.07', 'null', 'Inactive drug at 1.07'), ('revtop_0.05', 'null', 'Top-dose rebound 0.05'), ('revtop_0.15', 'null', 'Top-dose rebound 0.15'), ('revmid_0.15', 'null', 'Middle reversal 0.15'), ('base', 'bumpC_0.25', 'Raised cell 0.25'), ('base', 'bumpR_0.20', 'Raised row 0.20')]

def panel_f(ax):
    d = rd('fig6_monotone_size.parquet')
    d = d[(d.design == 'S') & (d.I == 5)]
    yy = np.arange(len(MONO_ROWS))[::-1]
    NUM['F'] = {}
    for (col, c, mk, off, lab, mfc) in (('p', ST.COL['tstd'], 'o', 0.15, ST.TSTD, None), ('p_mean', ST.COL['mean'], 'D', -0.15, 'V2Sp', 'white')):
        (k, n) = ([], [])
        for (s, cd, _) in MONO_ROWS:
            y = d[(d.surface == s) & (d.cond == cd)]
            k.append(int((y[col] <= A05).sum()))
            n.append(len(y))
        p = hbar_points(ax, yy, k, n, c, mk, lab, mfc=mfc, off=off)
        NUM['F'][lab] = {r[2]: float(v) for (r, v) in zip(MONO_ROWS, p)}
        NUM['F']['n'] = n[0]
    ax.axvline(A05, ls='--', lw=0.7, color=ST.COL['gray'])
    ax.set_yticks(yy)
    ax.set_yticklabels([r[2] for r in MONO_ROWS])
    ax.set_xlim(-0.003, 0.08)
    ax.set_xticks([0, 0.02, 0.04, 0.06, 0.08])
    ax.set_xticklabels(['0', '0.02', '0.04', '0.06', '0.08'])
    ax.set_ylim(-0.6, len(MONO_ROWS) - 0.4)
    ax.set_xlabel('Rejection rate ($\\alpha = 0.05$)')
    ax.legend(loc='lower right', handletextpad=0.3, borderaxespad=0.1)
(FIG_W, FIG_H) = (ST.TEXTWIDTH, 7.0)
BOXES = {'A': (1.62, 5.25, 1.45, 1.55), 'B': (3.62, 5.25, 3.0, 1.55), 'C': (0.52, 3.08, 2.55, 1.6), 'D': (3.62, 3.08, 3.0, 1.6), 'E': (1.62, 0.42, 1.45, 1.95), 'F': (4.55, 0.42, 1.97, 1.95)}
LABEL_X = {'A': 0.0, 'B': 3.17, 'C': 0.0, 'D': 3.17, 'E': 0.0, 'F': 3.3}

def main():
    fig = plt.figure(figsize=(FIG_W, FIG_H))
    for (f, lab) in ((panel_a, 'A'), (panel_b, 'B'), (panel_c, 'C'), (panel_d, 'D'), (panel_e, 'E'), (panel_f, 'F')):
        (l, b, w, h) = BOXES[lab]
        ax = fig.add_axes([l / FIG_W, b / FIG_H, w / FIG_W, h / FIG_H])
        f(ax)
        ST.panel_label(ax, lab, x=(LABEL_X[lab] - l) / w, y=1.0 + 0.1 / h)
    ST.save(fig, 'fig6_missing_outliers')
    with open(os.path.join(R, 'fig6_numbers.json'), 'w') as fh:
        json.dump(NUM, fh, indent=1, default=float)
    print(json.dumps(NUM, indent=1, default=float))
if __name__ == '__main__':
    main()
