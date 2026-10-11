"""Figure 2: the SIR procedure, effect estimates and family calls for one O'Neil unit."""
import os
import sys
import mc
import numpy as np
import pandas as pd
sys.path.insert(0, mc.FINAL)
import common as C
import runner as Rn
import sirsyn.core as SC
import style as ST
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyBboxPatch, PathPatch, Polygon
from matplotlib.path import Path
UNIT = 'TEMOZOLOMIDE|MK-4827|EFM192B|batch0'
S = C.S
CAP = []
_orig_stats = SC._Grid.stats

def _stats(self, Ys):
    out = _orig_stats(self, Ys)
    if out.shape[0] > 1:
        CAP.append(np.array(out, copy=True))
    return out
SC._Grid.stats = _stats
Rn.init('oneil')
meta = Rn.G['meta']
i = int(np.where(meta.unit_id.values == UNIT)[0][0])
Yst = np.array(Rn.G['Y'][i], float, copy=True)
Yst = Yst / Yst[..., 0, 0][..., None, None]
doses = (Rn.G['DA'][i], Rn.G['DB'][i])
seed = (C.SEED_TAG, 'screen', 'oneil', UNIT)
res = S.fit(Yst, doses, noise=Rn.G['bank'], plate_var=Rn.G['pvt'], group=Rn.G['cell'][i], seed=seed, **C.FIT_ONEIL)
SC._Grid.stats = _orig_stats
prod = pd.read_parquet(os.path.join(C.RES, 'screen_oneil_q.parquet'))
row = prod[prod.unit_id == UNIT].iloc[0]
for k in ('t_std', 'p_tstd', 'p_scan', 'mean_excess', 'mean_excess_se', 'activity'):
    assert abs(float(getattr(res, k)) - float(row[k])) < 1e-12, (k, getattr(res, k), row[k])
assert abs(res.p_mean - row['p_companion']) < 1e-12
Tref = CAP[-1][:res.B_used, 0]
assert abs((1 + np.sum(Tref >= res.t_std)) / (res.B_used + 1) - res.p_tstd) < 1e-12
fam = prod[prod.family == row['family']]
pf = np.sort(fam.p_tstd.values)
nf = pf.size
n_bh = int((fam.q_tstd <= 0.05).sum())
top = [tuple((int(x) for x in c.split(','))) for c in row['top_region'].split(';')]
Y = res.analyzed_matrix
(a, b) = (np.r_[1.0, res.a_hat], np.r_[1.0, res.b_hat])
pred = np.outer(a, b)
mu = res.fitted_surface
eta = np.full_like(mu, np.nan)
eta[1:, 1:] = res.excess_map
raw = np.full_like(mu, np.nan)
raw[1:, 1:] = res.raw_excess_map
(I, J) = Y.shape
print(f"unit {UNIT}: T_SIR {res.t_std:.1f}, p {res.p_tstd:.4g} (B = {res.B_used}), scan p {res.p_scan:.4g}, mean excess {res.mean_excess:.3f} (SE {res.mean_excess_se:.3f}, {res.mean_test['name']}, p {res.p_mean:.2g}), activity {res.activity:.3f}, top region {row['top_region']}, family {row['family']} ({nf} units, {n_bh} BH)")
from matplotlib.patches import FancyArrowPatch, ArrowStyle
TSYM = 'T_{\\mathrm{SIR}}'
plt.rcParams.update({'font.size': 7, 'font.family': 'DejaVu Sans', 'mathtext.fontset': 'dejavusans', 'axes.linewidth': 0.6})
t_obs = float(res.t_std)
tb = Tref
(mean_exc, se_exc) = (float(res.mean_excess), float(res.mean_excess_se))
activity = float(res.activity)
Yr = Y
fig = plt.figure(figsize=(ST.TEXTWIDTH, 5.0))
W = 0.118
H = W * 7.0 / 5.0
pos = {'A': (0.07, 0.545), 'B': (0.325, 0.715), 'C': (0.55, 0.715), 'D': (0.425, 0.375), 'E': (0.8, 0.545)}
cmap_v = 'YlGnBu_r'

def heat(key, M, cmap, vmin, vmax, title, ylab=False):
    ax = fig.add_axes([*pos[key], W, H])
    im = ax.imshow(M, origin='lower', cmap=cmap, vmin=vmin, vmax=vmax, aspect='auto')
    ax.set_xticks([0, J - 1])
    ax.set_yticks([0, I - 1])
    ax.set_xticklabels(['0', 'max'], fontsize=6.5)
    ax.set_yticklabels(['0', 'max'], fontsize=6.5)
    ax.tick_params(length=0, pad=1)
    ax.set_xlabel('MK-4827', labelpad=0, fontsize=6.8)
    if ylab:
        ax.set_ylabel('Temozolomide', labelpad=0, fontsize=6.8)
    ax.set_title(title, fontsize=7, pad=3)
    return (ax, im)
(axA, imv) = heat('A', Y, cmap_v, 0, 1.15, 'Observed viability $Y$', ylab=True)
axA.add_patch(Rectangle((-0.5, -0.5), 5, 1, fill=False, ec='black', lw=0.9, ls='--'))
axA.add_patch(Rectangle((-0.5, -0.5), 1, 5, fill=False, ec='black', lw=0.9, ls='--'))
axB = fig.add_axes([*pos['B'], W, H])
xd = np.arange(1, 5)
axB.axhline(1.0, color='0.6', lw=0.6, ls=':')
axB.plot(xd, Y[1:, 0], 'o', ms=2.4, color='#b2182b', alpha=0.55)
axB.step(xd, a[1:], where='mid', color='#b2182b', lw=1.2)
axB.plot(xd, Y[0, 1:], 's', ms=2.4, color='#2166ac', alpha=0.55)
axB.step(xd, b[1:], where='mid', color='#2166ac', lw=1.2)
axB.text(1.0, a[1] + 0.02, 'temozolomide', color='#b2182b', fontsize=6.5, va='bottom')
axB.text(1.0, b[1] - 0.025, 'MK-4827', color='#2166ac', fontsize=6.5, va='top')
axB.set_ylim(0.8, 1.2)
axB.set_xticks([1, 4])
axB.set_xticklabels(['low', 'max'], fontsize=6.5)
axB.tick_params(labelsize=6.5, length=2, pad=1)
axB.set_xlabel('Dose alone', labelpad=0, fontsize=6.8)
axB.set_ylabel('Viability', labelpad=0, fontsize=6.8)
axB.set_title('Single agents $\\hat a,\\ \\hat b$', fontsize=7, pad=3)
for s in ('top', 'right'):
    axB.spines[s].set_visible(False)
(axC, _) = heat('C', pred, cmap_v, 0, 1.15, 'Bliss reference $\\hat a\\hat b$')
(axD, _) = heat('D', mu, cmap_v, 0, 1.15, 'Isotonic surface $\\hat\\mu$')
(axE, ime) = heat('E', eta, 'RdBu_r', -0.8, 0.8, 'Excess $\\hat\\eta=\\hat a\\hat b-\\hat\\mu$')
rr = [c[0] for c in top]
cc = [c[1] for c in top]
axE.add_patch(Rectangle((min(cc) - 0.5, min(rr) - 0.5), max(cc) - min(cc) + 1, max(rr) - min(rr) + 1, fill=False, ec='black', lw=1.3))
cb1 = fig.add_axes([pos['A'][0] + 0.01, pos['A'][1] - 0.075, W - 0.02, 0.012])
fig.colorbar(imv, cax=cb1, orientation='horizontal').ax.tick_params(labelsize=6.5, length=2, pad=1)
cb2 = fig.add_axes([pos['E'][0] + W + 0.008, pos['E'][1] + 0.01, 0.009, H - 0.02])
fig.colorbar(ime, cax=cb2).ax.tick_params(labelsize=6.5, length=2, pad=1)
panel_labels = {}
for (key, dx) in zip('ABCDE', [-0.06, -0.055, -0.04, -0.04, -0.04]):
    (x0, y0) = pos[key]
    panel_labels[key] = fig.text(x0 + dx, y0 + H + 0.052, key + '.', fontweight='bold', fontsize=8.5)
(FW, FH) = (ST.TEXTWIDTH, 5.0)
(LIGHT, DARK) = ('#a9bfdc', '#46648f')

def routed(points, r_in=0.09, w_out=6.2, w_in=4.4, head_len=0.085, head_w=0.11):
    """Thick arrow along an orthogonal polyline (figure fractions) with rounded corners and a block head."""
    P = [np.array([x * FW, y * FH]) for (x, y) in points]
    d_end = (P[-1] - P[-2]) / np.linalg.norm(P[-1] - P[-2])
    tip = P[-1].copy()
    P[-1] = tip - d_end * head_len
    (verts, codes) = ([P[0]], [Path.MOVETO])
    for k_ in range(1, len(P) - 1):
        (A_, C_, B_) = (P[k_ - 1], P[k_], P[k_ + 1])
        u = (C_ - A_) / np.linalg.norm(C_ - A_)
        v = (B_ - C_) / np.linalg.norm(B_ - C_)
        r = min(r_in, np.linalg.norm(C_ - A_) / 2, np.linalg.norm(B_ - C_) / 2)
        verts += [C_ - u * r, C_, C_ + v * r]
        codes += [Path.LINETO, Path.CURVE3, Path.CURVE3]
    verts.append(P[-1])
    codes.append(Path.LINETO)
    path = Path(np.array(verts), codes)
    for (lw, col) in ((w_out, DARK), (w_in, LIGHT)):
        fig.patches.append(PathPatch(path, transform=fig.dpi_scale_trans, fill=False, lw=lw, ec=col, capstyle='butt', joinstyle='round', zorder=2))
    n = np.array([-d_end[1], d_end[0]])
    base = tip - d_end * head_len
    head = np.array([base + n * head_w / 2, tip, base - n * head_w / 2])
    fig.patches.append(Polygon(head, closed=True, transform=fig.dpi_scale_trans, fc=LIGHT, ec=DARK, lw=0.7, zorder=3))

def label(x, y, txt, ha='center', rot=0):
    fig.text(x, y, txt, ha=ha, va='center', fontsize=6.8, color='#2b4469', rotation=rot)
cy = lambda k_: pos[k_][1] + H / 2
xA_out = pos['A'][0] + W + 0.012
trunk = xA_out + 0.035
routed([(xA_out, cy('A')), (trunk, cy('A')), (trunk, cy('B')), (pos['B'][0] - 0.072, cy('B'))])
routed([(xA_out, cy('A')), (trunk, cy('A')), (trunk, cy('D')), (pos['D'][0] - 0.03, cy('D'))])
label(trunk + 0.027, (cy('A') + cy('B')) / 2 + 0.01, '1D isotonic', rot=90)
label(trunk + 0.027, (cy('A') + cy('D')) / 2 + 0.004, '2D isotonic', rot=90)
routed([(pos['B'][0] + W + 0.012, cy('B')), (pos['C'][0] - 0.035, cy('B'))])
label((pos['B'][0] + W + pos['C'][0]) / 2 - 0.011, cy('B') + 0.051, 'product')
cxE = pos['E'][0] + W / 2
routed([(pos['C'][0] + W + 0.012, cy('C')), (cxE, cy('C')), (cxE, pos['E'][1] + H + 0.05)])
routed([(pos['D'][0] + W + 0.012, cy('D')), (cxE, cy('D')), (cxE, pos['E'][1] - 0.058)])
blk = np.array([np.nanmean((np.outer(Yst[k][:, 0], Yst[k][0, :]) - Yst[k])[1:, 1:]) for k in range(Yst.shape[0])])
assert abs(blk.mean() - mean_exc) < 1e-12 and abs(blk.std(ddof=1) / np.sqrt(blk.size) - se_exc) < 1e-12
from scipy import stats as _st
half = _st.t.ppf(0.975, blk.size - 1) * se_exc
(ci_lo, ci_hi) = (mean_exc - half, mean_exc + half)
called = np.sort(fam.p_tstd.values[(fam.q_tstd <= 0.05).values])
assert called.size == n_bh and np.all(pf[:n_bh] == called)
cards = [('Synergy anywhere?', f'${TSYM}$ vs. simulated\nexact Bliss: $p={res.p_tstd:.4f}$'), ('Where?', 'high-dose block with a\nsynergistic cell (FWER)'), ('On average?', f'mean excess {mean_exc:.2f}\n(95% CI {ci_lo:.2f} to {ci_hi:.2f})'), ('How much killing?', f'activity {activity:.2f}\n(mean killing)'), ('Many matrices?', f'BH in the cell line:\n{n_bh} of {nf} called')]
(cw, ch, cy0, gap, x0) = (0.172, 0.215, 0.03, 0.017, 0.035)
fig.text(x0 - 0.02, cy0 + ch + 0.035, 'F.', fontweight='bold', fontsize=8.5)
for (i, (q, ans)) in enumerate(cards):
    xl = x0 + i * (cw + gap)
    fig.patches.append(FancyBboxPatch((xl, cy0), cw, ch, boxstyle='round,pad=0.004,rounding_size=0.012', transform=fig.transFigure, fc='#f3f5fa', ec='#46648f', lw=0.7, zorder=-5))
    fig.text(xl + cw / 2, cy0 + ch - 0.018, q, ha='center', va='top', fontsize=7, fontweight='bold', color='#1f3354')
    fig.text(xl + cw / 2, cy0 + 0.012, ans, ha='center', va='bottom', fontsize=6.6, color='0.15', linespacing=1.2)
    if i == 1:
        rect = [xl + 0.03, cy0 + 0.078, cw - 0.06, 0.082]
    elif i in (2, 4):
        rect = [xl + 0.052, cy0 + 0.078, cw - 0.077, 0.082]
    elif i == 0:
        rect = [xl + 0.035, cy0 + 0.09, cw - 0.06, 0.072]
    else:
        rect = [xl + 0.035, cy0 + 0.078, cw - 0.06, 0.082]
    ax = fig.add_axes(rect)
    ax.set_zorder(5)
    ax.patch.set_alpha(0)
    for s in ('top', 'right'):
        ax.spines[s].set_visible(False)
    ax.tick_params(labelsize=6.5, length=1.5, pad=1)
    if i == 0:
        ax.hist(tb[tb > 0], bins=np.logspace(-2, np.log10(t_obs * 1.8), 46), color='0.6', lw=0)
        ax.axvline(t_obs, color='#b2182b', lw=1.1)
        ax.set_xscale('log')
        ax.set_yticks([])
        ax.set_xticks([1, 100])
        ax.set_xticklabels(['1', '100'])
        ax.xaxis.set_minor_locator(matplotlib.ticker.NullLocator())
    elif i == 1:
        ax.imshow(raw[1:, 1:], origin='lower', cmap='RdBu_r', vmin=-0.8, vmax=0.8, aspect='auto')
        rr = [c[0] - 1 for c in top]
        cc = [c[1] - 1 for c in top]
        ax.add_patch(Rectangle((min(cc) - 0.5, min(rr) - 0.5), max(cc) - min(cc) + 1, max(rr) - min(rr) + 1, fill=False, ec='black', lw=1.1))
        ax.set_xticks([])
        ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_visible(False)
    elif i == 2:
        ax.axhline(0, color='0.5', lw=0.6, ls=':')
        ax.plot(np.linspace(-0.62, -0.18, blk.size), blk, 'o', ms=2.8, color='0.5', lw=0)
        ax.errorbar([0.45], [mean_exc], yerr=[[mean_exc - ci_lo], [ci_hi - mean_exc]], fmt='o', ms=3.2, color='#b2182b', capsize=2.5, lw=1.1, capthick=1.1)
        ax.set_xlim(-1, 1)
        ax.set_xticks([])
        ax.set_ylim(-0.04, 0.46)
        ax.set_yticks([0, 0.4])
        ax.set_yticklabels(['0', '0.4'])
        ax.set_ylabel('Excess', fontsize=6.5, labelpad=1)
    elif i == 3:
        ax.barh([0], [activity], color='#5e81b5', height=0.5)
        ax.set_xlim(0, 1)
        ax.set_yticks([])
    else:
        rk = np.arange(1, nf + 1)
        ax.plot(rk[n_bh - 1:], pf[n_bh - 1:], '-', color='0.6', lw=1.1)
        ax.plot(rk[:n_bh], pf[:n_bh], '-', color='#2166ac', lw=1.5)
        ax.plot(rk, 0.05 * rk / nf, color='#b2182b', lw=0.9)
        tie = (np.searchsorted(pf, res.p_tstd, 'left') + np.searchsorted(pf, res.p_tstd, 'right') + 1) / 2
        ax.plot([tie], [res.p_tstd], 'o', ms=3.2, color='#b2182b', zorder=6)
        ax.text(nf, 0.05 * 0.32, 'BH line', color='#b2182b', fontsize=6.5, ha='right', va='top')
        ax.set_yscale('log')
        ax.set_ylim(4e-05, 1.6)
        ax.set_xlim(0, nf + 1)
        ax.set_xticks([])
        ax.set_yticks([0.0001, 1])
        ax.set_yticklabels(['$10^{-4}$', '1'])
        ax.yaxis.set_minor_locator(matplotlib.ticker.NullLocator())
        ax.set_ylabel('p', fontsize=6.5, labelpad=1, style='italic')
fig.text(x0 + 0.02, cy0 + ch + 0.035, 'Questions SIR answers', fontsize=7.5, fontweight='bold', color='#1f3354')
fig.canvas.draw()
rend = fig.canvas.get_renderer()
inv = fig.transFigure.inverted()
others = [t for t in fig.findobj(matplotlib.text.Text) if t.get_text().strip() and t not in panel_labels.values()]
moved = {}
for key in 'ABCDE':
    (px0, py1) = (pos[key][0], pos[key][1] + H)
    tb = panel_labels[key].get_window_extent(rend)
    ((tx0, ty0), (tx1, ty1)) = inv.transform([[tb.x0, tb.y0], [tb.x1, tb.y1]])
    (ccx, ccy) = ((tx0 + tx1) / 2, (ty0 + ty1) / 2)
    (lx, ly) = panel_labels[key].get_position()
    for frac in np.arange(0.5, -0.001, -0.025):
        panel_labels[key].set_position((lx - (ccx - px0) * frac, ly - (ccy - py1) * frac))
        lb = panel_labels[key].get_window_extent(rend).expanded(1.15, 1.15)
        if not any((o.get_visible() and lb.overlaps(o.get_window_extent(rend)) for o in others)):
            break
    moved[key] = round(float(frac), 3)
print('panel label move fractions:', moved)
out = str(mc.paths.FIGURES / 'fig2_schematic')
fig.savefig(out + '.pdf', bbox_inches='tight', pad_inches=0.04)
fig.savefig(out + '.png', dpi=170, bbox_inches='tight', pad_inches=0.04)
print('done')
