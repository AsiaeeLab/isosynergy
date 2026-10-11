"""style calculations."""
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.text import Text
import paths
REPO = str(paths.REPO)
FIGDIR = str(paths.FIGURES)
TABDIR = str(paths.TABLES)
PREVIEW = str(paths.OUT / 'previews')
RESDIR = str(paths.SIMS)
TEXTWIDTH = 6.7
MIN_PT = 7.0
plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 8, 'axes.labelsize': 8, 'axes.titlesize': 8, 'xtick.labelsize': 7, 'ytick.labelsize': 7, 'legend.fontsize': 7, 'legend.title_fontsize': 7, 'mathtext.fontset': 'dejavusans', 'axes.spines.top': False, 'axes.spines.right': False, 'axes.linewidth': 0.6, 'xtick.major.width': 0.6, 'ytick.major.width': 0.6, 'lines.linewidth': 1.1, 'lines.markersize': 4, 'errorbar.capsize': 1.5, 'legend.frameon': False, 'figure.dpi': 150, 'savefig.dpi': 300, 'pdf.fonttype': 42, 'ps.fonttype': 42})
COL = {'tstd': '#0072B2', 'scan': '#D55E00', 'mean': '#009E73', 'bigl': '#CC79A7', 'bigl2': '#882255', 'sfp': '#E69F00', 'dm': '#56B4E9', 'oracle': '0.15', 'bb': '#AA4499', 'gray': '0.45', 'oneil': '#4C72B0', 'almanac': '#DD8452'}
TSTD = '$T_{\\mathrm{SIR}}$'
LAB = {'tstd': TSTD, 'scan': 'Region scan', 'mean': 'Mean test'}

def panel_label(ax, letter, x=-0.16, y=1.04, size=9):
    ax.text(x, y, f'{letter}.', transform=ax.transAxes, fontsize=size, fontweight='bold', va='bottom', ha='left')

def _check(fig, name, allow_titles=False):
    fig.canvas.draw()
    bad = []
    for t in fig.findobj(Text):
        if not t.get_visible() or not t.get_text().strip():
            continue
        if t.get_fontsize() < MIN_PT - 1e-06:
            bad.append((t.get_text()[:40], t.get_fontsize()))
    if bad:
        raise ValueError(f'{name}: text below {MIN_PT} pt: {bad[:8]}')
    for ax in [] if allow_titles else fig.axes:
        if ax.get_title() or ax.get_title(loc='left') or ax.get_title(loc='right'):
            raise ValueError(f'{name}: axes title present ({ax.get_title()!r}); move it to the caption')
    if getattr(fig, '_suptitle', None) is not None and fig._suptitle.get_text():
        raise ValueError(f'{name}: suptitle present; move it to the caption')
    bb = fig.get_tightbbox(fig.canvas.get_renderer())
    (w, h) = (bb.width, bb.height)
    if w > TEXTWIDTH + 0.02:
        raise ValueError(f'{name}: tight width {w:.2f} in exceeds {TEXTWIDTH} in')
    return (w, h)

def save(fig, name, pad=0.05, allow_titles=False):
    (w, h) = _check(fig, name, allow_titles)
    os.makedirs(FIGDIR, exist_ok=True)
    out = os.path.join(FIGDIR, f'{name}.pdf')
    fig.savefig(out, bbox_inches='tight', pad_inches=pad)
    if PREVIEW:
        os.makedirs(PREVIEW, exist_ok=True)
        fig.savefig(os.path.join(PREVIEW, f'{name}.png'), bbox_inches='tight', pad_inches=pad, dpi=200)
    plt.close(fig)
    print(f'wrote {out}  ({w:.2f} x {h:.2f} in)')
    return out

def write_table(name, body):
    os.makedirs(TABDIR, exist_ok=True)
    out = os.path.join(TABDIR, f'{name}.tex')
    with open(out, 'w') as f:
        f.write(body.rstrip() + '\n')
    print(f'wrote {out}')
    return out

def fmt_rate(p, nd=3):
    return f'{p:.{nd}f}'

def fmt_ci(p, lo, hi, nd=3):
    return f'{p:.{nd}f} [{lo:.{nd}f}, {hi:.{nd}f}]'
