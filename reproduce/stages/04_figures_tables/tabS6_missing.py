"""tabS6_missing calculations."""
import os
import mc
import style as ST
import numpy as np
import pandas as pd
ROWS = [('full', 'None'), ('int1', '1 combination well missing'), ('int2', '2 combination wells missing'), ('int3', '3 combination wells missing'), ('rowdrop', 'Single-agent dose row dropped')]
TEX_LABEL = {}

def signed(x, nd=4):
    return f'${x:+.{nd}f}$'

def main():
    d = pd.read_parquet(os.path.join(mc.RESDIR, 'fig6_missing_size.parquet'))
    w = pd.read_parquet(os.path.join(mc.RESDIR, 'fig6_missing_power.parquet'))
    n = len(d)
    r0 = (d.p_full <= 0.05).values.astype(float)
    pw0 = w.groupby(['I', 'law', 'surface', 'cond']).p_full.apply(lambda v: (v <= 0.05).mean())
    (lines, rec) = ([], [])
    for (key, lab) in ROWS:
        x = (d[f'p_{key}'] <= 0.05).values.astype(float)
        k = int(x.sum())
        (lo, hi) = mc.wilson(k, n)
        r01 = float((d[f'p_{key}'] <= 0.01).mean())
        cell05 = f'{k / n:.4f} [{lo:.4f}, {hi:.4f}]'
        if key == 'full':
            (diff_s, dm, dl, dh) = ('', np.nan, np.nan, np.nan)
        else:
            dd = x - r0
            dm = dd.mean()
            se = dd.std(ddof=1) / np.sqrt(n)
            (dl, dh) = (dm - 1.959964 * se, dm + 1.959964 * se)
            diff_s = f'{signed(dm)} [{signed(dl)}, {signed(dh)}]'
        if key == 'full':
            (pw_s, pwm) = ('', np.nan)
        else:
            pw = w.groupby(['I', 'law', 'surface', 'cond'])[f'p_{key}'].apply(lambda v: (v <= 0.05).mean())
            delta = pw - pw0
            pwm = float(delta.mean())
            pw_s = signed(pwm, 3)
            rec_extra = dict(power_change_min=float(delta.min()), power_change_max=float(delta.max()), worst_row='/'.join(map(str, delta.idxmin())))
        rec.append(dict(test=key, label=lab, n=n, rate05=k / n, lo=lo, hi=hi, rate01=r01, diff=dm, diff_lo=dl, diff_hi=dh, power_change=pwm, **rec_extra if key != 'full' else {}))
        tex_lab = TEX_LABEL.get(key, lab)
        lines.append(f'{tex_lab} & {cell05} & {r01:.4f} & {diff_s} & {pw_s}\\\\')
    body = '\n'.join(['\\begin{tabular}{@{}llllp{0.1\\linewidth}@{}}', '\\toprule', 'Missing wells & $T_{\\mathrm{SIR}}$, 0.05 & $T_{\\mathrm{SIR}}$, 0.01 & Paired difference, 0.05 & Mean power change\\\\', '\\midrule', *lines, '\\bottomrule', '\\end{tabular}'])
    ST.write_table('tabS6_missing', body)
    out = pd.DataFrame(rec)
    out.to_csv(os.path.join(mc.RESDIR, 'tabS6_missing.csv'), index=False)
    print(body)
    print(out.to_string())
if __name__ == '__main__':
    main()
