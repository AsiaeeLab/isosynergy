"""tabS2_misspec calculations."""
import mc
import style as ST
import h2h_lib as L
ROWS = [('D', 'Gaussian reference, $t_4$ data'), ('C', "Constant SD 0.08, O'Neil-shaped data"), ('B', 'Constant-CV Gaussian reference, log-normal data')]

def main():
    df = L.load('size')
    g = df[(df.ref == 'misspecified') & (df.cond == 'null')]
    R = L.rates(g, ['design', 'I', 'law', 'surface'], 0.05)
    R01 = L.rates(g, ['design', 'I', 'law', 'surface'], 0.01)
    lines = ['\\begin{tabular}{@{}llrrr@{}}', '\\toprule', 'Misspecification & Design & $T_{\\mathrm{SIR}}$ & Region scan & Mean test\\\\', '\\midrule']
    for (tl, lab) in ROWS:
        for (k, des) in enumerate(('S', 'P')):
            m = (R.law == tl) & (R.design == des)
            w = {t: L.worst(R, t, m) for t in ('tstd', 'scan', 'mean')}
            w01 = {t: L.worst(R01, t, (R01.law == tl) & (R01.design == des)) for t in ('tstd', 'scan', 'mean')}
            print(tl, des, {t: (round(v['rate'], 4), v['cell']) for (t, v) in w.items()}, '| at 0.01:', {t: round(v['rate'], 4) for (t, v) in w01.items()})
            lines.append(f"{(lab if k == 0 else '')} & {des} & {L.fr(w['tstd']['rate'])} & {L.fr(w['scan']['rate'])} & {L.fr(w['mean']['rate'])}\\\\")
    lines += ['\\bottomrule', '\\end{tabular}']
    ST.write_table('tabS2_misspec', '\n'.join(lines))
    print('cells per row', R.groupby(['law', 'design']).size().to_dict(), 'datasets per cell', sorted(set(R.n)))
if __name__ == '__main__':
    main()
