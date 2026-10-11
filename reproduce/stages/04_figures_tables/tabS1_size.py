"""tabS1_size calculations."""
import mc
import numpy as np
import style as ST
import h2h_lib as L

def ci(w, nd=3):
    return L.fci(w, nd)

def main():
    df = L.load('size')
    m = df[df.ref == 'matched']
    keys = ['design', 'I', 'law', 'surface', 'cond']
    R05 = L.rates(m, keys, 0.05)
    R01 = L.rates(m, keys, 0.01)
    cf = L.load('confirm')
    Rc = L.rates(cf, ['design', 'I', 'law', 'surface'], 0.05)
    Rc01 = L.rates(cf, ['design', 'I', 'law', 'surface'], 0.01)
    cols = {}
    for des in ('S', 'P'):
        null05 = L.worst(R05, 'tstd', (R05.design == des) & (R05.cond == 'null'))
        ant05 = L.worst(R05, 'tstd', (R05.design == des) & (R05.cond != 'null'))
        null01 = L.worst(R01, 'tstd', (R01.design == des) & (R01.cond == 'null'))
        conf = L.worst(Rc, 'tstd', Rc.design == des)
        conf01 = L.worst(Rc01, 'tstd', Rc01.design == des)
        cols[des] = dict(null05=null05, ant05=ant05, null01=null01, conf=conf, conf01=conf01)
        print(des, 'null05', null05)
        print(des, 'confirm', conf)
        print(des, 'confirm at 0.01', conf01)
        print(des, 'ant05', ant05)
        print(des, 'null01', null01)
        for t in ('scan', 'mean'):
            print(f"  {t}: null05 {L.worst(R05, t, (R05.design == des) & (R05.cond == 'null'))}")
            print(f"  {t}: ant05 {L.worst(R05, t, (R05.design == des) & (R05.cond != 'null'))}")
            print(f'  {t}: confirm cell {L.worst(Rc, t, Rc.design == des)}')
    (S, P) = (cols['S'], cols['P'])
    body = '\n'.join(['\\begin{tabular}{@{}lll@{}}', '\\toprule', ' & Design S & Design P\\\\', '\\midrule', f"Exact Bliss, level 0.05, worst cell & {ci(S['null05'])} & {ci(P['null05'])}\\\\", f"Re-estimate of worst cell (20,000 datasets) & {ci(S['conf'], 4)} & {ci(P['conf'], 4)}\\\\", f"Antagonistic surfaces, level 0.05, worst & {ci(S['ant05'])} & {ci(P['ant05'])}\\\\", f"Exact Bliss, level 0.01, worst cell & {ci(S['null01'])} & {ci(P['null01'])}\\\\", '\\bottomrule', '\\end{tabular}'])
    ST.write_table('tabS1_size', body)
    n_cells = R05[R05.cond == 'null'].groupby('design').size().to_dict()
    n_ant = R05[R05.cond != 'null'].groupby('design').size().to_dict()
    print('exact-Bliss cells per design', n_cells, 'antagonism cells per design', n_ant)
    print('datasets per cell', sorted(set(R05.n)), 'confirm n', sorted(set(Rc.n)))
if __name__ == '__main__':
    main()
