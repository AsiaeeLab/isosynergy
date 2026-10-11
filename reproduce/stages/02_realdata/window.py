"""DrugCombDB window calculations."""
import argparse
import json
import os
import time
from multiprocessing import Pool
import numpy as np
import pandas as pd
import common as C
import runner as Rn
S = C.S
MULT = 3.0
SYN = {'15DEOXYPENTOFURANOSYL5FLUORO4PENTYLOXYCARBONYLAMINOPYRIMIDIN21HONE': 'CAPECITABINE', '5FLUORO2DEOXYURIDINE': 'FLOXURIDINE', '5FU': '5FLUOROURACIL', 'ACTINOMYCIND': 'DACTINOMYCIN', 'ADMHYDROCHLORIDE': 'DOXORUBICIN', 'AMINOLEVULINICACIDHYDROCHLORIDE': 'AMINOLEVULINICACID', 'ANTIBIOTICAD32': 'VALRUBICIN', 'ANTIBIOTICAY22989': 'RAPAMYCIN', 'AZACYTIDINE': 'AZACITIDINE', 'BENDAMUSTINEHYDROCHLORIDE': 'BENDAMUSTINE', 'BORTEZOMIB': 'NSC681239', 'CISPLATINO': 'CISPLATIN', 'DAUNORUBICINHYDROCHLORIDE': 'DAUNORUBICIN', 'DEXRAZOXANE': 'DEXRAZOXONE', 'ELOXATINTNSANOFISYNTHELAB': 'OXALIPLATIN', 'EMCYTPHARMACIA': 'ESTRAMUSTINE', 'ERLOTINIBHYDROCHLORIDE': 'ERLOTINIB', 'FLUDARABINEBASE': '2FLUOROADENOSINE', 'GEMZARLILLY': 'GEMCITABINE', 'MECHLORETHAMINEHYDROCHLORIDE': 'NITROGENMUSTARD', 'MERCAPTOPURINE': '6MERCAPTOPURINE', 'MITOMYCINE': 'MITOMYCINC', 'NAVELBINEDITARTRATETN': 'VINORELBINE', 'NSC127716': 'DECITABINE', 'NSC141540': 'ETOPOSIDE', 'NSC733504': 'EVEROLIMUS', 'NSC85998': 'STREPTOZOCIN', 'PAZOPANIBHYDROCHLORIDE': 'PAZOPANIB', 'PEMETREXED': 'NSC698037', 'PROCARBAZINEHYDROCHLORIDE': 'PROCARBAZINE', 'ROMIDEPSIN': 'DEPSIPEPTIDE', 'SN38': '7ETHYL10HYDROXYCAMPTOTHECIN', 'TAMOXIFENCITRATE': 'TAMOXIFEN', 'TOPOTECANHYDROCHLORIDE': 'TOPOTECAN', 'VINBLASTINESULFATE': 'VINBLASTINE', 'VINCRISTINESULFATE': 'VINCRISTINE', 'ZOLEDRONICACID': 'ZOLENDRONICACID', 'ZOLINZA': 'VORINOSTAT'}

def canon(names):
    return np.array([SYN.get(n, n) for n in C.norm_name(names)])

def plan(study, cm, prov):
    (meta, Y, DA, DB) = C.load_oneil() if study == 'oneil' else C.load_almanac()
    (ca, cb) = (canon(meta.drugA.values), canon(meta.drugB.values))
    xa = np.array([cm.get(x, np.nan) for x in ca])
    xb = np.array([cm.get(x, np.nan) for x in cb])
    elig = np.isfinite(xa) & np.isfinite(xb)
    rows = []
    for i in np.where(elig)[0]:
        ka = DA[i, 1:] <= MULT * xa[i] * (1 + 1e-09)
        kb = DB[i, 1:] <= MULT * xb[i] * (1 + 1e-09)
        rows.append(dict(idx=int(i), unit_id=meta.unit_id.iloc[i], cell_line=meta.cell_line.iloc[i], drugA=meta.drugA.iloc[i], drugB=meta.drugB.iloc[i], canonA=ca[i], canonB=cb[i], cmaxA_uM=xa[i], cmaxB_uM=xb[i], n_keep_a=int(ka.sum()), n_keep_b=int(kb.sum()), n_doses_a=int(ka.size), n_doses_b=int(kb.size), keep_a=ka, keep_b=kb, full_window=bool(ka.all() and kb.all()), testable=bool(ka.sum() >= 2 and kb.sum() >= 2)))
    P = pd.DataFrame(rows)
    used = sorted(set(P.canonA) | set(P.canonB)) if len(P) else []
    return (meta, P, used)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--n', type=int, default=0)
    ap.add_argument('--workers', type=int, default=16)
    a = ap.parse_args()
    t0 = time.time()
    cc = pd.read_csv(C.CMAX)
    cc['canon'] = canon(cc.drug_drugcombdb.values)
    cm = dict(zip(cc.canon, cc.cmax_uM))
    prov = dict(zip(cc.canon, cc.cmax_source))
    (out_units, summ, drugs) = ([], {}, [])
    for study in ('oneil', 'almanac'):
        (meta, P, used) = plan(study, cm, prov)
        scr = pd.read_parquet(os.path.join(C.RES, f'screen_{study}_q.parquet')).set_index('unit_id')
        P['family'] = scr.loc[P.unit_id, 'family'].values
        for dn in used:
            r = cc[cc.canon == dn].iloc[0]
            k = (P.canonA == dn) | (P.canonB == dn)
            drugs.append(dict(study=study, drug_table_name=r.drug_drugcombdb, canonical=dn, cmax_uM=r.cmax_uM, window_upper_uM=MULT * r.cmax_uM, cmax_source=r.cmax_source, units_with_drug=int(k.sum()), study_names=';'.join(sorted(set(P.drugA[P.canonA == dn]) | set(P.drugB[P.canonB == dn])))))
        T = P[P.testable & ~P.full_window]
        if a.n:
            T = T.sample(n=min(a.n, len(T)), random_state=1)
        Rn.init(study)
        items = [(r.idx, r.keep_a, r.keep_b) for r in T.itertuples()]
        chunks = [items[k:k + 30] for k in range(0, len(items), 30)]
        res = []
        if chunks:
            with Pool(min(a.workers, C.MAX_WORKERS)) as pool:
                for df in pool.imap_unordered(Rn.job_window, chunks):
                    res.append(df)
        W = pd.concat(res, ignore_index=True) if res else pd.DataFrame(columns=['idx'])
        F = P[P.testable & P.full_window]
        chk = F.sample(n=min(20, len(F)), random_state=2)
        ident = []
        for r in chk.itertuples():
            w = Rn.fit_one(r.idx, r.keep_a, r.keep_b)
            ident.append(abs(w['p_tstd'] - scr.loc[r.unit_id, 'p_tstd']) + abs(w['t_std'] - scr.loc[r.unit_id, 't_std']))
        wcols = ['p_tstd', 't_std', 'p_companion', 'mean_excess', 'activity', 'n_regions_claimed', 'analyzed_rows', 'analyzed_cols', 'stage', 'log_notes']
        W = W.set_index('idx')
        for c in wcols:
            P['win_' + c] = np.nan if c != 'log_notes' else ''
        for c in wcols:
            if c in W:
                P.loc[P.idx.isin(W.index), 'win_' + c] = W.loc[P.idx[P.idx.isin(W.index)], c].values
            fk = P.testable & P.full_window
            P.loc[fk, 'win_' + c] = scr.loc[P.unit_id[fk], c].values
        nerr = int(W['error'].notna().sum()) if 'error' in W else 0
        for c in ['p_tstd', 't_std', 'p_companion', 'mean_excess', 'activity', 'n_regions_claimed']:
            P['full_' + c] = scr.loc[P.unit_id, c].values
        Tt = P[P.testable].copy()
        for pre in ('full', 'win'):
            Tt[pre + '_q_tstd'] = np.nan
            Tt[pre + '_q_companion'] = np.nan
            for (f, idx) in Tt.groupby('family').groups.items():
                Tt.loc[idx, pre + '_q_tstd'] = S.bh_qvalues(Tt.loc[idx, pre + '_p_tstd'].astype(float).values)
                Tt.loc[idx, pre + '_q_companion'] = S.bh_qvalues(Tt.loc[idx, pre + '_p_companion'].astype(float).values)
        P = P.merge(Tt[['unit_id', 'full_q_tstd', 'win_q_tstd', 'full_q_companion', 'win_q_companion']], on='unit_id', how='left')
        (fc, wc) = (Tt.full_q_tstd <= 0.05, Tt.win_q_tstd <= 0.05)
        (fm, wm) = (Tt.full_q_companion <= 0.05, Tt.win_q_companion <= 0.05)
        summ[study] = dict(drugs_in_cmax_table=int(len(cc)), drugs_with_cmax_present=len(used), units_total=int(len(meta)), units_both_drugs_with_cmax=int(len(P)), units_untestable_after_restriction=int((~P.testable).sum()), untestable_reason_counts=P[~P.testable].apply(lambda r: f'A keeps {r.n_keep_a}, B keeps {r.n_keep_b}', axis=1).value_counts().to_dict(), units_testable=int(P.testable.sum()), units_window_keeps_all_doses=int((P.testable & P.full_window).sum()), units_refit_restricted=int(len(T)), refit_errors=nerr, identity_check_max_abs_diff=float(max(ident)) if ident else None, families=int(Tt.family.nunique()), tstd_bh_calls_full=int(fc.sum()), tstd_bh_calls_window=int(wc.sum()), tstd_bh_calls_both=int((fc & wc).sum()), companion_bh_calls_full=int(fm.sum()), companion_bh_calls_window=int(wm.sum()), companion_bh_calls_both=int((fm & wm).sum()), tstd_nominal_full=int((Tt.full_p_tstd <= 0.05).sum()), tstd_nominal_window=int((Tt.win_p_tstd.astype(float) <= 0.05).sum()), spearman_mean_excess_full_vs_window=float(Tt[['full_mean_excess', 'win_mean_excess']].astype(float).corr('spearman').iloc[0, 1]), median_doses_kept_a=float(Tt.n_keep_a.median()), median_doses_kept_b=float(Tt.n_keep_b.median()), analyzed_grid_counts=Tt.apply(lambda r: f'{r.win_analyzed_rows}x{r.win_analyzed_cols}', axis=1).value_counts().to_dict())
        P['study'] = study.upper()
        out_units.append(P.drop(columns=['keep_a', 'keep_b']))
    U = pd.concat(out_units, ignore_index=True)
    tag = '' if not a.n else f'_n{a.n}'
    U.to_parquet(os.path.join(C.RES, f'window_units{tag}.parquet'), index=False)
    pd.DataFrame(drugs).to_csv(os.path.join(C.RES, f'window_drugs{tag}.csv'), index=False)
    summ['window'] = f'nonzero doses <= {MULT} x Cmax (uM) for both drugs; Cmax from {os.path.relpath(C.CMAX, C.REPO)}'
    summ['seconds'] = round(time.time() - t0, 1)
    with open(os.path.join(C.RES, f'window_summary{tag}.json'), 'w') as f:
        json.dump(summ, f, indent=1, default=str)
    print(json.dumps(summ, indent=1, default=str))
if __name__ == '__main__':
    main()
