"""Recompute paper quantities and fail on any mismatch with the stated expectations."""
import paths
import argparse
import csv
import hashlib
import json
import math
import re
import sys
import numpy as np
import pandas as pd
from check_metrics import release_metrics
from stage_checks import stage_metrics


def within_tolerance(value, expected, tolerance):
    difference = abs(value - expected)
    # Decimal rounding boundaries can differ by one floating-point bit.
    return math.isfinite(value) and (difference <= tolerance or
        math.isclose(difference, tolerance, rel_tol=1e-12, abs_tol=0.0))


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--real-only',action='store_true');a=ap.parse_args()
    table=paths.REAL/'release/sir_drugcombdb_unit_results.parquet'
    d=pd.read_parquet(table)
    assert len(d.columns)==61, f'Release has {len(d.columns)} columns'
    assert d.unit_id.is_unique
    assert set(d.study)=={'ONEIL','ALMANAC'}
    for col in ['p_tstd','q_tstd','p_scan','p_companion','mean_excess','activity','t_std']:
        assert np.isfinite(d[col]).all(), f'Non-finite release values in {col}'
    for col in ['p_tstd','q_tstd','p_scan','p_companion']:
        assert d[col].between(0,1).all(), f'Probability range: {col}'
    for line in (table.parent/'SHA256SUMS.txt').read_text().splitlines():
        checksum,name=line.split(maxsplit=1)
        assert hashlib.sha256((table.parent/name).read_bytes()).hexdigest()==checksum,name
    released_csv=pd.read_csv(table.with_suffix('.csv'),keep_default_na=False,low_memory=False)
    assert list(released_csv)==list(d) and len(released_csv)==len(d), 'CSV/Parquet structure'
    for col in d:
        if pd.api.types.is_integer_dtype(d[col]):
            present=d[col].notna()
            assert released_csv.loc[~present,col].eq('').all(), f'CSV/Parquet empty integer values: {col}'
            x=d.loc[present,col].to_numpy();y=pd.to_numeric(released_csv.loc[present,col],errors='raise').to_numpy()
            assert np.array_equal(x,y), f'CSV/Parquet integer values: {col}'
        elif pd.api.types.is_numeric_dtype(d[col]) and not pd.api.types.is_bool_dtype(d[col]):
            x=d[col].to_numpy(float);y=pd.to_numeric(released_csv[col],errors='coerce').to_numpy(float)
            assert np.allclose(x,y,rtol=5e-8,atol=5e-12,equal_nan=True), f'CSV/Parquet values: {col}'
        else:
            x=d[col].fillna('').astype(str);y=released_csv[col].fillna('').astype(str)
            assert x.equals(y), f'CSV/Parquet values: {col}'
    figures=['fig1_baseline_disagreement','fig2_schematic','fig7_realdata_overview',
             'figS3_positive_controls','figS4_baseline_scatter','figS5_baseline_nci_almanac','figS6_null_checks']
    if not a.real_only:
        figures+=['fig3_head_to_head','fig4_grid_power','fig5_bigl','fig6_missing_outliers',
                  'figS1_reference_components','figS2_berger_boos']
    for name in figures:assert (paths.FIGURES/(name+'.pdf')).is_file(),name
    for name in ['tabS8_controls','tabS9_null','tabS10_reproducibility','tabS10_calls','tabS11_window']:
        assert (paths.TABLES/(name+'.tex')).is_file(),name
    if not a.real_only:
        for i in range(1,8):assert list(paths.TABLES.glob(f'tabS{i}_*.tex')),f'Table S{i}'
    if paths.SMOKE:
        required=['null_oneil.parquet','null_almanac.parquet','window_units.parquet','controls_summary.csv','missing_well_sets.parquet','counts_reconciliation.csv']
        for name in required:assert (paths.REAL/name).is_file(),name
        if not a.real_only:
            for name in ['h2h_size.parquet','grid_counts.csv','bigl_ours.parquet','bb_plugin.parquet','stress_reps.parquet','fig6_text_numbers.json']:
                assert (paths.SIMS/name).is_file(),name
        result={'mode':'smoke','rows':len(d),'columns':len(d.columns),'figures':len(list(paths.FIGURES.glob('*.pdf'))),'status':'passed'}
        (paths.OUT/'checks.json').write_text(json.dumps(result,indent=2)+'\n')
        print(json.dumps(result));return
    observed=release_metrics(d)
    stages=stage_metrics(d)
    # Reading stage outputs is separate from the released-table computation.
    cache={};results=[]
    def read(name):
        if name not in cache:
            p=paths.OUT/name
            cache[name]=json.loads(p.read_text()) if p.suffix=='.json' else pd.read_csv(p) if p.suffix=='.csv' else pd.read_parquet(p) if p.suffix=='.parquet' else p.read_text()
        return cache[name]
    def evaluate(recipe):
        parts=recipe.split('|',2) if recipe.startswith('json|') else recipe.split('|')
        if parts[0]=='release':return observed[parts[1]]
        if parts[0]=='stage':return stages[parts[1]]
        if parts[0]=='json':
            v=read(parts[1])
            for k in json.loads(parts[2]):v=v[int(k)] if isinstance(v,list) else v[k]
            return float(v)
        if parts[0]=='aggregate':
            data=read(parts[1])
            for key,values in json.loads(parts[2]).items():
                data=data[data[key].isin(values if isinstance(values,list) else [values])]
            values=data[parts[3]] if parts[3] else data
            operation=json.loads(parts[4])
            if isinstance(operation,list):
                if operation[0]=='group_rate_max':
                    threshold=operation[2]*(1+operation[3])
                    data=data.assign(_rejected=values.to_numpy(float)<=threshold)
                    return float(data.groupby(operation[1])._rejected.mean().max())
                threshold=operation[1]*(1+(operation[2] if len(operation)>2 else 0))
                comparable=values.to_numpy(float) if len(operation)>2 else values
                return float((comparable<=threshold).mean())
            if operation=='count':return len(values)
            return float(getattr(values,operation)())
        if parts[0]=='csv':return float(read(parts[1]).iloc[int(parts[2])][parts[3]])
        if parts[0]=='tex':
            body='\n'.join(line for line in read(parts[1]).splitlines() if not line.startswith('Runtime'))
            numbers=re.findall(r'(?<![A-Za-z])[-+]?\d+(?:\.\d+)?',body)
            return float(numbers[int(parts[2])])
        if parts[0]=='table':
            body=read(parts[1]).split(r'\midrule',1)[1].split(r'\bottomrule',1)[0]
            values=' '.join(' & '.join(line.split('&')[int(parts[3]):])
                            for line in body.splitlines() if '&' in line)
            values=re.sub(r'\\multicolumn\{[^}]*\}\{[^}]*\}', '', values)
            numbers=re.findall(r'[-+]?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?',values)
            return float(numbers[int(parts[2])].replace(',',''))
        raise ValueError(recipe)
    failures=[]
    for key,value in stages.items():
        if value!=0:
            message=f'stage.{key}: expected 0; observed {value}'
            failures.append(message);print(message,file=sys.stderr)
    with (paths.ROOT/'expected_numbers.csv').open() as stream:
        for row in csv.DictReader(stream):
            if a.real_only and row['scope']=='simulation':continue
            try:
                value=evaluate(row['computation']);expected=float(row['value']);tolerance=float(row['tolerance'])
                ok=within_tolerance(value, expected, tolerance)
                message='' if ok else f"{row['id']}: expected {expected} +/- {tolerance}; observed {value}"
            except Exception as exc:
                value=None;ok=False;message=f"{row['id']}: {type(exc).__name__}: {exc}"
            results.append(dict(id=row['id'],expected=row['value'],observed=value,tolerance=row['tolerance'],passed=ok))
            if not ok:failures.append(message);print(message,file=sys.stderr)
    (paths.OUT/'checks.json').write_text(json.dumps({'mode':'full','checks':results,'failures':len(failures)},indent=2)+'\n')
    print(f'{len(results)} numerical checks; {len(failures)} mismatches.')
    if failures:raise SystemExit(1)
    if not a.real_only:
        import simulation_cache
        simulation_cache.seal()

if __name__=='__main__':main()
