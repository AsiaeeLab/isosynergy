"""Numbered stages for all paper calculations, with per-command wall times."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('stages', nargs='*', help='Stage numbers, 00 through 05; default all.')
    ap.add_argument('--smoke', action='store_true')
    ap.add_argument('--workers', type=int, default=int(os.environ.get('SIR_WORKERS','16')))
    ap.add_argument('--item', choices=['bigl','grid','missing','head-to-head','references','bands','bigl-r','propositions','null-search','band-search','companions','resampling','runtime','band-order'])
    ap.add_argument('--reuse-simulations',action='store_true',help='Reuse simulation outputs only after validating their numerical-check seal.')
    ap.add_argument('--real-only', action='store_true', help='Stage 04 or 05: only real-data outputs.')
    ap.add_argument('--skip-bigl', action='store_true', help='Omit the optional R comparison; its figures require existing results.')
    args = ap.parse_args()
    if not 1 <= args.workers <= 16:
        ap.error('--workers must be between 1 and 16')
    os.environ['SIR_WORKERS'] = str(args.workers)
    os.environ['SIR_SMOKE'] = '1' if args.smoke else '0'
    if args.reuse_simulations and args.smoke:
        ap.error('Simulation reuse is for full runs only')
    import paths
    selected = args.stages or ['00','01','02','03','04','05']
    if any(s not in ['00','01','02','03','04','05'] for s in selected):
        ap.error('Stages must be 00, 01, 02, 03, 04 or 05')
    manifest = paths.OUT / 'runs.jsonl'

    def run(stage, script, *params, label=None, env=None):
        file = paths.ROOT / 'stages' / stage / script
        name = label or Path(script).stem + ('_' + '_'.join(str(p).replace('/','-') for p in params) if params else '')
        command = (['Rscript','--vanilla'] if file.suffix == '.R' else [sys.executable]) + [str(file)] + list(map(str,params))
        log = paths.LOGS / (name + '.log')
        print(f'[{stage}] {name} ({args.workers} worker limit)',flush=True)
        start=time.monotonic()
        with log.open('w') as stream:
            completed=subprocess.run(command,cwd=paths.REPO,env=dict(os.environ,**(env or {})),stdout=stream,stderr=subprocess.STDOUT)
            elapsed=time.monotonic()-start
            stream.write(f'\nElapsed (wall clock): {int(elapsed//60)}:{elapsed%60:.3f}\n')
        record=dict(stage=stage,script=str(file.relative_to(paths.ROOT)),args=list(params),workers=args.workers,seconds=elapsed,returncode=completed.returncode,smoke=paths.SMOKE)
        with manifest.open('a') as stream:stream.write(json.dumps(record)+'\n')
        print(f'  {elapsed:.1f} s; exit {completed.returncode}; {log.relative_to(paths.REPO)}',flush=True)
        if completed.returncode:
            print('\n'.join(log.read_text().splitlines()[-25:]),file=sys.stderr)
            raise SystemExit(completed.returncode)

    if '00' in selected:
        run('00_data','verify.py')
    if '01' in selected:
        for script in ('matrices','noise_models','noise_summary','donors','panels','bigl_data','bigl_real'):
            run('01_inputs',script+'.py')
        run('02_realdata','prepare.py')
        run('01_inputs','nci_baselines.R')
        run('01_inputs','baseline_summary.py')
    if '02' in selected:
        for study in ('oneil','almanac'):
            run('02_realdata','null_checks.py','--study',study,'--workers',args.workers,label='null_'+study)
        run('02_realdata','null_summary.py')
        for study in ('oneil','almanac'):
            run('02_realdata','screen.py','--study',study,'--workers',args.workers,label='screen_'+study)
        for script in ('families','release','reproducibility','controls','window','missing','counts','extra_numbers'):
            run('02_realdata',script+'.py')
    if '03' in selected and args.reuse_simulations:
        import simulation_cache
        simulation_cache.verify()
        with manifest.open('a') as stream:stream.write(json.dumps(dict(stage='03_simulations',reuse=True,seconds=0,workers=0,returncode=0))+'\n')
    if '03' in selected and not args.reuse_simulations:
        def sim(script,*params,**kw):run('03_simulations',script+'.py',*params,**kw)
        item=lambda name:args.item is None or args.item==name
        if item('bigl'):
            sim('sim_bigl')
        if item('grid'):
            sim('sim_grid','run','--n',2000,'--chunk',250,'--workers',args.workers,'--tag','main')
            sim('sim_scan_partial','run','--workers',args.workers,'--n',1000)
            sim('sim_grid','collect','--tag','main')
            sim('sim_scan_partial','collect')
        if item('missing'):
            jobs=[('sim_missing','size'),('sim_screen','size'),('sim_fragility',),('sim_screen','check'),('sim_missing','power'),('sim_screen','power'),('sim_monotone','size'),('sim_monotone','power'),('sim_missing','sizeP')]
            jobs += [('sim_missing','confirm','--cell','4,B,moderate'),('sim_screen','confirm','--cell','5,B,weak'),('sim_monotone','confirm','--cell','S,5,B,horm_0.07'),('sim_screen_bank',),('sim_missing','confirm','--cell','4,B,weak'),('sim_screen','confirm','--cell','5,A,moderate'),('sim_monotone','confirm','--cell','S,5,B,hormA_0.07'),('sim_screen','confirm','--cell','4,B,weak')]
            for job in jobs:sim(*job)
        if item('head-to-head'):
            sim('sim_h2h_realpower','--study','almanac','--workers',args.workers,'--chunk',60)
            for part in ('power','size','confirm'):sim('sim_h2h_synth',part)
            sim('sim_h2h_realpower','--study','oneil','--interleave','--chunk',25,'--batch',42,'--conds','syn_broad_0.05','syn_highdose_0.10','sat_0.5','--workers',args.workers)
        if item('references'):
            sim('sim_stress_shared','--scen','null','--screens',400)
            sim('sim_stress_shift','--n',20000)
            sim('sim_stress_shared','--scen','mixed','--screens',400)
            sim('sim_stress_reps','--n',30000)
        if item('bands'):
            for part in ('refs','calib','oc','search','ocext','searchext'):sim('bb_operating',part,'--workers',args.workers)
            sim('bb_coverage')
            sim('bb_coverage','--ext','--n',200000)
            sim('bb_collect')
            sim('sim_bb_plugin')
        if item('resampling'):sim('resampling','bcheck')
        if item('runtime'):
            sim('runtime_grid');sim('runtime_python')
            run('03_simulations','runtime.R',env={'RETICULATE_PYTHON':sys.executable})
        if item('companions'):
            sim('companion','null');sim('companion','comp');sim('companion_replicates');sim('comparator_checks');sim('companion_real')
        if item('null-search'):sim('null_search')
        if item('band-search'):sim('bb_search_check')
        if item('band-order'):sim('band_order')
        if item('propositions'):
            for part in ('order','coupling','patterns','hill','mean','localization'):sim('propositions',part)
        if item('bigl-r') and not args.skip_bigl:
            run('03_simulations','bigl_reference.R')
            jobs=[('ship_eq,vb_eq','r4_.*__null',0),('vb_eq','r4_.*__(ant|syn|sat)',0),('ship_eq','r4_(strong|moderate)_.*__(ant|syn|sat)',0),('ship_eq,vb_eq','r1_.*__null',500),('vb_model','r4_.*_oneil__null',500),('vb_model','r4_.*_oneil__(ant|syn|sat)',0),('ship_eq,ship_model,vb_eq','real_oneil300',0),('ship_eq_residual','r1_.*__null',100)]
            for k,(config,pattern,nmax) in enumerate(jobs):
                run('03_simulations','bigl.R',config,pattern,args.workers,100,50,label=f'bigl_r_{k}',env={'BIGL_NMAX':str(4 if paths.SMOKE else nmax)})
            sim('bigl_runtime')
    if '04' in selected:
        real=['fig1_baseline_disagreement','fig2_schematic','fig7_realdata_overview','figS3_positive_controls','figS4_baseline_scatter','figS5_baseline_nci_almanac','figS6_null_checks','tabS8_controls','tabS9_S11_realdata']
        simulated=['tabS1_size','tabS2_misspec','tabS3_power','tabS4_stress','tabS5_bigl','tabS6_missing','tabS7_grid','sec6_numbers','fig3_head_to_head','fig4_grid_power','fig5_bigl','fig6_missing_outliers','figS1_reference_components','figS2_berger_boos','resampling_summary','supplement_numbers']
        for script in real + ([] if args.real_only else simulated):
            run('04_figures_tables',script+'.py')
    if '05' in selected:
        run('05_check','check.py',*(['--real-only'] if args.real_only else []))

if __name__=='__main__':main()
