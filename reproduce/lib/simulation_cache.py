"""Content hashes for reuse of a numerically checked simulation run."""
import paths
import hashlib
import json
from pathlib import Path


def digest(file):
    h=hashlib.sha256()
    with file.open('rb') as stream:
        for chunk in iter(lambda:stream.read(8<<20),b''):h.update(chunk)
    return h.hexdigest()


def sources():
    files=[]
    for folder in [paths.ROOT/'lib',paths.ROOT/'stages',paths.REPO/'inst/python',paths.REPO/'R']:
        files.extend(p for p in folder.rglob('*') if p.is_file() and p.suffix in ['.py','.R'])
    files.extend(paths.ROOT/name for name in ['driver.py','run_all.sh','requirements.txt','expected_numbers.csv'])
    files.extend(p for p in paths.DATA.iterdir() if p.suffix=='.csv')
    return {str(p.relative_to(paths.REPO)):digest(p) for p in sorted(set(files))}


def outputs():
    files=list(paths.SIMS.rglob('*'))+list((paths.OUT/'runtime').rglob('*'))
    for pattern in ['bb*','band_order*','bigl_raw','bigl_runtime.csv']:
        for p in paths.INPUTS.glob(pattern):files.extend(p.rglob('*') if p.is_dir() else [p])
    return {str(p.relative_to(paths.OUT)):digest(p) for p in sorted(set(files)) if p.is_file()}


def seal():
    record=dict(format=1,validation='Full numerical check passed',sources=sources(),outputs=outputs())
    (paths.OUT/'simulation_cache.json').write_text(json.dumps(record,indent=2)+'\n')
    print(f'Simulation cache: {len(record["outputs"])} files recorded.')


def verify():
    p=paths.OUT/'simulation_cache.json'
    if not p.is_file():raise SystemExit('No checked simulation cache. Run stages 03, 04 and 05 first.')
    record=json.loads(p.read_text())
    if record['sources']!=sources():raise SystemExit('Simulation cache source hashes differ. Run stage 03 again.')
    for name,expected in record['outputs'].items():
        file=paths.OUT/name
        if not file.is_file() or digest(file)!=expected:raise SystemExit(f'Simulation cache mismatch: {name}')
    print(f'Reusing {len(record["outputs"])} simulation files after checking their source and content hashes.',flush=True)
