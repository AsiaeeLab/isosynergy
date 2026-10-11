"""Single-thread benchmark of the bundled Python implementation."""
import paths
import os
import tempfile
os.environ["NUMBA_CACHE_DIR"] = tempfile.mkdtemp(prefix="runtime_", dir=paths.OUT / "cache")
import csv
import sys
import time
from pathlib import Path
import numpy as np
import sirsyn
out = paths.OUT / 'runtime'
out.mkdir(parents=True, exist_ok=True)
noise = sirsyn.ParametricNoise.additive(0.04)

def matrix(k):
    a = np.linspace(1, 0.3, k)
    b = np.linspace(1, 0.25, k)
    Y = np.outer(a, b) + np.sin(np.arange(k * k)).reshape(k, k) * 0.04
    Y[0, 0] = 1
    return Y
t0 = time.perf_counter()
sirsyn.fit(matrix(5), design='S', noise=noise, B=399, seed=0, two_stage=False, fragility='never')
first_call_s = time.perf_counter() - t0
rows = []
for k in (4, 5, 6, 8, 10):
    Y = matrix(k)
    for B in (399, 9999):

        def run(seed):
            return sirsyn.fit(Y, design='S', noise=noise, B=B, seed=seed, two_stage=False, fragility='never')
        run(0)
        timing = []
        for seed in range(1, 3 if paths.SMOKE else 21):
            start = time.perf_counter()
            run(seed)
            timing.append(1000 * (time.perf_counter() - start))
        rows.append(dict(engine='Python', design='S', rows=k, columns=k, B=B, n_std=200, two_stage=False, fragility='never', repetitions=len(timing), median_ms=np.median(timing), mean_ms=np.mean(timing), p90_ms=np.quantile(timing, 0.9)))
        print(rows[-1], flush=True)
with (out / 'runtime_grid.csv').open('w') as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0]))
    w.writeheader()
    w.writerows(rows)
units = len(np.unique(np.load(paths.INPUTS/'noise_bank_oneil.npz')['group_key']))
timings = {(row['rows'],row['B']):row['median_ms'] for row in rows}
estimates = [dict(rows=k,columns=k,units=units,second_stage_share=.05,
    core_minutes=units*(timings[k,399]+.05*timings[k,9999])/60000) for k in (4,5,6,8,10)]
with (out / 'runtime_screen_estimates.csv').open('w') as f:
    w = csv.DictWriter(f,fieldnames=list(estimates[0]))
    w.writeheader()
    w.writerows(estimates)
(out / 'runtime_first_call.txt').write_text(f'first call including compilation: {first_call_s:.1f} s\n')
print(f'first call including compilation: {first_call_s:.1f} s')
