"""Single-thread benchmark of the bundled Python implementation."""
import paths
import csv
import json
import os
import platform
import sys
import subprocess
import time
from pathlib import Path
import numpy as np
import scipy
import numba
import sirsyn
out = paths.OUT / 'runtime'
out.mkdir(parents=True, exist_ok=True)
a = np.array([1, 0.95, 0.8, 0.55, 0.3])
b = np.array([1, 0.9, 0.7, 0.45, 0.25])
Y = np.outer(a, b) + np.sin(np.arange(25)).reshape(5, 5) * 0.04
Y[0, 0] = 1
noise = sirsyn.ParametricNoise.additive(0.04)
rows = []
for design in ('S', 'P'):
    for B in (399, 9999):

        def run(seed):
            return sirsyn.fit(Y, design=design, noise=noise, B=B, seed=seed, two_stage=False, fragility='never')
        run(0)
        timing = []
        for seed in range(1, 3 if paths.SMOKE else 31):
            start = time.perf_counter()
            run(seed)
            timing.append(1000 * (time.perf_counter() - start))
        rows.append(dict(engine='Python', design=design, rows=5, columns=5, B=B, n_std=200, two_stage=False, fragility='never', repetitions=len(timing), median_ms=np.median(timing), mean_ms=np.mean(timing), p90_ms=np.quantile(timing, 0.9)))
with (out / 'runtime_python.csv').open('w') as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0]))
    w.writeheader()
    w.writerows(rows)
model = platform.processor() or platform.machine()
try:
    cpu = json.loads(subprocess.check_output(['lscpu','--json'],text=True))
    model = next((row['data'] for row in cpu['lscpu'] if row['field'].strip(':').lower() == 'model name'),model)
except (OSError,subprocess.CalledProcessError,ValueError,KeyError):
    pass
ram_gib = os.sysconf('SC_PHYS_PAGES') * os.sysconf('SC_PAGE_SIZE') / 1024 ** 3
info = dict(cpu_model=model, logical_cores=os.cpu_count(), ram_gib=ram_gib, benchmark_affinity=sorted(os.sched_getaffinity(0)), platform=platform.platform(), python=sys.version, executable=Path(sys.executable).name, numpy=np.__version__, scipy=scipy.__version__, numba=numba.__version__, sirsyn=sirsyn.__version__, thread_limits={key: os.environ.get(key) for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMBA_NUM_THREADS')})
(out / 'hardware.json').write_text(json.dumps(info, indent=2) + '\n')
