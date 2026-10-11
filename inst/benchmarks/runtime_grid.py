"""Single-core time per matrix by grid size for the complete sirsyn.fit call.

Design S, additive Gaussian SD 0.04, fixed B (two_stage=False, fragility="never"), 200 scan
standardization plates; the primary test, region scan and companion test are all included.
Each configuration is warmed once, then timed over seeds 1..20. Usage:

    taskset -c 0 python3 runtime_grid.py OUTDIR
"""
import csv
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python"))
import sirsyn

out = Path(sys.argv[1] if len(sys.argv) > 1 else ".")
out.mkdir(parents=True, exist_ok=True)
noise = sirsyn.ParametricNoise.additive(.04)


def matrix(k):
    a = np.linspace(1, .3, k)
    b = np.linspace(1, .25, k)
    Y = np.outer(a, b) + np.sin(np.arange(k * k)).reshape(k, k) * .04
    Y[0, 0] = 1
    return Y


t0 = time.perf_counter()
sirsyn.fit(matrix(5), design="S", noise=noise, B=399, seed=0, two_stage=False, fragility="never")
first_call_s = time.perf_counter() - t0

rows = []
for k in (4, 5, 6, 8, 10):
    Y = matrix(k)
    for B in (399, 9999):
        def run(seed):
            return sirsyn.fit(Y, design="S", noise=noise, B=B, seed=seed, two_stage=False, fragility="never")
        run(0)
        timing = []
        for seed in range(1, 21):
            start = time.perf_counter()
            run(seed)
            timing.append(1000 * (time.perf_counter() - start))
        rows.append(dict(engine="Python", design="S", rows=k, columns=k, B=B, n_std=200, two_stage=False,
                         fragility="never", repetitions=20, median_ms=np.median(timing),
                         mean_ms=np.mean(timing), p90_ms=np.quantile(timing, .9)))
        print(rows[-1], flush=True)
with (out / "runtime_grid.csv").open("w") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0]))
    w.writeheader()
    w.writerows(rows)
(out / "runtime_first_call.txt").write_text(f"first call including compilation: {first_call_s:.1f} s\n")
print(f"first call including compilation: {first_call_s:.1f} s")
