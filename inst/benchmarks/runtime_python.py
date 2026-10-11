"""Pure-Python comparison; use the same CPU affinity and environment as runtime.R."""
import csv
import json
import os
import platform
import sys
import time
from pathlib import Path

import numpy as np
import scipy
import numba

# Works both in the source package and in an installed R package.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python"))
import sirsyn

out = Path(sys.argv[1] if len(sys.argv) > 1 else ".")
out.mkdir(parents=True, exist_ok=True)
a = np.array([1, .95, .8, .55, .3])
b = np.array([1, .9, .7, .45, .25])
Y = np.outer(a, b) + np.sin(np.arange(25)).reshape(5, 5) * .04
Y[0, 0] = 1
noise = sirsyn.ParametricNoise.additive(.04)
rows = []
for design in ("S", "P"):
    for B in (399, 9999):
        def run(seed):
            return sirsyn.fit(Y, design=design, noise=noise, B=B, seed=seed,
                              two_stage=False, fragility="never")
        run(0)
        timing = []
        for seed in range(1, 31):
            start = time.perf_counter()
            run(seed)
            timing.append(1000 * (time.perf_counter() - start))
        rows.append(dict(engine="Python", design=design, rows=5, columns=5,
                         B=B, n_std=200, two_stage=False, fragility="never", repetitions=30,
                         median_ms=np.median(timing), mean_ms=np.mean(timing),
                         p90_ms=np.quantile(timing, .9)))
with (out / "runtime_python.csv").open("w") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0]))
    w.writeheader()
    w.writerows(rows)
cpuinfo = Path("/proc/cpuinfo").read_text()
model = next(line.split(":", 1)[1].strip() for line in cpuinfo.splitlines() if line.startswith("model name"))
meminfo = Path("/proc/meminfo").read_text()
mem_kb = int(next(line.split()[1] for line in meminfo.splitlines() if line.startswith("MemTotal:")))
info = dict(cpu_model=model, logical_cores=os.cpu_count(), ram_gib=mem_kb / 1024**2,
            benchmark_affinity=sorted(os.sched_getaffinity(0)), platform=platform.platform(),
            python=sys.version, executable=sys.executable, numpy=np.__version__,
            scipy=scipy.__version__, numba=numba.__version__, sirsyn=sirsyn.__version__,
            thread_limits={key: os.environ.get(key) for key in
                           ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMBA_NUM_THREADS")})
(out / "hardware.json").write_text(json.dumps(info, indent=2) + "\n")
