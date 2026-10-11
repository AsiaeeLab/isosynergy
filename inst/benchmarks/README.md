# Single-core runtime

`runtime_r.csv` compares the complete `sir_fit()` R call (including conversion to
ordinary R values) with pure `sirsyn.fit()`. Both use the same deterministic 5 by
5 matrix, additive Gaussian SD 0.04, seeds 1 through 30, and 200 independent scan
standardization plates. Each configuration is warmed once before timing.

The requested B values are measured as fixed budgets: `two_stage = FALSE` and
`fragility = "never"`. Thus B = 9,999 means 9,999 reference draws, not 399 plus
9,999. These timings include the primary test, companion test and region scan.
Default adaptive second stages and fragility can increase runtime; first-call
Python initialization and Numba compilation are excluded. R elapsed timing has
millisecond resolution. CPU affinity was restricted to logical CPU 0; all BLAS,
OpenMP and Numba thread counts were one. Runs were sequential, without the
calibration suite competing for CPU. Hardware and dependency versions are in
`hardware.json` and `r_session.txt`.

To reproduce after installing SIR, from this directory:

```sh
export RETICULATE_PYTHON=/usr/bin/python3
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMBA_NUM_THREADS=1
taskset -c 0 Rscript runtime.R /tmp/sir-timing
taskset -c 0 python3 runtime_python.py /tmp/sir-timing
```

Combine the resulting `runtime_r_only.csv` and `runtime_python.csv` rows to obtain
`runtime_r.csv`.

`runtime_grid.csv` times the complete `sirsyn.fit()` call by grid size (4 by 4 to 10
by 10, design S, the same noise and fixed B, seeds 1 through 20, warmed once), and
`runtime_first_call.txt` records the first call in a fresh process with an empty
Numba cache, which includes compilation:

```sh
taskset -c 0 python3 runtime_grid.py /tmp/sir-timing
```

Other platforms can omit `taskset` while retaining thread limits.
