# sirsyn

Synergy testing for drug-combination viability matrices. Bliss independence is
the reference: E(Y_ij) = E(Y_i0) E(Y_0j), with control normalized to one.
This is the Python implementation bundled with SIR 0.2.0.

```sh
pip install .
pip install '.[test]'   # optional pytest and cvxpy
```

```python
import numpy as np
import sirsyn

a = np.array([1, .9, .7, .5, .3])
Y = np.outer(a, a)
noise = sirsyn.ParametricNoise.additive(.08)
result = sirsyn.fit(Y, design="S", noise=noise, seed=1)
print(result.summary())
```

Y has shape (rows, columns) or (replicates, rows, columns). The first row and
column hold single-agent curves at increasing doses, starting with the zero-dose
control. NaN means missing. S measures single agents on one plate; R measures
single agents on each replicate; P uses external curves. P holds those curves
fixed by default (sa_replicates=inf); finite sa_replicates and sa_rho specify
edge uncertainty. S with replicate plates becomes R.

The primary test, based on the SIR statistic (`t_std` in the result), uses a
plug-in reference at exact Bliss, with B=399 then B2=9999 fresh draws if either
the primary or region-scan p-value is <=0.05.
The companion mean test uses a design-matched SE; for replicate stacks the
block-level test treats a shared single-agent reference as fixed, and on
incomplete single-plate S grids the bootstrap mean p-value is exploratory. The
least-favorable-null result covers replicate means and other replicate summaries
nondecreasing in each replicate; the replicate screen is evaluated by
simulation. Missing single-agent wells
remove their dose rows/columns; missing combination wells are excluded.
Replicate screening is mirrored in the reference. Results include activity,
mean_excess, mean_excess_se, p_mean, p_tstd, p_scan, regions, claims, excess_map,
raw_excess_map, fitted_surface, fragility, log, and settings.

Noise options: ParametricNoise.additive, .lognormal, .constant_cv, .sd_table,
.oneil; SDCurve; sd_curve_from_replicates; NoiseBank.from_replicates with input
shape (sets, replicates, rows, columns) and group labels; PlateVarianceTable.
Gaussian lognormal noise preserves the mean; lognormal t4 draws do not have a
finite mean. Banks rescale residuals by SD ratios and sample across group folds
within potency strata. Supply the target group to enable cross-fitting.

`sirsyn.screen(items, family=labels, fit_kwargs=dict(noise=noise), seed=1)` uses
BH within declared families. A relative empirical null is available only for the
mean test. Screen calls require independent confirmation. Region-scan calibration
was evaluated by simulation on complete grids with parametric noise in designs S
and P; it has not been evaluated with residual banks, missing wells, replicate
screening or propagated single-agent uncertainty. Plate shifts were evaluated in
design P, not in design S.

```sh
python3 -m pytest -q          # fast invariants; cvxpy check skipped if unavailable
python3 -m pytest -m slow -q  # self-contained Monte Carlo calibration checks
```

Set NUMBA_NUM_THREADS=1 for single-core timing; parallelize independent matrices.
Initial calls compile Numba kernels; subsequent calls use the cache.
