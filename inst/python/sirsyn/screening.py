"""Screen declared families with within-family BH adjustment.

Every input is fit using two-stage Monte Carlo. The relative empirical null
applies only to the companion mean test. Screen calls require independent
experimental confirmation."""
import os
from dataclasses import dataclass, field
from typing import Any, Dict, List

import numpy as np

from .core import fit
from .multiplicity import bh_qvalues, empirical_null
from .result import FitResult

MAX_WORKERS = 12


@dataclass
class ScreenResult:
    test: str
    q_level: float
    family: np.ndarray
    p: np.ndarray
    q: np.ndarray
    call: np.ndarray
    results: List[FitResult]
    relative_null: Dict[Any, Dict[str, float]] = field(default_factory=dict)
    notes: List[str] = field(default_factory=list)

    def summary(self):
        L = [f"screen: {len(self.p)} matrices, {np.unique(self.family).size} family(ies), BH on p_{self.test} at q = "
             f"{self.q_level}: {int(self.call.sum())} call(s)"]
        for f in np.unique(self.family):
            k = self.family == f
            L.append(f"  family {f}: {int(k.sum())} matrices, {int(self.call[k].sum())} call(s)")
        for n in self.notes:
            L.append(f"  note: {n}")
        return "\n".join(L)


def _fit_one(args):
    Y, kw = args
    for k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMBA_NUM_THREADS"):
        os.environ.setdefault(k, "1")
    return fit(Y, **kw)


def screen(items, family=None, q=0.05, test="tstd", relative_null=False, nan_policy="one", fit_kwargs=None,
           n_workers=1, seed=None):
    """items: list of FitResult, or of inputs to `fit` (an array, or a dict with key 'Y' and per-matrix fit
    arguments such as 'doses' or 'group'). family: labels (default: one family). test: 'tstd' (primary),
    'scan' (scan global p) or 'mean' (companion; relative_null=True uses the within-family empirical null).
    Raw inputs are fit with fit_kwargs (two_stage forced on) and per-matrix seeds spawned from `seed`."""
    n = len(items)
    fam = np.zeros(n, dtype=object) if family is None else np.asarray(family, dtype=object)
    if fam.shape != (n,):
        raise ValueError("family must have one label per item")
    notes = []
    todo = [k for k, it in enumerate(items) if not isinstance(it, FitResult)]
    results = list(items)
    if todo:
        base = dict(fit_kwargs or {})
        if base.get("two_stage") is False:
            notes.append("two_stage=False was overridden: screens need the second Monte Carlo stage")
        base["two_stage"] = True
        jobs = []
        for k in todo:
            it = items[k]
            kw = dict(base)
            if isinstance(it, dict):
                it = dict(it)
                Y = it.pop("Y")
                kw.update(it)
            else:
                Y = it
            kw.setdefault("seed", [0 if seed is None else int(seed), int(k)])
            jobs.append((Y, kw))
        nw = int(min(max(1, n_workers), MAX_WORKERS))
        if nw > 1 and len(jobs) > 1:
            import multiprocessing as mp
            with mp.get_context("fork").Pool(nw) as pool:
                out = pool.map(_fit_one, jobs, chunksize=max(1, len(jobs) // (4 * nw)))
        else:
            out = [_fit_one(j) for j in jobs]
        for k, r in zip(todo, out):
            results[k] = r
    if test == "tstd":
        p = np.array([r.p_tstd for r in results], float)
    elif test == "scan":
        p = np.array([r.p_scan for r in results], float)
    elif test == "mean":
        p = np.array([r.p_mean for r in results], float)
    else:
        raise ValueError("test must be 'tstd', 'scan' or 'mean'")
    if test != "mean" and any(not r.settings.get("two_stage", True) for r in results):
        notes.append("some results were fit without the second stage; BH thresholds may not be resolved")
    qv = np.full(n, np.nan)
    rel = {}
    for f in np.unique(fam):
        k = fam == f
        pf = p[k]
        if test == "mean" and relative_null:
            pf, mu0, s0 = empirical_null(pf)
            rel[f] = dict(center=mu0, scale=s0)
        qv[k] = bh_qvalues(pf, nan_policy)
        nf = int(k.sum())
        if test != "mean":
            pmin = min(1.0 / (r.settings["B2"] + 1.0) for r, kk in zip(results, k) if kk)
            if pmin > q / nf:
                notes.append(f"family {f}: smallest attainable p {pmin:.2g} exceeds the BH threshold q / n = {q / nf:.2g}")
    call = np.where(np.isfinite(qv), qv <= q, False)
    return ScreenResult(test=test, q_level=q, family=fam, p=p, q=qv, call=call, results=results,
                        relative_null=rel, notes=notes)
