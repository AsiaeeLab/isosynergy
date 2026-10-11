"""Self-contained Monte Carlo regression checks against fixed reference rates.

Stage 1 uses B=399. Each comparison uses a two-sample z statistic (absolute
value below 3), accounting for simulation error in both reference and new runs.
Run explicitly with pytest -m slow.
"""

import numpy as np
import pytest

import _mc as M

pytestmark = pytest.mark.slow
ZMAX = 3.0


def _check(p, alpha, ref, n_ref, label):
    k, n = M.rate(p, alpha)
    z = M.compare(k, n, ref, n_ref)
    lo, hi = M.wilson(k, n)
    print(f"{label}: {k / n:.4f} [{lo:.4f}, {hi:.4f}] (n = {n}) vs {ref} (n = {n_ref}), z = {z:+.2f}")
    assert abs(z) < ZMAX, f"{label}: {k / n:.4f} vs {ref}, z = {z:.2f}"


def test_tstd_size_design_S():
    p = M.run(M._job_global, ("S", 5, "strong", "B"), 20000, 500, "global-S")
    _check(p[:, 0], 0.05, 0.0448, 20000, "global T_std design S 5x5 lognormal strong")


def test_tstd_size_design_P():
    p = M.run(M._job_global, ("P", 5, "strong", "A"), 20000, 500, "global-P")
    _check(p[:, 0], 0.05, 0.0505, 20000, "global T_std design P 5x5 additive strong")
    _check(p[:, 0], 0.01, 0.0090, 20000, "global T_std design P 5x5 additive strong, 0.01")


@pytest.mark.parametrize("law,ref,key", [("A", 0.036, "scan-A"), ("C", 0.034, "scan-C")])
def test_scan_weak_5x5(law, ref, key):
    p = M.run(M._job_global, ("S", 5, "weak", law), 12000, 500, key)
    _check(p[:, 1], 0.05, ref, 4000, f"scan scan (ii) design S 5x5 weak law {law}")


def test_missing_wells_and_rule_R():
    p = M.run(M._job_missing, (4, "moderate", "B"), 20000, 250, "handling-miss")
    _check(p[:, 0], 0.05, 0.0413, 20000, "handling complete grid 4x4 lognormal moderate")
    _check(p[:, 1], 0.05, 0.0420, 20000, "handling 3 missing combination wells")
    _check(p[:, 2], 0.05, 0.0445, 20000, "handling rule R")
    for c in (1, 2):
        d = (p[:, c] <= 0.05).astype(float) - (p[:, 0] <= 0.05)
        assert abs(d.mean()) < 3 * d.std() / np.sqrt(d.size) + 0.002


def test_replicate_screen_clean_data_invariance():
    p = M.run(M._job_screen, (5, "weak", "B"), 20000, 250, "handling-screen")
    _check(p[:, 0], 0.05, 0.0465, 20000, "handling replicated, screen, clean")
    _check(p[:, 1], 0.05, 0.0466, 20000, "handling replicated, no screen, clean")
    d = (p[:, 0] <= 0.05).astype(float) - (p[:, 1] <= 0.05)
    print(f"paired screen - no screen: {d.mean():+.5f} (SE {d.std() / np.sqrt(d.size):.5f}); "
          f"data flags per dataset {p[:, 2].mean():.4f}")
    assert abs(d.mean()) < 3 * d.std() / np.sqrt(d.size) + 0.001

