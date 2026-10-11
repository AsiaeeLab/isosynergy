"""Fast unit tests: invariants of the fits, data-handling rules, reproducibility, multiplicity fixes."""
import os

for _k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMBA_NUM_THREADS"):
    os.environ[_k] = "1"

import numpy as np  # noqa: E402
import pytest  # noqa: E402

import sirsyn as S  # noqa: E402
from sirsyn import _kernels as K  # noqa: E402
from sirsyn import simulate as sim  # noqa: E402
from sirsyn.core import _Grid  # noqa: E402
from sirsyn.geometry import order_pairs, adjacent_pairs, regions  # noqa: E402
from sirsyn.meantests import fixed_edge_t, block_t  # noqa: E402
from sirsyn.replicates import screen  # noqa: E402

NZ = S.ParametricNoise.oneil()


def _plate(rng, cond="null", surf="moderate", I=5, design="S", nz=NZ):
    return sim.simulate(sim.truth(cond, surf, I), nz, design, 1, rng)[0]


# ---------------------------------------------------------------------------
# isotonic fits
# ---------------------------------------------------------------------------
def _iso_dec_brute(y):
    """Nonincreasing isotonic regression by the min-max formula f_i = min_{j<=i} max_{k>=i} mean(y_j..y_k)."""
    n = y.size
    f = np.empty(n)
    for i in range(n):
        f[i] = min(max(y[j:k + 1].mean() for k in range(i, n)) for j in range(i + 1))
    return f


def test_pava_matches_minmax_formula_and_is_nonincreasing():
    rng = np.random.default_rng(1)
    for _ in range(200):
        y = rng.normal(size=rng.integers(1, 10))
        f = K.pava_dec(y.copy())
        assert np.all(np.diff(f) <= 1e-12)
        assert np.allclose(f, _iso_dec_brute(y), atol=1e-12)


def test_2d_fit_order_preserving_and_idempotent():
    rng = np.random.default_rng(2)
    for I, J in ((4, 4), (5, 5), (4, 6), (8, 8)):
        pa, pb = adjacent_pairs(I, J)
        for _ in range(20):
            z = rng.normal(size=I * J)
            th = K.dykstra(z, np.ones(I * J), pa, pb, 5000, 1e-12)
            T = th.reshape(I, J)
            assert np.all(np.diff(T, axis=0) <= 1e-9) and np.all(np.diff(T, axis=1) <= 1e-9)
            th2 = K.dykstra(th, np.ones(I * J), pa, pb, 5000, 1e-12)
            assert np.allclose(th, th2, atol=1e-10)


def test_restricted_order_fit_matches_cvxpy():
    cp = pytest.importorskip("cvxpy")
    rng = np.random.default_rng(3)
    for t in range(15):
        I = J = 5
        obs = np.ones((I, J), bool)
        for c in sim.mcar_cells(rng, I, J, int(rng.integers(1, 4))):
            obs[c] = False
        y = rng.normal(size=I * J)
        pa, pb = order_pairs(I, J, obs.ravel())
        th = K.dykstra(y, np.ones(I * J), pa, pb, 5000, 1e-12)
        idx = np.where(obs.ravel())[0]
        x = cp.Variable(idx.size)
        pos = {k: q for q, k in enumerate(idx)}
        cons = []
        for u in idx:
            for v in idx:
                ui, uj = divmod(u, J)
                vi, vj = divmod(v, J)
                if u != v and vi >= ui and vj >= uj:
                    cons.append(x[pos[u]] >= x[pos[v]])
        cp.Problem(cp.Minimize(cp.sum_squares(x - y[idx])), cons).solve(
            solver="OSQP", eps_abs=1e-10, eps_rel=1e-10, max_iter=200000)
        assert np.max(np.abs(th[idx] - x.value)) < 1e-7
        # product order among observed wells is preserved
        for u in idx:
            for v in idx:
                ui, uj = divmod(u, J)
                vi, vj = divmod(v, J)
                if vi >= ui and vj >= uj:
                    assert th[u] >= th[v] - 1e-9


def test_order_pairs_complete_grid_and_region_family_sizes():
    for I, J in ((4, 4), (5, 5), (4, 6)):
        pa, pb = order_pairs(I, J, np.ones(I * J, bool))
        qa, qb = adjacent_pairs(I, J)
        assert np.array_equal(pa, qa) and np.array_equal(pb, qb)
    assert regions(5, 5).nR == 24 and regions(4, 4).nR == 12
    assert regions(8, 8).nR == 84 and regions(10, 10).nR == 144


# ---------------------------------------------------------------------------
# reproducibility and stages
# ---------------------------------------------------------------------------
def test_identical_seeds_identical_results():
    rng = np.random.default_rng(4)
    Y = _plate(rng, "syn_broad_0.20")
    a = S.fit(Y, noise=NZ, seed=11)
    b = S.fit(Y, noise=NZ, seed=11)
    c = S.fit(Y, noise=NZ, seed=12)
    for f in ("p_tstd", "p_tstd_stage1", "p_scan", "p_mean", "p_mean_boot", "t_std", "stage"):
        assert getattr(a, f) == getattr(b, f)
    assert [r["p_adj"] for r in a.regions] == [r["p_adj"] for r in b.regions]
    assert a.fragility == b.fragility
    assert (a.p_tstd_stage1, a.p_scan_stage1) != (c.p_tstd_stage1, c.p_scan_stage1)


def test_two_stage_trigger():
    rng = np.random.default_rng(5)
    strong = S.fit(_plate(rng, "syn_broad_0.60"), noise=NZ, seed=1)
    assert strong.stage == 2 and strong.B_used == 9999 and strong.p_tstd < 0.01
    null = S.fit(_plate(rng, "null"), noise=NZ, seed=2, stage2_threshold=1e-9)
    assert null.stage == 1 and null.B_used == 399


def test_tstd_zero_for_antagonism_free_of_excess():
    """T_std is a positive-part statistic: an exactly Bliss (noise-free) plate has T_std = 0 and p = 1."""
    mu = sim.truth("null", "moderate", 5)
    r = S.fit(mu, noise=NZ, seed=1)
    assert r.t_std == 0.0 and r.p_tstd == 1.0


# ---------------------------------------------------------------------------
# data handling
# ---------------------------------------------------------------------------
def test_rule_R_equals_reduced_grid():
    rng = np.random.default_rng(6)
    Y = _plate(rng)
    d = sim.dose_grid(4)
    Ym = Y.copy()
    Ym[2, 0] = np.nan  # drug A, dose index 2 missing
    r = S.fit(Ym, (d, d), noise=NZ, seed=7, fragility="never")
    keep = [0, 1, 3, 4]
    r2 = S.fit(Y[keep], (d[keep], d), noise=NZ, seed=7, fragility="never")
    assert r.log.dropped_doses == [dict(drug="A", index=2, dose=float(d[2]))]
    assert r.log.analyzed_shape == (4, 5) and r.log.rows_kept == keep
    assert r.p_tstd == r2.p_tstd and r.t_std == r2.t_std and r.p_scan == r2.p_scan and r.p_mean == r2.p_mean
    assert np.array_equal(r.doses_a, d[keep])


def test_missing_combination_wells_are_missing():
    rng = np.random.default_rng(8)
    Y = _plate(rng)
    Ym = Y.copy()
    Ym[2, 3] = np.nan
    Ym[4, 4] = np.nan
    r = S.fit(Ym, noise=NZ, seed=9, fragility="never")
    assert sorted(r.log.missing_cells) == [(2, 3), (4, 4)]
    assert np.isnan(r.fitted_surface[2, 3]) and np.isnan(r.excess_map[1, 2])
    # the statistic only sees observed wells: any value at the missing wells gives the same result
    obs = np.isfinite(Ym)
    G = _Grid(obs, NZ.sd)
    t1 = G.stats(np.where(obs, Y, 0.3)[None])[0]
    t2 = G.stats(np.where(obs, Y, 5.0)[None])[0]
    assert np.array_equal(t1, t2) and t1[0] == r.t_std
    # V2Sp needs a complete grid: the companion falls back to the bootstrap mean excess
    assert r.mean_test["name"].startswith("bootstrap")


def test_control_check():
    rng = np.random.default_rng(10)
    Y = _plate(rng)
    Y[0, 0] = 1.2
    with pytest.raises(ValueError):
        S.fit(Y, noise=NZ)
    r = S.fit(Y, noise=NZ, normalize="divide", seed=1, two_stage=False)
    assert any("divided by its control" in n for n in r.log.notes)


def test_screen_flags_gross_error_and_ignores_common_shift():
    rng = np.random.default_rng(12)
    sd = S.SDCurve("constant", 0.08)
    R = 0.6 + 0.08 * rng.standard_normal((2000, 4, 5, 5))
    R[:, :, 0, 0] = 1.0
    m0, d0 = screen(R, 5.0, sd)
    # a common shift of all replicates of a well (what synergy does) leaves the flags exactly unchanged
    shift = np.zeros((5, 5))
    shift[1:, 1:] = rng.uniform(-0.3, 0.3, (4, 4))
    m1, d1 = screen(R + shift[None, None], 5.0, sd)
    assert np.array_equal(d0, d1)
    assert np.allclose(m1 - m0, shift[None], atol=1e-12)
    # a gross error is dropped
    Rg = R.copy()
    Rg[0, 2, 3, 3] += 0.8
    mg, dg = screen(Rg, 5.0, sd)
    assert dg[0, 3, 3] == 2 and abs(mg[0, 3, 3] - np.delete(Rg[0, :, 3, 3], 2).mean()) < 1e-12


def test_screen_in_fit_logs_and_is_applied_in_reference():
    rng = np.random.default_rng(13)
    nz = S.ParametricNoise.lognormal(0.12)
    Rr = sim.simulate(sim.truth("null", "weak", 5), nz, "S", 1, rng, m=4)[0]
    Rr[1, 2, 2] += 0.9
    r = S.fit(Rr, design="R", noise=nz, seed=3, two_stage=False)
    assert r.log.screen_applied and any(d["i"] == 2 and d["j"] == 2 and d["replicate"] == 1 for d in r.log.screened)
    assert r.log.reference_flags_per_plate > 0  # lognormal: the reference plates flag at the model rate
    assert r.mean_test["name"].startswith("block-level")
    r3 = S.fit(Rr[:2], design="R", noise=nz, seed=3, two_stage=False)
    assert not r3.log.screen_applied and any("at least 3 replicates" in n for n in r3.log.notes)


def test_partial_replicates_pattern():
    rng = np.random.default_rng(14)
    Rr = sim.simulate(sim.truth("null", "moderate", 5), NZ, "S", 1, rng, m=4)[0]
    Rr[0, 3, 3] = np.nan
    Rr[:, 2, 2] = np.nan
    r = S.fit(Rr, design="R", noise=NZ, seed=1, two_stage=False)
    assert r.log.partial_replicates == 1 and r.log.missing_cells == [(2, 2)]


# ---------------------------------------------------------------------------
# companion tests and multiplicity fixes
# ---------------------------------------------------------------------------
def test_se_zero_gives_nan_not_zero():
    Y = np.ones((5, 5))
    out = fixed_edge_t(Y, np.ones((4, 4), bool))
    assert out["degenerate"] and np.isnan(out["p"])
    R = np.ones((3, 5, 5))
    out = block_t(R, np.ones((4, 4), bool), "S")
    assert np.isnan(out["p"])


def test_bh_with_nan():
    p = np.array([0.001, np.nan, 0.02, 0.04, 0.5])
    q = S.bh_qvalues(p)  # NaN counted as p = 1
    assert np.isnan(q[1]) and np.all(np.isfinite(q[[0, 2, 3, 4]]))
    assert np.allclose(q[[0, 2, 3, 4]], [0.005, 0.05, 0.04 * 5 / 3, 0.625])
    q2 = S.bh_qvalues(p, "omit")
    assert np.allclose(q2[[0, 2, 3, 4]], [0.004, 0.04, 0.04 * 4 / 3, 0.5])
    assert list(S.bh_reject(p, 0.05)) == [True, False, True, False, False]


def test_empirical_null_no_saturation():
    p = np.concatenate([np.full(60, 1e-30), np.linspace(0.01, 0.99, 40)])  # more than half beyond 1e-16
    out, mu0, s0 = S.empirical_null(p)
    assert np.all(np.isfinite(out)) and s0 > 0
    p2 = np.full(10, 1e-40)
    out2, _, s2 = S.empirical_null(p2)
    assert s2 == 1.0 and np.all(np.isfinite(out2))


# ---------------------------------------------------------------------------
# designs, fragility, noise bank, screens
# ---------------------------------------------------------------------------
def test_design_P_reference_keeps_edges_and_redraws_when_reported():
    rng = np.random.default_rng(15)
    Y = _plate(rng, design="P")
    r = S.fit(Y, design="P", noise=NZ, seed=1, two_stage=False, plate_var=0.0004)
    assert r.mean_test["name"].startswith("fixed-edge") and r.mean_test["plate_var"] == 0.0004
    r2 = S.fit(Y, design="P", noise=NZ, seed=1, two_stage=False, sa_replicates=4)
    assert r2.mean_test["edge_var"] > 0
    assert r2.p_tstd_stage1 != r.p_tstd_stage1  # the reference redraws the reported curves


def test_fragility_single_cell_call():
    rng = np.random.default_rng(16)
    mu = sim.surface("weak", 5)
    mu = mu.copy()
    mu[4, 4] *= np.exp(-3.0)  # one strongly synergistic top-dose cell
    Y = sim.simulate(mu, NZ, "P", 1, rng)[0]
    r = S.fit(Y, design="P", noise=NZ, seed=2)
    assert r.p_tstd <= 0.05 and r.fragility["computed"]
    assert r.fragility["loco_cell"] == (4, 4) and r.fragility["loco_flag"]
    assert any(c["claimed"] and (4, 4) in c["cells"] for c in r.regions)
    r_no = S.fit(_plate(rng), noise=NZ, seed=3, stage2_threshold=0)
    if r_no.p_tstd > 0.05:
        assert not r_no.fragility["computed"]


def test_noise_bank_from_replicates_and_cross_fitting():
    rng = np.random.default_rng(17)
    groups = np.repeat([f"cl{k}" for k in range(8)], 40)
    mus = [sim.truth("null", s, 5) for s in ("weak", "moderate", "strong")]
    Yb = np.stack([sim.simulate(mus[k % 3], NZ, "S", 1, rng, m=4)[0] for k in range(groups.size)])
    bank = S.NoiseBank.from_replicates(Yb, groups, n_strata=3, include_edges=True)
    mu0 = mus[1]
    pool, st = bank.pool(mu0, group="cl0")
    f0 = bank._fold_of("cl0")
    assert pool.size > 0 and np.all(bank.fold[pool] != f0) and np.all(bank.stratum_of_set[pool] == st)
    Y = _plate(rng)
    r = S.fit(Y, noise=bank, group="cl0", seed=1, two_stage=False, min_bank_pool=5)
    assert 0 < r.p_tstd <= 1 and any("noise bank" in n for n in r.log.notes)
    with pytest.raises(ValueError):
        S.fit(Y, noise=S.NoiseBank.from_replicates(Yb, groups, n_strata=3), group="cl0", min_bank_pool=5)


def test_screen_family_bh():
    rng = np.random.default_rng(18)
    items = [_plate(rng, "syn_broad_0.50") for _ in range(3)] + [_plate(rng) for _ in range(5)]
    sr = S.screen(items, family=["a"] * 4 + ["b"] * 4, fit_kwargs=dict(noise=NZ, fragility="never"), seed=5)
    assert sr.call[:3].all() and len(sr.results) == 8
    again = S.screen(sr.results, family=["a"] * 4 + ["b"] * 4)
    assert np.array_equal(again.q, sr.q)
    rel = S.screen(sr.results, test="mean", relative_null=True)
    assert np.all(np.isfinite(rel.q))

