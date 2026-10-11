"""Analyze a drug-combination matrix with the plug-in exact-Bliss reference."""
import numpy as np

from . import _kernels as K
from .geometry import order_pairs, regions as make_regions
from .meantests import v2sp, block_t, fixed_edge_t, edge_var_mean
from .noise import NoiseBank, ParametricNoise, PlateVarianceTable, seed_for
from .reference import ReferenceSpec, ref_mean, tail_p, scan_pvalues
from .replicates import screen as rep_screen
from .result import DataLog, FitResult

DESIGNS = ("S", "P", "R")


def _sd_args(sd):
    sdk, sdp, skv, sks = sd.kernel_args()
    return int(sdk), float(sdp), np.ascontiguousarray(skv, float), np.ascontiguousarray(sks, float)


class _Grid:
    """Analyzed grid: observed mask, covering pairs, region family, SD arguments."""

    def __init__(self, obs, sd):
        I, J = obs.shape
        self.I, self.J = I, J
        self.obs = np.ascontiguousarray(obs.ravel())
        self.pa, self.pb = order_pairs(I, J, self.obs)
        self.reg = make_regions(I, J, self.obs)
        self.sda = _sd_args(sd)

    def stats(self, Ys):
        return K.stats_many(np.ascontiguousarray(Ys, float), self.obs, self.pa, self.pb, *self.sda,
                            self.reg.cells, self.reg.cptr)

    def tstd(self, Ys):
        return K.tstd_many(np.ascontiguousarray(Ys, float), self.obs, self.pa, self.pb, *self.sda)


def _resolve_plate_var(plate_var, noise, mu0, Y, obs_int, group, log):
    if plate_var is None:
        if isinstance(noise, ParametricNoise):
            v = noise.shift_sd(mu0) ** 2
            if v > 0:
                log.note(f"plate variance component from the noise model's plate shift: {v:.4g}")
            return v
        return 0.0
    if callable(plate_var):
        mint = float(np.clip(Y[1:, 1:][obs_int].mean(), 0.0, 1.5))
        if isinstance(plate_var, PlateVarianceTable):
            return float(plate_var(mint, group))
        return float(plate_var(mint))
    return float(plate_var)


def fit(Y, doses=None, design="S", noise=None, *, alpha=0.05, B=399, B2=9999, n_std=200, two_stage=True,
        stage2_threshold=0.05, sa_replicates=np.inf, sa_rho=0.10, screen_replicates=True, screen_c=5.0,
        plate_var=None, group=None, fragility="auto", sd=None, seed=None, normalize="check",
        min_bank_pool=50):
    """Analyze one drug-combination matrix using the exact-Bliss reference.

    Y        viability, control normalized to 1: (I, J) single plate or (m, I, J) replicate plates; row and
             column 0 hold the single agents (index 0 = zero dose, doses increasing); NaN marks a missing well.
    doses    optional (doses_a (I,), doses_b (J,)) for labeling.
    design   'S' single agents measured on the plate(s); 'R' replicated design S (m >= 2 plates, each with its
             own single-agent wells); 'P' single agents external (reported curves in row/column 0, exact when
             sa_replicates = inf, otherwise the mean of sa_replicates replicate curves with shift share sa_rho).
    noise    ParametricNoise or NoiseBank (the reference law; its SD curve standardizes T_std unless `sd`).
    B, B2    reference plates of stage 1 and of the second stage, run when min(p_tstd, p_scan) of stage 1 is at
             most stage2_threshold (B2 = 9,999 resolves BH thresholds in screens).
    n_std    separate plates that standardize the region scan.
    screen_replicates, screen_c   replicate screen (m >= 3) at screen_c model SDs, identical in the reference.
    plate_var  plate (block-shift) variance component of the mean combination viability for the design-P mean
             test: number, callable(mean viability), or PlateVarianceTable (cross-fitted with `group`).
    group    the matrix's group (for example cell line) for cross-fitting the noise bank / plate variance table.
    fragility 'auto' (single plates with a call), 'always' (any call), or 'never'.
    seed     int or None; identical seeds give identical results.
    normalize 'check' (error if a finite control differs from 1 by more than 1e-6), 'divide' (divide each plate
             by its control), or 'ignore' (set the control to 1).
    """
    if design not in DESIGNS:
        raise ValueError(f"design must be one of {DESIGNS}")
    if noise is None:
        raise ValueError("a noise model is required (ParametricNoise or NoiseBank)")
    log = DataLog()
    Y = np.array(Y, dtype=float, copy=True)
    if Y.ndim == 2:
        R = Y[None]
    elif Y.ndim == 3:
        R = Y
    else:
        raise ValueError("Y must be (I, J) or (m, I, J)")
    m, I0, J0 = R.shape
    if I0 < 3 or J0 < 3:
        raise ValueError("need at least two nonzero doses of each drug")
    if design == "R" and m < 2:
        raise ValueError("design 'R' needs replicate plates (m >= 2)")
    if design == "S" and m >= 2:
        design = "R"
        log.note("design 'S' with replicate plates analyzed as replicated design S ('R')")
    dkey = "P" if design == "P" else "S"
    log.design, log.n_replicates, log.input_shape = design, m, (I0, J0)
    if isinstance(noise, NoiseBank):
        if noise.shape != (I0, J0):
            raise ValueError(f"noise bank grid {noise.shape} does not match the matrix grid {(I0, J0)}")
        if dkey == "S" and not noise.has_edges:
            raise ValueError("design S needs a noise bank with single-agent noise (include_edges=True)")
    sd_std = sd if sd is not None else noise.sd

    # ---- control --------------------------------------------------------------------------------
    for k in range(m):
        c = R[k, 0, 0]
        if not np.isfinite(c) or abs(c - 1.0) <= 1e-6:
            R[k, 0, 0] = 1.0
            continue
        if normalize == "check":
            raise ValueError(f"control of plate {k} is {c:.6g}, not 1; normalize the plate or pass normalize='divide'")
        if normalize == "divide":
            R[k] = R[k] / c
            log.note(f"plate {k} divided by its control {c:.6g}")
        R[k, 0, 0] = 1.0

    if dkey == "P" and m > 1:
        E = np.concatenate([R[:, 1:, 0], R[:, 0, 1:]], 1)
        with np.errstate(all="ignore"):
            spread = np.nanmax(E, 0) - np.nanmin(E, 0)
        if np.any(spread[np.isfinite(spread)] > 1e-9):
            log.note("external single-agent values differ across replicate plates; their mean is used")

    # ---- rule R: a missing single-agent dose drops its row or column -----------------------------
    avail = np.isfinite(R)
    avail[:, 0, 0] = True
    keepA = [0] + [i for i in range(1, I0) if avail[:, i, 0].any()]
    keepB = [0] + [j for j in range(1, J0) if avail[:, 0, j].any()]
    da = None if doses is None else np.asarray(doses[0], float)
    db = None if doses is None else np.asarray(doses[1], float)
    for i in range(1, I0):
        if i not in keepA:
            log.dropped_doses.append(dict(drug="A", index=i, dose=None if da is None else float(da[i])))
    for j in range(1, J0):
        if j not in keepB:
            log.dropped_doses.append(dict(drug="B", index=j, dose=None if db is None else float(db[j])))
    if len(keepA) < 3 or len(keepB) < 3:
        raise ValueError("fewer than two observed single-agent doses remain for a drug after rule R")
    if log.dropped_doses:
        log.note(f"rule R: {len(log.dropped_doses)} missing single-agent dose(s) dropped with their row/column; "
                 f"analyzed as {len(keepA)} x {len(keepB)}")
        if len(log.dropped_doses) > 1:
            log.note("more than one missing single-agent dose: rule R was validated for one")
    rows0, cols0 = np.array(keepA), np.array(keepB)
    R = R[:, rows0[:, None], cols0[None, :]]
    avail = avail[:, rows0[:, None], cols0[None, :]]
    I, J = R.shape[1:]
    log.rows_kept, log.cols_kept, log.analyzed_shape = rows0.tolist(), cols0.tolist(), (I, J)

    # ---- replicates: screen and well means --------------------------------------------------------
    cnt = avail.sum(0)
    log.partial_replicates = int(((cnt > 0) & (cnt < m)).sum()) if m > 1 else 0
    do_screen = bool(screen_replicates) and m >= 3
    dropped = None
    if m == 1:
        Yd = R[0].copy()
    elif do_screen:
        mean, dropped = rep_screen(R[None], screen_c, noise.sd, skip_edges=(dkey == "P"), avail=avail[None])
        Yd, dropped = mean[0], dropped[0]
        log.screen_applied = True
        for i, j in zip(*np.where(dropped >= 0)):
            k = int(dropped[i, j])
            others = [R[q, i, j] for q in range(m) if q != k and avail[q, i, j]]
            med = float(np.median(others))
            log.screened.append(dict(i=int(rows0[i]), j=int(cols0[j]), replicate=k, value=float(R[k, i, j]),
                                     median_others=med, model_sds=float(abs(R[k, i, j] - med) / noise.sd(med))))
    else:
        with np.errstate(all="ignore"):
            Yd = np.nanmean(np.where(avail, R, np.nan), 0)
        if m == 2 and screen_replicates:
            log.note("replicate screen not applied: it needs at least 3 replicates")
    Yd[0, 0] = 1.0
    if m > 1 and log.partial_replicates:
        log.note(f"{log.partial_replicates} well(s) with some replicates missing; the reference uses the same pattern")

    obs = np.isfinite(Yd)
    obs[0, 0] = True
    obs_int = obs[1:, 1:]
    log.missing_cells = [(int(rows0[i + 1]), int(cols0[j + 1])) for i, j in zip(*np.where(~obs_int))]
    if obs_int.sum() < 2:
        raise ValueError("fewer than two observed combination wells")
    if len(log.missing_cells) > 3:
        log.note(f"{len(log.missing_cells)} missing combination wells: validated for 1 to 3 MCAR wells")
    G = _Grid(obs, sd_std)

    # ---- observed statistics ----------------------------------------------------------------------
    T0 = G.stats(Yd[None])[0]
    t_obs, mean_obs, reg_obs = T0[0], T0[1], T0[2:]
    a_hat, b_hat, th = K.iso_fit(np.ascontiguousarray(Yd), G.obs, G.pa, G.pb)
    mu0 = ref_mean(a_hat, b_hat)

    pool = None
    if isinstance(noise, NoiseBank):
        pool, st = noise.pool(mu0, group, min_size=max(min_bank_pool, m))
        log.note(f"noise bank: potency stratum {st}, {pool.size} eligible sets"
                 + ("" if group is None else f", cross-fitted against group {group!r}"))

    spec = ReferenceSpec(dkey, m, noise, (I, J), avail=avail if m > 1 else None, screen=do_screen,
                         screen_c=screen_c, sa_reps=sa_replicates if dkey == "P" else np.inf, sa_rho=sa_rho,
                         pool=pool)
    if isinstance(noise, NoiseBank):
        spec_noise = _BankView(noise, rows0, cols0)
        spec.noise = spec_noise

    ss = _seed_seq(seed)
    s1, s2 = ss.spawn(2)

    def stage(sseed, nB):
        Ys, nflag = spec.plates(sseed, nB + n_std, Yd[1:, 0], Yd[0, 1:])
        T = G.stats(Ys)
        p_t = tail_p(t_obs, T[:nB, 0])
        p_m = tail_p(mean_obs, T[:nB, 1])
        p_s, z, padj = scan_pvalues(reg_obs, T[:nB, 2:], T[nB:, 2:])
        return dict(p_t=p_t, p_m=p_m, p_s=p_s, z=z, padj=padj, nflag=float(nflag[:nB].mean()), T=T, Ys=Ys,
                    seed=sseed, B=nB)

    st1 = stage(s1, B)
    fin = st1
    if two_stage and min(st1["p_t"], st1["p_s"]) <= stage2_threshold:
        fin = stage(s2, B2)
    log.reference_flags_per_plate = fin["nflag"]

    # ---- companion mean test ----------------------------------------------------------------------
    tests = {}
    if m == 1 and dkey == "S":
        if obs_int.all():
            tests["V2Sp"] = v2sp(Yd)
            main = "V2Sp"
        else:
            tests["bootstrap mean excess"] = dict(name="bootstrap-calibrated raw mean excess (incomplete grid)",
                                                  est=float(mean_obs), se=np.nan, df=np.nan, t=np.nan,
                                                  p=fin["p_m"], degenerate=False)
            main = "bootstrap mean excess"
            log.note("V2Sp needs a complete grid; companion p-value from the bootstrap-calibrated mean excess")
    else:
        if m >= 2:
            Rb = np.where(avail, R, np.nan)
            tests["block t"] = block_t(Rb, obs_int, dkey, dropped)
            main = "block t"
        if dkey == "P":
            pv = _resolve_plate_var(plate_var, noise, mu0, Yd, obs_int, group, log)
            ev = 0.0
            if np.isfinite(sa_replicates):
                V1 = edge_var_mean(a_hat, noise.sd, sa_replicates, sa_rho)
                V2 = edge_var_mean(b_hat, noise.sd, sa_replicates, sa_rho)
                ev = Yd[0, 1:].mean() ** 2 * V1 + Yd[1:, 0].mean() ** 2 * V2 + V1 * V2
            tests["fixed-edge t"] = fixed_edge_t(Yd, obs_int, pv, m, ev)
            tests["fixed-edge t"]["plate_var"] = pv
            tests["fixed-edge t"]["edge_var"] = ev
            if m == 1:
                main = "fixed-edge t"
                if pv == 0:
                    log.note("design P mean test without a plate variance component (plate_var not supplied)")
        if main == "block t" and tests["block t"].get("degenerate"):
            log.note("block-level t degenerate (fewer than 2 usable replicate plates or zero spread)")
    mt = tests[main]

    # ---- fragility --------------------------------------------------------------------------------
    frag = dict(computed=False)
    want = fragility == "always" or (fragility == "auto" and m == 1)
    if want and fin["p_t"] <= alpha:
        frag = _fragility(Yd, G, sd_std, spec, fin, t_obs, alpha, rows0, cols0)

    # ---- descriptive ------------------------------------------------------------------------------
    TH = th.reshape(I, J).copy()
    TH[~obs] = np.nan
    exc = np.outer(a_hat, b_hat) - TH[1:, 1:]
    raw = np.outer(Yd[1:, 0], Yd[0, 1:]) - Yd[1:, 1:]
    act = float(np.mean(1.0 - Yd[1:, 1:][obs_int]))
    nc = obs.copy()
    nc[0, 0] = False
    act_all = float(np.mean(1.0 - Yd[nc]))

    reg = G.reg
    regs = []
    for q in range(reg.nR):
        cells = [(int(rows0[i]), int(cols0[j])) for i, j in reg.sets[q]]
        regs.append(dict(label=";".join(f"{i},{j}" for i, j in cells), cells=cells, type=reg.types[q],
                         n_observed=int(reg.cptr[q + 1] - reg.cptr[q]), sum=float(reg_obs[q]), z=float(fin["z"][q]),
                         p_adj=float(fin["padj"][q]), claimed=bool(fin["padj"][q] <= alpha)))

    settings = dict(B=B, B2=B2, n_std=n_std, two_stage=two_stage, stage2_threshold=stage2_threshold,
                    sa_replicates=float(sa_replicates), sa_rho=sa_rho, screen_c=screen_c,
                    screen_replicates=screen_replicates, noise=noise.describe(), sd=sd_std.describe(),
                    seed_entropy=ss.entropy, group=group)
    return FitResult(
        design=design, alpha=alpha, activity=act, activity_all_wells=act_all, a_hat=a_hat, b_hat=b_hat,
        fitted_surface=TH, excess_map=exc, raw_excess_map=raw, analyzed_matrix=Yd,
        doses_a=None if da is None else da[rows0], doses_b=None if db is None else db[cols0],
        mean_excess=float(mt["est"]), mean_excess_se=float(mt["se"]), p_mean=float(mt["p"]), mean_test=mt,
        mean_tests=tests, p_mean_boot=float(fin["p_m"]), t_std=float(t_obs), p_tstd=float(fin["p_t"]),
        p_tstd_stage1=float(st1["p_t"]), p_scan=float(fin["p_s"]), p_scan_stage1=float(st1["p_s"]),
        regions=regs, stage=1 if fin is st1 else 2, B_used=int(fin["B"]), fragility=frag, log=log,
        settings=settings)


def _seed_seq(seed):
    """int, sequence of ints, SeedSequence, None, or any keys (strings) hashed with seed_for."""
    if isinstance(seed, np.random.SeedSequence):
        return seed
    if seed is None or isinstance(seed, (int, np.integer)):
        return np.random.SeedSequence(seed)
    if isinstance(seed, (list, tuple)) and all(isinstance(x, (int, np.integer)) and x >= 0 for x in seed):
        return np.random.SeedSequence([int(x) for x in seed])
    keys = seed if isinstance(seed, (list, tuple)) else (seed,)
    return np.random.SeedSequence(seed_for(*keys))


class _BankView:
    """A noise bank seen through the rule-R row/column selection of the analyzed grid."""
    kind = "bank"

    def __init__(self, bank, rows0, cols0):
        self.bank, self.rows0, self.cols0 = bank, rows0, cols0
        self.sd = bank.sd

    def draw(self, rng, n, m, pool):
        return self.bank.draw(rng, n, m, pool)

    def noise(self, sets, pos, mu0, rows, cols):
        return self.bank.noise(sets, pos, mu0, self.rows0[rows], self.cols0[cols])

    def shift_sd(self, mu0):
        return 0.0

    def describe(self):
        return self.bank.describe()


def _fragility(Yd, G, sd_std, spec, fin, t_obs, alpha, rows0, cols0):
    """Leave-one-cell-out and leave-one-single-agent-row-out fragility of a T_std call, with
    the reference draws of the reported stage."""
    I, J = Yd.shape
    nB = fin["B"]
    Yref = fin["Ys"][:nB]
    out = dict(computed=True)
    # leave one combination cell out
    cells = [(i, j) for i in range(1, I) for j in range(1, J) if G.obs[i * J + j]]
    best = None
    for (i, j) in cells:
        o = G.obs.reshape(I, J).copy()
        o[i, j] = False
        g = _Grid(o, sd_std)
        t = g.tstd(Yd[None])[0]
        infl = t_obs - t
        if best is None or infl > best[0]:
            best = (infl, (i, j), g, t)
    infl, (i, j), g, t = best
    p_loco = tail_p(t, g.tstd(Yref))
    out.update(loco_cell=(int(rows0[i]), int(cols0[j])), loco_influence_share=float(infl / max(t_obs, 1e-300)),
               loco_p=float(p_loco), loco_flag=bool(p_loco > alpha))
    # leave one single-agent dose (with its row or column) out
    best = None
    for side, n in (("A", I), ("B", J)):
        if n - 1 < 3:
            continue
        for k in range(1, n):
            rows = np.arange(I) if side == "B" else np.array([q for q in range(I) if q != k])
            cols = np.arange(J) if side == "A" else np.array([q for q in range(J) if q != k])
            Yr = np.ascontiguousarray(Yd[rows[:, None], cols[None, :]])
            o = G.obs.reshape(I, J)[rows[:, None], cols[None, :]]
            g = _Grid(o, sd_std)
            t = g.tstd(Yr[None])[0]
            infl = t_obs - t
            if best is None or infl > best[0]:
                best = (infl, side, k, rows, cols, g, t, Yr)
    if best is None:
        out.update(sa_dose=None, sa_p=np.nan, sa_flag=False, sa_influence_share=np.nan)
        return out
    infl, side, k, rows, cols, g, t, Yr = best
    Ys, _ = spec.plates(fin["seed"], fin["Ys"].shape[0], Yr[1:, 0], Yr[0, 1:], rows=rows, cols=cols)
    p_sa = tail_p(t, g.tstd(Ys[:nB]))
    orig = int(rows0[k]) if side == "A" else int(cols0[k])
    out.update(sa_dose=(side, orig), sa_influence_share=float(infl / max(t_obs, 1e-300)), sa_p=float(p_sa),
               sa_flag=bool(p_sa > alpha))
    return out
