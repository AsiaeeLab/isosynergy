"""Noise models: SD curves, parametric laws and replicate-residual banks.

Additive, lognormal, tabulated-SD and constant-CV laws support Gaussian or
variance-standardized t4 draws and optional plate shifts. Banks sample replicate
residuals within potency strata, across group folds, without selection on realized
noise. Cellwise SD ratios are clipped to [0.5, 2]."""
import hashlib

import numpy as np

# O'Neil pooled single-replicate SD by mean viability; flat outside the knots.
ONEIL_KV = np.array([0.021335, 0.074527, 0.148886, 0.250160, 0.349643, 0.449982, 0.550349,
                     0.650214, 0.751085, 0.852027, 0.925525, 0.974200, 1.021731])
ONEIL_KS = np.array([0.009054, 0.022849, 0.041241, 0.062327, 0.077293, 0.090443, 0.102209,
                     0.110316, 0.117421, 0.112216, 0.105203, 0.102541, 0.107957])
MFLOOR = 0.02


def seed_for(*keys):
    """Stable integer from arbitrary keys (SHA-256, never Python's hash)."""
    h = hashlib.sha256("|".join(map(str, keys)).encode()).hexdigest()
    return int(h[:15], 16)


# ---------------------------------------------------------------------------
# SD curves
# ---------------------------------------------------------------------------
class SDCurve:
    """Single-replicate noise SD as a function of mean viability.
    kind 'constant' (par), 'proportional' (par * max(m, 0.02)) or 'table' (interpolation in (kv, ks),
    flat outside the knots)."""

    _KIND = {"constant": 0, "proportional": 1, "table": 2}

    def __init__(self, kind, par=0.0, kv=None, ks=None):
        if kind not in self._KIND:
            raise ValueError(f"unknown SD curve kind {kind!r}")
        self.kind, self.par = kind, float(par)
        if kind == "table":
            kv = np.ascontiguousarray(kv, dtype=float)
            ks = np.ascontiguousarray(ks, dtype=float)
            if kv.ndim != 1 or kv.shape != ks.shape or kv.size < 2 or np.any(np.diff(kv) <= 0):
                raise ValueError("table SD curve needs increasing knots kv and values ks of equal length")
            if np.any(ks <= 0):
                raise ValueError("SD table values must be positive")
        else:
            kv = np.zeros(2)
            ks = np.ones(2)
            if self.par <= 0:
                raise ValueError("SD parameter must be positive")
        self.kv, self.ks = kv, ks

    def __call__(self, m):
        m = np.asarray(m, dtype=float)
        if self.kind == "constant":
            return np.full(m.shape, self.par)
        if self.kind == "proportional":
            return self.par * np.maximum(m, MFLOOR)
        return np.interp(m, self.kv, self.ks)

    def kernel_args(self):
        return self._KIND[self.kind], self.par, self.kv, self.ks

    def describe(self):
        if self.kind == "table":
            return f"table SD curve ({self.kv.size} knots, {self.ks.min():.4g} to {self.ks.max():.4g})"
        return f"{self.kind} SD {self.par:.4g}"


def oneil_sd_curve():
    """The O'Neil pooled single-replicate SD curve."""
    return SDCurve("table", kv=ONEIL_KV, ks=ONEIL_KS)


def sd_curve_from_replicates(Y, n_bins=13, min_wells=20, wells="combination"):
    """Pooled single-replicate SD of viability by mean viability from replicate sets.
    Y: (S, m, I, J) replicate plates (m >= 2). Wells are binned by their replicate mean into n_bins
    quantile bins; each knot is (mean of the bin's well means, sqrt of the mean within-well variance).
    wells: 'combination' (interior only) or 'all' (every non-control well with replicate spread)."""
    Y = np.asarray(Y, float)
    if Y.ndim != 4 or Y.shape[1] < 2:
        raise ValueError("Y must be (S, m, I, J) with m >= 2")
    sub = Y[:, :, 1:, 1:] if wells == "combination" else Y
    mean = np.nanmean(sub, 1).ravel()
    var = np.nanvar(sub, 1, ddof=1).ravel()
    ok = np.isfinite(mean) & np.isfinite(var)
    if wells != "combination":
        ok &= var > 0
    mean, var = mean[ok], var[ok]
    edges = np.quantile(mean, np.linspace(0, 1, n_bins + 1))
    b = np.clip(np.digitize(mean, edges[1:-1]), 0, n_bins - 1)
    kv, ks = [], []
    for j in range(n_bins):
        k = b == j
        if k.sum() >= min_wells:
            kv.append(mean[k].mean())
            ks.append(np.sqrt(var[k].mean()))
    kv, ks = np.array(kv), np.array(ks)
    o = np.argsort(kv)
    kv, ks = kv[o], ks[o]
    keep = np.concatenate([[True], np.diff(kv) > 1e-12])
    return SDCurve("table", kv=kv[keep], ks=ks[keep])


# ---------------------------------------------------------------------------
# parametric noise
# ---------------------------------------------------------------------------
class ParametricNoise:
    """A parametric noise law for the plug-in bootstrap reference (and for simulation)."""

    _GEN = {"additive": 0, "lognormal": 1, "additive_table": 2, "constant_cv": 3}
    kind = "parametric"

    def __init__(self, family="additive", par=0.08, table=None, tails="gaussian",
                 plate_shift_ratio=0.0, plate_shift_sd=None, sd=None):
        if family not in self._GEN:
            raise ValueError(f"unknown family {family!r}")
        if tails not in ("gaussian", "t4"):
            raise ValueError("tails must be 'gaussian' or 't4'")
        self.family, self.par, self.tails = family, float(par), tails
        if family == "additive_table":
            if table is None:
                table = oneil_sd_curve()
            if table.kind != "table":
                raise ValueError("additive_table needs a table SDCurve")
        self.table = table
        self.plate_shift_ratio = float(plate_shift_ratio)
        self.plate_shift_sd = None if plate_shift_sd is None else float(plate_shift_sd)
        if sd is None:
            if family == "additive":
                sd = SDCurve("constant", par)
            elif family == "lognormal":
                sd = SDCurve("proportional", float(np.sqrt(np.exp(par ** 2) - 1.0)))
            elif family == "additive_table":
                sd = table
            else:
                sd = SDCurve("proportional", par)
        self.sd = sd

    # constructors -------------------------------------------------------------
    @classmethod
    def additive(cls, sd=0.08, **kw):
        return cls("additive", sd, **kw)

    @classmethod
    def lognormal(cls, log_sd=0.12, **kw):
        return cls("lognormal", log_sd, **kw)

    @classmethod
    def sd_table(cls, curve=None, **kw):
        return cls("additive_table", 1.0, table=curve if curve is not None else oneil_sd_curve(), **kw)

    @classmethod
    def oneil(cls, **kw):
        return cls.sd_table(oneil_sd_curve(), **kw)

    @classmethod
    def constant_cv(cls, cv, **kw):
        return cls("constant_cv", cv, **kw)

    # sampling -----------------------------------------------------------------
    def standard(self, rng, shape):
        if self.tails == "gaussian":
            return rng.standard_normal(shape)
        return rng.standard_t(4, shape) / np.sqrt(2.0)

    def generate(self, mu, e):
        """Noisy values with mean mu (broadcast against e)."""
        k = self._GEN[self.family]
        if k == 0:
            return mu + self.par * e
        if k == 1:
            return np.maximum(mu, 1e-6) * np.exp(self.par * e - 0.5 * self.par * self.par)
        if k == 2:
            return mu + np.interp(mu, self.table.kv, self.table.ks) * e
        return mu + self.par * np.maximum(mu, 0.0) * e

    def shift_sd(self, mu0):
        if self.plate_shift_sd is not None:
            return self.plate_shift_sd
        if self.plate_shift_ratio > 0:
            return self.plate_shift_ratio * float(self.sd(np.mean(mu0[1:, 1:])))
        return 0.0

    def describe(self):
        s = f"{self.family} ({self.par:.4g}), {self.tails} tails, working SD: {self.sd.describe()}"
        if self.plate_shift_sd or self.plate_shift_ratio:
            s += (f", plate shift SD {self.plate_shift_sd:.4g}" if self.plate_shift_sd is not None
                  else f", plate shift ratio tau/sigma {self.plate_shift_ratio:.4g}")
        return s


# ---------------------------------------------------------------------------
# noise bank
# ---------------------------------------------------------------------------
class NoiseBank:
    """Replicate-residual noise bank.

    resid: (S, R, I, J) residuals with single-replicate variance ((Y - ybar) sqrt(R / (R - 1))); entries
           that carry no noise (the control; the single-agent wells of design-P banks) are 0 and
           `has_edges` is False.
    ybar:  (S, I, J) set means (for the cellwise SD-ratio rescaling).
    group: (S,) group labels (for example cell lines); fold: (S,) cross-fitting fold of each set.
    """

    kind = "bank"

    def __init__(self, resid, ybar, group, sd, n_strata=10, n_folds=2, fold_key="sirsyn", has_edges=False,
                 clip=(0.5, 2.0), fold_of=None):
        self.resid = np.ascontiguousarray(resid, dtype=float)
        self.ybar = np.ascontiguousarray(ybar, dtype=float)
        S, R, I, J = self.resid.shape
        if self.ybar.shape != (S, I, J):
            raise ValueError("ybar must be (S, I, J)")
        self.group = np.asarray(group).astype(str)
        if self.group.shape != (S,):
            raise ValueError("group must have one label per set")
        self.sd = sd
        self.has_edges = bool(has_edges)
        self.clip = clip
        self.n_folds = int(n_folds)
        self.fold_key = fold_key
        self._fold_of = fold_of if fold_of is not None else (
            lambda g: seed_for(fold_key, "half", g) % self.n_folds)
        ug = np.unique(self.group)
        fmap = {g: int(self._fold_of(g)) for g in ug}
        self.fold = np.array([fmap[g] for g in self.group], dtype=np.int64)
        self.sdy = self.sd(self.ybar)
        mb = self.ybar[:, 1:, 1:].mean((1, 2))
        self.mean_int = mb
        e = np.quantile(mb, np.linspace(0, 1, n_strata + 1))
        e[0], e[-1] = -np.inf, np.inf
        self.st_edges = e
        self.n_strata = n_strata
        self.stratum_of_set = np.clip(np.digitize(mb, e) - 1, 0, n_strata - 1)
        self.shape = (I, J)
        self.R = R

    # constructors -------------------------------------------------------------
    @classmethod
    def from_replicates(cls, Y, groups, sd=None, include_edges=False, **kw):
        """Y: (S, R, I, J) replicate plates (R >= 2), control normalized to 1. Residuals of the combination
        wells (and of the single-agent wells if include_edges, for design-S use) about the set mean, scaled
        by sqrt(R / (R - 1)) to single-replicate variance. sd: SD curve for rescaling and standardization
        (default: estimated from the same replicates with `sd_curve_from_replicates`)."""
        Y = np.asarray(Y, float)
        if Y.ndim != 4 or Y.shape[1] < 2:
            raise ValueError("Y must be (S, R, I, J) with R >= 2")
        if not np.all(np.isfinite(Y)):
            raise ValueError("the bank needs complete replicate sets (no NaN)")
        R = Y.shape[1]
        ybar = Y.mean(1)
        res = (Y - ybar[:, None]) * np.sqrt(R / (R - 1.0))
        if not include_edges:
            res[:, :, 0, :] = 0.0
            res[:, :, :, 0] = 0.0
        res[:, :, 0, 0] = 0.0
        if sd is None:
            sd = sd_curve_from_replicates(Y)
        return cls(res, ybar, groups, sd, has_edges=include_edges, **kw)

    @classmethod
    def from_oneil_npz(cls, path, sd=None, group_size=4, **kw):
        """Read an O'Neil-format NPZ bank: resid_raw, ybar_raw, cell_line, group_size.
        Residuals are unclipped. The default fold key preserves existing bank partitions.
        """
        B = np.load(path)
        sel = np.where(B["group_size"] == group_size)[0]
        I, J = B["ybar_raw"].shape[1:]
        res = np.zeros((sel.size, B["resid_raw"].shape[1], I, J))
        res[:, :, 1:, 1:] = B["resid_raw"][sel]
        kw.setdefault("fold_key", "r2b")
        kw.setdefault("n_folds", 2)
        bank = cls(res, B["ybar_raw"][sel], B["cell_line"][sel], sd if sd is not None else oneil_sd_curve(),
                   has_edges=False, **kw)
        bank.set_index = sel
        return bank

    # sampling -----------------------------------------------------------------
    def stratum(self, mean_int):
        return int(np.clip(np.digitize(mean_int, self.st_edges) - 1, 0, self.n_strata - 1))

    def pool(self, mu0, group=None, min_size=None):
        """Indices of the sets eligible as donors for a target with Bliss mean mu0: same potency stratum,
        and a different cross-fitting fold from `group` when it is given."""
        st = self.stratum(float(np.mean(mu0[1:, 1:])))
        ok = self.stratum_of_set == st
        if group is not None:
            ok &= self.fold != int(self._fold_of(str(group)))
        idx = np.where(ok)[0]
        if min_size is not None and idx.size < min_size:
            raise ValueError(f"noise-bank pool too small ({idx.size} sets, need {min_size})")
        return idx, st

    def draw(self, rng, n, m, pool):
        """(sets (n, m), positions (n, m)): m = 1 random set and replicate position; m > 1 distinct sets,
        positions 0..m-1 (random positions if m exceeds the bank's replicate count)."""
        if m == 1:
            s = rng.choice(pool, (n, 1))
            k = rng.integers(0, self.R, (n, 1))
            return s, k
        if pool.size < m:
            raise ValueError("noise-bank pool smaller than the number of replicates")
        S = rng.choice(pool, (n, m))
        for r in range(n):
            while np.unique(S[r]).size < m:
                S[r] = rng.choice(pool, m)
        if m <= self.R:
            K = np.broadcast_to(np.arange(m), (n, m)).copy()
        else:
            K = rng.integers(0, self.R, (n, m))
        return S, K

    def noise(self, sets, pos, mu0, rows=None, cols=None):
        """Rescaled residuals (n, m, I', J') for target mean mu0 (I', J'), rows / cols selecting the bank grid
        positions of the analyzed grid (default: all)."""
        rows = np.arange(self.shape[0]) if rows is None else np.asarray(rows)
        cols = np.arange(self.shape[1]) if cols is None else np.asarray(cols)
        Rz = self.resid[sets, pos][..., rows[:, None], cols[None, :]]
        sdy = self.sdy[sets][..., rows[:, None], cols[None, :]]
        ratio = np.clip(self.sd(mu0)[None, None] / sdy, self.clip[0], self.clip[1])
        return Rz * ratio

    def describe(self):
        return (f"noise bank: {self.resid.shape[0]} sets x {self.R} replicates, {self.shape[0]} x {self.shape[1]}, "
                f"{np.unique(self.group).size} groups in {self.n_folds} folds, {self.n_strata} potency strata, "
                f"{'with' if self.has_edges else 'without'} single-agent noise; SD: {self.sd.describe()}")


# ---------------------------------------------------------------------------
# plate (block-shift) variance component for the fixed-edge mean test
# ---------------------------------------------------------------------------
VC_BINS = np.array([-1.0, 0.1, 0.3, 0.5, 0.7, 0.9, 10.0])


class PlateVarianceTable:
    """Block-shift variance of the mean combination viability of one plate, by bin of mean combination
    viability, estimated per cross-fitting fold from replicate sets:
    sigma_b^2 = max(0, Var_k(block mean) - mean over cells of Var_k(cell) / r). Calling it with a group
    uses the table of the other fold(s) (cross-fitting)."""

    def __init__(self, Y, groups, n_folds=2, fold_key="sirsyn", bins=VC_BINS, fold_of=None):
        Y = np.asarray(Y, float)
        Yi = Y[:, :, 1:, 1:]
        r = Yi.shape[2] * Yi.shape[3]
        V = Yi.mean((2, 3)).var(1, ddof=1)
        pe = Yi.var(1, ddof=1).mean((1, 2)) / r
        m = Y.mean(1)[:, 1:, 1:].mean((1, 2))
        b = np.digitize(m, bins) - 1
        nb = bins.size - 1
        self.n_folds = n_folds
        self._fold_of = fold_of if fold_of is not None else (lambda g: seed_for(fold_key, "half", g) % n_folds)
        fold = np.array([int(self._fold_of(str(g))) for g in np.asarray(groups).astype(str)])
        self.cen = np.zeros(nb)
        self.tab = np.zeros((n_folds, nb))
        for j in range(nb):
            k = b == j
            self.cen[j] = m[k].mean() if k.any() else 0.5 * (max(bins[j], 0) + min(bins[j + 1], 1.5))
            for h in range(n_folds):
                kk = k & (fold == h)
                if kk.any():
                    self.tab[h, j] = max(0.0, V[kk].mean() - pe[kk].mean())

    def __call__(self, mean_int, group=None):
        x = float(np.clip(mean_int, 0.0, 1.5))
        if group is None:
            return float(np.interp(x, self.cen, self.tab.mean(0)))
        f = int(self._fold_of(str(group)))
        other = [h for h in range(self.n_folds) if h != f]
        return float(np.mean([np.interp(x, self.cen, self.tab[h]) for h in other]))
