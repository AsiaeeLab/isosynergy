"""Plug-in bootstrap at exact Bliss.

The reference reproduces single-agent uncertainty, plate shifts, replicate
structure, availability and replicate screening. Design P fixes external edges
when sa_reps is infinite; otherwise it redraws them with the supplied correlation.
The first B plates calibrate the tests, and n_std independent plates standardize
the scan. Full-grid draws are reused for fragility calculations."""
import numpy as np

from . import _kernels as K
from .replicates import screen as rep_screen

TIE_RTOL = 1e-12


def ref_mean(a, b):
    """Exact-Bliss mean matrix from single-agent curves at nonzero doses (control 1)."""
    I, J = a.size + 1, b.size + 1
    M = np.empty((I, J))
    M[0, 0] = 1.0
    M[1:, 0] = a
    M[0, 1:] = b
    M[1:, 1:] = np.outer(a, b)
    return M


class ReferenceSpec:
    """Everything that defines the reference law of one analysis except the observed values."""

    def __init__(self, design, m, noise, full_shape, avail=None, screen=True, screen_c=5.0, min_avail=3,
                 sa_reps=np.inf, sa_rho=0.10, pool=None):
        self.design = design
        self.m = int(m)
        self.noise = noise
        self.full_shape = tuple(full_shape)
        self.avail = None if avail is None else np.asarray(avail, bool)  # (m, I0, J0)
        self.screen = bool(screen) and self.m >= min_avail
        self.screen_c = float(screen_c)
        self.min_avail = int(min_avail)
        self.sa_reps = float(sa_reps)
        self.sa_rho = float(sa_rho)
        self.pool = pool

    def plates(self, seed, n, edgeA, edgeB, rows=None, cols=None):
        """n reference plates (n, I, J) on the grid rows x cols of the full analyzed grid, plus the number of
        screened replicates per plate. edgeA, edgeB: the analyzed single-agent values at the nonzero doses of
        that grid (data values; a_hat, b_hat are their PAVA fits)."""
        I0, J0 = self.full_shape
        rows = np.arange(I0) if rows is None else np.asarray(rows)
        cols = np.arange(J0) if cols is None else np.asarray(cols)
        I, J = rows.size, cols.size
        rng = np.random.default_rng(seed)
        nz = self.noise
        m = self.m
        # ---- draws on the full grid, fixed order -------------------------------------------------
        if nz.kind == "parametric":
            e = nz.standard(rng, (n, m, I0, J0))
        else:
            sets, pos = nz.draw(rng, n, m, self.pool)
        g = rng.standard_normal((n, m))
        if self.design == "P" and np.isfinite(self.sa_reps):
            ua = rng.standard_normal((n, I0 - 1))
            ua0 = rng.standard_normal((n, 1))
            ub = rng.standard_normal((n, J0 - 1))
            ub0 = rng.standard_normal((n, 1))
        # ---- assemble on the requested grid ------------------------------------------------------
        a = K.pava_dec(np.ascontiguousarray(edgeA, dtype=float))
        b = K.pava_dec(np.ascontiguousarray(edgeB, dtype=float))
        mu0 = ref_mean(a, b)
        if nz.kind == "parametric":
            R = nz.generate(mu0, e[..., rows[:, None], cols[None, :]])
            tau = nz.shift_sd(mu0)
        else:
            R = mu0 + nz.noise(sets, pos, mu0, rows, cols)
            tau = 0.0
        if tau > 0:
            if self.design == "S":
                R = R + tau * g[:, :, None, None]
            else:
                R[..., 1:, 1:] += tau * g[:, :, None, None]
        if self.design == "P":
            if np.isfinite(self.sa_reps):
                ms, rho = self.sa_reps, self.sa_rho
                sa, sb = nz.sd(a), nz.sd(b)
                ea = a[None] + sa[None] * (np.sqrt((1 - rho) / ms) * ua[:, rows[1:] - 1] + np.sqrt(rho / ms) * ua0)
                eb = b[None] + sb[None] * (np.sqrt((1 - rho) / ms) * ub[:, cols[1:] - 1] + np.sqrt(rho / ms) * ub0)
                R[:, :, 1:, 0] = ea[:, None, :]
                R[:, :, 0, 1:] = eb[:, None, :]
            else:
                R[:, :, 1:, 0] = np.asarray(edgeA, float)
                R[:, :, 0, 1:] = np.asarray(edgeB, float)
        R[:, :, 0, 0] = 1.0
        av = None
        if self.avail is not None:
            av = self.avail[:, rows[:, None], cols[None, :]]
            if not av.all():
                R = np.where(av[None], R, np.nan)
            else:
                av = None
        if m == 1:
            Ys = R[:, 0]
            nflag = np.zeros(n, np.int64)
        elif self.screen:
            Ys, dropped = rep_screen(R, self.screen_c, nz.sd, skip_edges=(self.design == "P"), avail=av,
                                     min_avail=self.min_avail)
            nflag = (dropped >= 0).sum((1, 2))
        else:
            with np.errstate(all="ignore"):
                Ys = np.nanmean(R, 1) if av is not None else R.mean(1)
            nflag = np.zeros(n, np.int64)
        Ys = np.ascontiguousarray(Ys)
        Ys[:, 0, 0] = 1.0
        return Ys, nflag


# ---------------------------------------------------------------------------
# Monte Carlo p-values
# ---------------------------------------------------------------------------
def tail_p(t0, tref):
    """(1 + #{T* >= T}) / (B + 1), ties counted as exceedances (relative tolerance 1e-12)."""
    thr = t0 - TIE_RTOL * max(1.0, abs(t0))
    return (1.0 + float(np.sum(np.asarray(tref) >= thr))) / (np.size(tref) + 1.0)


def scan_pvalues(t0, Tref, Tstd):
    """Single-step max-T region scan.
    t0: (nR,) observed region sums; Tref: (B, nR) reference rows; Tstd: (n_std, nR) standardization rows.
    S_U = (T_U - m_U) / s_U with m_U, s_U the mean and SD (ddof 1) over the standardization rows.
    Returns (global p, z (nR,), adjusted region p (nR,)); region U is claimed at level alpha when its adjusted
    p <= alpha, which controls the familywise error rate of the claims "U contains at least one synergistic
    cell" in the strong sense (single-step max-T, exchangeable reference)."""
    B = Tref.shape[0]
    k = Tstd.shape[0]
    m = Tstd.mean(0)
    s = np.sqrt(((Tstd - m) ** 2).sum(0) / (k - 1))
    s = np.maximum(s, 1e-14)
    z0 = (t0 - m) / s
    Mb = np.sort(((Tref - m) / s).max(1))
    cnt = B - np.searchsorted(Mb, z0 - 1e-12, side="left")
    padj = (1.0 + cnt) / (B + 1.0)
    return float(padj.min()), z0, padj
