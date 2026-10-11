"""Design-matched companion tests for mean Bliss excess.

Single-plate design S uses V2Sp. Replicate stacks use a block-level viability t
test. Single-plate design P uses a fixed-edge t test with plate and edge variance.
Zero or nonfinite SE yields a missing p-value and a degeneracy flag."""
import numpy as np
from scipy import stats


def _pt(est, se, df):
    est = float(est)
    se = float(se)
    if not (np.isfinite(se) and se > 0 and np.isfinite(est)):
        return dict(est=est, se=se, df=float(df), t=np.nan, p=np.nan, p_ant=np.nan, degenerate=True)
    t = est / se
    if np.isinf(df):
        return dict(est=est, se=se, df=float(df), t=t, p=float(stats.norm.sf(t)), p_ant=float(stats.norm.cdf(t)),
                    degenerate=False)
    return dict(est=est, se=se, df=float(df), t=t, p=float(stats.t.sf(t, df)), p_ant=float(stats.t.cdf(t, df)),
                degenerate=False)


# ---------------------------------------------------------------------------
# V2Sp (design S, complete grid)
# ---------------------------------------------------------------------------
def rank1_fit(Y, W=None, iters=300, tol=1e-12):
    """Least-squares fit E Y_i0 = a_i, E Y_0j = b_j, E Y_ij = a_i b_j (control fixed) by alternating least
    squares, vectorized over the first axis."""
    YA, YB, YI = Y[:, 1:, 0], Y[:, 0, 1:], Y[:, 1:, 1:]
    if W is None:
        wA, wB, wI = np.ones_like(YA), np.ones_like(YB), np.ones_like(YI)
    else:
        wA, wB, wI = W[:, 1:, 0], W[:, 0, 1:], W[:, 1:, 1:]
    a, b = YA.copy(), YB.copy()
    for _ in range(iters):
        a_new = (wA * YA + (wI * YI * b[:, None, :]).sum(2)) / (wA + (wI * b[:, None, :] ** 2).sum(2))
        b_new = (wB * YB + (wI * YI * a_new[:, :, None]).sum(1)) / (wB + (wI * a_new[:, :, None] ** 2).sum(1))
        d = max(np.abs(a_new - a).max(), np.abs(b_new - b).max())
        a, b = a_new, b_new
        if d < tol:
            break
    return a, b


def _fitted(a, b):
    n = a.shape[0]
    I, J = a.shape[1] + 1, b.shape[1] + 1
    F = np.empty((n, I, J))
    F[:, 0, 0] = 1.0
    F[:, 1:, 0] = a
    F[:, 0, 1:] = b
    F[:, 1:, 1:] = a[:, :, None] * b[:, None, :]
    return F


def _cvec(a, b):
    n, I1 = a.shape
    J1 = b.shape[1]
    r = I1 * J1
    c = np.empty((n, I1 + 1, J1 + 1))
    c[:, 0, 0] = 0.0
    c[:, 1:, 0] = (b.mean(1) / I1)[:, None]
    c[:, 0, 1:] = (a.mean(1) / J1)[:, None]
    c[:, 1:, 1:] = -1.0 / r
    return c


def v2sp_many(Y):
    """V2Sp for a stack (n, I, J) of complete design-S plates. Returns dict of arrays est, se, df, p."""
    Y = np.array(Y, float, copy=True)
    Y[:, 0, 0] = 1.0
    n, I, J = Y.shape
    r = (I - 1) * (J - 1)
    est = Y[:, 1:, 0].mean(1) * Y[:, 0, 1:].mean(1) - Y[:, 1:, 1:].mean((1, 2))
    a, b = rank1_fit(Y)
    F = _fitted(a, b)
    c = _cvec(a, b)
    res = Y - F
    res[:, 0, 0] = 0.0
    N = I * J - 1
    idx = np.arange(1, I * J)
    isE = np.zeros(I * J, bool)
    isE[1:J] = True
    isE[np.arange(1, I) * J] = True
    isE = isE[idx]
    isI = ~isE
    T = np.zeros((n, I, J, (I - 1) + (J - 1) + 1))
    for i in range(1, I):
        T[:, i, 0, i - 1] = 1.0
        T[:, i, 1:, i - 1] = b
    for j in range(1, J):
        T[:, 0, j, (I - 1) + j - 1] = 1.0
        T[:, 1:, j, (I - 1) + j - 1] = a
    T[..., -1] = c
    T = T.reshape(n, I * J, -1)[:, idx, :]
    Q, _ = np.linalg.qr(T)
    Rm = np.eye(N)[None] - Q @ np.swapaxes(Q, 1, 2)
    rv = res.reshape(n, -1)[:, idx]
    rr = np.einsum("nkl,nl->nk", Rm, rv)
    qE = (rr[:, isE] ** 2).sum(1)
    qI = (rr[:, isI] ** 2).sum(1)
    R2 = Rm * Rm
    mEE = R2[:, isE][:, :, isE].sum((1, 2))
    mEI = R2[:, isE][:, :, isI].sum((1, 2))
    mII = R2[:, isI][:, :, isI].sum((1, 2))
    det = mEE * mII - mEI * mEI
    sE = (mII * qE - mEI * qI) / det
    sI = (mEE * qI - mEI * qE) / det
    tot = qE + qI
    negE = sE < 0
    negI = sI < 0
    sE = np.where(negE, 0.0, sE)
    sI = np.where(negE, tot / (mEI + mII), sI)
    sI = np.where(negI & ~negE, 0.0, sI)
    sE = np.where(negI & ~negE, tot / (mEE + mEI), sE)
    cv = c.reshape(n, -1)[:, idx]
    lE = (cv[:, isE] ** 2).sum(1)
    lI = (cv[:, isI] ** 2).sum(1)
    VA = sE / (I - 1)
    VB = sE / (J - 1)
    hE = (mII * lE - mEI * lI) / det
    hI = np.maximum((mEE * lI - mEI * lE) / det, 0.0)
    v = hE * qE + hI * qI + VA * VB
    se = np.sqrt(np.maximum(v, 0.0))
    h = np.where(isE[None, :], hE[:, None], hI[:, None])
    S = np.where(isE[None, :], sE[:, None], sI[:, None])
    QS = (Rm * h[:, None, :]) @ (Rm * S[:, None, :])
    num = np.einsum("nkk->n", QS)
    den = (QS * np.swapaxes(QS, 1, 2)).sum((1, 2))
    df = np.clip(num * num / np.maximum(den, 1e-300), 1.0, r - 1)
    with np.errstate(divide="ignore", invalid="ignore"):
        ok = np.isfinite(se) & (se > 0)
        t = np.where(ok, est / np.where(ok, se, 1.0), np.nan)
        p = np.where(ok, stats.t.sf(t, df), np.nan)
    return dict(est=est, se=se, df=df, t=t, p=p)


def v2sp(Y):
    o = v2sp_many(np.asarray(Y, float)[None])
    out = _pt(o["est"][0], o["se"][0], o["df"][0])
    out["name"] = "V2Sp (design S)"
    return out


# ---------------------------------------------------------------------------
# replicated designs: block-level viability t
# ---------------------------------------------------------------------------
def block_t(R, obs_int, design, dropped=None):
    """R: (m, I, J) replicate plates (NaN = unavailable); obs_int: (I - 1, J - 1) analyzed combination cells.
    Per replicate k, mean over its available, analyzed, unscreened combination cells of the viability Bliss
    score (design S / R: its own single-agent wells; design P: the shared external single agents, taken as
    the replicate mean of the edges). t over replicates, df m' - 1 (m' replicates with at least one cell)."""
    R = np.array(R, float, copy=True)
    m, I, J = R.shape
    if dropped is not None:
        for k in range(m):
            R[k][dropped == k] = np.nan
    if design == "P":
        with np.errstate(all="ignore"):
            eA = np.nanmean(R[:, 1:, 0], 0)
            eB = np.nanmean(R[:, 0, 1:], 0)
        S = np.outer(eA, eB)[None] - R[:, 1:, 1:]
    else:
        S = R[:, 1:, :1] * R[:, :1, 1:] - R[:, 1:, 1:]
    S = np.where(obs_int[None], S, np.nan)
    with np.errstate(all="ignore"):
        E = np.nanmean(S.reshape(m, -1), 1)
    E = E[np.isfinite(E)]
    if E.size < 2:
        out = _pt(np.nan, np.nan, np.nan)
    else:
        out = _pt(E.mean(), E.std(ddof=1) / np.sqrt(E.size), E.size - 1)
    out["name"] = "block-level viability t (replicated design)"
    out["n_blocks"] = int(E.size)
    return out


# ---------------------------------------------------------------------------
# design P: fixed-edge viability t with plate variance component and edge variance
# ---------------------------------------------------------------------------
def edge_var_mean(a, sd, m_s, rho):
    """Variance of the mean over doses of a reported curve averaged over m_s replicate curves with
    replicate-level shift share rho."""
    if not np.isfinite(m_s):
        return 0.0
    s = sd(a)
    return ((1 - rho) / m_s) * np.sum(s ** 2) / a.size ** 2 + (rho / m_s) * np.mean(s) ** 2


def fixed_edge_t(Y, obs_int, plate_var=0.0, m=1, edge_var=0.0):
    """Y: (I, J) analyzed matrix; scores eA_i eB_j - Y_ij over analyzed combination cells; SE^2 =
    var(scores, ddof 1) / r + plate_var / m + edge_var; df r - 1."""
    S = np.outer(Y[1:, 0], Y[0, 1:]) - Y[1:, 1:]
    s = S[obs_int]
    r = s.size
    if r < 2:
        out = _pt(np.nan, np.nan, np.nan)
    else:
        out = _pt(s.mean(), np.sqrt(s.var(ddof=1) / r + plate_var / m + edge_var), r - 1)
    out["name"] = "fixed-edge viability t" + (" with plate variance component" if plate_var > 0 else "") + \
        (" and edge variance" if edge_var > 0 else "")
    return out
