"""Synthetic surfaces and viability generation for examples and validation.

Bliss products use Hill single agents. Synergy and antagonism perturb the
combination wells multiplicatively; generators also support gross errors and
missing completely at random wells."""
import numpy as np

SURF_E = {"weak": 0.3, "moderate": 0.7, "strong": 0.95}
SURF_AB = {"flat": (0.0, 0.0), "half": (0.7, 0.0)}
GROSS_NEG_SHARE = 0.44


def dose_grid(n_nonzero, ec50=1.0, span=2.0):
    return np.concatenate([[0.0], ec50 * np.logspace(-span / 2, span / 2, n_nonzero)])


def hill(d, E):
    return 1 - E * d ** 1.5 / (d ** 1.5 + 1)


def surface(kind, I, J=None):
    J = I if J is None else J
    da, db = dose_grid(I - 1), dose_grid(J - 1)
    if kind in SURF_AB:
        ea, eb = SURF_AB[kind]
    else:
        ea = eb = SURF_E[kind]
    return np.outer(hill(da, ea), hill(db, eb))


def interaction_pattern(I, J, kind):
    """nonnegative interior pattern, zero on the edges, max 1."""
    ii, jj = np.meshgrid(np.arange(I), np.arange(J), indexing="ij")
    x = ii / (I - 1)
    y = jj / (J - 1)
    inter = (ii > 0) & (jj > 0)
    if kind == "broad":
        p = np.ones((I, J))
    elif kind == "highdose":
        p = x * y
    elif kind == "lowdose":
        p = (1 - x + 1.0 / (I - 1)) * (1 - y + 1.0 / (J - 1))
    elif kind == "corner":
        p = ((ii == I - 1) & (jj == J - 1)).astype(float)
    elif kind == "bump":
        ci, cj = (I - 1) / 2.0, (J - 1) / 2.0
        w = max(I, J) / 4.0
        p = np.exp(-(((ii - ci) ** 2 + (jj - cj) ** 2) / (2 * w * w)))
    else:
        raise ValueError(kind)
    p = np.where(inter, p, 0.0)
    return p / np.max(np.abs(p))


def pattern(kind, I):
    m = I - 1
    P = np.zeros((I, I))
    if kind == "hdblock":
        P[m - 1:, m - 1:] = 1.0
    elif kind == "lshape":
        P[m, 1:] = 1.0
        P[1:, m] = 1.0
    else:
        P = interaction_pattern(I, I, kind)
    return P


def truth(cond, surf, I):
    mu = surface(surf, I)
    if cond == "null":
        return mu
    parts = cond.split("_")
    if parts[0] == "sat":
        out = mu.copy()
        out[1:, 1:] = mu[1:, 1:] ** (1.0 + float(parts[1]))
        return out
    s, pat, amp = parts
    return mu * np.exp((-float(amp) if s == "syn" else float(amp)) * pattern(pat, I))


def bliss_excess(mu):
    return np.outer(mu[1:, 0], mu[0, 1:]) - mu[1:, 1:]


def simulate(mu, noise, design, n, rng, m=1):
    """Data plates with mean mu under a ParametricNoise law: design 'S' every non-control well noisy;
    'P' single agents exact (noise free). Returns (n, I, J) for m = 1, else (n, m, I, J). Control 1."""
    I, J = mu.shape
    e = noise.standard(rng, (n, m, I, J))
    Y = noise.generate(mu, e)
    tau = noise.shift_sd(mu)
    if tau > 0:
        g = rng.standard_normal((n, m))
        if design == "P":
            Y[..., 1:, 1:] += tau * g[:, :, None, None]
        else:
            Y = Y + tau * g[:, :, None, None]
    if design == "P":
        Y[..., 1:, 0] = mu[1:, 0]
        Y[..., 0, 1:] = mu[0, 1:]
    Y[..., 0, 0] = 1.0
    return Y[:, 0] if m == 1 else Y


def contaminate(R, rng, p, magnitudes, edges=True):
    """Gross single-replicate errors at per-well rate p on replicate plates R (n, m, I, J) (a copy is
    returned with the contamination mask (n, I, J))."""
    n, m, I, J = R.shape
    U = rng.random((n, I, J))
    Kr = rng.integers(0, m, (n, I, J))
    D = rng.choice(np.asarray(magnitudes, float), (n, I, J))
    S = rng.random((n, I, J))
    Rc = R.copy()
    cm = U < p
    cm[:, 0, 0] = False
    if not edges:
        cm[:, 0, :] = False
        cm[:, :, 0] = False
    idx = np.nonzero(cm)
    kk = Kr[idx]
    y = Rc[idx[0], kk, idx[1], idx[2]]
    d = D[idx]
    neg = (S[idx] < GROSS_NEG_SHARE) & (y - d >= 0.0)
    Rc[idx[0], kk, idx[1], idx[2]] = np.where(neg, y - d, y + d)
    return Rc, cm


def mcar_cells(rng, I, J, k):
    """k distinct combination cells (i, j >= 1) missing completely at random."""
    cells = [(i, j) for i in range(1, I) for j in range(1, J)]
    sel = rng.choice(len(cells), k, replace=False)
    return [cells[s] for s in sel]
