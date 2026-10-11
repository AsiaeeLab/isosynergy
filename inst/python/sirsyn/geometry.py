"""Grid geometry and scan regions.

Covering pairs restrict the product order to observed wells. Regions comprise
high-dose rectangles and 2 by 2 combination blocks, with duplicates removed
and sorted by size and cells (24 regions on a 5 by 5 grid)."""
import numpy as np

_PAIRS = {}
_REGIONS = {}


def adjacent_pairs(I, J):
    """Adjacent-pair constraints theta[u] >= theta[v] of the complete I x J grid."""
    key = (I, J)
    if key not in _PAIRS:
        pairs = []
        for i in range(I):
            for j in range(J):
                if i + 1 < I:
                    pairs.append((i * J + j, (i + 1) * J + j))
                if j + 1 < J:
                    pairs.append((i * J + j, i * J + (j + 1)))
        P = np.array(pairs, dtype=np.int64)
        _PAIRS[key] = (np.ascontiguousarray(P[:, 0]), np.ascontiguousarray(P[:, 1]))
    return _PAIRS[key]


def order_pairs(I, J, obs):
    """Covering pairs (u, v) of the componentwise order restricted to the observed wells: u <= v, u != v,
    and no other observed well w with u <= w <= v. obs: (I * J,) bool. Constraint theta[u] >= theta[v]."""
    obs = np.asarray(obs, bool).reshape(I, J)
    if obs.all():
        return adjacent_pairs(I, J)
    C = np.zeros((I + 1, J + 1), np.int64)
    C[1:, 1:] = np.cumsum(np.cumsum(obs.astype(np.int64), 0), 1)

    def count(i0, j0, i1, j1):
        return C[i1 + 1, j1 + 1] - C[i0, j1 + 1] - C[i1 + 1, j0] + C[i0, j0]

    cells = [(i, j) for i in range(I) for j in range(J) if obs[i, j]]
    pa, pb = [], []
    for (i0, j0) in cells:
        for (i1, j1) in cells:
            if (i1, j1) == (i0, j0) or i1 < i0 or j1 < j0:
                continue
            if count(i0, j0, i1, j1) == 2:
                pa.append(i0 * J + j0)
                pb.append(i1 * J + j1)
    return np.array(pa, np.int64), np.array(pb, np.int64)


class Regions:
    """The rb region family on an I x J grid, optionally restricted to observed combination cells.
    sets: list of lists of (i, j); cells / cptr: flat CSR layout of the OBSERVED cells of each region for
    the numba kernel; regions without any observed cell are dropped."""

    def __init__(self, I, J, obs=None):
        self.I, self.J = I, J
        seen = {}
        for i0 in range(1, I):
            for j0 in range(1, J):
                s = tuple(sorted((a, b) for a in range(i0, I) for b in range(j0, J)))
                seen.setdefault(s, set()).add("rectangle")
        for i in range(1, I - 1):
            for j in range(1, J - 1):
                s = tuple(sorted([(i, j), (i + 1, j), (i, j + 1), (i + 1, j + 1)]))
                seen.setdefault(s, set()).add("block")
        ob = np.ones((I, J), bool) if obs is None else np.asarray(obs, bool).reshape(I, J)
        sets, types, cells, cptr = [], [], [], [0]
        for s in sorted(seen, key=lambda s: (len(s), s)):
            oc = [(i, j) for i, j in s if ob[i, j]]
            if not oc:
                continue
            sets.append(list(s))
            types.append("+".join(sorted(seen[s])))
            cells += [i * J + j for i, j in oc]
            cptr.append(len(cells))
        self.sets, self.types = sets, types
        self.nR = len(sets)
        self.cells = np.array(cells, dtype=np.int64)
        self.cptr = np.array(cptr, dtype=np.int64)
        self.size = np.array([len(s) for s in sets])

    def label(self, r):
        return ";".join(f"{i},{j}" for i, j in self.sets[r])


def regions(I, J, obs=None):
    if obs is None or np.asarray(obs, bool).all():
        key = (I, J)
        if key not in _REGIONS:
            _REGIONS[key] = Regions(I, J)
        return _REGIONS[key]
    return Regions(I, J, obs)
