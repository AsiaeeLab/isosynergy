"""Select donor sets with deterministic sampling and viability distances."""
import paths
import numpy as np
import sirsyn as X
import time
_B = np.load(paths.INPUTS / 'noise_bank_oneil.npz')
SEL4 = np.where(_B['group_size'] == 4)[0]
CELL = _B['cell_line']
YBAR = _B['ybar_raw']
EA = _B['edge_drugA_raw'] / _B['edge_drugA_raw'][:, :1]
EB = _B['edge_drugB_raw'] / _B['edge_drugB_raw'][:, :1]
DONOR_FILE = paths.INPUTS / 'oneil_donors.npz'

def build_donors():
    n = SEL4.size
    donors = np.empty((n, 20), dtype=np.int64)
    ks = np.empty((n, 5), dtype=np.int8)
    ybi = YBAR[:, 1:, 1:].reshape(YBAR.shape[0], -1)
    t0 = time.time()
    for (ti, s) in enumerate(SEL4):
        rng = np.random.default_rng(X.seed_for(bytes([114, 50, 98]).decode('ascii'), 'donors', int(s)))
        cand = rng.choice(SEL4, 600, replace=False)
        cand = cand[CELL[cand] != CELL[s]]
        mu = (EA[s, 1:, None] * EB[s, None, 1:]).ravel()
        d2 = ((ybi[cand] - mu[None]) ** 2).sum(1)
        o = np.argsort(d2, kind='stable')[:20]
        donors[ti] = cand[o]
        ks[ti] = rng.integers(0, 4, size=5)
    np.savez_compressed(DONOR_FILE, donors=donors, k=ks, targets=SEL4)
    dist = []
    for ti in range(0, n, 50):
        s = SEL4[ti]
        mu = EA[s, 1:, None] * EB[s, None, 1:]
        dist.append(np.abs(YBAR[donors[ti, 0], 1:, 1:] - mu).mean())
    print(f'donors built for {n} targets in {time.time() - t0:.0f} s; median mean |mu - donor mean| {np.median(dist):.3f}')
if __name__ == '__main__':
    build_donors()
