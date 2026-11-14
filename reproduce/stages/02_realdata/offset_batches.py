"""Offset between the two O'Neil batches on groups whose reported curves are identical (Supplementary Section S13)."""
import json
import os
import numpy as np
import pandas as pd
import common as C
NB = 30 if C.paths.SMOKE else 2000


def summary(d):
    D = (d.mean_excess_b0 - d.mean_excess_b1).to_numpy(float)
    v = (d.mean_excess_se_b0 ** 2 + d.mean_excess_se_b1 ** 2).to_numpy(float)
    z = D / np.sqrt(v)
    sd = float(np.std(D, ddof=1))
    # Moment estimate: Var(D) = mean SE^2 of the difference + 2 tau^2 when each experiment has its own offset.
    return dict(n=int(D.size), mean_diff=float(D.mean()), median_diff=float(np.median(D)), sd_diff=sd,
                sd_diff_no_offset=float(np.sqrt(v.mean())), median_se_b0=float(np.median(d.mean_excess_se_b0)),
                median_se_b1=float(np.median(d.mean_excess_se_b1)), sd_z=float(np.std(z, ddof=1)),
                robust_sd_z=float(1.4826 * np.median(np.abs(z - np.median(z)))),
                tau_equal_var=float(np.sqrt(max(sd ** 2 - v.mean(), 0.0) / 2)))


def cluster_ci(d, seed=15):
    D = (d.mean_excess_b0 - d.mean_excess_b1).to_numpy(float)
    (u, inv) = np.unique(d.cell_line.to_numpy(), return_inverse=True)
    members = [np.where(inv == k)[0] for k in range(u.size)]
    rng = np.random.default_rng(seed)
    bs = np.array([D[np.concatenate([members[k] for k in rng.integers(0, u.size, u.size)])].mean() for _ in range(NB)])
    (lo, hi) = np.percentile(bs, [2.5, 97.5])
    return dict(n_cell_lines=int(u.size), n_boot=NB, mean_diff_lo=float(lo), mean_diff_hi=float(hi))


def main():
    p = pd.read_parquet(os.path.join(C.RES, 'repro_pairs_oneil.parquet'))
    same = p[p.edges_identical]
    out = dict(identical=dict(summary(same), **cluster_ci(same)), all=summary(p))
    with open(os.path.join(C.RES, 'offset_batches.json'), 'w') as f:
        json.dump(out, f, indent=1)
    print(json.dumps(out, indent=1))


if __name__ == '__main__':
    main()
