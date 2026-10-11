"""Collapse the large Monte Carlo draws of reproduce/output into the small count tables that the
notebooks 03 and 04 read from data/aggregates/. Run from reproduce/notebooks/data after the pipeline:
    python3 make_aggregates.py
Each table holds rejection counts k out of n per cell; rates and intervals are computed in the notebooks."""
import glob, os
import numpy as np
import pandas as pd

OUT = "../../output"
SIM, INP, REAL = f"{OUT}/simulations", f"{OUT}/inputs", f"{OUT}/realdata"
os.makedirs("aggregates", exist_ok=True)
TOL = 1 + 1e-6

def cat2str(d):
    for c in d.columns:
        if str(d[c].dtype) == "category":
            d[c] = d[c].astype(str)
    return d

def counts(d, keys, cols, thr, suffix=""):
    g = d.assign(**{f"k_{n}{suffix}": (d[c].fillna(1.0) <= thr) for n, c in cols.items()}).groupby(keys, observed=True, sort=False)
    out = g.agg(n=(list(cols.values())[0], "size"), **{f"k_{n}{suffix}": (f"k_{n}{suffix}", "sum") for n in cols}).reset_index()
    return out

P3 = {"tstd": "p_tstd", "scan": "p_scan", "mean": "p_mean"}
# h2h size and confirm: levels 0.05 and 0.01
for name in ("size", "confirm"):
    d = cat2str(pd.read_parquet(f"{SIM}/h2h_{name}.parquet"))
    keys = ["ref", "design", "I", "law", "surface", "cond"] if name == "size" else ["design", "I", "law", "surface"]
    a = counts(d, keys, P3, 0.05 * TOL, "_05")
    b = counts(d, keys, P3, 0.01 * TOL, "_01").drop(columns="n")
    a.merge(b, on=keys).to_csv(f"aggregates/h2h_{name}_cells.csv", index=False)
# real power
rows = []
for study in ("oneil", "almanac"):
    d = pd.read_parquet(f"{SIM}/h2h_realpower_{study}.parquet")
    r = d.groupby("cond").apply(lambda g: pd.Series(dict(n=len(g), eff_mean=g.eff_mean.mean(),
        k_tstd=(g.p_tstd <= 0.05 * TOL).sum(), k_scan=(g.p_scan <= 0.05 * TOL).sum(), k_companion=(g.p_companion <= 0.05 * TOL).sum()))).reset_index()
    r.insert(0, "study", study); rows.append(r)
pd.concat(rows).to_csv("aggregates/h2h_realpower_cells.csv", index=False)
# stress studies
d = pd.read_parquet(f"{SIM}/stress_shared_null.parquet")
d.assign(k_t1=d.p_t1 <= 0.05, k_s1=d.p_s1 <= 0.05, k_m=d.p_m <= 0.05).groupby(["share", "m_s", "ref", "screen"]).agg(
    n=("p_t1", "size"), k_t1=("k_t1", "sum"), k_s1=("k_s1", "sum"), k_m=("k_m", "sum")).reset_index().to_csv("aggregates/stress_null_cells.csv", index=False)
d = pd.read_parquet(f"{SIM}/stress_shared_mixed.parquet")
d.assign(k_t1=d.p_t1 <= 0.05).groupby(["m_s", "cls", "cond"]).agg(n=("p_t1", "size"), k_t1=("k_t1", "sum")).reset_index().to_csv("aggregates/stress_mixed_cells.csv", index=False)
d = pd.read_parquet(f"{SIM}/stress_shift.parquet")
d.assign(k_t=d.p_t <= 0.05, k_s=d.p_s <= 0.05, k_m=d.p_m <= 0.05).groupby(["m_s", "ratio", "ref"]).agg(
    n=("p_t", "size"), k_t=("k_t", "sum"), k_s=("k_s", "sum"), k_m=("k_m", "sum")).reset_index().to_csv("aggregates/stress_shift_cells.csv", index=False)
d = pd.read_parquet(f"{SIM}/bb_plugin.parquet")
d.assign(k=d.p_tstd <= 0.05).groupby(["noise", "base", "ms", "cond"]).agg(n=("k", "size"), k=("k", "sum")).reset_index().to_csv("aggregates/bb_plugin_cells.csv", index=False)
# adversarial search
ns = pd.read_csv(f"{SIM}/null_search.csv"); nsp = pd.read_csv(f"{SIM}/null_search_paths.csv")
cand = ns[ns.cand != "bliss"]
pd.DataFrame([dict(searches=len(nsp.groupby(["design", "I", "surface", "truth", "stat", "alpha"])), climbs=len(nsp),
    climbs_above_bliss=int((nsp.crn_hard > nsp.crn_hard_bliss + 1e-12).sum()), candidates=len(cand), datasets_per_candidate=int(cand.n.iloc[0]),
    candidates_new_rejection=int((cand.n_new_rej > 0).sum()), max_diff_vs_bliss=float(cand.diff_vs_bliss.max()))]).to_csv("aggregates/null_search_summary.csv", index=False)
# BIGL: our side
d = pd.read_parquet(f"{SIM}/bigl_ours.parquet")
d.assign(k_tstd=d.p_tstd <= 0.05 * (1 + 1e-9), k_scan=d.p_scan <= 0.05 * (1 + 1e-9), k_mean=d.p_mean <= 0.05 * (1 + 1e-9)).groupby(["r", "setting", "cond"]).agg(
    n=("p_tstd", "size"), k_tstd=("k_tstd", "sum"), k_scan=("k_scan", "sum"), k_mean=("k_mean", "sum")).reset_index().to_csv("aggregates/bigl_ours_cells.csv", index=False)
# BIGL: its per-matrix decisions
keep = ["fit_err", "surf_err", "sigma0", "maxR_B", "maxR_nge_abs", "maxR_rej_syn_005", "maxR_rej_syn_010", "meanR_p", "c_h1", "c_h2", "c_b", "c_m1", "c_m2", "c_e1", "c_e2", "idx", "file", "config"]
raw = pd.concat([pd.read_csv(f, usecols=keep, low_memory=False) for f in sorted(glob.glob(f"{INP}/bigl_raw/*/*.csv"))], ignore_index=True)
base = raw.file.str.replace(".f64", "", regex=False)
raw["setting"] = np.where(base.str.contains("__"), base.str.split("__").str[0], base)
raw["cond"] = np.where(base.str.contains("__"), base.str.split("__").str[1], "real")
ok = (raw.fit_err == 0) & (raw.surf_err == 0)
B = raw.maxR_B.fillna(100); nge = raw.maxR_nge_abs.fillna(B)
raw["ok"] = ok.astype(int)
raw["syn05"] = (ok & (raw.maxR_rej_syn_005.fillna(0) == 1)).astype(int)
raw["syn10"] = (ok & (raw.maxR_rej_syn_010.fillna(0) == 1)).astype(int)
raw["rank05"] = (ok & ((1 + nge) / (B + 1) <= 0.05 + 1e-12)).astype(int)
raw["mean05"] = (ok & (raw.meanR_p.fillna(1) < 0.05 - 1e-12)).astype(int)
raw[["config", "setting", "cond", "idx", "fit_err", "surf_err", "sigma0", "ok", "syn05", "syn10", "rank05", "mean05",
     "c_h1", "c_h2", "c_b", "c_m1", "c_m2", "c_e1", "c_e2"]].round(6).to_csv("aggregates/bigl_decisions.csv.gz", index=False)
# production calls on the 300 real groups
meta = pd.read_csv(f"{INP}/bigl_real_meta.csv"); q = pd.read_parquet(f"{REAL}/screen_oneil_q.parquet")
m = meta.assign(bkey=meta.block_ids.astype(str).str.replace(";", ",")).merge(q.assign(bkey=q.block_ids.astype(str)), on="bkey", suffixes=("", "_q"), validate="one_to_one")
m.sort_values("k")[["k", "group_key", "p_tstd", "p_scan", "p_companion"]].to_csv("aggregates/bigl_real_production.csv", index=False)
# plain copies of small results
import shutil
for rel in ["simulations/h2h_power.parquet", "simulations/stress_reps.parquet", "simulations/stress_shared_null_family.csv",
            "simulations/stress_shared_mixed_family.csv", "simulations/grid_counts.csv", "simulations/grid_effect80.csv",
            "simulations/grid_size.csv", "simulations/grid_numbers.json", "simulations/figS2_summary.csv", "simulations/supplement_numbers.json",
            "simulations/tabS4_stress_numbers.csv", "simulations/bigl_summary.json", "simulations/bigl_reference.csv", "simulations/pooling_size.csv",
            "simulations/complete_pairwise_size.csv", "simulations/comparator_numbers.json", "realdata/null_summary.csv",
            "inputs/bb_table_rows.csv", "inputs/bb_table_rows_ext.csv", "inputs/bb_coverage.csv", "inputs/bb_coverage_ext.csv",
            "inputs/bigl_real_meta.csv", "inputs/bigl_runtime.csv", "inputs/bigl_comparison_data/real_oneil300.f64",
            "runtime/runtime_grid.csv", "runtime/runtime_r_only.csv"] + [f"tables/tabS{i}_{n}.tex" for i, n in
            ((1, "size"), (2, "misspec"), (3, "power"), (4, "stress"), (5, "bigl"), (7, "grid"))]:
    os.makedirs(os.path.dirname(rel), exist_ok=True)
    shutil.copy(f"{OUT}/{rel}", rel)
