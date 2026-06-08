"""Adaptive larger-B wild bootstrap, reusing the existing 200 draws.
Escalates per-cell-line candidates (p<=0.05) to B_TARGET, pools with the
production 200 via recovered exceedance count, recomputes per-cell-line BH.
15-core cap, chunk checkpointing, resumable. Writes results/drugcombdb_sir_adaptive.parquet.
Never overwrites the universe results."""
import os, sys, time, json, hashlib, numpy as np, pandas as pd
import pyarrow.dataset as pds, pyarrow.parquet as pq, pyarrow as pa
from multiprocessing import Pool
sys.path.insert(0, "theory")
import step5_numeric as S

B_TARGET = int(os.environ.get("B_TARGET", "20000"))
NWORK    = min(15, os.cpu_count()-1)
CHUNK    = 1500
OUT      = "results/drugcombdb_sir_adaptive.parquet"
SIDE     = "results/drugcombdb_sir_adaptive.checkpoint.json"
PROG     = "dataset_release/PROGRESS.md"
GSEED    = 42

def log(msg):
    with open(PROG,"a") as f: f.write(f"| {time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())} | adaptive | {msg} |\n"); f.flush(); os.fsync(f.fileno())

def seed_for(eid):
    h=int(hashlib.sha1(str(eid).encode()).hexdigest()[:8],16)
    return (h ^ GSEED) & 0x7fffffff

def boot_counts(geom, z, w, B, seed):
    tm0,ta0=geom.fit_both(z,w); d0=tm0-ta0; s2=float(np.sum(w*d0*d0)); r=z-ta0
    u,v=S.additive_uv(ta0.reshape(geom.I,geom.J))
    dfn=max(0,min(S.n_distinct_runs(u)+S.n_distinct_runs(v)-1, geom.N-1))
    rs=r*np.sqrt(geom.N/max(1,geom.N-dfn)); rng=np.random.default_rng(seed); cnt=nfin=0
    for _ in range(B):
        xi=rng.choice([-1.0,1.0],size=geom.N); tmb,tab=geom.fit_both(ta0+xi*rs,w); db=tmb-tab
        s2b=float(np.sum(w*db*db))
        if np.isfinite(s2b):
            nfin+=1
            if s2b>=s2: cnt+=1
    return s2,cnt,nfin

_MT=None
def _init(mt_path): 
    global _MT; _MT=pd.read_parquet(mt_path, columns=["experiment_id","row_index","col_index","response"])

def work(rec):
    eid,gs,p_old,s2_prod=rec
    geom=S.GEOM[gs]; w=np.full(geom.N,S.W_COMMON)
    g=_MT[_MT.experiment_id==eid]
    M=np.full((geom.I,geom.J),np.nan); M[g.row_index.values-1,g.col_index.values-1]=g.response.values
    if np.isnan(M).any(): return None
    z=S.logit(M.ravel())
    Bnew=B_TARGET-200
    s2,k_new,nfin=boot_counts(geom,z,w,Bnew,seed_for(eid))
    k200=int(round(p_old*201)-1); tb=200+nfin
    return dict(experiment_id=eid,n_i=gs,n_j=gs,sir_S2=s2,s2_prod=s2_prod,
                total_B=tb,total_exceed=k200+k_new,p_pooled=(1+k200+k_new)/(tb+1),
                s2_ratio=s2/s2_prod if s2_prod else np.nan)

def bh(p):
    p=np.asarray(p); n=len(p); o=np.argsort(p); q=np.empty(n)
    r=p[o]*n/np.arange(1,n+1); q[o]=np.clip(np.minimum.accumulate(r[::-1])[::-1],0,1); return q

def main():
    cov=pd.read_csv("theory/filter_covariates.csv")
    cov=cov[(cov.cell_line!="not_available")]
    cand=cov[cov.sir_p_value<=0.05].copy()
    done=set()
    if os.path.exists(OUT):
        prev=pd.read_parquet(OUT); done=set(prev.experiment_id)
        log(f"RESUMING: {len(done)} already done")
    todo=cand[~cand.experiment_id.isin(done)]
    recs=list(zip(todo.experiment_id,todo.n_i.astype(int),todo.sir_p_value,todo.sir_S2))
    log(f"START B_TARGET={B_TARGET} workers={NWORK} candidates={len(cand)} todo={len(recs)}")
    t0=time.time(); buf=[]; total=len(done)
    with Pool(NWORK, initializer=_init, initargs=("data/processed/drugcomb_matrices.parquet",)) as pool:
        for i,r in enumerate(pool.imap_unordered(work, recs, chunksize=20)):
            if r: buf.append(r)
            if len(buf)>=CHUNK:
                _flush(buf); total+=len(buf); 
                log(f"chunk: {total} done, {time.time()-t0:.0f}s, max|s2ratio-1|={max(abs(x['s2_ratio']-1) for x in buf):.1e}")
                buf=[]
    if buf: _flush(buf); total+=len(buf)
    log(f"COMPUTE DONE: {total} matrices, {time.time()-t0:.0f}s")
    _finalize(cov)

def _flush(buf):
    df=pd.DataFrame(buf)
    if os.path.exists(OUT):
        df=pd.concat([pd.read_parquet(OUT),df],ignore_index=True)
    tmp=OUT+".tmp"; pq.write_table(pa.Table.from_pandas(df),tmp); os.replace(tmp,OUT)
    json.dump({"done":len(df),"B_TARGET":B_TARGET,"workers":NWORK,"ts":time.time()},open(SIDE,"w"))

def _finalize(cov):
    adp=pd.read_parquet(OUT)[["experiment_id","p_pooled","total_B"]]
    m=cov.merge(adp,on="experiment_id",how="left")
    m["p_final"]=m.p_pooled.fillna(m.sir_p_value)  # non-candidates keep old p
    hits=0; per={}
    for cl,sub in m.groupby("cell_line"):
        q=bh(sub.p_final.values); h=int((q<=0.05).sum())
        if h>0: per[cl]=h; hits+=h
    log(f"FINAL per-cell-line FDR hits @0.05: total={hits} (baseline was 2686). breakdown={per}")
    print("TOTAL HITS:",hits,"baseline 2686"); print("breakdown:",per)
    m[["experiment_id","cell_line","p_final","p_pooled","total_B"]].to_parquet("results/drugcombdb_sir_adaptive_final.parquet")

if __name__=="__main__": main()
