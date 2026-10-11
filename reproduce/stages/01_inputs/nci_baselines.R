script <- sub("^--file=", "", grep("^--file=",commandArgs(FALSE),value=TRUE)[1])
root <- normalizePath(file.path(dirname(script), "..", ".."))
repo <- dirname(root)
output <- file.path(root, "output")
smoke <- Sys.getenv("SIR_SMOKE", "0") == "1"
if (smoke) output <- file.path(output, "smoke")
inputs <- file.path(output, "inputs")
suppressPackageStartupMessages({library(data.table); library(arrow)})
data.table::setDTthreads(1L)
source(file.path(root, "lib", "nci_ingest.R"))
source(file.path(root, "lib", "baseline_scores.R"))
raw <- file.path(repo, "data", "raw", "nci_almanac")
nci <- standardize_nci_almanac(file.path(raw,"ComboDrugGrowth_Nov2017.csv"),
  cell_map_path=file.path(raw,"NCI60_CELLNAME_to_Combo.txt"),
  drug_map_path=file.path(raw,"NCI_IOA_AOA_drugs.tsv"))
write_parquet(nci$grid,file.path(inputs,"nci_grid.parquet"))
grid_ok <- nci$grid[has_monoA & has_monoB & is.finite(n_i) & is.finite(n_j) & n_i >= 3 & n_j >= 3]
set.seed(42L)
ids <- grid_ok[sample.int(nrow(grid_ok),min(20000L,nrow(grid_ok)))]$experiment_id
if (smoke) ids <- ids[seq_len(min(100L,length(ids)))]
write.csv(data.frame(experiment_id=ids),file.path(inputs,"nci_sample.csv"),row.names=FALSE)
dt_list <- split(nci$long[experiment_id %in% ids],by="experiment_id",keep.by=TRUE)
interior_mean <- function(M) {
  if (!is.matrix(M) || nrow(M)<2 || ncol(M)<2) return(NA_real_)
  mean(M[-1,-1],na.rm=TRUE)
}
compute_one <- function(df) {
  df <- df[is.finite(response) & is.finite(doseA) & is.finite(doseB)]
  data.table(experiment_id=df$experiment_id[1],
    Bliss=interior_mean(bliss_synergy(df,"viability")$synergy),
    HSA=interior_mean(hsa_synergy(df,"viability")$synergy),
    Loewe=interior_mean(loewe_synergy(df,"viability","decreasing")$synergy),
    ZIP=interior_mean(zip_synergy(df,"viability",TRUE)$synergy))
}
workers <- max(1L,min(16L,as.integer(Sys.getenv("SIR_WORKERS","16"))))
scores <- rbindlist(parallel::mclapply(dt_list,compute_one,mc.cores=workers,mc.set.seed=FALSE))
write_parquet(scores,file.path(inputs,"nci_almanac_baseline_scores.parquet"))
cat("NCI baseline matrices:",nrow(scores),"\n")
