#!/usr/bin/env Rscript

get_script_path <- function() {
  cmd <- commandArgs(trailingOnly = FALSE)
  file_arg <- grep("^--file=", cmd, value = TRUE)
  if (length(file_arg) == 0) return(NULL)
  sub("^--file=", "", file_arg[1])
}

script_path <- get_script_path()
if (!is.null(script_path)) {
  root <- normalizePath(file.path(dirname(script_path), ".."), winslash = "/", mustWork = FALSE)
  if (file.exists(file.path(root, "renv.lock"))) setwd(root)
}

suppressPackageStartupMessages({
  library(optparse)
  library(data.table)
  library(arrow)
})

source("R/pipeline.R")
source("R/config.R")

option_list <- list(
  make_option("--config", type = "character", default = "configs/default.yaml",
              help = "YAML config path", metavar = "path"),
  make_option("--grid-summary", type = "character", default = "results/drugcombdb_grid_summary.parquet",
              help = "Parquet with per-experiment grid sizes", metavar = "path"),
  make_option("--data", type = "character", default = "data/processed/drugcomb_matrices.parquet",
              help = "DrugCombDB standardized long parquet", metavar = "path"),
  make_option("--target-sizes", type = "character", default = "6x6,8x8,10x10",
              help = "Comma-separated grid sizes to try (e.g., 6x6,8x8,10x10)", metavar = "sizes"),
  make_option("--n-total", type = "integer", default = 5L,
              help = "Total matrices to benchmark (fills from common sizes if targets unavailable)", metavar = "int"),
  make_option("--B", type = "integer", default = 200L,
              help = "Bootstrap replicates per matrix", metavar = "int"),
  make_option("--B-big", type = "integer", default = 1000L,
              help = "Bootstrap replicates for one matrix (scaling demo)", metavar = "int"),
  make_option("--screen-n", type = "integer", default = 400000L,
              help = "Number of matrices in a full screen (for extrapolation)", metavar = "int"),
  make_option("--screen-cores", type = "integer", default = 8L,
              help = "Cores assumed for screen-time extrapolation", metavar = "int"),
  make_option("--seed", type = "integer", default = 42L,
              help = "Random seed for selection", metavar = "int"),
  make_option("--out", type = "character", default = "results/runtime_benchmark.csv",
              help = "Output CSV path", metavar = "path")
)

opt <- parse_args(OptionParser(option_list = option_list))
set.seed(opt$seed)

cfg <- read_config(opt$config)
transform <- make_transform(cfg$transform)

if (!file.exists(opt$`grid-summary`)) stop("Missing grid summary: ", opt$`grid-summary`)
if (!file.exists(opt$data)) stop("Missing data file: ", opt$data)

grid <- as.data.table(read_parquet(opt$`grid-summary`))
if (!all(c("experiment_id", "n_i", "n_j") %in% names(grid))) {
  stop("grid-summary must contain: experiment_id, n_i, n_j")
}

parse_size <- function(s) {
  s <- trimws(s)
  parts <- strsplit(s, "x", fixed = TRUE)[[1]]
  if (length(parts) != 2) return(NULL)
  ij <- suppressWarnings(as.integer(parts))
  if (any(is.na(ij))) return(NULL)
  list(i = ij[1], j = ij[2], label = s)
}

pick_ids_for_size <- function(grid, i, j, n_pick, used) {
  sub <- grid[n_i == i & n_j == j & !(experiment_id %in% used)]
  if (nrow(sub) == 0) return(character(0))
  n_take <- min(n_pick, nrow(sub))
  sub[sample.int(nrow(sub), n_take)]$experiment_id
}

targets <- Filter(Negate(is.null), lapply(strsplit(opt$`target-sizes`, ",", fixed = TRUE)[[1]], parse_size))
picked <- character(0)
notes <- character(0)

for (t in targets) {
  ids <- pick_ids_for_size(grid, t$i, t$j, n_pick = 1L, used = picked)
  if (length(ids) == 0) {
    notes <- c(notes, paste0("target size missing: ", t$label))
  } else {
    picked <- c(picked, ids)
  }
  if (length(picked) >= opt$`n-total`) break
}

if (length(picked) < opt$`n-total`) {
  counts <- grid[, .N, by = .(n_i, n_j)][order(-N)]
  for (k in seq_len(nrow(counts))) {
    i <- counts$n_i[k]; j <- counts$n_j[k]
    if (!is.finite(i) || !is.finite(j)) next
    ids <- pick_ids_for_size(grid, i, j, n_pick = opt$`n-total` - length(picked), used = picked)
    picked <- c(picked, ids)
    if (length(picked) >= opt$`n-total`) break
  }
}

picked <- unique(picked)
if (length(picked) == 0) stop("No experiments selected for benchmarking.")

message("[", now_utc(), "] Selected experiments: ", length(picked))
if (length(notes) > 0) message("[note] ", paste(unique(notes), collapse = " | "))

ds <- arrow::open_dataset(opt$data)
dt_sel <- as.data.table(ds |>
  dplyr::filter(experiment_id %in% picked) |>
  dplyr::collect())
if (nrow(dt_sel) == 0) stop("No rows loaded for selected experiment_ids.")
dt_list <- split(dt_sel, by = "experiment_id", keep.by = TRUE)

run_one <- function(df, B) {
  df <- df[is.finite(response) & is.finite(doseA) & is.finite(doseB)]
  exp_id <- df$experiment_id[1]

  t_sum <- system.time({
    ms <- summarise_matrix(
      df_long = df,
      transform = transform,
      response_mode = cfg$responses$mode,
      method_direction = cfg$method$monotone_direction,
      tau = cfg$weights$tau,
      winsor_q = cfg$weights$winsor_q
    )
  })[["elapsed"]]

  osqp_pars <- cfg$method$osqp %||% list()
  synergy_sign <- cfg$interaction$synergy_sign %||% "auto"
  if (synergy_sign == "auto") {
    synergy_sign <- infer_synergy_sign(cfg$responses$mode, ms$direction)
  }
  threshold <- cfg$interaction$threshold %||% 0

  t_fit <- system.time({
    invisible(interaction_fit(
      barZ = ms$barZ,
      w = ms$w,
      direction = ms$direction,
      osqp_pars = osqp_pars,
      synergy_sign = synergy_sign,
      threshold = threshold
    ))
  })[["elapsed"]]

  t_boot <- system.time({
    invisible(wild_bootstrap(
      barZ = ms$barZ,
      w = ms$w,
      direction = ms$direction,
      B = B,
      seed = cfg$bootstrap$seed %||% 1,
      osqp_pars = osqp_pars,
      n_cores = 1,
      use_parallel = FALSE,
      stat = cfg$bootstrap$stat %||% "S2",
      synergy_sign = synergy_sign,
      threshold = threshold,
      residual_scale = cfg$bootstrap$residual_scale %||% "none",
      df_null = cfg$bootstrap$df_null %||% NULL,
      df_tol = cfg$bootstrap$df_tol %||% 1e-8
    ))
  })[["elapsed"]]

  total <- t_sum + t_fit + t_boot
  data.table(
    experiment_id = exp_id,
    n_i = ms$I,
    n_j = ms$J,
    B = as.integer(B),
    time_summarise_s = t_sum,
    time_fit_s = t_fit,
    time_bootstrap_s = t_boot,
    time_total_s = total,
    time_per_boot_s = t_boot / max(1, B)
  )
}

rows <- list()
for (id in names(dt_list)) {
  message("[", now_utc(), "] Timing ", id, " (B=", opt$B, ")")
  rows[[length(rows) + 1L]] <- run_one(dt_list[[id]], B = opt$B)
}

big_id <- names(dt_list)[1]
message("[", now_utc(), "] Timing scaling run ", big_id, " (B=", opt$`B-big`, ")")
rows[[length(rows) + 1L]] <- run_one(dt_list[[big_id]], B = opt$`B-big`)

res <- rbindlist(rows, fill = TRUE)
res[, grid_size := paste0(n_i, "x", n_j)]

screen_n <- as.numeric(opt$`screen-n`)
screen_cores <- max(1, as.numeric(opt$`screen-cores`))
res[, estimated_screen_time_s := time_total_s * screen_n / screen_cores]
res[, estimated_screen_time_hours := estimated_screen_time_s / 3600]

dir_create(dirname(opt$out))
fwrite(res, opt$out)
message("[", now_utc(), "] Wrote: ", opt$out)

fmt_hours <- function(h) {
  if (!is.finite(h)) return(NA_character_)
  if (h < 1) return(sprintf("%.1f min", 60 * h))
  if (h < 48) return(sprintf("%.1f h", h))
  sprintf("%.1f days", h / 24)
}

print(res[order(n_i, n_j, B), .(
  grid_size,
  B,
  time_total_s = round(time_total_s, 3),
  time_per_boot_s = signif(time_per_boot_s, 3),
  estimated_screen_time = vapply(estimated_screen_time_hours, fmt_hours, character(1))
)])
