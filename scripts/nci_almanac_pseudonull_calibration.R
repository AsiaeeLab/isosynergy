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
  library(ggplot2)
  library(future)
  library(future.apply)
})

source("R/pipeline.R")
source("R/config.R")

option_list <- list(
  make_option("--config", type = "character", default = "configs/default.yaml",
              help = "YAML config path", metavar = "path"),
  make_option("--data", type = "character", default = NA_character_,
              help = "NCI-ALMANAC standardized long parquet (defaults to config.data.standard_files.nci_almanac)", metavar = "path"),
  make_option("--grid-summary", type = "character", default = "results/nci_almanac_grid_summary.parquet",
              help = "Parquet with per-experiment grid stats", metavar = "path"),
  make_option("--n", type = "integer", default = 200L,
              help = "Number of matrices for pseudo-null calibration", metavar = "int"),
  make_option("--boot-B", type = "integer", default = 200L,
              help = "Bootstrap iterations per matrix", metavar = "int"),
  make_option("--seed", type = "integer", default = 42L,
              help = "Random seed", metavar = "int"),
  make_option("--n-cores", type = "integer", default = 8L,
              help = "Parallel workers across matrices", metavar = "int"),
  make_option("--out", type = "character", default = "results/nci_almanac_pseudonull_pvals.parquet",
              help = "Output parquet path", metavar = "path"),
  make_option("--out-fig", type = "character", default = "figures/public/pseudonull_calibration_nci_almanac.pdf",
              help = "Output figure PDF", metavar = "path")
)

opt <- parse_args(OptionParser(option_list = option_list))
set.seed(opt$seed)

cfg <- read_config(opt$config)
transform <- make_transform(cfg$transform)

data_path <- opt$data
if (is.na(data_path) || !nzchar(data_path)) {
  data_path <- (cfg$data$standard_files %||% list())$nci_almanac
}
if (is.null(data_path) || !file.exists(data_path)) stop("Missing NCI-ALMANAC data parquet: ", data_path %||% "<null>")
if (!file.exists(opt$`grid-summary`)) stop("Missing grid summary: ", opt$`grid-summary`)

out_results <- cfg$project$out_dir_results %||% "results"
out_figures <- cfg$project$out_dir_figures %||% "figures"
fig_dir <- file.path(out_figures, "public")
dir_create(out_results)
dir_create(fig_dir)

grid <- as.data.table(read_parquet(opt$`grid-summary`))
req <- c("experiment_id", "has_monoA", "has_monoB", "n_i", "n_j")
miss <- setdiff(req, names(grid))
if (length(miss) > 0) stop("grid-summary missing columns: ", paste(miss, collapse = ", "))

grid_ok <- grid[
  !is.na(has_monoA) & has_monoA &
    !is.na(has_monoB) & has_monoB &
    is.finite(n_i) & is.finite(n_j) & n_i >= 3 & n_j >= 3
]
if (nrow(grid_ok) == 0) stop("No eligible experiments after filtering for mono edges and min grid.")

n_target <- min(opt$n, nrow(grid_ok))
ids <- grid_ok[sample.int(nrow(grid_ok), n_target)]$experiment_id
message("[", now_utc(), "] Pseudo-null matrices: ", length(ids),
        " | bootstrap B: ", opt$`boot-B`,
        " | residual_scale: ", cfg$bootstrap$residual_scale %||% "none",
        " | df_null override: ", cfg$bootstrap$df_null %||% "<auto>")

ds <- arrow::open_dataset(data_path)
dt_sel <- as.data.table(ds |>
  dplyr::filter(experiment_id %in% ids) |>
  dplyr::collect())
if (nrow(dt_sel) == 0) stop("No rows loaded for selected experiment_ids.")
dt_list <- split(dt_sel, by = "experiment_id", keep.by = TRUE)

scale_residuals_df <- function(r, w, fit_add, df_null = NULL, df_tol = 1e-8) {
  n_distinct_runs <- function(x, tol) {
    if (length(x) == 0) return(0L)
    1L + sum(abs(diff(x)) > tol)
  }

  wv <- as.numeric(w)
  rv <- as.numeric(r)
  ok <- is.finite(wv) & is.finite(rv) & (wv > 0)
  n_eff <- sum(ok)
  if (n_eff <= 1) return(r)
  if (is.null(df_null)) {
    Ku <- n_distinct_runs(fit_add$u, df_tol)
    Kv <- n_distinct_runs(fit_add$v, df_tol)
    df_null <- Ku + Kv - 1L
  }
  df_null <- max(0, min(as.integer(df_null), n_eff - 1L))
  scale <- sqrt(n_eff / max(1, n_eff - df_null))
  r * scale
}

pseudonull_one <- function(df, idx) {
  df <- df[is.finite(response) & is.finite(doseA) & is.finite(doseB)]
  exp_id <- df$experiment_id[1]
  seed_exp <- opt$seed + (sum(utf8ToInt(exp_id)) %% 100000) + idx
  set.seed(seed_exp)

  ms <- summarise_matrix(
    df_long = df,
    transform = transform,
    response_mode = cfg$responses$mode,
    method_direction = cfg$method$monotone_direction,
    tau = cfg$weights$tau,
    winsor_q = cfg$weights$winsor_q
  )

  osqp_pars <- cfg$method$osqp %||% list()
  add_fit <- additive_ordered_fit(ms$barZ, ms$w, direction = ms$direction, osqp_pars = osqp_pars)
  r <- ms$barZ - add_fit$theta

  r_scaled <- r
  if ((cfg$bootstrap$residual_scale %||% "none") == "df") {
    r_scaled <- scale_residuals_df(
      r = r,
      w = ms$w,
      fit_add = add_fit,
      df_null = cfg$bootstrap$df_null %||% NULL,
      df_tol = cfg$bootstrap$df_tol %||% 1e-8
    )
  }

  xi <- matrix(sample(c(-1, 1), length(r), replace = TRUE), nrow = nrow(r), ncol = ncol(r))
  Z_star <- add_fit$theta + xi * r_scaled

  synergy_sign <- cfg$interaction$synergy_sign %||% "auto"
  if (synergy_sign == "auto") {
    synergy_sign <- infer_synergy_sign(cfg$responses$mode, ms$direction)
  }

  boot <- wild_bootstrap(
    barZ = Z_star,
    w = ms$w,
    direction = ms$direction,
    B = opt$`boot-B`,
    seed = seed_exp,
    osqp_pars = osqp_pars,
    n_cores = 1,
    use_parallel = FALSE,
    stat = cfg$bootstrap$stat %||% "S2",
    synergy_sign = synergy_sign,
    threshold = cfg$interaction$threshold %||% 0,
    residual_scale = cfg$bootstrap$residual_scale %||% "none",
    df_null = cfg$bootstrap$df_null %||% NULL,
    df_tol = cfg$bootstrap$df_tol %||% 1e-8
  )

  data.table(
    experiment_id = exp_id,
    p_value = boot$p_value,
    df_null = boot$df_null
  )
}

workers <- max(1L, min(opt$`n-cores`, as.integer(future::availableCores(methods = "mc.cores"))))
oplan <- future::plan()
on.exit(future::plan(oplan), add = TRUE)
if (workers > 1) future::plan(future::multisession, workers = workers) else future::plan(future::sequential)

res <- future_lapply(seq_along(dt_list), function(i) pseudonull_one(dt_list[[i]], i), future.seed = TRUE)
pseudo_dt <- rbindlist(res, fill = TRUE)

write_table(pseudo_dt, opt$out)
message("[", now_utc(), "] Wrote: ", opt$out, " (n=", nrow(pseudo_dt), ")")

if (nrow(pseudo_dt) > 0) {
  m <- nrow(pseudo_dt)
  ecdf_band <- function(p) {
    eps <- sqrt(log(2 / 0.05) / (2 * m))
    data.table(p = p, lo = pmax(0, p - eps), hi = pmin(1, p + eps))
  }
  grid_p <- seq(0, 1, length.out = 200)
  band <- ecdf_band(grid_p)
  p_cal <- ggplot(pseudo_dt[is.finite(p_value)], aes(x = p_value)) +
    stat_ecdf(geom = "step", color = "#0C6291") +
    geom_ribbon(data = band, aes(x = p, ymin = lo, ymax = hi), fill = "#F28C28", alpha = 0.2, inherit.aes = FALSE) +
    geom_abline(intercept = 0, slope = 1, linetype = "dashed") +
    labs(
      title = "Pseudo-null calibration (NCI-ALMANAC)",
      subtitle = paste0("n=", nrow(pseudo_dt), ", B=", opt$`boot-B`, ", residual_scale=", cfg$bootstrap$residual_scale %||% "none"),
      x = "p-value",
      y = "ECDF"
    ) +
    theme_minimal(base_size = 10)

  dir_create(dirname(opt$`out-fig`))
  ggsave(opt$`out-fig`, p_cal, width = 5, height = 4)
  message("[", now_utc(), "] Wrote: ", opt$`out-fig`)
}

message("[", now_utc(), "] Done.")
