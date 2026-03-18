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

source("R/utils.R")
source("R/config.R")
source("R/transforms.R")
source("R/matrix_stats.R")
source("R/osqp_helpers.R")
source("R/isotonic_2d.R")
source("R/additive_ordered.R")
source("R/interaction.R")
source("R/bootstrap.R")

option_list <- list(
  make_option("--config", type = "character", default = "configs/default.yaml",
              help = "YAML config path", metavar = "path"),
  make_option("--source", type = "character", default = "drugcombdb",
              help = "Dataset source key in config.data.standard_files", metavar = "name"),
  make_option("--pseudonull-n", type = "integer", default = 300L,
              help = "Number of matrices for pseudo-null calibration", metavar = "int"),
  make_option("--boot-B", type = "integer", default = 200L,
              help = "Bootstrap iterations per matrix", metavar = "int"),
  make_option("--seed", type = "integer", default = 42L,
              help = "Random seed", metavar = "int"),
  make_option("--n-cores", type = "integer", default = 8L,
              help = "Parallel workers across matrices", metavar = "int")
)

opt <- parse_args(OptionParser(option_list = option_list))
set.seed(opt$seed)

cfg <- read_config(opt$config)
transform <- make_transform(cfg$transform)

standard_path <- (cfg$data$standard_files %||% list())[[opt$source]]
if (is.null(standard_path) || !file.exists(standard_path)) {
  stop("Missing standard file for source '", opt$source, "': ", standard_path %||% "<null>")
}

out_results <- cfg$project$out_dir_results %||% "results"
out_figures <- cfg$project$out_dir_figures %||% "figures"
fig_dir <- file.path(out_figures, "public")
dir_create(out_results)
dir_create(fig_dir)

message("[", now_utc(), "] Reading standard long file: ", standard_path)
dt_all <- as.data.table(read_parquet(standard_path))

if (!"experiment_id" %in% names(dt_all)) stop("Expected column 'experiment_id' in: ", standard_path)

exp_ids <- unique(dt_all$experiment_id)
if (length(exp_ids) == 0) stop("No experiments found in: ", standard_path)
target_n <- min(opt$`pseudonull-n`, length(exp_ids))
exp_ids <- if (length(exp_ids) <= target_n) exp_ids else sample(exp_ids, target_n)

message("[", now_utc(), "] Pseudo-null matrices: ", length(exp_ids),
        " | bootstrap B: ", opt$`boot-B`,
        " | residual_scale: ", cfg$bootstrap$residual_scale %||% "none",
        " | df_null override: ", cfg$bootstrap$df_null %||% "<auto>")

dt_list <- split(dt_all[experiment_id %in% exp_ids], by = "experiment_id", keep.by = TRUE)

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

  xi <- matrix(sample(c(-1, 1), length(r_scaled), replace = TRUE), nrow = nrow(r_scaled), ncol = ncol(r_scaled))
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

oplan <- future::plan()
on.exit(future::plan(oplan), add = TRUE)
workers <- max(1L, min(opt$`n-cores`, as.integer(future::availableCores(methods = "mc.cores"))))
future::plan(future::multisession, workers = workers)

res <- future_lapply(seq_along(dt_list), function(i) pseudonull_one(dt_list[[i]], i), future.seed = TRUE)
pseudo_dt <- rbindlist(res, fill = TRUE)

out_p <- file.path(out_results, "public_pseudonull_pvals.parquet")
write_table(pseudo_dt, out_p)

message("[", now_utc(), "] Wrote: ", out_p, " (n=", nrow(pseudo_dt), ")")

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
    labs(title = "Pseudo-null p-value distribution", x = "p-value", y = "ECDF") +
    theme_minimal(base_size = 10)

  out_fig <- file.path(fig_dir, "pseudonull_calibration.pdf")
  ggsave(out_fig, p_cal, width = 5, height = 4)
  message("[", now_utc(), "] Wrote: ", out_fig)
}

message("[", now_utc(), "] Done.")

