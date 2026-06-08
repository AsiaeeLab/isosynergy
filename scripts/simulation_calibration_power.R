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
  library(patchwork)
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
source("R/simulation.R")

option_list <- list(
  make_option("--config", type = "character", default = "configs/default.yaml",
              help = "YAML config path (used for transform defaults and output dirs)", metavar = "path"),
  make_option("--seed", type = "integer", default = 42L,
              help = "Random seed", metavar = "int"),
  make_option("--I", type = "integer", default = 8L,
              help = "Grid rows (drug A doses)", metavar = "int"),
  make_option("--J", type = "integer", default = 8L,
              help = "Grid cols (drug B doses)", metavar = "int"),
  make_option("--noise-sigma", type = "double", default = 0.1,
              help = "Noise SD on transformed Z-scale", metavar = "num"),
  make_option("--n-null", type = "integer", default = 200L,
              help = "Number of null simulations for calibration", metavar = "int"),
  make_option("--n-power", type = "integer", default = 30L,
              help = "Simulations per delta for power curve", metavar = "int"),
  make_option("--delta-values", type = "character", default = "0,0.8,1.2,1.4,1.6,1.8",
              help = "Comma-separated interaction strengths (bump amplitudes)", metavar = "list"),
  make_option("--boot-B", type = "integer", default = 200L,
              help = "Bootstrap replicates per simulation", metavar = "int"),
  make_option("--replot-only", action = "store_true", default = FALSE,
              help = "Skip recompute; read precomputed null/power parquet and only redraw the figure"),
  make_option("--out-fig", type = "character", default = NA_character_,
              help = "Output figure PDF (defaults to <figures>/public/simulation_calibration_power.pdf)", metavar = "path")
)

opt <- parse_args(OptionParser(option_list = option_list))
set.seed(opt$seed)

cfg <- read_config(opt$config)
transform <- make_transform(cfg$transform)

out_results <- cfg$project$out_dir_results %||% "results"
out_figures <- cfg$project$out_dir_figures %||% "figures"
fig_dir <- file.path(out_figures, "public")
dir_create(out_results)
dir_create(fig_dir)

delta_values <- as.numeric(strsplit(opt$`delta-values`, ",", fixed = TRUE)[[1]])
delta_values <- delta_values[is.finite(delta_values)]
if (length(delta_values) == 0) stop("No valid --delta-values provided.")

# DKW band for uniform(0,1)
ecdf_band <- function(p, alpha = 0.05, m) {
  eps <- sqrt(log(2 / alpha) / (2 * m))
  data.table(p = p, lo = pmax(0, p - eps), hi = pmin(1, p + eps))
}
alpha_test <- 0.05

if (opt$`replot-only`) {
  null_p <- file.path(out_results, "simulation_null_pvals.parquet")
  power_p <- file.path(out_results, "simulation_power_pvals.parquet")
  if (!file.exists(null_p)) stop("Missing: ", null_p)
  if (!file.exists(power_p)) stop("Missing: ", power_p)
  null_dt <- as.data.table(read_parquet(null_p))
  power_dt <- as.data.table(read_parquet(power_p))
  message("[", now_utc(), "] Replot-only from: ", null_p, " and ", power_p)

  m <- nrow(null_dt)
  grid_p <- seq(0, 1, length.out = 200)
  band <- ecdf_band(grid_p, alpha = 0.05, m = m)

  p_cal <- ggplot(null_dt, aes(x = p_value)) +
    stat_ecdf(geom = "step", color = "#0C6291") +
    geom_ribbon(data = band, aes(x = p, ymin = lo, ymax = hi), fill = "#F28C28", alpha = 0.2, inherit.aes = FALSE) +
    geom_abline(intercept = 0, slope = 1, linetype = "dashed") +
    labs(title = "Null p-value distribution", x = "p-value", y = "ECDF") +
    theme_minimal(base_size = 11)

  pow_summ <- power_dt[, {
    p <- mean(p_value <= alpha_test, na.rm = TRUE)
    .(power = p, se = sqrt(p * (1 - p) / .N),
      true_max_abs_delta = mean(true_max_abs_delta, na.rm = TRUE), n = .N)
  }, by = .(delta)]

  p_pow <- ggplot(pow_summ, aes(x = delta, y = power)) +
    geom_hline(yintercept = alpha_test, linetype = "dashed", color = "gray50") +
    geom_ribbon(aes(ymin = pmax(0, power - 1.96 * se), ymax = pmin(1, power + 1.96 * se)),
                fill = "steelblue", alpha = 0.2) +
    geom_line(color = "steelblue", linewidth = 1) +
    geom_point(color = "steelblue", size = 2) +
    scale_y_continuous(limits = c(0, 1), labels = scales::percent) +
    labs(title = "Power curve", x = expression("Interaction strength (" * delta * ")"), y = "Power") +
    theme_minimal(base_size = 11)

  out_fig <- if (is.na(opt$`out-fig`)) file.path(fig_dir, "simulation_calibration_power.pdf") else opt$`out-fig`
  dir_create(dirname(out_fig))
  # Two panels side by side (landscape) for panel (B) of the two-up float.
  ggsave(out_fig, p_cal | p_pow, width = 11, height = 4.4)
  message("[", now_utc(), "] Wrote: ", out_fig)
  message("[", now_utc(), "] Done.")
  quit(save = "no", status = 0)
}

response_mode <- "viability"
direction <- "decreasing"
synergy_sign <- infer_synergy_sign(response_mode, direction)
threshold <- 0
stat_choice <- cfg$bootstrap$stat %||% "S2"

boot_residual_scale <- cfg$bootstrap$residual_scale %||% "df"
boot_df_null <- cfg$bootstrap$df_null %||% NULL
boot_df_tol <- cfg$bootstrap$df_tol %||% 1e-8

osqp_pars <- cfg$method$osqp %||% list(eps_abs = 1e-8, eps_rel = 1e-8, max_iter = 20000)

message("[", now_utc(), "] Simulation calibration/power")
message("  Grid: ", opt$I, "x", opt$J,
        " | sigma=", opt$`noise-sigma`,
        " | B=", opt$`boot-B`,
        " | stat=", stat_choice,
        " | residual_scale=", boot_residual_scale)

# -------------------------
# Null calibration
# -------------------------
p_values_null <- numeric(opt$`n-null`)
for (sim in seq_len(opt$`n-null`)) {
  set.seed(opt$seed + 20000 + sim)
  surfaces_null <- simulate_true_surfaces(
    I = opt$I, J = opt$J,
    interaction_strength = 0,
    response_mode = response_mode,
    interaction_mode = "projected",
    interaction_sign = "auto"
  )
  Z_true <- surfaces_null$theta_iso_true
  set.seed(opt$seed + 21000 + sim)
  Z_obs <- as.numeric(Z_true) + rnorm(opt$I * opt$J, 0, opt$`noise-sigma`)
  Z_obs_mat <- matrix(Z_obs, opt$I, opt$J)
  w <- matrix(1, opt$I, opt$J)

  boot_result <- wild_bootstrap(
    barZ = Z_obs_mat,
    w = w,
    direction = direction,
    B = opt$`boot-B`,
    seed = opt$seed + sim,
    osqp_pars = osqp_pars,
    n_cores = 1,
    use_parallel = FALSE,
    stat = stat_choice,
    synergy_sign = synergy_sign,
    threshold = threshold,
    residual_scale = boot_residual_scale,
    df_null = boot_df_null,
    df_tol = boot_df_tol
  )

  p_values_null[sim] <- boot_result$p_value
  if (sim %% 25 == 0) message("  [null] completed ", sim, "/", opt$`n-null`)
}

null_dt <- data.table(sim = seq_len(opt$`n-null`), delta = 0, p_value = p_values_null)
write_table(null_dt, file.path(out_results, "simulation_null_pvals.parquet"))

# DKW band for uniform(0,1)
ecdf_band <- function(p, alpha = 0.05, m) {
  eps <- sqrt(log(2 / alpha) / (2 * m))
  data.table(p = p, lo = pmax(0, p - eps), hi = pmin(1, p + eps))
}

m <- length(p_values_null)
grid_p <- seq(0, 1, length.out = 200)
band <- ecdf_band(grid_p, alpha = 0.05, m = m)

p_cal <- ggplot(null_dt, aes(x = p_value)) +
  stat_ecdf(geom = "step", color = "#0C6291") +
  geom_ribbon(data = band, aes(x = p, ymin = lo, ymax = hi), fill = "#F28C28", alpha = 0.2, inherit.aes = FALSE) +
  geom_abline(intercept = 0, slope = 1, linetype = "dashed") +
  labs(title = "Null p-value distribution", x = "p-value", y = "ECDF") +
  theme_minimal(base_size = 11)

# -------------------------
# Power curve
# -------------------------
power_rows <- list()
for (delta in delta_values) {
  if (!is.finite(delta) || delta <= 0) next
  message("  [power] delta=", delta)
  for (sim in seq_len(opt$`n-power`)) {
    # Fix nuisance surfaces across sims for each delta to isolate effect size.
    set.seed(opt$seed + 10000 + round(delta * 1000))
    surfaces_alt <- simulate_true_surfaces(
      I = opt$I, J = opt$J,
      interaction_strength = delta,
      response_mode = response_mode,
      interaction_mode = "projected",
      interaction_sign = "auto"
    )
    Z_true <- surfaces_alt$theta_iso_true
    set.seed(opt$seed + 21000 + sim + round(delta * 1000))
    Z_obs <- as.numeric(Z_true) + rnorm(opt$I * opt$J, 0, opt$`noise-sigma`)
    Z_obs_mat <- matrix(Z_obs, opt$I, opt$J)
    w <- matrix(1, opt$I, opt$J)

    boot_result <- wild_bootstrap(
      barZ = Z_obs_mat,
      w = w,
      direction = direction,
      B = opt$`boot-B`,
      seed = opt$seed + sim + round(delta * 1000),
      osqp_pars = osqp_pars,
      n_cores = 1,
      use_parallel = FALSE,
      stat = stat_choice,
      synergy_sign = synergy_sign,
      threshold = threshold,
      residual_scale = boot_residual_scale,
      df_null = boot_df_null,
      df_tol = boot_df_tol
    )

    power_rows[[length(power_rows) + 1L]] <- data.table(
      delta = delta,
      sim = sim,
      p_value = boot_result$p_value,
      t_obs = boot_result$t0,
      true_max_abs_delta = max(abs(surfaces_alt$delta_true))
    )
  }
}
power_dt <- rbindlist(power_rows, fill = TRUE)
write_table(power_dt, file.path(out_results, "simulation_power_pvals.parquet"))

alpha_test <- 0.05
pow_summ <- power_dt[, {
  p <- mean(p_value <= alpha_test, na.rm = TRUE)
  .(
    power = p,
    se = sqrt(p * (1 - p) / .N),
    true_max_abs_delta = mean(true_max_abs_delta, na.rm = TRUE),
    n = .N
  )
}, by = .(delta)]

p_pow <- ggplot(pow_summ, aes(x = delta, y = power)) +
  geom_hline(yintercept = alpha_test, linetype = "dashed", color = "gray50") +
  geom_ribbon(aes(ymin = pmax(0, power - 1.96 * se),
                  ymax = pmin(1, power + 1.96 * se)),
              fill = "steelblue", alpha = 0.2) +
  geom_line(color = "steelblue", linewidth = 1) +
  geom_point(color = "steelblue", size = 2) +
  scale_y_continuous(limits = c(0, 1), labels = scales::percent) +
  labs(title = "Power curve", x = expression("Interaction strength (" * delta * ")"), y = "Power") +
  theme_minimal(base_size = 11)

out_fig <- if (is.na(opt$`out-fig`)) file.path(fig_dir, "simulation_calibration_power.pdf") else opt$`out-fig`
dir_create(dirname(out_fig))
# Two panels side by side (landscape) for panel (B) of the two-up float.
ggsave(out_fig, p_cal | p_pow, width = 11, height = 4.4)
message("[", now_utc(), "] Wrote: ", out_fig)

message("[", now_utc(), "] Done.")
