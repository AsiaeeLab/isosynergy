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
source("R/baselines_common.R")
source("R/baselines_synergyfinder.R")
source("R/metrics.R")

option_list <- list(
  make_option("--config", type = "character", default = "configs/default.yaml",
              help = "YAML config path", metavar = "path"),
  make_option("--replicate-groups", type = "integer", default = 300L,
              help = "Target number of replicate groups", metavar = "int"),
  make_option("--max-reps", type = "integer", default = 6L,
              help = "Max replicates per group (downsample if larger)", metavar = "int"),
  make_option("--boot-B", type = "integer", default = 200L,
              help = "Bootstrap replicates for subset p-values", metavar = "int"),
  make_option("--boot-subset", type = "integer", default = 50L,
              help = "Number of experiments to bootstrap for p-values", metavar = "int"),
  make_option("--missing-n", type = "integer", default = 200L,
              help = "Number of matrices for missingness benchmark", metavar = "int"),
  make_option("--missing-frac", type = "double", default = 0.20,
              help = "Fraction of interior wells to hold out", metavar = "num"),
  make_option("--pseudonull-n", type = "integer", default = 300L,
              help = "Number of matrices for pseudo-null calibration", metavar = "int"),
  make_option("--seed", type = "integer", default = 42L,
              help = "Random seed", metavar = "int")
)

opt <- parse_args(OptionParser(option_list = option_list))
set.seed(opt$seed)

cfg <- read_config(opt$config)
transform <- make_transform(cfg$transform)
n_workers <- max(1L, min(cfg$project$n_cores %||% 1L, as.integer(future::availableCores(methods = "mc.cores"))))
cfg$project$n_cores <- n_workers

if (!synergyfinder_available()) {
  stop("SynergyFinder is required for baseline surfaces. Install via renv::install('hly89/synergyfinder').")
}

dir_create(cfg$project$out_dir_results %||% "results")
fig_dir <- file.path(cfg$project$out_dir_figures %||% "figures", "public")
dir_create(fig_dir)

mode_int <- function(x) {
  x <- x[is.finite(x)]
  if (length(x) == 0) return(NA_integer_)
  ux <- unique(x)
  ux[which.max(tabulate(match(x, ux)))]
}

align_surfaces <- function(s1, s2) {
  a <- intersect(s1$doseA, s2$doseA)
  b <- intersect(s1$doseB, s2$doseB)
  if (length(a) < 2 || length(b) < 2) return(NULL)
  i1 <- match(a, s1$doseA); j1 <- match(b, s1$doseB)
  i2 <- match(a, s2$doseA); j2 <- match(b, s2$doseB)
  list(
    v1 = as.numeric(s1$mat[i1, j1]),
    v2 = as.numeric(s2$mat[i2, j2])
  )
}

surface_compare <- function(orig, pert, eps = 1e-8) {
  aligned <- align_surfaces(orig, pert)
  if (is.null(aligned)) return(list(corr = NA_real_, rmse = NA_real_, rel_rmse = NA_real_, rms1 = NA_real_, rms2 = NA_real_))
  v1 <- aligned$v1; v2 <- aligned$v2
  ok <- is.finite(v1) & is.finite(v2)
  v1 <- v1[ok]; v2 <- v2[ok]
  if (length(v1) == 0) return(list(corr = NA_real_, rmse = NA_real_, rel_rmse = NA_real_, rms1 = NA_real_, rms2 = NA_real_))
  rms1 <- sqrt(mean(v1^2)); rms2 <- sqrt(mean(v2^2))
  corr <- if (rms1 > eps && rms2 > eps) suppressWarnings(stats::cor(v1, v2)) else NA_real_
  rmse <- sqrt(mean((v1 - v2)^2))
  rel_rmse <- rmse / max(eps, rms1)
  list(corr = corr, rmse = rmse, rel_rmse = rel_rmse, rms1 = rms1, rms2 = rms2)
}

topk_overlap <- function(m1, m2, k = 5) {
  v1 <- as.numeric(m1); v2 <- as.numeric(m2)
  ord1 <- order(-abs(v1), na.last = NA)
  ord2 <- order(-abs(v2), na.last = NA)
  idx1 <- head(ord1, k); idx2 <- head(ord2, k)
  if (length(idx1) == 0 || length(idx2) == 0) return(NA_real_)
  length(intersect(idx1, idx2)) / min(length(idx1), length(idx2))
}

standard_path <- cfg$data$standard_files$drugcombdb
if (is.null(standard_path) || !file.exists(standard_path)) {
  stop("Missing DrugCombDB standard file. Check configs/default.yaml data.standard_files.drugcombdb")
}
grid_path <- "results/drugcombdb_grid_summary.parquet"
if (!file.exists(grid_path)) stop("Missing grid summary: ", grid_path)

grid <- as.data.table(read_parquet(grid_path))
dt_all <- as.data.table(read_parquet(standard_path))

exclusions <- list()
ex_missing <- grid[is.na(cell_line) | !nzchar(cell_line), .(experiment_id, reason = "missing_cell_line")]
if (nrow(ex_missing) > 0) exclusions[[length(exclusions) + 1L]] <- ex_missing
ex_self <- grid[!is.na(drugA) & !is.na(drugB) & drugA == drugB, .(experiment_id, reason = "self_combo")]
if (nrow(ex_self) > 0) exclusions[[length(exclusions) + 1L]] <- ex_self
ex_dt <- if (length(exclusions) > 0) rbindlist(exclusions) else data.table()
if (nrow(ex_dt) > 0) {
  write_table(ex_dt, file.path(cfg$project$out_dir_results, "drugcombdb_exclusions.parquet"))
}

keep_ids <- setdiff(grid$experiment_id, ex_dt$experiment_id)
grid <- grid[experiment_id %in% keep_ids]
dt_all <- dt_all[experiment_id %in% keep_ids]

# -------------------------
# Replicate benchmark
# -------------------------
grp <- grid[, .(
  n = .N,
  n_i_mode = mode_int(n_i),
  n_j_mode = mode_int(n_j),
  mono_activity = mean(mono_activity, na.rm = TRUE)
), by = .(drugA, drugB, cell_line)]
grp <- grp[n >= 4]
if (nrow(grp) == 0) stop("No replicate groups with N>=4 after filtering.")

grp[, grid_stratum := paste(pmin(n_i_mode, 6L), pmin(n_j_mode, 6L), sep = "x")]
cuts <- quantile(grp$mono_activity, probs = c(0, 0.33, 0.66, 1), na.rm = TRUE, names = FALSE)
cuts <- unique(cuts)
if (length(cuts) < 2) cuts <- c(0, 1)
grp[, mono_stratum := cut(mono_activity, breaks = cuts, include.lowest = TRUE, labels = FALSE)]
grp[is.na(mono_stratum), mono_stratum := 1L]
grp[, stratum := paste(grid_stratum, mono_stratum, sep = "__")]

total_groups <- nrow(grp)
target_groups <- min(opt$`replicate-groups`, total_groups)
grp[, take := ceiling((.N / total_groups) * target_groups), by = stratum]
sampled_groups <- grp[, .SD[sample.int(.N, min(.N, take[1L]))], by = stratum]
if (nrow(sampled_groups) > target_groups) sampled_groups <- sampled_groups[sample.int(.N, target_groups)]

sampled_groups[, group_id := paste(drugA, drugB, cell_line, sep = "__")]
group_ids <- sampled_groups$group_id

grid_grp <- merge(
  grid,
  sampled_groups[, .(drugA, drugB, cell_line, group_id)],
  by = c("drugA", "drugB", "cell_line"),
  all.x = FALSE
)
grid_grp <- grid_grp[, .(experiment_id = if (.N > opt$`max-reps`) sample(experiment_id, opt$`max-reps`) else experiment_id),
                     by = .(group_id, drugA, drugB, cell_line)]

selected_ids <- unique(grid_grp$experiment_id)
dt_sel <- dt_all[experiment_id %in% selected_ids]
dt_list <- split(dt_sel, by = "experiment_id", keep.by = TRUE)

boot_ids <- if (length(selected_ids) <= opt$`boot-subset`) selected_ids else sample(selected_ids, opt$`boot-subset`)

compute_surfaces_one <- function(df) {
  df <- df[is.finite(response) & is.finite(doseA) & is.finite(doseB)]
  exp_id <- df$experiment_id[1]
  do_boot <- exp_id %in% boot_ids
  prop <- compute_proposed_metrics(df, cfg, transform, boot_override = if (do_boot) opt$`boot-B` else 0, skip_boot = !do_boot)

  surfaces <- list(
    delta_Z = list(mat = prop$delta, doseA = prop$doseA_levels, doseB = prop$doseB_levels),
    S_proposed = list(mat = prop$S_proposed, doseA = prop$doseA_levels, doseB = prop$doseB_levels)
  )

  baseline_stats <- list()
  for (m in c("Bliss", "HSA", "Loewe", "ZIP")) {
    res <- tryCatch(
      synergyfinder_synergy(df, method = m, response_mode = cfg$responses$mode),
      error = function(e) NULL
    )
    if (is.null(res)) {
      surfaces[[m]] <- list(mat = matrix(NA_real_, nrow = length(prop$doseA_levels), ncol = length(prop$doseB_levels)),
                            doseA = prop$doseA_levels, doseB = prop$doseB_levels)
      baseline_stats[[m]] <- list(finite_frac = NA_real_)
    } else {
      surfaces[[m]] <- list(mat = res$synergy, doseA = res$doseA_levels, doseB = res$doseB_levels)
      baseline_stats[[m]] <- list(finite_frac = mean(is.finite(as.numeric(res$synergy))))
    }
  }

  energy <- function(mat) mean(as.numeric(mat)^2, na.rm = TRUE)
  mean_val <- function(mat) mean(as.numeric(mat), na.rm = TRUE)

  metric_row <- data.table(
    experiment_id = exp_id,
    p_value = prop$p_value,
    S2_mean = prop$S2_mean,
    delta_energy = energy(prop$delta),
    proposed_energy = energy(prop$S_proposed),
    delta_mean = mean_val(prop$delta),
    proposed_mean = mean_val(prop$S_proposed),
    bliss_finite = baseline_stats$Bliss$finite_frac %||% NA_real_,
    hsa_finite = baseline_stats$HSA$finite_frac %||% NA_real_,
    loewe_finite = baseline_stats$Loewe$finite_frac %||% NA_real_,
    zip_finite = baseline_stats$ZIP$finite_frac %||% NA_real_
  )

  list(surfaces = surfaces, metrics = metric_row)
}

oplan <- future::plan()
on.exit(future::plan(oplan), add = TRUE)
if (n_workers > 1) future::plan(future::multisession, workers = n_workers) else future::plan(future::sequential)

surface_list <- future_lapply(dt_list, compute_surfaces_one, future.seed = TRUE)
names(surface_list) <- names(dt_list)

metrics_dt <- rbindlist(lapply(surface_list, `[[`, "metrics"), fill = TRUE)
metrics_dt <- merge(metrics_dt, grid_grp[, .(experiment_id, group_id, drugA, drugB, cell_line)], by = "experiment_id", all.x = TRUE)
write_table(metrics_dt, file.path(cfg$project$out_dir_results, "replicate_experiment_metrics.parquet"))

methods <- c("delta_Z", "S_proposed", "Bliss", "HSA", "Loewe", "ZIP")
pair_rows <- list()
for (gid in unique(grid_grp$group_id)) {
  exps <- grid_grp[group_id == gid, experiment_id]
  if (length(exps) < 2) next
  pairs <- combn(exps, 2, simplify = FALSE)
  for (pair in pairs) {
    s1 <- surface_list[[pair[1]]]$surfaces
    s2 <- surface_list[[pair[2]]]$surfaces
    for (m in methods) {
      comp <- surface_compare(
        list(mat = s1[[m]]$mat, doseA = s1[[m]]$doseA, doseB = s1[[m]]$doseB),
        list(mat = s2[[m]]$mat, doseA = s2[[m]]$doseA, doseB = s2[[m]]$doseB)
      )
      pair_rows[[length(pair_rows) + 1L]] <- data.table(
        group_id = gid,
        experiment_a = pair[1],
        experiment_b = pair[2],
        method = m,
        corr = comp$corr,
        rel_rmse = comp$rel_rmse,
        top5 = topk_overlap(s1[[m]]$mat, s2[[m]]$mat, k = 5),
        top10 = topk_overlap(s1[[m]]$mat, s2[[m]]$mat, k = 10),
        rms_a = comp$rms1,
        rms_b = comp$rms2
      )
    }
  }
}

rep_dt <- if (length(pair_rows) > 0) rbindlist(pair_rows, fill = TRUE) else data.table()
write_table(rep_dt, file.path(cfg$project$out_dir_results, "replicate_concordance.parquet"))

cv_rows <- list()
for (gid in unique(metrics_dt$group_id)) {
  sub <- metrics_dt[group_id == gid]
  if (nrow(sub) < 2) next
  for (m in methods) {
    if (m == "delta_Z") {
      vals <- sub$delta_energy
      means <- sub$delta_mean
    } else if (m == "S_proposed") {
      vals <- sub$proposed_energy
      means <- sub$proposed_mean
    } else {
      vals <- NA_real_
      means <- NA_real_
    }
    cv_rows[[length(cv_rows) + 1L]] <- data.table(
      group_id = gid,
      method = m,
      energy_mean = mean(vals, na.rm = TRUE),
      energy_sd = stats::sd(vals, na.rm = TRUE),
      energy_cv = stats::sd(vals, na.rm = TRUE) / max(1e-8, mean(vals, na.rm = TRUE)),
      mean_surface = mean(means, na.rm = TRUE),
      mean_cv = stats::sd(means, na.rm = TRUE) / max(1e-8, abs(mean(means, na.rm = TRUE)))
    )
  }
}
rep_cv_dt <- if (length(cv_rows) > 0) rbindlist(cv_rows, fill = TRUE) else data.table()
write_table(rep_cv_dt, file.path(cfg$project$out_dir_results, "replicate_energy_cv.parquet"))

if (nrow(rep_dt) > 0) {
  rep_dt[method == "S_proposed", method := "S_SIR (viability)"]
  rep_dt[method == "delta_Z", method := "SIR (logit)"]
  corr_plot <- ggplot(rep_dt[is.finite(corr)], aes(x = method, y = corr, fill = method)) +
    geom_violin(alpha = 0.4, trim = FALSE) +
    geom_boxplot(width = 0.2, outlier.shape = NA, alpha = 0.6) +
    labs(title = "Replicate correlation", x = "Method", y = "Correlation") +
    theme_minimal(base_size = 10) +
    theme(legend.position = "none", axis.text.x = element_text(angle = 30, hjust = 1))

  # NOTE: NA in *_finite means the method completely failed (not just partial NA in surface).

  # Treat complete failures as 0% finite (100% NA rate) by replacing NA with 0 before computing.
  na_rates <- metrics_dt[, .(
    Bliss = mean(1 - fifelse(is.na(bliss_finite), 0, bliss_finite)),
    HSA = mean(1 - fifelse(is.na(hsa_finite), 0, hsa_finite)),
    Loewe = mean(1 - fifelse(is.na(loewe_finite), 0, loewe_finite)),
    ZIP = mean(1 - fifelse(is.na(zip_finite), 0, zip_finite))
  )]
  na_long <- melt(na_rates, measure.vars = names(na_rates), variable.name = "method", value.name = "na_rate")
  na_plot <- ggplot(na_long, aes(x = method, y = na_rate, fill = method)) +
    geom_col(alpha = 0.7) +
    labs(title = "Failure rate", x = "Method", y = "NA rate") +
    theme_minimal(base_size = 10) +
    theme(legend.position = "none")

  ggsave(file.path(fig_dir, "replicate_concordance.pdf"), corr_plot + na_plot, width = 10, height = 4.5)
}

# -------------------------
# Missingness-as-prediction
# -------------------------
miss_ids <- unique(grid$experiment_id)
miss_ids <- if (length(miss_ids) <= opt$`missing-n`) miss_ids else sample(miss_ids, opt$`missing-n`)
miss_list <- split(dt_all[experiment_id %in% miss_ids], by = "experiment_id", keep.by = TRUE)

missingness_one <- function(df) {
  df <- df[is.finite(response) & is.finite(doseA) & is.finite(doseB)]
  exp_id <- df$experiment_id[1]
  grid_full <- matrix_mean_response(df, response_mode = cfg$responses$mode)
  doseA_levels <- grid_full$doseA_levels
  doseB_levels <- grid_full$doseB_levels
  Y_full <- grid_full$Y

  combos <- unique(df[, .(doseA, doseB)])
  interior <- combos[doseA > 0 & doseB > 0]
  if (nrow(interior) < 5) {
    return(data.table(experiment_id = exp_id, rmse_response = NA_real_, rmse_S_proposed = NA_real_, n_holdout = 0L))
  }
  n_hold <- max(1L, floor(opt$`missing-frac` * nrow(interior)))
  hold <- interior[sample.int(nrow(interior), n_hold)]

  df_train <- df[!hold, on = .(doseA, doseB)]
  if (nrow(df_train) < 10) {
    return(data.table(experiment_id = exp_id, rmse_response = NA_real_, rmse_S_proposed = NA_real_, n_holdout = n_hold))
  }

  prop_train <- compute_proposed_metrics(df_train, cfg, transform, boot_override = 0, skip_boot = TRUE)
  iso_hat <- transform$inverse(prop_train$theta_iso)
  S_prop_train <- prop_train$S_proposed

  # Full-surface truth (for S_proposed)
  prop_full <- compute_proposed_metrics(df, cfg, transform, boot_override = 0, skip_boot = TRUE)
  S_prop_full <- prop_full$S_proposed

  i_idx <- match(hold$doseA, doseA_levels)
  j_idx <- match(hold$doseB, doseB_levels)
  ok <- is.finite(i_idx) & is.finite(j_idx)
  if (!any(ok)) {
    return(data.table(experiment_id = exp_id, rmse_response = NA_real_, rmse_S_proposed = NA_real_, n_holdout = n_hold))
  }
  pred_resp <- as.numeric(iso_hat[cbind(i_idx[ok], j_idx[ok])])
  true_resp <- as.numeric(Y_full[cbind(i_idx[ok], j_idx[ok])])
  rmse_response <- sqrt(mean((pred_resp - true_resp)^2, na.rm = TRUE))

  pred_S <- as.numeric(S_prop_train[cbind(i_idx[ok], j_idx[ok])])
  true_S <- as.numeric(S_prop_full[cbind(i_idx[ok], j_idx[ok])])
  rmse_S <- sqrt(mean((pred_S - true_S)^2, na.rm = TRUE))

  data.table(experiment_id = exp_id, rmse_response = rmse_response, rmse_S_proposed = rmse_S, n_holdout = n_hold)
}

miss_dt <- future_lapply(miss_list, missingness_one, future.seed = TRUE)
miss_dt <- rbindlist(miss_dt, fill = TRUE)
write_table(miss_dt, file.path(cfg$project$out_dir_results, "missingness_prediction.parquet"))

if (nrow(miss_dt) > 0) {
  p_miss1 <- ggplot(miss_dt[is.finite(rmse_response)], aes(x = rmse_response)) +
    geom_histogram(bins = 30, fill = "#0C6291", color = "white") +
    labs(title = "Viability prediction RMSE", x = "RMSE", y = "Count") +
    theme_minimal(base_size = 10)
  p_miss2 <- ggplot(miss_dt[is.finite(rmse_S_proposed)], aes(x = rmse_S_proposed)) +
    geom_histogram(bins = 30, fill = "#F28C28", color = "white") +
    labs(title = expression(S[SIR]~"prediction RMSE"), x = "RMSE", y = "Count") +
    theme_minimal(base_size = 10)
  ggsave(file.path(fig_dir, "missingness_prediction.pdf"), p_miss1 + p_miss2, width = 9, height = 4)
}

# -------------------------
# Pseudo-null calibration
# -------------------------
pseudo_ids <- unique(grid$experiment_id)
pseudo_ids <- if (length(pseudo_ids) <= opt$`pseudonull-n`) pseudo_ids else sample(pseudo_ids, opt$`pseudonull-n`)
pseudo_list <- split(dt_all[experiment_id %in% pseudo_ids], by = "experiment_id", keep.by = TRUE)

pseudo_one <- function(df) {
  df <- df[is.finite(response) & is.finite(doseA) & is.finite(doseB)]
  exp_id <- df$experiment_id[1]
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

  # Generate one pseudo-null draw under the fitted monotone-additive null.
  # Use the same residual scaling strategy as the wild bootstrap to avoid
  # underestimating noise (anti-conservative calibration).
  r_scaled <- r
  if ((cfg$bootstrap$residual_scale %||% "none") == "df") {
    n_distinct_runs <- function(x, tol) {
      if (length(x) == 0) return(0L)
      1L + sum(abs(diff(x)) > tol)
    }
    wv <- as.numeric(ms$w)
    rv <- as.numeric(r)
    ok <- is.finite(wv) & is.finite(rv) & (wv > 0)
    n_eff <- sum(ok)
    if (n_eff > 1) {
      df_null <- cfg$bootstrap$df_null %||% NULL
      if (is.null(df_null)) {
        Ku <- n_distinct_runs(add_fit$u, cfg$bootstrap$df_tol %||% 1e-8)
        Kv <- n_distinct_runs(add_fit$v, cfg$bootstrap$df_tol %||% 1e-8)
        df_null <- Ku + Kv - 1L
      }
      df_null <- max(0, min(as.integer(df_null), n_eff - 1L))
      scale <- sqrt(n_eff / max(1, n_eff - df_null))
      r_scaled <- r * scale
    }
  }

  xi <- matrix(sample(c(-1, 1), length(r_scaled), replace = TRUE), nrow = nrow(r_scaled), ncol = ncol(r_scaled))
  Z_star <- add_fit$theta + xi * r_scaled

  synergy_sign <- cfg$interaction$synergy_sign %||% "auto"
  if (synergy_sign == "auto") {
    synergy_sign <- infer_synergy_sign(cfg$responses$mode, ms$direction)
  }

  seed_exp <- opt$seed + (sum(utf8ToInt(exp_id)) %% 100000)
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

  data.table(experiment_id = exp_id, p_value = boot$p_value)
}

pseudo_dt <- future_lapply(pseudo_list, pseudo_one, future.seed = TRUE)
pseudo_dt <- rbindlist(pseudo_dt, fill = TRUE)
write_table(pseudo_dt, file.path(cfg$project$out_dir_results, "public_pseudonull_pvals.parquet"))

if (nrow(pseudo_dt) > 0) {
  m <- nrow(pseudo_dt)
  ecdf_band <- function(p) {
    eps <- sqrt(log(2 / 0.05) / (2 * m))
    cbind(p = p, lo = pmax(0, p - eps), hi = pmin(1, p + eps))
  }
  grid_p <- seq(0, 1, length.out = 200)
  band <- as.data.table(ecdf_band(grid_p))
  p_cal <- ggplot(pseudo_dt, aes(x = p_value)) +
    stat_ecdf(geom = "step", color = "#0C6291") +
    geom_ribbon(data = band, aes(x = p, ymin = lo, ymax = hi), fill = "#F28C28", alpha = 0.2, inherit.aes = FALSE) +
    geom_abline(intercept = 0, slope = 1, linetype = "dashed") +
    labs(title = "Pseudo-null p-value distribution", x = "p-value", y = "ECDF") +
    theme_minimal(base_size = 10)
  ggsave(file.path(fig_dir, "pseudonull_calibration.pdf"), p_cal, width = 5, height = 4)
}

message("Done. Replicate groups: ", length(unique(grid_grp$group_id)),
        " | Replicate pairs: ", nrow(rep_dt),
        " | Missingness matrices: ", nrow(miss_dt),
        " | Pseudo-null matrices: ", nrow(pseudo_dt))
