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
source("R/baselines_bliss.R")
source("R/baselines_hsa.R")
source("R/baselines_loewe.R")
source("R/baselines_zip.R")
source("R/baselines_synergyfinder.R")
source("R/data_standard.R")
source("R/metrics.R")

option_list <- list(
  make_option("--config", type = "character", default = "configs/default.yaml",
              help = "YAML config path", metavar = "path"),
  make_option("--sample-n", type = "integer", default = 400L,
              help = "Number of matrices for prevalence analysis (300–1000 recommended)", metavar = "int"),
  make_option("--stability-n", type = "integer", default = 60L,
              help = "Number of matrices for stability/fragility experiments", metavar = "int"),
  make_option("--boot1", type = "integer", default = 200L,
              help = "Bootstrap replicates for pass 1", metavar = "int"),
  make_option("--boot2", type = "integer", default = 1000L,
              help = "Bootstrap replicates for pass 2 (refinement)", metavar = "int"),
  make_option("--seed", type = "integer", default = 42L,
              help = "Random seed", metavar = "int")
)

opt <- parse_args(OptionParser(option_list = option_list))
set.seed(opt$seed)

cfg <- read_config(opt$config)
transform <- make_transform(cfg$transform)
n_workers <- max(
  1L,
  min(cfg$project$n_cores %||% 1, as.integer(future::availableCores(methods = "mc.cores")))
)
cfg$project$n_cores <- n_workers

dir_create(cfg$project$out_dir_results %||% "results")
fig_dir <- file.path(cfg$project$out_dir_figures %||% "figures", "public")
dir_create(fig_dir)

load_source <- function(path, source) {
  dt <- as.data.table(read_parquet(path))
  if (!"source" %in% names(dt)) dt[, source := source]
  dt
}

standard_paths <- cfg$data$standard_files %||% list()
loaded <- list()
for (nm in names(standard_paths)) {
  path <- standard_paths[[nm]]
  if (is.null(path) || is.na(path) || identical(path, "")) next
  loaded[[nm]] <- load_source(path, nm)
}
all_long <- rbindlist(loaded, use.names = TRUE, fill = TRUE)
if (nrow(all_long) == 0) stop("No public datasets loaded; set data.standard_files.* in the config.")

exp_meta <- all_long[, .(
  n = .N,
  n_i = uniqueN(doseA),
  n_j = uniqueN(doseB),
  has_monoA = any(doseB == 0, na.rm = TRUE),
  has_monoB = any(doseA == 0, na.rm = TRUE),
  mono_activity = mean(1 - response[(doseA == 0 | doseB == 0)], na.rm = TRUE)
), by = .(source, experiment_id, drugA, drugB, cell_line)]

add_strata <- function(dt) {
  dt[, grid_stratum := paste(pmin(n_i, 6L), pmin(n_j, 6L), sep = "x")]
  cuts <- quantile(dt$mono_activity, probs = c(0, 0.33, 0.66, 1), na.rm = TRUE, names = FALSE)
  cuts <- unique(cuts)
  if (length(cuts) < 2) cuts <- c(0, 1)
  dt[, mono_stratum := cut(mono_activity, breaks = cuts, include.lowest = TRUE, labels = FALSE)]
  dt[is.na(mono_stratum), mono_stratum := 1L]
  dt[, stratum := paste(source, grid_stratum, mono_stratum, sep = "__")]
  dt
}

sample_by_strata <- function(meta, n_target) {
  meta <- add_strata(meta)
  total <- nrow(meta)
  message("[sample] total experiments=", total, " target=", n_target)
  if (total <= n_target) return(meta$experiment_id)
  meta[, take := ceiling((.N / total) * n_target), by = stratum]
  message("[sample] take range: ", paste(range(meta$take), collapse = " - "))
  message("[sample] strata head:"); print(head(unique(meta[, .(stratum, N = .N, take = take[1]), by = stratum][order(-take)])))
  sampled <- meta[, .SD[sample.int(.N, min(.N, take[1L]))], by = stratum]
  message("[sample] sampled rows: ", nrow(sampled))
  ids <- unique(sampled$experiment_id)
  if (length(ids) > n_target) ids <- sample(ids, n_target)
  ids
}

split_by_experiment <- function(dt) split(dt, by = "experiment_id", keep.by = TRUE)

rms <- function(x) {
  if (is.null(x)) return(NA_real_)
  sqrt(mean(as.numeric(x)^2, na.rm = TRUE))
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

topk_overlap <- function(m1, m2, k = 5, mode = c("abs", "pos"), dosesA, dosesB) {
  mode <- match.arg(mode)
  v1 <- as.numeric(m1); v2 <- as.numeric(m2)
  if (mode == "abs") {
    ord1 <- order(-abs(v1), na.last = NA)
    ord2 <- order(-abs(v2), na.last = NA)
  } else {
    ord1 <- order(-pmax(v1, 0), na.last = NA)
    ord2 <- order(-pmax(v2, 0), na.last = NA)
  }
  idx1 <- head(ord1, k); idx2 <- head(ord2, k)
  if (length(idx1) == 0 || length(idx2) == 0) return(NA_real_)
  length(intersect(idx1, idx2)) / min(length(idx1), length(idx2))
}

surface_compare <- function(orig, pert, eps = 1e-8) {
  aligned <- align_surfaces(orig, pert)
  if (is.null(aligned)) return(list(corr = NA_real_, rmse = NA_real_, rel_rmse = NA_real_))
  v1 <- aligned$v1; v2 <- aligned$v2
  ok <- is.finite(v1) & is.finite(v2)
  v1 <- v1[ok]; v2 <- v2[ok]
  if (length(v1) == 0) return(list(corr = NA_real_, rmse = NA_real_, rel_rmse = NA_real_))
  rms1 <- sqrt(mean(v1^2))
  rms2 <- sqrt(mean(v2^2))
  corr <- if (rms1 > eps && rms2 > eps) suppressWarnings(stats::cor(v1, v2)) else NA_real_
  rmse <- sqrt(mean((v1 - v2)^2))
  rel_rmse <- rmse / max(eps, rms1)
  list(corr = corr, rmse = rmse, rel_rmse = rel_rmse, rms1 = rms1, rms2 = rms2)
}

compute_metrics_one <- function(df, boot_B) {
  df <- df[is.finite(response) & is.finite(doseA) & is.finite(doseB)]
  meta <- df[1, .(experiment_id, source, drugA, drugB, cell_line)]
  prop <- compute_proposed_metrics(df, cfg, transform, boot_override = boot_B)
  base <- compute_baselines(df, cfg)
  sf_stats <- synergyfinder_na_stats(df, response_mode = cfg$responses$mode)
  S_prop <- prop$S_proposed
  corr_fun <- function(mat, dosesA, dosesB) {
    aligned <- align_surfaces(
      list(mat = S_prop, doseA = prop$doseA_levels, doseB = prop$doseB_levels),
      list(mat = mat, doseA = dosesA, doseB = dosesB)
    )
    if (is.null(aligned)) return(NA_real_)
    v1 <- aligned$v1; v2 <- aligned$v2
    ok <- is.finite(v1) & is.finite(v2)
    v1 <- v1[ok]; v2 <- v2[ok]
    if (!any(ok)) return(NA_real_)
    rms1 <- sqrt(mean(v1^2)); rms2 <- sqrt(mean(v2^2))
    if (rms1 < 1e-8 || rms2 < 1e-8) return(NA_real_)
    suppressWarnings(stats::cor(v1, v2))
  }
  data.table(
    meta,
    S2 = prop$S2,
    S2_mean = prop$S2_mean,
    p_value = prop$p_value,
    synergy_index = prop$synergy_index,
    rms_proposed = rms(S_prop),
    bliss_energy = rms(base$bliss$synergy)^2,
    hsa_energy = rms(base$hsa$synergy)^2,
    loewe_energy = rms(base$loewe$synergy)^2,
    zip_energy = rms(base$zip$synergy)^2,
    bliss_corr = corr_fun(base$bliss$synergy, base$bliss$doseA_levels, base$bliss$doseB_levels),
    hsa_corr = corr_fun(base$hsa$synergy, base$hsa$doseA_levels, base$hsa$doseB_levels),
    loewe_corr = corr_fun(base$loewe$synergy, base$loewe$doseA_levels, base$loewe$doseB_levels),
    zip_corr = corr_fun(base$zip$synergy, base$zip$doseA_levels, base$zip$doseB_levels),
    bliss_finite = sf_stats$Bliss$finite_frac %||% NA_real_,
    hsa_finite = sf_stats$HSA$finite_frac %||% NA_real_,
    loewe_finite = sf_stats$Loewe$finite_frac %||% NA_real_,
    zip_finite = sf_stats$ZIP$finite_frac %||% NA_real_
  )
}

run_metrics <- function(dt_list, ids, boot_B) {
  mats <- dt_list[names(dt_list) %in% ids]
  oplan <- future::plan()
  on.exit(future::plan(oplan), add = TRUE)
  if (n_workers > 1) {
    future::plan(future::multisession, workers = n_workers)
  } else {
    future::plan(future::sequential)
  }
  res <- future_lapply(mats, compute_metrics_one, boot_B = boot_B, future.seed = TRUE)
  rbindlist(res, fill = TRUE)
}

perturb_df <- function(df, type, outlier_k = 4, jitter_sd = 0.02, drop_frac = 0.1) {
  dt <- copy(as.data.table(df))
  dosesA <- sort(unique(dt$doseA))
  dosesB <- sort(unique(dt$doseB))
  dt[, i := match(doseA, dosesA)]
  dt[, j := match(doseB, dosesB)]
  interior <- dt[i > 1 & j > 1]
  if (type == "outlier") {
    if (nrow(interior) == 0) return(dt)
    pick <- interior[sample.int(nrow(interior), 1)]
    s <- stats::sd(dt$response, na.rm = TRUE); if (!is.finite(s) || s == 0) s <- 0.05
    sign <- sample(c(-1, 1), 1)
    dt[doseA == pick$doseA & doseB == pick$doseB, response := clamp01(response + sign * outlier_k * s, eps = 0)]
    return(dt)
  }
  if (type == "drop_row") {
    if (length(dosesA) <= 2) return(dt)
    drop_i <- sample(dosesA[-1], 1)
    return(dt[doseA != drop_i])
  }
  if (type == "drop_col") {
    if (length(dosesB) <= 2) return(dt)
    drop_j <- sample(dosesB[-1], 1)
    return(dt[doseB != drop_j])
  }
  if (type == "missingness") {
    if (nrow(interior) == 0) return(dt)
    n_drop <- max(1L, ceiling(nrow(interior) * drop_frac))
    drop_idx <- interior[sample.int(nrow(interior), n_drop)]
    return(dt[!(doseA %in% drop_idx$doseA & doseB %in% drop_idx$doseB)])
  }
  if (type == "jitter") {
    dt[, response := clamp01(response + stats::rnorm(.N, sd = jitter_sd), eps = 0)]
    return(dt)
  }
  if (type == "edge_ablate") {
    # Drop a random subset of edge wells (monotherapy)
    edges <- dt[(doseA == 0 | doseB == 0)]
    if (nrow(edges) == 0) return(dt)
    n_drop <- max(1L, ceiling(0.3 * nrow(edges)))
    drop_rows <- edges[sample.int(nrow(edges), n_drop)]
    return(dt[!(doseA %in% drop_rows$doseA & doseB %in% drop_rows$doseB)])
  }
  if (type == "edge_noise") {
    dt[(doseA == 0 | doseB == 0), response := clamp01(response + stats::rnorm(.N, sd = jitter_sd * 2), eps = 0)]
    return(dt)
  }
  dt
}

compute_surfaces <- function(df) {
  prop <- compute_proposed_metrics(df, cfg, transform, skip_boot = TRUE)
  base <- compute_baselines(df, cfg)
  list(
    proposed = list(mat = prop$S_proposed, doseA = prop$doseA_levels, doseB = prop$doseB_levels),
    bliss = list(mat = base$bliss$synergy, doseA = base$bliss$doseA_levels, doseB = base$bliss$doseB_levels),
    hsa = list(mat = base$hsa$synergy, doseA = base$hsa$doseA_levels, doseB = base$hsa$doseB_levels),
    loewe = list(mat = base$loewe$synergy, doseA = base$loewe$doseA_levels, doseB = base$loewe$doseB_levels),
    zip = list(mat = base$zip$synergy, doseA = base$zip$doseA_levels, doseB = base$zip$doseB_levels)
  )
}

analyze_stability <- function(dt_list, ids, pert_types = c("outlier", "jitter", "edge_noise")) {
  rows <- list()
  for (id in ids) {
    df <- dt_list[[id]]
    if (is.null(df)) next
    base_surfaces <- compute_surfaces(df)
    for (tp in pert_types) {
      dfp <- perturb_df(df, tp)
      pert_surfaces <- compute_surfaces(dfp)
      for (m in names(base_surfaces)) {
        comp <- surface_compare(base_surfaces[[m]], pert_surfaces[[m]])
        rows[[length(rows) + 1L]] <- data.table(
          experiment_id = id,
          method = m,
          perturbation = tp,
          corr = comp$corr,
          rmse = comp$rmse,
          rel_rmse = comp$rel_rmse,
          rms_orig = comp$rms1,
          rms_pert = comp$rms2,
          top5_abs = topk_overlap(base_surfaces[[m]]$mat, pert_surfaces[[m]]$mat, k = 5, mode = "abs"),
          top10_abs = topk_overlap(base_surfaces[[m]]$mat, pert_surfaces[[m]]$mat, k = 10, mode = "abs"),
          top5_pos = topk_overlap(base_surfaces[[m]]$mat, pert_surfaces[[m]]$mat, k = 5, mode = "pos"),
          top10_pos = topk_overlap(base_surfaces[[m]]$mat, pert_surfaces[[m]]$mat, k = 10, mode = "pos")
        )
      }
    }
  }
  rbindlist(rows, fill = TRUE)
}

# -------------------------
# Main prevalence analysis
# -------------------------
sample_ids <- sample_by_strata(exp_meta, opt$`sample-n`)
dt_list <- split_by_experiment(all_long[experiment_id %in% sample_ids])
message("[public] sampled ", length(sample_ids), " experiments across ", length(unique(exp_meta$stratum)), " strata.")

# Pass 1
pass1 <- run_metrics(dt_list, sample_ids, boot_B = opt$boot1)

# Identify for refinement: borderline p or top effects
quant_S2 <- stats::quantile(pass1$S2_mean, probs = 0.9, na.rm = TRUE, names = FALSE)
refine_ids <- unique(pass1[(p_value > 0.01 & p_value < 0.2) | (S2_mean >= quant_S2), experiment_id])

pass2 <- data.table()
if (length(refine_ids) > 0) {
  pass2 <- run_metrics(dt_list, refine_ids, boot_B = opt$boot2)
}

metrics_dt <- pass1
if (nrow(pass2) > 0) {
  # replace refined rows
  metrics_dt <- metrics_dt[!experiment_id %in% pass2$experiment_id]
  metrics_dt <- rbind(metrics_dt, pass2, fill = TRUE)
}
metrics_dt[, q_value := p.adjust(p_value, method = "BH")]
write_table(metrics_dt, file.path(cfg$project$out_dir_results, "public_sample_metrics.parquet"))

# -------------------------
# Stability (perturbations)
# -------------------------
stab_ids <- sample_ids
if (length(stab_ids) > opt$`stability-n`) stab_ids <- sample(stab_ids, opt$`stability-n`)
stab_dt <- analyze_stability(dt_list, stab_ids)
stab_dt[, rms_quartile := cut(rms_orig, breaks = quantile(rms_orig, probs = seq(0, 1, 0.25), na.rm = TRUE), include.lowest = TRUE)]
write_table(stab_dt, file.path(cfg$project$out_dir_results, "public_stability.parquet"))

# Edge ablation fragility on a small subset
frag_ids <- sample(stab_ids, min(20, length(stab_ids)))
frag_dt <- analyze_stability(dt_list, frag_ids, pert_types = c("edge_ablate", "edge_noise"))
frag_dt[, rms_quartile := cut(rms_orig, breaks = quantile(rms_orig, probs = seq(0, 1, 0.25), na.rm = TRUE), include.lowest = TRUE)]
write_table(frag_dt, file.path(cfg$project$out_dir_results, "public_edge_fragility.parquet"))

# -------------------------
# Figures
# -------------------------
metrics_plot <- metrics_dt[is.finite(p_value)]
p1 <- ggplot(metrics_plot, aes(x = p_value)) +
  geom_histogram(bins = 30, fill = "#0C6291", color = "white") +
  labs(title = sprintf("Bootstrap p-values (proposed, B=%d/%d)", opt$boot1, opt$boot2), x = "p-value", y = "count") +
  theme_minimal(base_size = 11)
ggsave(file.path(fig_dir, "pvalue_hist.pdf"), p1, width = 5, height = 4)

p2 <- ggplot(metrics_dt[is.finite(S2_mean)], aes(x = S2_mean)) +
  stat_ecdf(geom = "step", color = "#F28C28") +
  labs(title = "ECDF of S2_mean (proposed)", x = "S2_mean", y = "ECDF") +
  theme_minimal(base_size = 11)
ggsave(file.path(fig_dir, "S2_ecdf.pdf"), p2, width = 6, height = 4)

corr_long <- melt(
  metrics_dt,
  id.vars = c("experiment_id", "drugA", "drugB", "cell_line", "source"),
  measure.vars = c("bliss_corr", "hsa_corr", "loewe_corr", "zip_corr"),
  variable.name = "baseline",
  value.name = "corr"
)
p3 <- ggplot(corr_long[is.finite(corr)], aes(x = corr, fill = baseline)) +
  geom_density(alpha = 0.5) +
  labs(title = "Correlation of baseline synergy vs proposed (sign-aligned)", x = "correlation", y = "density") +
  theme_minimal(base_size = 11)
ggsave(file.path(fig_dir, "baseline_corr.pdf"), p3, width = 7, height = 4)

stab_summary <- stab_dt[, .(
  corr_med = median(corr, na.rm = TRUE),
  rel_rmse_med = median(rel_rmse, na.rm = TRUE),
  top5_abs_med = median(top5_abs, na.rm = TRUE)
), by = .(method, rms_quartile, perturbation)]
gg <- ggplot(stab_summary, aes(x = perturbation, y = corr_med, fill = method)) +
  geom_col(position = position_dodge()) +
  facet_wrap(~ rms_quartile) +
  labs(title = "Stability (corr) by energy quartile", y = "median corr", x = "perturbation") +
  theme_minimal(base_size = 9) +
  theme(axis.text.x = element_text(angle = 45, hjust = 1))
ggsave(file.path(fig_dir, "stability_corr_energy.pdf"), gg, width = 9, height = 5)

frag_summary <- frag_dt[, .(
  rel_rmse_med = median(rel_rmse, na.rm = TRUE),
  top5_abs_med = median(top5_abs, na.rm = TRUE),
  na_rate = mean(is.na(rel_rmse))
), by = .(method, perturbation)]
gg_frag <- ggplot(frag_summary, aes(x = perturbation, y = rel_rmse_med, fill = method)) +
  geom_col(position = position_dodge()) +
  labs(title = "Edge fragility (relative RMSE)", y = "median rel RMSE", x = "") +
  theme_minimal(base_size = 10)
ggsave(file.path(fig_dir, "edge_fragility.pdf"), gg_frag, width = 8, height = 4.5)

# -------------------------
# Baseline NA rate vs monotherapy coverage
# -------------------------
na_df <- merge(metrics_dt, exp_meta[, .(experiment_id, mono_activity)], by = "experiment_id", all.x = TRUE)
na_long <- melt(
  na_df,
  id.vars = c("experiment_id", "mono_activity"),
  measure.vars = c("bliss_finite", "hsa_finite", "loewe_finite", "zip_finite"),
  variable.name = "baseline",
  value.name = "finite_frac"
)
na_long[, na_rate := 1 - finite_frac]
if (nrow(na_long) > 0) {
  p_na <- ggplot(na_long, aes(x = mono_activity, y = na_rate, color = baseline)) +
    geom_point(alpha = 0.6) +
    geom_smooth(se = FALSE) +
    labs(title = "Baseline NA rate vs monotherapy activity", x = "mean inhibition on edges", y = "NA rate (SynergyFinder)") +
    theme_minimal(base_size = 10)
  ggsave(file.path(fig_dir, "baseline_na_rate.pdf"), p_na, width = 7, height = 4.5)
}

# -------------------------
# Pseudo-null calibration
# -------------------------
pseudo_ids <- sample(sample_ids, min(300, length(sample_ids)))
pseudo_rows <- list()
for (id in pseudo_ids) {
  df <- dt_list[[id]]
  if (is.null(df)) next
  ms <- summarise_matrix(
    df_long = df,
    transform = transform,
    response_mode = cfg$responses$mode,
    method_direction = cfg$method$monotone_direction,
    tau = cfg$weights$tau,
    winsor_q = cfg$weights$winsor_q
  )
  add_fit <- additive_ordered_fit(ms$barZ, ms$w, direction = ms$direction, osqp_pars = cfg$method$osqp %||% list())
  r <- ms$barZ - add_fit$theta
  xi <- matrix(sample(c(-1, 1), length(r), replace = TRUE), nrow = nrow(r), ncol = ncol(r))
  Z_star <- add_fit$theta + xi * r
  y_star <- transform$inverse(Z_star)
  df_star <- copy(df)
  i_idx <- match(df_star$doseA, ms$doseA_levels)
  j_idx <- match(df_star$doseB, ms$doseB_levels)
  df_star$response <- as.numeric(y_star[cbind(i_idx, j_idx)])
  prop_star <- compute_proposed_metrics(df_star, cfg, transform, boot_override = 200)
  pseudo_rows[[length(pseudo_rows) + 1L]] <- data.table(
    experiment_id = id,
    p_value = prop_star$p_value
  )
}
pseudo_dt <- rbindlist(pseudo_rows, fill = TRUE)
write_table(pseudo_dt, file.path(cfg$project$out_dir_results, "public_pseudonull_pvals.parquet"))

if (nrow(pseudo_dt) > 0) {
  m <- nrow(pseudo_dt)
  ecdf_band <- function(p) {
    eps <- sqrt(log(2 / 0.05) / (2 * m))
    cbind(p = p, lo = pmax(0, p - eps), hi = pmin(1, p + eps))
  }
  grid_p <- seq(0, 1, length.out = 200)
  band <- ecdf_band(grid_p)
  p_cal <- ggplot(pseudo_dt, aes(x = p_value)) +
    stat_ecdf(geom = "step", color = "#0C6291") +
    geom_ribbon(data = as.data.table(band), aes(x = p, ymin = lo, ymax = hi), fill = "#F28C28", alpha = 0.2, inherit.aes = FALSE) +
    geom_abline(intercept = 0, slope = 1, linetype = "dashed") +
    labs(title = "Pseudo-null calibration (one wild draw)", x = "p-value", y = "ECDF") +
    theme_minimal(base_size = 10)
  ggsave(file.path(fig_dir, "pseudonull_calibration.pdf"), p_cal, width = 5, height = 4)
}

# -------------------------
# Replicate concordance (matching grids)
# -------------------------
sig_dt <- rbindlist(
  lapply(dt_list, function(df) {
    df <- as.data.table(df)
    data.table(
      experiment_id = df$experiment_id[1],
      doseA_sig = paste(sort(unique(as.numeric(df$doseA))), collapse = ";"),
      doseB_sig = paste(sort(unique(as.numeric(df$doseB))), collapse = ";")
    )
  }),
  fill = TRUE
)
meta_sig <- merge(exp_meta[experiment_id %in% sample_ids], sig_dt, by = "experiment_id", all.x = TRUE)
rep_groups <- meta_sig[, .(n = .N, experiments = list(experiment_id)),
  by = .(source, drugA, drugB, cell_line, doseA_sig, doseB_sig)
]
rep_groups <- rep_groups[n >= 2]
rep_rows <- list()
for (i in seq_len(nrow(rep_groups))) {
  exps <- rep_groups$experiments[[i]]
  if (length(exps) < 2) next
  pair <- exps[1:2]
  dfs <- dt_list[names(dt_list) %in% pair]
  if (length(dfs) < 2) next
  s1 <- compute_surfaces(dfs[[1]])
  s2 <- compute_surfaces(dfs[[2]])
  for (m in names(s1)) {
    comp <- surface_compare(s1[[m]], s2[[m]])
    rep_rows[[length(rep_rows) + 1L]] <- data.table(
      source = rep_groups$source[i],
      drugA = rep_groups$drugA[i],
      drugB = rep_groups$drugB[i],
      cell_line = rep_groups$cell_line[i],
      method = m,
      corr = comp$corr,
      rel_rmse = comp$rel_rmse
    )
  }
}
rep_dt <- if (length(rep_rows) > 0) rbindlist(rep_rows, fill = TRUE) else data.table()
write_table(rep_dt, file.path(cfg$project$out_dir_results, "replicate_concordance.parquet"))

if (nrow(rep_dt) > 0) {
  p_rep <- ggplot(rep_dt, aes(x = method, y = corr, fill = method)) +
    geom_boxplot(outlier.shape = NA, alpha = 0.5) +
    geom_jitter(width = 0.15, alpha = 0.6, size = 1) +
    labs(title = "Replicate concordance (corr)", x = "method", y = "correlation") +
    theme_minimal(base_size = 10) +
    theme(legend.position = "none")
  ggsave(file.path(fig_dir, "replicate_concordance.pdf"), p_rep, width = 6, height = 4)
}

message("Done. Metrics: ", nrow(metrics_dt), " matrices; stability: ", nrow(stab_dt), " rows.")
