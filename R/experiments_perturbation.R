run_perturbation_experiment <- function(cfg, transform, out_results, out_figures) {
  if (!requireNamespace("data.table", quietly = TRUE)) {
    stop("Package 'data.table' is required. Install with: install.packages('data.table')")
  }
  # Default to simulation until real-data loaders are configured.
  mats <- simulate_matrices(cfg, transform = transform)
  n <- min(length(mats), cfg$experiments$perturbation$n_matrices %||% length(mats))
  mats <- mats[seq_len(n)]

  compute_scores <- function(df) {
    prop <- compute_proposed_metrics(df, cfg, transform)
    base <- compute_baselines(df, cfg)

    # Matrix-level summaries: mean synergy over interior wells (exclude edges).
    interior_mean <- function(M) {
      if (!is.matrix(M)) return(NA_real_)
      if (nrow(M) < 2 || ncol(M) < 2) return(NA_real_)
      mean(M[-1, -1], na.rm = TRUE)
    }

    data.table::data.table(
      experiment_id = df$experiment_id[1],
      t_int = prop$t_int,
      p_value = prop$p_value,
      S2 = prop$S2,
      Splus = prop$Splus,
      Sminus = prop$Sminus,
      max_synergy = prop$max_synergy,
      area_synergy = prop$area_synergy,
      area_synergy_weighted = prop$area_synergy_weighted,
      max_abs_delta = prop$max_abs_delta,
      mean_delta = prop$mean_delta,
      boot_stat = prop$boot_stat,
      bliss_mean = interior_mean(base$bliss$synergy),
      hsa_mean = interior_mean(base$hsa$synergy),
      loewe_mean = interior_mean(base$loewe$synergy),
      zip_mean = interior_mean(base$zip$synergy)
    )
  }

  perturb_df <- function(df, type, cfg) {
    dt <- data.table::as.data.table(df)
    set.seed(cfg$project$seed %||% 1)
    dosesA <- sort(unique(dt$doseA))
    dosesB <- sort(unique(dt$doseB))
    dt[, i := match(doseA, dosesA)]
    dt[, j := match(doseB, dosesB)]
    interior <- dt[i > 1 & j > 1]

    if (type == "outlier") {
      if (nrow(interior) == 0) return(dt)
      k <- cfg$experiments$perturbation$outlier_k %||% 4
      s <- stats::sd(dt$response, na.rm = TRUE)
      if (!is.finite(s) || s == 0) s <- 0.05
      pick <- interior[sample.int(nrow(interior), 1)]
      sign <- sample(c(-1, 1), 1)
      dt[experiment_id == pick$experiment_id & doseA == pick$doseA & doseB == pick$doseB,
         response := clamp01(response + sign * k * s, eps = 0)]
      return(dt[])
    }
    if (type == "leave_one_out") {
      if (nrow(interior) == 0) return(dt)
      pick <- interior[sample.int(nrow(interior), 1)]
      return(dt[!(doseA == pick$doseA & doseB == pick$doseB)])
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
    stop("Unknown perturbation type: ", type)
  }

  types <- c("outlier", "leave_one_out", "drop_row", "drop_col")
  rows <- list()
  scores_to_long <- function(scores_dt) {
    cols <- setdiff(names(scores_dt), "experiment_id")
    data.table::data.table(
      experiment_id = scores_dt$experiment_id[1],
      metric = cols,
      value = as.numeric(scores_dt[1, ..cols])
    )
  }
  for (df in mats) {
    base <- compute_scores(df)
    base_long <- scores_to_long(base)
    for (tp in types) {
      dfp <- perturb_df(df, tp, cfg)
      pert <- compute_scores(dfp)
      pert_long <- scores_to_long(pert)
      row <- merge(base_long, pert_long, by = c("experiment_id", "metric"), suffixes = c("_base", "_pert"), all = TRUE)
      row[, perturbation := tp]
      rows[[length(rows) + 1L]] <- row[, .(experiment_id, metric, perturbation, base = value_base, perturbed = value_pert)]
    }
  }

  dt <- data.table::rbindlist(rows, fill = TRUE)
  dt[, delta := perturbed - base]
  write_table(dt, file.path(out_results, "perturbation_deltas.parquet"))

  summ <- dt[, .(
    abs_delta_median = stats::median(abs(delta), na.rm = TRUE),
    abs_delta_mean = mean(abs(delta), na.rm = TRUE),
    sign_flip = mean(sign(base) != sign(perturbed), na.rm = TRUE)
  ), by = .(metric, perturbation)]
  write_table(summ, file.path(out_results, "perturbation_summary.parquet"))

  fig_paths <- plot_perturbation_stability(dt, summ, out_dir = file.path(out_figures, "perturbation"))

  # Diagnostic plots for a few matrices
  diag_dir <- file.path(out_figures, "diagnostics")
  dir_create(diag_dir)
  for (df in mats[seq_len(min(12, length(mats)))]) {
    prop <- compute_proposed_metrics(df, cfg, transform)
    pdf(file.path(diag_dir, paste0(df$experiment_id[1], ".pdf")), width = 10, height = 8)
    print(plot_diagnostics_matrix(prop, title = df$experiment_id[1]))
    dev.off()
  }

  invisible(list(deltas = dt, summary = summ, figures = fig_paths))
}
