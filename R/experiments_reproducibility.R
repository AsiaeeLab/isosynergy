run_reproducibility <- function(cfg, out_results, out_figures) {
  metrics_path <- file.path(out_results, "public_metrics.parquet")
  if (!file.exists(metrics_path)) {
    message("[", now_utc(), "] Missing public metrics table; run with --steps public first.")
    return(invisible(NULL))
  }
  if (!requireNamespace("arrow", quietly = TRUE) || !requireNamespace("data.table", quietly = TRUE)) {
    stop("Packages 'arrow' and 'data.table' are required.")
  }
  dt <- data.table::as.data.table(arrow::read_parquet(metrics_path))
  key_cols <- c("source", "drugA", "drugB", "cell_line")
  dt <- dt[complete.cases(dt[, ..key_cols])]

  dt[, group_id := do.call(paste, c(.SD, sep = "||")), .SDcols = key_cols]
  dt <- dt[is.finite(S2)]
  groups <- dt[, .N, by = group_id][N >= 2]$group_id
  if (length(groups) == 0) {
    message("[", now_utc(), "] No replicate groups found; skipping reproducibility.")
    return(invisible(NULL))
  }

  pick2 <- function(g) {
    sub <- dt[group_id == g][order(experiment_id)]
    sub[1:2]
  }
  pairs <- data.table::rbindlist(lapply(groups, pick2))
  pairs[, rep_idx := rep(1:2, times = length(groups))]
  wide <- data.table::dcast(pairs, group_id + source + drugA + drugB + cell_line ~ rep_idx,
                            value.var = c("S2", "t_int", "p_value", "bliss_mean", "hsa_mean", "loewe_mean", "zip_mean"))

  corr_one <- function(a, b) {
    ok <- is.finite(a) & is.finite(b)
    if (sum(ok) < 5) return(NA_real_)
    stats::cor(a[ok], b[ok])
  }

  metrics <- c("S2", "t_int", "bliss_mean", "hsa_mean", "loewe_mean", "zip_mean")
  corrs <- data.table::data.table(metric = metrics)
  corrs[, correlation := vapply(metrics, function(m) {
    corr_one(wide[[paste0(m, "_1")]], wide[[paste0(m, "_2")]])
  }, numeric(1))]

  write_table(corrs, file.path(out_results, "reproducibility_correlations.csv"))
  fig_paths <- plot_reproducibility_scatter(wide, out_dir = file.path(out_figures, "reproducibility"))
  invisible(list(correlations = corrs, figures = fig_paths))
}

