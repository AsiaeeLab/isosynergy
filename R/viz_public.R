plot_public_benchmarks <- function(dt, out_dir) {
  if (!requireNamespace("ggplot2", quietly = TRUE)) {
    stop("Package 'ggplot2' is required. Install with: install.packages('ggplot2')")
  }
  if (!requireNamespace("data.table", quietly = TRUE)) {
    stop("Package 'data.table' is required. Install with: install.packages('data.table')")
  }
  dir_create(out_dir)
  d <- data.table::copy(dt)
  d <- d[is.finite(S2)]

  p1_path <- file.path(out_dir, "baseline_vs_S2.pdf")
  long <- data.table::melt(
    d,
    id.vars = c("experiment_id", "source", "drugA", "drugB", "cell_line", "S2"),
    measure.vars = c("bliss_mean", "hsa_mean", "loewe_mean", "zip_mean"),
    variable.name = "baseline",
    value.name = "baseline_mean"
  )
  gg <- ggplot2::ggplot(long[is.finite(baseline_mean)], ggplot2::aes(x = baseline_mean, y = S2)) +
    ggplot2::geom_point(alpha = 0.25, size = 0.6) +
    ggplot2::facet_wrap(~ baseline, scales = "free_x") +
    ggplot2::labs(x = "Baseline mean synergy (interior)", y = "Proposed S2", title = "Baselines vs proposed interaction") +
    ggplot2::theme_minimal(base_size = 11)
  ggplot2::ggsave(p1_path, gg, width = 10, height = 4)

  p2_path <- file.path(out_dir, "pvalue_hist.pdf")
  gg2 <- ggplot2::ggplot(d[is.finite(p_value)], ggplot2::aes(x = p_value)) +
    ggplot2::geom_histogram(bins = 30) +
    ggplot2::labs(x = "p-value", y = "Count", title = "Bootstrap p-values") +
    ggplot2::theme_minimal(base_size = 11)
  ggplot2::ggsave(p2_path, gg2, width = 5, height = 4)

  c(baseline_vs_S2 = p1_path, pvalue_hist = p2_path)
}

