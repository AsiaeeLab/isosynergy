plot_clinical_overlap <- function(metrics_dt, out_dir) {
  if (!requireNamespace("ggplot2", quietly = TRUE)) {
    stop("Package 'ggplot2' is required. Install with: install.packages('ggplot2')")
  }
  if (!requireNamespace("data.table", quietly = TRUE)) {
    stop("Package 'data.table' is required. Install with: install.packages('data.table')")
  }
  dir_create(out_dir)
  dt <- data.table::copy(metrics_dt)
  cols <- c("S2", "t_int", "bliss_mean", "hsa_mean", "loewe_mean", "zip_mean")
  long <- data.table::melt(
    dt,
    id.vars = c("experiment_id", "source", "drugA", "drugB", "cell_line", "clinical"),
    measure.vars = cols,
    variable.name = "metric",
    value.name = "value"
  )
  path <- file.path(out_dir, "clinical_vs_nonclinical.pdf")
  gg <- ggplot2::ggplot(long[is.finite(value)], ggplot2::aes(x = clinical, y = value)) +
    ggplot2::geom_boxplot(outlier.size = 0.4) +
    ggplot2::facet_wrap(~ metric, scales = "free_y") +
    ggplot2::labs(x = "Clinical drug pair", y = NULL, title = "Clinical overlap (descriptive)") +
    ggplot2::theme_minimal(base_size = 11)
  ggplot2::ggsave(path, gg, width = 10, height = 5)
  c(clinical_boxplot = path)
}

