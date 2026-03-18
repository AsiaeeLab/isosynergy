plot_reproducibility_scatter <- function(wide_dt, out_dir) {
  if (!requireNamespace("ggplot2", quietly = TRUE)) {
    stop("Package 'ggplot2' is required. Install with: install.packages('ggplot2')")
  }
  if (!requireNamespace("data.table", quietly = TRUE)) {
    stop("Package 'data.table' is required. Install with: install.packages('data.table')")
  }
  dir_create(out_dir)
  d <- data.table::copy(wide_dt)
  path <- file.path(out_dir, "S2_replicate_scatter.pdf")
  gg <- ggplot2::ggplot(d[is.finite(S2_1) & is.finite(S2_2)], ggplot2::aes(x = S2_1, y = S2_2)) +
    ggplot2::geom_point(alpha = 0.3, size = 0.7) +
    ggplot2::geom_abline(slope = 1, intercept = 0, linetype = "dashed") +
    ggplot2::labs(x = "S2 (replicate 1)", y = "S2 (replicate 2)", title = "Replicate concordance (scalar)") +
    ggplot2::theme_minimal(base_size = 11)
  ggplot2::ggsave(path, gg, width = 5, height = 4)
  c(S2_scatter = path)
}

