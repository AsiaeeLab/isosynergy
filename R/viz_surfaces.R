plot_surface <- function(mat, title = NULL) {
  if (!requireNamespace("ggplot2", quietly = TRUE)) {
    stop("Package 'ggplot2' is required. Install with: install.packages('ggplot2')")
  }
  if (!requireNamespace("data.table", quietly = TRUE)) {
    stop("Package 'data.table' is required. Install with: install.packages('data.table')")
  }
  dt <- data.table::data.table(
    i = rep(seq_len(nrow(mat)), times = ncol(mat)),
    j = rep(seq_len(ncol(mat)), each = nrow(mat)),
    value = as.numeric(mat)
  )
  ggplot2::ggplot(dt, ggplot2::aes(x = j, y = i, fill = value)) +
    ggplot2::geom_raster() +
    ggplot2::scale_y_reverse() +
    ggplot2::labs(title = title, x = "Dose B index", y = "Dose A index", fill = NULL) +
    ggplot2::theme_minimal(base_size = 11) +
    ggplot2::theme(panel.grid = ggplot2::element_blank())
}

plot_diagnostics_matrix <- function(prop, title = NULL) {
  if (!requireNamespace("patchwork", quietly = TRUE)) {
    stop("Package 'patchwork' is required. Install with: install.packages('patchwork')")
  }
  p1 <- plot_surface(prop$barZ, title = paste0(title, " | barZ"))
  p2 <- plot_surface(prop$theta_iso, title = "Theta_iso")
  p3 <- plot_surface(prop$theta_add, title = "Theta_add")
  p4 <- plot_surface(prop$delta, title = "Delta = iso - add")
  (p1 | p2) / (p3 | p4)
}

