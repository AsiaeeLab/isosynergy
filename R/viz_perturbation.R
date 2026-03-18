plot_perturbation_stability <- function(delta_dt, summary_dt, out_dir) {
  if (!requireNamespace("ggplot2", quietly = TRUE)) {
    stop("Package 'ggplot2' is required. Install with: install.packages('ggplot2')")
  }
  if (!requireNamespace("data.table", quietly = TRUE)) {
    stop("Package 'data.table' is required. Install with: install.packages('data.table')")
  }
  dir_create(out_dir)
  dt <- data.table::copy(delta_dt)

  box_path <- file.path(out_dir, "abs_delta_boxplot.pdf")
  gg <- ggplot2::ggplot(dt[is.finite(delta)], ggplot2::aes(x = metric, y = abs(delta))) +
    ggplot2::geom_boxplot(outlier.size = 0.5) +
    ggplot2::facet_wrap(~ perturbation, scales = "free_y") +
    ggplot2::labs(x = NULL, y = "|delta|", title = "Perturbation sensitivity") +
    ggplot2::theme_minimal(base_size = 11) +
    ggplot2::theme(axis.text.x = ggplot2::element_text(angle = 45, hjust = 1))
  ggplot2::ggsave(box_path, gg, width = 10, height = 6)

  flip_path <- file.path(out_dir, "sign_flip.pdf")
  gg2 <- ggplot2::ggplot(summary_dt, ggplot2::aes(x = metric, y = sign_flip, fill = perturbation)) +
    ggplot2::geom_col(position = "dodge") +
    ggplot2::scale_y_continuous(limits = c(0, 1)) +
    ggplot2::labs(x = NULL, y = "Sign-flip frequency", title = "Sign flips under perturbations") +
    ggplot2::theme_minimal(base_size = 11) +
    ggplot2::theme(axis.text.x = ggplot2::element_text(angle = 45, hjust = 1))
  ggplot2::ggsave(flip_path, gg2, width = 10, height = 4)

  c(abs_delta = box_path, sign_flip = flip_path)
}
