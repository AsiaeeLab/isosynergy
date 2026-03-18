plot_simulation_benchmark <- function(sim_dt, out_dir) {
  if (!requireNamespace("ggplot2", quietly = TRUE)) {
    stop("Package 'ggplot2' is required. Install with: install.packages('ggplot2')")
  }
  if (!requireNamespace("data.table", quietly = TRUE)) {
    stop("Package 'data.table' is required. Install with: install.packages('data.table')")
  }
  dir_create(out_dir)
  dt <- data.table::copy(sim_dt)

  null <- dt[sim_interaction_strength == 0 & is.finite(p_value)]
  calib_path <- file.path(out_dir, "calibration_ecdf.pdf")
  if (nrow(null) > 0) {
    gg <- ggplot2::ggplot(null, ggplot2::aes(x = p_value)) +
      ggplot2::stat_ecdf(geom = "step") +
      ggplot2::geom_abline(slope = 1, intercept = 0, linetype = "dashed") +
      ggplot2::labs(x = "p-value", y = "ECDF", title = "Type I calibration (null)") +
      ggplot2::theme_minimal(base_size = 11)
    ggplot2::ggsave(calib_path, gg, width = 5, height = 4)
  }

  pow <- dt[is.finite(p_value), .(power_0.05 = mean(p_value < 0.05)), by = .(sim_interaction_strength)]
  power_path <- file.path(out_dir, "power_curve.pdf")
  gg2 <- ggplot2::ggplot(pow, ggplot2::aes(x = sim_interaction_strength, y = power_0.05)) +
    ggplot2::geom_line() +
    ggplot2::geom_point() +
    ggplot2::scale_y_continuous(limits = c(0, 1)) +
    ggplot2::labs(x = "Interaction strength", y = "Power at 0.05", title = "Power curve") +
    ggplot2::theme_minimal(base_size = 11)
  ggplot2::ggsave(power_path, gg2, width = 5, height = 4)

  c(calibration = calib_path, power = power_path)
}

