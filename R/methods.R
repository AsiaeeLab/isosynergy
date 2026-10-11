#' Summarize a SIR Fit
#' @param object,x A `sir_fit` object, or its summary for the summary print method.
#' @param ... Additional arguments (unused).
#' @param digits Number of significant digits to print.
#' @return `summary()` returns a `summary.sir_fit` list of scalar results,
#'   region claims, fragility flags, and the data log. Print methods return their
#'   input invisibly.
#' @export
summary.sir_fit <- function(object, ...) {
  fields <- c("design", "activity", "mean_excess", "mean_excess_se", "p_mean",
              "mean_test", "t_std", "p_tstd", "p_scan", "stage", "B_used",
              "claims", "fragility", "log")
  structure(object[fields], class = "summary.sir_fit")
}

#' @rdname summary.sir_fit
#' @export
print.sir_fit <- function(x, ..., digits = 4L) {
  print(summary(x), digits = digits, ...)
  invisible(x)
}

#' @rdname summary.sir_fit
#' @export
print.summary.sir_fit <- function(x, ..., digits = 4L) {
  f <- function(z) format(signif(z, digits), trim = TRUE)
  cat("SIR fit: design ", x$design, ", ", paste(unlist(x$log$analyzed_shape), collapse = " x "),
      ", ", x$log$n_replicates, " replicate(s)\n", sep = "")
  cat("  Activity: ", f(x$activity), "\n", sep = "")
  cat("  Mean Bliss excess: ", f(x$mean_excess), " (SE ", f(x$mean_excess_se),
      "); companion p = ", f(x$p_mean), "\n", sep = "")
  cat("  SIR statistic: ", f(x$t_std), "; p = ", f(x$p_tstd),
      " (stage ", x$stage, ", B = ", x$B_used, ")\n", sep = "")
  cat("  Scan p = ", f(x$p_scan), "; ", length(x$claims), " region claim(s)\n", sep = "")
  if (isTRUE(x$fragility$computed))
    cat("  Fragility: cell flag ", x$fragility$loco_flag,
        "; single-agent flag ", x$fragility$sa_flag, "\n", sep = "")
  cat("  Data handling: ", length(x$log$dropped_doses), " dose(s) dropped, ",
      length(x$log$missing_cells), " missing combination well(s), ",
      length(x$log$screened), " replicate value(s) screened\n", sep = "")
  for (note in x$log$notes) cat("  Note: ", note, "\n", sep = "")
  invisible(x)
}

#' Plot Observed Viability, Bliss Prediction and Isotonic Excess
#'
#' Panels use equal spacing by dose index, with the supplied dose labels when
#' present. The observed and Bliss panels share a color scale; positive excess
#' is red and negative excess is blue. Missing wells are blank. The excess panel
#' contains only combination wells on the analyzed grid.
#' @param x A `sir_fit` object.
#' @param ... Additional arguments passed to [graphics::image()].
#' @return Invisibly, the three plotted matrices.
#' @export
plot.sir_fit <- function(x, ...) {
  observed <- x$analyzed_matrix
  bliss <- outer(c(1, as.vector(x$a_hat)), c(1, as.vector(x$b_hat)))
  excess <- x$excess_map
  da <- if (is.null(x$doses_a)) unlist(x$log$rows_kept) else as.vector(x$doses_a)
  db <- if (is.null(x$doses_b)) unlist(x$log$cols_kept) else as.vector(x$doses_b)
  old <- graphics::par(mfrow = c(1, 3), mar = c(4, 4, 4, 1))
  on.exit(graphics::par(old))
  lim <- range(c(observed, bliss), finite = TRUE)
  if (diff(lim) == 0) lim <- lim + c(-0.01, 0.01)
  em <- max(abs(excess), na.rm = TRUE)
  if (!is.finite(em) || em == 0) em <- 0.01
  draw <- function(z, title, a, b, zlim, col) {
    graphics::image(seq_len(nrow(z)), seq_len(ncol(z)), z, xlab = "Drug A dose",
                    ylab = "Drug B dose", axes = FALSE, main = title,
                    zlim = zlim, col = col, ...)
    graphics::axis(1, at = seq_along(a), labels = signif(a, 3))
    graphics::axis(2, at = seq_along(b), labels = signif(b, 3))
    graphics::box()
    graphics::mtext(paste("Color range:", paste(signif(zlim, 3), collapse = " to ")),
                    side = 3, line = 0.25, cex = 0.65)
  }
  cols <- grDevices::hcl.colors(64, "YlGnBu", rev = TRUE)
  draw(observed, "Observed viability", da, db, lim, cols)
  draw(bliss, "Bliss prediction", da, db, lim, cols)
  draw(excess, "Isotonic excess", da[-1], db[-1], c(-em, em),
       grDevices::colorRampPalette(c("#2166ac", "white", "#b2182b"))(65))
  invisible(list(observed = observed, bliss = bliss, excess = excess))
}
