# Run with SIR installed and RETICULATE_PYTHON set. Set BLAS/OpenMP/Numba
# thread counts to one before starting R; optionally pin the process to one CPU.
# Usage: Rscript runtime.R /path/to/output-directory
args <- commandArgs(trailingOnly = TRUE)
outdir <- if (length(args)) args[1L] else "."
dir.create(outdir, recursive = TRUE, showWarnings = FALSE)
library(SIR)
a <- c(1, .95, .8, .55, .3)
b <- c(1, .9, .7, .45, .25)
# Fixed deterministic data shared with the Python benchmark.
Y <- outer(a, b) + matrix(sin(0:24) * .04, 5, 5, byrow = TRUE)
Y[1, 1] <- 1
noise <- sir_noise_parametric(sd = .04)
rows <- list()
for (design in c("S", "P")) for (B in c(399L, 9999L)) {
  run <- function(seed) sir_fit(Y, design = design, noise = noise, B = B,
                               seed = seed, two_stage = FALSE, fragility = "never")
  invisible(run(0L)) # compile/warm up each path, not timed
  timing <- vapply(1:30, function(i) {
    start <- proc.time()[["elapsed"]]
    invisible(run(as.integer(i)))
    (proc.time()[["elapsed"]] - start) * 1000
  }, numeric(1))
  rows[[length(rows) + 1L]] <- data.frame(engine = "R", design = design, rows = 5,
    columns = 5, B = B, n_std = 200, two_stage = FALSE, fragility = "never",
    repetitions = length(timing), median_ms = median(timing),
    mean_ms = mean(timing), p90_ms = unname(quantile(timing, .9)))
}
utils::write.csv(do.call(rbind, rows), file.path(outdir, "runtime_r_only.csv"), row.names = FALSE)
writeLines(capture.output(sessionInfo()), file.path(outdir, "r_session.txt"))
