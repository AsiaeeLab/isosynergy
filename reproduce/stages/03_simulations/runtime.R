script <- sub("^--file=", "", grep("^--file=",commandArgs(FALSE),value=TRUE)[1])
root <- normalizePath(file.path(dirname(script), "..", ".."))
repo <- dirname(root)
smoke <- Sys.getenv("SIR_SMOKE", "0") == "1"
outdir <- file.path(root, "output", if (smoke) "smoke" else "", "runtime")
dir.create(outdir, recursive=TRUE, showWarnings=FALSE)
cache <- file.path(dirname(outdir), "cache", "numba")
dir.create(cache, recursive=TRUE, showWarnings=FALSE)
Sys.setenv(NUMBA_CACHE_DIR=cache, PYTHONDONTWRITEBYTECODE="1")
for (file in c("python.R", "noise.R", "fit.R", "methods.R")) source(file.path(repo,"R",file))
.sir_env$module <- reticulate::import_from_path("sirsyn", path=file.path(repo,"inst","python"), convert=FALSE)
a <- c(1, .95, .8, .55, .3)
b <- c(1, .9, .7, .45, .25)
Y <- outer(a, b) + matrix(sin(0:24) * .04, 5, 5, byrow = TRUE)
Y[1, 1] <- 1
noise <- sir_noise_parametric(sd = .04)
rows <- list()
for (design in c("S", "P")) for (B in c(399L, 9999L)) {
  run <- function(seed) sir_fit(Y, design = design, noise = noise, B = B,
                               seed = seed, two_stage = FALSE, fragility = "never")
  invisible(run(0L)) # compile/warm up each path, not timed
  timing <- vapply(seq_len(if (smoke) 2L else 30L), function(i) {
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
