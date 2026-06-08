# Pipeline driver. When this file is loaded as part of the SIR package, the
# `source()` calls below are no-ops because R/ does not exist relative to the
# install/load directory. When the file is sourced directly from the repo
# root (e.g., by scripts/run_pipeline.R), the calls chain-load all helpers.
local({
  files <- c(
    "R/utils.R", "R/cache.R", "R/config.R", "R/transforms.R",
    "R/matrix_stats.R", "R/osqp_helpers.R", "R/isotonic_2d.R",
    "R/additive_ordered.R", "R/interaction.R", "R/bootstrap.R",
    "R/baselines_common.R", "R/baselines_bliss.R", "R/baselines_hsa.R",
    "R/baselines_loewe.R", "R/baselines_zip.R", "R/simulation.R",
    "R/metrics.R", "R/experiments_simulation.R",
    "R/experiments_perturbation.R", "R/data_standard.R",
    "R/public_data_ingest.R", "R/experiments_public.R", "R/viz_public.R",
    "R/experiments_clinical.R", "R/viz_clinical.R",
    "R/experiments_reproducibility.R", "R/viz_reproducibility.R",
    "R/viz_surfaces.R", "R/viz_simulation.R", "R/viz_perturbation.R"
  )
  if (all(file.exists(files))) {
    for (f in files) source(f)
  }
})

run_pipeline <- function(cfg, steps = c("all")) {
  out_results <- cfg$project$out_dir_results %||% "results"
  out_figures <- cfg$project$out_dir_figures %||% "figures"
  cache_dir <- cfg$project$cache_dir %||% "cache"
  dir_create(out_results)
  dir_create(out_figures)
  dir_create(cache_dir)

  transform <- make_transform(cfg$transform)
  if (is.null(cfg$project$n_cores)) cfg$project$n_cores <- 1L
  if (requireNamespace("future", quietly = TRUE)) {
    n_avail <- as.integer(future::availableCores(methods = "mc.cores"))
    if (is.finite(n_avail) && n_avail > 0) {
      cfg$project$n_cores <- max(1L, min(as.integer(cfg$project$n_cores), n_avail))
    } else {
      cfg$project$n_cores <- 1L
    }
  } else {
    cfg$project$n_cores <- max(1L, as.integer(cfg$project$n_cores))
  }

  if ("all" %in% steps || "public" %in% steps) {
    pub <- run_public_benchmarks(cfg, transform = transform, out_results = out_results, out_figures = out_figures)
    invisible(pub)
  }

  if ("all" %in% steps || "clinical" %in% steps) {
    clin <- run_clinical_overlap(cfg, out_results = out_results, out_figures = out_figures)
    invisible(clin)
  }

  if ("all" %in% steps || "reproducibility" %in% steps) {
    rep <- run_reproducibility(cfg, out_results = out_results, out_figures = out_figures)
    invisible(rep)
  }

  if ("all" %in% steps || "simulation" %in% steps) {
    sim <- run_simulation_benchmark(cfg, transform = transform, out_results = out_results, out_figures = out_figures)
    invisible(sim)
  }

  if ("all" %in% steps || "perturbation" %in% steps) {
    pert <- run_perturbation_experiment(cfg, transform = transform, out_results = out_results, out_figures = out_figures)
    invisible(pert)
  }

  invisible(TRUE)
}
