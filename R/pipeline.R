source("R/utils.R")
source("R/cache.R")
source("R/config.R")
source("R/transforms.R")
source("R/matrix_stats.R")
source("R/osqp_helpers.R")
source("R/isotonic_2d.R")
source("R/additive_ordered.R")
source("R/interaction.R")
source("R/bootstrap.R")
source("R/baselines_common.R")
source("R/baselines_bliss.R")
source("R/baselines_hsa.R")
source("R/baselines_loewe.R")
source("R/baselines_zip.R")
source("R/simulation.R")
source("R/metrics.R")
source("R/experiments_simulation.R")
source("R/experiments_perturbation.R")
source("R/data_standard.R")
source("R/public_data_ingest.R")
source("R/experiments_public.R")
source("R/viz_public.R")
source("R/experiments_clinical.R")
source("R/viz_clinical.R")
source("R/experiments_reproducibility.R")
source("R/viz_reproducibility.R")
source("R/viz_surfaces.R")
source("R/viz_simulation.R")
source("R/viz_perturbation.R")

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
