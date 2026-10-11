# Shared setup for the notebooks in this folder.
#
# The notebooks never rerun the pipeline. Every result is read from the pipeline output
# (../output, present after `bash reproduce/run_all.sh`) when it exists, and otherwise from the
# small copies shipped in data/. Figures are drawn here from those tables.

suppressPackageStartupMessages({
  library(ggplot2)
  library(knitr)
  library(kableExtra)
})

nb_roots <- c(file.path("..", "output"), "data")

# Path of a pipeline result, e.g. result_path("simulations/grid_size.csv").
result_path <- function(rel) {
  for (root in nb_roots) {
    p <- file.path(root, rel)
    if (file.exists(p)) return(p)
  }
  stop(sprintf(paste0("Result '%s' not found in ../output or data/. ",
                      "Run `bash reproduce/run_all.sh` from the repository root, ",
                      "or restore reproduce/notebooks/data/."), rel), call. = FALSE)
}

# Read a CSV, JSON or Parquet result into memory.
read_result <- function(rel, ...) {
  p <- result_path(rel)
  switch(tolower(tools::file_ext(p)),
    csv = utils::read.csv(p, check.names = FALSE, ...),
    json = jsonlite::fromJSON(p, ...),
    parquet = {
      pd <- reticulate::import("pandas", convert = FALSE)
      reticulate::py_to_r(pd$read_parquet(p))
    },
    stop("Unsupported file type: ", p, call. = FALSE))
}

# The released per-unit DrugCombDB table (Zenodo 10.5281/zenodo.23302266). Uses the copy in
# dataset_release/ when present; otherwise downloads it once into data/.
release_table_path <- function() {
  local <- file.path("..", "..", "dataset_release", "sir_drugcombdb_unit_results.parquet")
  if (file.exists(local)) return(local)
  dest <- file.path("data", "sir_drugcombdb_unit_results.parquet")
  if (!file.exists(dest)) {
    dir.create("data", showWarnings = FALSE)
    utils::download.file(
      "https://zenodo.org/records/23302266/files/sir_drugcombdb_unit_results.parquet?download=1",
      dest, mode = "wb")
  }
  dest
}

# Tables and figures share one look.
kbl_report <- function(x, ...) {
  kableExtra::kable_styling(knitr::kable(x, ...),
                            bootstrap_options = c("striped", "hover"), full_width = FALSE)
}

theme_report <- theme_minimal(base_size = 12) +
  theme(panel.grid.minor = element_blank(),
        plot.title = element_text(face = "bold"),
        legend.position = "bottom")
