#!/usr/bin/env Rscript

get_script_path <- function() {
  cmd <- commandArgs(trailingOnly = FALSE)
  file_arg <- grep("^--file=", cmd, value = TRUE)
  if (length(file_arg) == 0) return(NULL)
  sub("^--file=", "", file_arg[1])
}

script_path <- get_script_path()
if (!is.null(script_path)) {
  root <- normalizePath(file.path(dirname(script_path), ".."), winslash = "/", mustWork = FALSE)
  if (file.exists(file.path(root, "renv.lock"))) setwd(root)
}

suppressPackageStartupMessages({
  library(data.table)
  library(arrow)
  library(synergyfinder)
})
source("R/utils.R")

dir.create("data/raw/drugcomb_demo", recursive = TRUE, showWarnings = FALSE)
dir.create("data/processed", showWarnings = FALSE)

data(mathews_screening_data)
dt <- as.data.table(mathews_screening_data)

raw_path <- "data/raw/drugcomb_demo/mathews_screening_data.csv"
fwrite(dt, raw_path)

dt[, response := clamp01(Response / 100, eps = 0)]
dt[, doseA := ConcRow]
dt[, doseB := ConcCol]
dt[, replicate := as.character(Replicate)]
dt[, cell_line := "mathews_panel"]
dt[, source := "drugcomb"]
dt[, response_mode := "viability"]
dt[, drugA := DrugRow]
dt[, drugB := DrugCol]
dt[, experiment_id := sprintf("drugcomb_demo__%s__%s__block%s", drugA, drugB, BlockID)]

std <- dt[, .(experiment_id, source, drugA, drugB, cell_line, doseA, doseB, response, response_mode, replicate)]
out_path <- "data/processed/drugcomb_demo_matrices.parquet"
arrow::write_parquet(std, out_path)

grid <- std[, .(
  n = .N,
  n_i = uniqueN(doseA),
  n_j = uniqueN(doseB),
  has_monoA = any(doseB == 0),
  has_monoB = any(doseA == 0),
  mono_activity = mean(1 - response[doseA == 0 | doseB == 0], na.rm = TRUE)
), by = experiment_id]
arrow::write_parquet(grid, "results/drugcomb_demo_grid_summary.parquet")

writeLines(c(
  "# DrugComb demo raw data",
  "- Source: `synergyfinder` package dataset `mathews_screening_data` (public demo combo matrix).",
  paste0("- Accessed: ", Sys.Date()),
  "- Raw file: mathews_screening_data.csv (response in % viability)."
), "data/raw/drugcomb_demo/README.md")

message("Wrote ", out_path, " and summary results/drugcomb_demo_grid_summary.parquet")
