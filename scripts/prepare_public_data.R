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
  library(optparse)
})

source("R/public_data_ingest.R")

option_list <- list(
  make_option("--raw", type = "character", default = "data/raw/nci_almanac/ComboDrugGrowth_Nov2017.csv",
              help = "Path to NCI-ALMANAC raw combo CSV", metavar = "path"),
  make_option("--cell-map", type = "character", default = "data/raw/nci_almanac/NCI60_CELLNAME_to_Combo.txt",
              help = "Path to cell-line mapping file", metavar = "path"),
  make_option("--drug-map", type = "character", default = "data/raw/nci_almanac/NCI_IOA_AOA_drugs.tsv",
              help = "Path to NSC drug metadata", metavar = "path"),
  make_option("--out", type = "character", default = "data/processed/nci_almanac_matrices.parquet",
              help = "Output standardized parquet", metavar = "path"),
  make_option("--summary", type = "character", default = "results/nci_almanac_grid_summary.parquet",
              help = "Output parquet with per-experiment grid stats", metavar = "path"),
  make_option("--no-clamp", action = "store_true", default = FALSE,
              help = "Disable [0,1] clamping of percent growth values"),
  make_option("--min-grid", type = "character", default = "3,3",
              help = "Minimum doses in each margin to keep an experiment (e.g., 3,3)", metavar = "I,J")
)

opt <- parse_args(OptionParser(option_list = option_list))
min_grid <- as.integer(strsplit(opt$`min-grid`, ",", fixed = TRUE)[[1]])
if (length(min_grid) != 2 || any(is.na(min_grid))) stop("Invalid --min-grid, expected two integers separated by a comma.")

out_dir <- normalizePath(dirname(opt$out), mustWork = FALSE)
summary_dir <- normalizePath(dirname(opt$summary), mustWork = FALSE)
dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)
dir.create(summary_dir, recursive = TRUE, showWarnings = FALSE)

std <- write_nci_almanac_standard(
  raw_csv = opt$raw,
  out_path = opt$out,
  cell_map_path = opt$`cell-map`,
  drug_map_path = opt$`drug-map`,
  clamp_01 = !isTRUE(opt$`no-clamp`),
  min_grid = min_grid,
  summary_path = opt$summary
)

grid <- std$grid
message("Standardized experiments: ", nrow(grid))
message("  median grid: ", stats::median(grid$n_i), " x ", stats::median(grid$n_j))
message("  fraction with mono edges: ",
        sprintf("%.1f%% / %.1f%%",
                100 * mean(grid$has_monoA, na.rm = TRUE),
                100 * mean(grid$has_monoB, na.rm = TRUE)))
