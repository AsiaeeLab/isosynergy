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
  library(data.table)
  library(arrow)
})

source("R/utils.R")
source("R/public_data_ingest.R")

option_list <- list(
  make_option("--raw-dir", dest = "raw_dir", type = "character", default = "data/raw/drugcombdb",
              help = "Raw folder containing DrugCombDB CSVs", metavar = "path"),
  make_option("--out", type = "character", default = "data/processed/drugcomb_matrices.parquet",
              help = "Output parquet path", metavar = "path"),
  make_option("--min-grid-i", dest = "min_grid_i", type = "integer", default = 3L,
              help = "Minimum unique doseA levels per matrix", metavar = "int"),
  make_option("--min-grid-j", dest = "min_grid_j", type = "integer", default = 3L,
              help = "Minimum unique doseB levels per matrix", metavar = "int"),
  make_option("--no-clamp", action = "store_true", default = FALSE,
              help = "Disable clamping response to [0,1] after converting from percent viability"),
  make_option("--no-unit-normalize", action = "store_true", default = FALSE,
              help = "Disable concentration unit normalization to uM")
)

opt <- parse_args(OptionParser(option_list = option_list))

dir_create(opt$raw_dir)
dir_create("data/processed")
dir_create("results")

response_csv <- file.path(opt$raw_dir, "drugcombs_response.csv")
scored_csv <- file.path(opt$raw_dir, "drugcombs_scored.csv")

if (!file.exists(response_csv) || !file.exists(scored_csv)) {
  stop(
    "Missing raw files. Download these from http://drugcombdb.denglab.org/download and place them at:\n",
    "  - ", response_csv, "\n",
    "  - ", scored_csv, "\n"
  )
}

summary_path <- "results/drugcombdb_grid_summary.parquet"
scores_path <- "results/drugcombdb_block_scores.parquet"

std <- write_drugcombdb_standard(
  response_csv = response_csv,
  scored_csv = scored_csv,
  out_path = opt$out,
  clamp_01 = !isTRUE(opt$`no-clamp`),
  min_grid = c(opt$min_grid_i, opt$min_grid_j),
  summary_path = summary_path,
  scores_path = scores_path,
  normalize_units = !isTRUE(opt$`no-unit-normalize`)
)

checksum_one <- function(path) {
  out <- tryCatch(system2("sha256sum", shQuote(path), stdout = TRUE, stderr = TRUE), error = function(e) NULL)
  if (!is.null(out) && length(out) > 0 && !any(grepl("not found", out, fixed = TRUE))) {
    return(strsplit(out[1], "\\s+")[[1]][1])
  }
  unname(tools::md5sum(path)[1])
}

readme_path <- file.path(opt$raw_dir, "README.md")
lines <- c(
  "# DrugCombDB raw downloads",
  "- Source page: http://drugcombdb.denglab.org/download",
  paste0("- Accessed: ", Sys.Date()),
  "",
  "## Files",
  paste0("- `drugcombs_response.csv` (dose–response matrices, percent viability; huge)"),
  paste0("  - sha256/md5: ", checksum_one(response_csv)),
  paste0("- `drugcombs_scored.csv` (per-matrix summary synergy scores + cell line metadata)"),
  paste0("  - sha256/md5: ", checksum_one(scored_csv)),
  "",
  "## Notes",
  "- `drugcombs_response.csv` does not include the cell line; we join `BlockID` (response) to `ID` (scored) to obtain `cell_line`.",
  "- We convert `Response` from percent viability to fraction viability (`response = clamp(Response/100, 0, 1)`) and standardize concentrations to `uM` when units are provided."
)
writeLines(lines, readme_path)

message("Wrote: ", opt$out)
message("Wrote: ", summary_path)
message("Wrote: ", scores_path)
message("Wrote: ", readme_path)
