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

source("R/pipeline.R")
source("R/config.R")

option_list <- list(
  make_option(c("-c", "--config"), type = "character", default = "configs/default.yaml",
              help = "Path to YAML config", metavar = "path"),
  make_option(c("-s", "--steps"), type = "character", default = "all",
              help = "Comma-separated: all,public,clinical,reproducibility,perturbation,simulation",
              metavar = "steps")
)

opt <- parse_args(OptionParser(option_list = option_list))
cfg <- read_config(opt$config)

steps <- strsplit(opt$steps, ",", fixed = TRUE)[[1]]
steps <- trimws(steps)

run_pipeline(cfg, steps = steps)
