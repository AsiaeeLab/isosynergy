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
  library(jsonlite)
})

`%||%` <- function(x, y) if (is.null(x)) y else x

release_dir <- "dataset_release"
progress_path <- file.path(release_dir, "PROGRESS.md")
sir_checkpoint_dir <- "results/drugcombdb_sir_universe.parquet"
sir_checkpoint_json <- "results/drugcombdb_sir_universe.checkpoint.json"
processed_path <- "data/processed/drugcomb_matrices.parquet"
block_scores_path <- "results/drugcombdb_block_scores.parquet"
main_dataset_path <- file.path(release_dir, "sir_drugcombdb_synergy_calls.parquet")
schema_path <- file.path(release_dir, "schema.json")
readme_path <- file.path(release_dir, "README.md")
citation_path <- file.path(release_dir, "CITATION.cff")
checksums_path <- file.path(release_dir, "checksums.txt")

iso_now <- function() {
  format(as.POSIXct(Sys.time(), tz = "UTC"), "%Y-%m-%dT%H:%M:%SZ", tz = "UTC")
}

sync_path <- function(path) {
  sync_bin <- Sys.which("sync")
  if (!nzchar(sync_bin)) return(invisible(FALSE))
  ok <- suppressWarnings(system2(sync_bin, c("-f", path), stdout = FALSE, stderr = FALSE))
  if (!identical(ok, 0L)) suppressWarnings(system2(sync_bin, stdout = FALSE, stderr = FALSE))
  invisible(TRUE)
}

ensure_progress <- function(path) {
  dir.create(dirname(path), showWarnings = FALSE, recursive = TRUE)
  if (!file.exists(path)) {
    con <- file(path, open = "a")
    on.exit(close(con), add = TRUE)
    writeLines(c(
      "# DrugCombDB SIR Universe Release Progress",
      "",
      "| timestamp | phase | event | details |",
      "| --- | --- | --- | --- |"
    ), con)
    flush(con)
    sync_path(path)
  }
}

escape_md_cell <- function(x) {
  x <- gsub("\n", "<br>", as.character(x), fixed = TRUE)
  gsub("|", "\\\\|", x, fixed = TRUE)
}

append_progress <- function(phase, event, details) {
  ensure_progress(progress_path)
  con <- file(progress_path, open = "a")
  on.exit(close(con), add = TRUE)
  writeLines(sprintf("| %s | %s | %s | %s |",
                     iso_now(), escape_md_cell(phase), escape_md_cell(event), escape_md_cell(details)), con)
  flush(con)
  sync_path(progress_path)
  invisible(TRUE)
}

part_files <- function(out_dir) {
  if (!dir.exists(out_dir)) return(character(0))
  sort(list.files(out_dir, pattern = "^chunk_[0-9]{6}\\.parquet$", full.names = TRUE))
}

read_sir_checkpoint <- function(path) {
  files <- part_files(path)
  if (length(files) == 0) stop("No SIR checkpoint part files found at: ", path)
  rbindlist(lapply(files, function(f) as.data.table(read_parquet(f, as_data_frame = TRUE))),
            use.names = TRUE, fill = TRUE)
}

atomic_write_text <- function(lines, path) {
  dir.create(dirname(path), showWarnings = FALSE, recursive = TRUE)
  tmp <- file.path(dirname(path), sprintf(".%s.tmp.%s", basename(path), Sys.getpid()))
  writeLines(lines, tmp, useBytes = TRUE)
  sync_path(tmp)
  if (!file.rename(tmp, path)) stop("Failed atomic rename for: ", path)
  sync_path(path)
  sync_path(dirname(path))
  invisible(path)
}

atomic_write_json <- function(x, path) {
  dir.create(dirname(path), showWarnings = FALSE, recursive = TRUE)
  tmp <- file.path(dirname(path), sprintf(".%s.tmp.%s", basename(path), Sys.getpid()))
  write_json(x, tmp, auto_unbox = TRUE, pretty = TRUE, null = "null")
  sync_path(tmp)
  if (!file.rename(tmp, path)) stop("Failed atomic rename for: ", path)
  sync_path(path)
  sync_path(dirname(path))
  invisible(path)
}

atomic_write_parquet <- function(dt, path) {
  dir.create(dirname(path), showWarnings = FALSE, recursive = TRUE)
  tmp <- file.path(dirname(path), sprintf(".%s.tmp.%s", basename(path), Sys.getpid()))
  write_parquet(dt, tmp, compression = "zstd")
  sync_path(tmp)
  if (!file.rename(tmp, path)) stop("Failed atomic rename for: ", path)
  sync_path(path)
  sync_path(dirname(path))
  invisible(path)
}

get_git_commit <- function() {
  out <- tryCatch(system2("git", c("rev-parse", "HEAD"), stdout = TRUE, stderr = FALSE), error = function(e) NA_character_)
  if (length(out) == 0 || !nzchar(out[1])) NA_character_ else out[1]
}

get_drugcombdb_version <- function() {
  readme <- "data/raw/drugcombdb/README.md"
  if (!file.exists(readme)) return("DrugCombDB snapshot used by data/processed/drugcomb_matrices.parquet")
  lines <- readLines(readme, warn = FALSE)
  accessed <- sub("^- Accessed: +", "", grep("^- Accessed:", lines, value = TRUE)[1] %||% "")
  if (nzchar(accessed)) paste0("DrugCombDB download accessed ", accessed) else
    "DrugCombDB snapshot used by data/processed/drugcomb_matrices.parquet"
}

schema_columns <- list(
  list(name = "matrix_id", type = "string", nullable = FALSE),
  list(name = "drug_a_name", type = "string", nullable = FALSE),
  list(name = "drug_b_name", type = "string", nullable = FALSE),
  list(name = "cell_line", type = "string", nullable = FALSE),
  list(name = "source_study", type = "string", nullable = TRUE),
  list(name = "n_doses_a", type = "integer", nullable = FALSE),
  list(name = "n_doses_b", type = "integer", nullable = FALSE),
  list(name = "n_replicates_total", type = "integer", nullable = FALSE),
  list(name = "sir_S2", type = "number", nullable = FALSE),
  list(name = "sir_p_value", type = "number", nullable = FALSE, minimum = 0, maximum = 1),
  list(name = "sir_q_value", type = "number", nullable = FALSE, minimum = 0, maximum = 1),
  list(name = "sir_hit_call", type = "string", nullable = FALSE, enum = c("synergy", "antagonism", "no_call")),
  list(name = "sir_failed", type = "boolean", nullable = FALSE),
  list(name = "bliss_score", type = "number", nullable = TRUE),
  list(name = "bliss_failed", type = "boolean", nullable = FALSE),
  list(name = "bliss_unavailable", type = "boolean", nullable = FALSE),
  list(name = "hsa_score", type = "number", nullable = TRUE),
  list(name = "hsa_failed", type = "boolean", nullable = FALSE),
  list(name = "hsa_unavailable", type = "boolean", nullable = FALSE),
  list(name = "loewe_score", type = "number", nullable = TRUE),
  list(name = "loewe_failed", type = "boolean", nullable = FALSE),
  list(name = "loewe_unavailable", type = "boolean", nullable = FALSE),
  list(name = "zip_score", type = "number", nullable = TRUE),
  list(name = "zip_failed", type = "boolean", nullable = FALSE),
  list(name = "zip_unavailable", type = "boolean", nullable = FALSE),
  list(name = "sir_version", type = "string", nullable = FALSE),
  list(name = "drugcombdb_version", type = "string", nullable = FALSE),
  list(name = "analysis_date", type = "string", nullable = FALSE, format = "date"),
  list(name = "bootstrap_B", type = "integer", nullable = FALSE)
)

column_names <- vapply(schema_columns, `[[`, character(1), "name")

make_schema_json <- function() {
  props <- list()
  for (col in schema_columns) {
    typ <- switch(col$type,
      string = "string",
      integer = "integer",
      number = "number",
      boolean = "boolean",
      col$type
    )
    prop <- list(type = if (isTRUE(col$nullable)) list(typ, "null") else typ)
    for (nm in intersect(names(col), c("minimum", "maximum", "enum", "format"))) prop[[nm]] <- col[[nm]]
    props[[col$name]] <- prop
  }
  list(
    `$schema` = "https://json-schema.org/draft/2020-12/schema",
    title = "SIR-derived DrugCombDB synergy calls row schema",
    type = "object",
    additionalProperties = FALSE,
    required = column_names,
    properties = props,
    `x-parquet-column-order` = column_names,
    `x-parquet-columns` = schema_columns
  )
}

validate_release <- function(path, schema_path, expected_n) {
  schema <- read_json(schema_path, simplifyVector = FALSE)
  expected_cols <- unlist(schema$`x-parquet-column-order`, use.names = FALSE)
  dt <- as.data.table(read_parquet(path, as_data_frame = TRUE))

  if (!identical(names(dt), expected_cols)) {
    stop("Schema validation failed: column order/name mismatch")
  }
  if (nrow(dt) != expected_n) stop("Schema validation failed: row count ", nrow(dt), " != ", expected_n)
  if (anyDuplicated(dt$matrix_id)) stop("Schema validation failed: matrix_id is not unique")

  for (col in schema$`x-parquet-columns`) {
    nm <- col$name
    x <- dt[[nm]]
    if (!isTRUE(col$nullable) && any(is.na(x))) stop("Schema validation failed: non-nullable column has NA: ", nm)
    if (!isTRUE(col$nullable) && all(is.na(x))) stop("Schema validation failed: required column all NA: ", nm)
    if (identical(col$type, "string") && !is.character(x)) stop("Schema validation failed: expected string: ", nm)
    if (identical(col$type, "integer") && !(is.integer(x) || (is.numeric(x) && all(is.na(x) | x == as.integer(x))))) {
      stop("Schema validation failed: expected integer: ", nm)
    }
    if (identical(col$type, "number") && !is.numeric(x)) stop("Schema validation failed: expected numeric: ", nm)
    if (identical(col$type, "boolean") && !is.logical(x)) stop("Schema validation failed: expected boolean: ", nm)
  }

  if (any(!is.finite(dt$sir_S2))) stop("Validation failed: sir_S2 contains non-finite values")
  if (any(!is.finite(dt$sir_p_value) | dt$sir_p_value < 0 | dt$sir_p_value > 1)) {
    stop("Validation failed: sir_p_value outside [0,1]")
  }
  if (any(!is.finite(dt$sir_q_value) | dt$sir_q_value < 0 | dt$sir_q_value > 1)) {
    stop("Validation failed: sir_q_value outside [0,1]")
  }
  bad_hits <- setdiff(unique(dt$sir_hit_call), c("synergy", "antagonism", "no_call"))
  if (length(bad_hits) > 0) stop("Validation failed: unexpected sir_hit_call values: ", paste(bad_hits, collapse = ", "))
  if (any(dt$sir_failed)) stop("Validation failed: sir_failed contains TRUE values")

  invisible(dt)
}

write_checksums <- function(dir_path) {
  files <- sort(list.files(dir_path, recursive = TRUE, full.names = TRUE, all.files = FALSE, no.. = TRUE))
  files <- files[file.info(files)$isdir == FALSE]
  files <- files[normalizePath(files, winslash = "/", mustWork = FALSE) != normalizePath(checksums_path, winslash = "/", mustWork = FALSE)]
  rel <- sub(paste0("^", normalizePath(dir_path, winslash = "/", mustWork = TRUE), "/?"),
             "", normalizePath(files, winslash = "/", mustWork = TRUE))
  out <- system2("sha256sum", files, stdout = TRUE)
  hashes <- sub("\\s+.*$", "", out)
  atomic_write_text(sprintf("%s  %s", hashes, rel), checksums_path)
}

append_progress("assemble", "started", "starting release dataset assembly")

if (!file.exists(sir_checkpoint_json)) stop("Missing SIR checkpoint sidecar: ", sir_checkpoint_json)
sidecar <- read_json(sir_checkpoint_json, simplifyVector = TRUE)
expected_n <- as.integer(sidecar$total_universe_size)
if (!isTRUE(as.integer(sidecar$completed_count) == expected_n)) {
  append_progress("assemble", "error", sprintf("SIR checkpoint incomplete: completed_count=%s expected=%s",
                                               sidecar$completed_count, expected_n))
  stop("SIR checkpoint is incomplete; run scripts/compute_drugcombdb_sir_universe.R first")
}

sir <- read_sir_checkpoint(sir_checkpoint_dir)
if (nrow(sir) != expected_n) stop("SIR checkpoint row count mismatch: ", nrow(sir), " != ", expected_n)
if (anyDuplicated(sir$experiment_id)) stop("SIR checkpoint has duplicate experiment_id rows")
if (any(sir$sir_failed)) stop("SIR checkpoint contains failed rows; inspect PROGRESS.md")
if (any(!is.finite(sir$sir_S2) | !is.finite(sir$sir_p_value) | sir$sir_p_value < 0 | sir$sir_p_value > 1)) {
  stop("SIR checkpoint has invalid SIR numeric outputs")
}

analysis_date <- substr(as.character(sidecar$last_completed_at %||% iso_now()), 1, 10)
drugcombdb_version <- get_drugcombdb_version()
git_commit <- get_git_commit()
sir_version <- "0.1.0"

processed_cols <- c("experiment_id", "drugA", "drugB", "cell_line", "study_source", "doseA", "doseB", "replicate", "block_id")
mat <- as.data.table(read_parquet(processed_path, col_select = processed_cols, as_data_frame = TRUE))
mat[, experiment_id := as.character(experiment_id)]

meta <- mat[, .(
  matrix_id = experiment_id[1],
  drug_a_name = as.character(drugA[1]),
  drug_b_name = as.character(drugB[1]),
  cell_line = as.character(cell_line[1]),
  source_study = as.character(study_source[1]),
  n_replicates_total = as.integer(.N)
), by = experiment_id]

missing_cell_line_n <- meta[is.na(cell_line) | !nzchar(cell_line), .N]
if (missing_cell_line_n > 0) {
  append_progress(
    "assemble", "decision:missing_cell_line",
    sprintf("DECISION: encoded %d matrices with missing processed-cache cell_line metadata as 'not_available' so the required identifier column remains a non-null string; affected rows retain their matrix_id, drugs, and source_study.",
            missing_cell_line_n)
  )
  meta[is.na(cell_line) | !nzchar(cell_line), cell_line := "not_available"]
}

scores <- as.data.table(read_parquet(block_scores_path, as_data_frame = TRUE))
scores[, experiment_id := as.character(experiment_id)]
scores <- unique(scores[, .(
  experiment_id,
  baseline_present = TRUE,
  bliss_score = as.numeric(Bliss),
  hsa_score = as.numeric(HSA),
  loewe_score = as.numeric(Loewe),
  zip_score = as.numeric(ZIP)
)], by = "experiment_id")

setkey(sir, experiment_id)
setkey(meta, experiment_id)
setkey(scores, experiment_id)
dt <- meta[sir]
dt <- scores[dt]
dt[is.na(baseline_present), baseline_present := FALSE]

dt[, sir_q_value := p.adjust(sir_p_value, method = "BH")]
dt[, sir_hit_call := fifelse(
  sir_q_value <= 0.05 & sir_direction == "synergy", "synergy",
  fifelse(sir_q_value <= 0.05 & sir_direction == "antagonism", "antagonism", "no_call")
)]

for (method in c("bliss", "hsa", "loewe", "zip")) {
  score_col <- paste0(method, "_score")
  failed_col <- paste0(method, "_failed")
  unavailable_col <- paste0(method, "_unavailable")
  dt[, (unavailable_col) := !baseline_present]
  dt[, (failed_col) := baseline_present & !is.finite(get(score_col))]
  dt[!is.finite(get(score_col)), (score_col) := NA_real_]
}

dt[, `:=`(
  n_doses_a = as.integer(n_i),
  n_doses_b = as.integer(n_j),
  sir_failed = as.logical(sir_failed),
  sir_version = sir_version,
  drugcombdb_version = drugcombdb_version,
  analysis_date = analysis_date,
  bootstrap_B = as.integer(B)
)]

release <- dt[, ..column_names]
setorder(release, matrix_id)

schema <- make_schema_json()
atomic_write_json(schema, schema_path)
atomic_write_parquet(release, main_dataset_path)

hit_counts <- release[, .N, by = sir_hit_call][order(sir_hit_call)]
missing_baseline_n <- release[bliss_unavailable | hsa_unavailable | loewe_unavailable | zip_unavailable, .N]
append_progress(
  "assemble", "completed",
  sprintf("wrote %s with %d rows; hit_counts=%s; baseline_unavailable_rows=%d",
          main_dataset_path, nrow(release),
          paste(sprintf("%s:%d", hit_counts$sir_hit_call, hit_counts$N), collapse = ", "),
          missing_baseline_n)
)

readme_lines <- c(
  "# SIR-derived DrugCombDB synergy calls",
  "",
  "This release contains matrix-level SIR synergy calls for 319,533 DrugCombDB dose-response matrices for which complete grid metadata was available in the SIR processed pipeline. SIR (Synergy via Isotonic Regression) fits a monotone-additive null and a fully monotone isotonic response surface, summarizes their departure by interaction energy S^2, and assigns calibrated p-values using a degrees-of-freedom-corrected wild bootstrap with B = 200 resamples.",
  "",
  "The larger DrugCombDB baseline score table used in the paper contains 391,653 baseline-only rows. The Zenodo SIR release is intentionally limited to the 319,533 matrices present in `data/processed/drugcomb_matrices.parquet`, because that is the universe with complete SIR-ready grid metadata.",
  "",
  "## Files",
  "",
  "- `sir_drugcombdb_synergy_calls.parquet`: main release table, one row per SIR-analyzed DrugCombDB matrix.",
  "- `schema.json`: JSON Schema row definition plus `x-parquet-*` metadata for column order and physical table checks.",
  "- `CITATION.cff`: citation metadata for the Zenodo record.",
  "- `checksums.txt`: SHA-256 checksums for every file in this folder except `checksums.txt`.",
  "",
  "An interaction-surface companion file is not included in this release. The existing pipeline did not persist per-cell `theta_iso`, `theta_add`, or `delta` surfaces for the full universe; producing that file would require an additional large compute/storage pass and is deferred to a future release.",
  "",
  "`sir_mean_delta` is intentionally omitted from the release table. The weighted mean interaction is zero by construction for the translation-invariant isotonic projection; see the Supplementary Information of the SIR paper for details. Keeping it in the public schema would add a numerically tiny column with no practical signal.",
  "",
  "## Provenance",
  "",
  paste0("- Method paper: Asiaee, Long, Pal, Pua, and Coombes, \"A shape-constrained regression and wild bootstrap framework for reproducible drug synergy testing\", bioRxiv doi:10.1101/2026.02.05.704019."),
  paste0("- Code repository: https://github.com/amir-as/synergy"),
  paste0("- Code commit used for this assembly: `", git_commit, "`."),
  paste0("- SIR version: `", sir_version, "`."),
  paste0("- DrugCombDB snapshot: ", drugcombdb_version, "."),
  paste0("- Analysis date: ", analysis_date, "."),
  "- Bootstrap resamples per matrix: 200.",
  "- DrugCombDB-native drug IDs and tissue annotations are not included because they were not available in the processed SIR cache. A future release may join these fields from a raw DrugCombDB drug dictionary or metadata table.",
  "- Two NIH matrices in the processed cache lacked cell-line metadata; their `cell_line` value is encoded as `not_available` rather than null so `cell_line` remains a required identifier column.",
  "",
  "## Schema",
  "",
  "| column | type | nullable | description |",
  "| --- | --- | --- | --- |",
  "| `matrix_id` | string | no | Stable SIR matrix identifier; equals `experiment_id` from the processed DrugCombDB cache. |",
  "| `drug_a_name` | string | no | Drug row/name as represented in the processed DrugCombDB cache. |",
  "| `drug_b_name` | string | no | Drug column/name as represented in the processed DrugCombDB cache. |",
  "| `cell_line` | string | no | Cell line name from DrugCombDB; `not_available` marks the two NIH matrices with missing processed-cache cell-line metadata. |",
  "| `source_study` | string | yes | DrugCombDB source tag, such as `ONEIL`; may be missing if unavailable upstream. |",
  "| `n_doses_a` | integer | no | Number of unique dose levels for drug A. |",
  "| `n_doses_b` | integer | no | Number of unique dose levels for drug B. |",
  "| `n_replicates_total` | integer | no | Number of dose-response measurements in the processed matrix. |",
  "| `sir_S2` | number | no | SIR interaction energy, the weighted squared departure between isotonic and monotone-additive fits. |",
  "| `sir_p_value` | number | no | Wild-bootstrap p-value for global interaction; B = 200. |",
  "| `sir_q_value` | number | no | Benjamini-Hochberg FDR-adjusted q-value over all 319,533 matrices. |",
  "| `sir_hit_call` | string | no | `synergy`, `antagonism`, or `no_call`; calls require `sir_q_value <= 0.05` and use the dominant signed interaction energy direction. |",
  "| `sir_failed` | boolean | no | TRUE if the SIR fit failed. This release should contain only FALSE values. |",
  "| `bliss_score` | number | yes | DrugCombDB Bliss score, if available. |",
  "| `bliss_failed` | boolean | no | TRUE when a DrugCombDB Bliss row was present but the score was non-finite. |",
  "| `bliss_unavailable` | boolean | no | TRUE when no DrugCombDB baseline score row was available for this SIR matrix. |",
  "| `hsa_score` | number | yes | DrugCombDB HSA score, if available. |",
  "| `hsa_failed` | boolean | no | TRUE when a DrugCombDB HSA row was present but the score was non-finite. |",
  "| `hsa_unavailable` | boolean | no | TRUE when no DrugCombDB baseline score row was available for this SIR matrix. |",
  "| `loewe_score` | number | yes | DrugCombDB Loewe score, if available. |",
  "| `loewe_failed` | boolean | no | TRUE when a DrugCombDB Loewe row was present but the score was non-finite. |",
  "| `loewe_unavailable` | boolean | no | TRUE when no DrugCombDB baseline score row was available for this SIR matrix. |",
  "| `zip_score` | number | yes | DrugCombDB ZIP score, if available. |",
  "| `zip_failed` | boolean | no | TRUE when a DrugCombDB ZIP row was present but the score was non-finite. |",
  "| `zip_unavailable` | boolean | no | TRUE when no DrugCombDB baseline score row was available for this SIR matrix. |",
  "| `sir_version` | string | no | SIR release/software version string. |",
  "| `drugcombdb_version` | string | no | DrugCombDB snapshot provenance string. |",
  "| `analysis_date` | string | no | ISO-8601 date for the completed SIR universe checkpoint. |",
  "| `bootstrap_B` | integer | no | Number of wild-bootstrap resamples. |",
  "",
  "## Usage",
  "",
  "R:",
  "",
  "```r",
  "library(arrow)",
  "sir <- read_parquet(\"sir_drugcombdb_synergy_calls.parquet\")",
  "table(sir$sir_hit_call)",
  "```",
  "",
  "Python:",
  "",
  "```python",
  "import pandas as pd",
  "sir = pd.read_parquet(\"sir_drugcombdb_synergy_calls.parquet\", engine=\"pyarrow\")",
  "sir[\"sir_hit_call\"].value_counts()",
  "```",
  "",
  "## License",
  "",
  "This derived SIR label dataset is released under CC-BY 4.0. Original DrugCombDB data and constituent studies remain subject to their upstream licenses and terms; please cite and respect those sources when using this release.",
  "",
  "## Citation",
  "",
  "Zenodo DOI placeholder: `10.5281/zenodo.TBD`.",
  "",
  "```bibtex",
  "@dataset{asiaee_sir_drugcombdb_2026,",
  "  title     = {SIR-derived synergy calls for DrugCombDB},",
  "  author    = {Asiaee, Amir and Long, James P. and Pal, Samhita and Pua, Heather H. and Coombes, Kevin R.},",
  "  year      = {2026},",
  "  publisher = {Zenodo},",
  "  doi       = {10.5281/zenodo.TBD},",
  "  url       = {https://doi.org/10.5281/zenodo.TBD}",
  "}",
  "",
  "@article{asiaee_sir_2026,",
  "  title   = {A shape-constrained regression and wild bootstrap framework for reproducible drug synergy testing},",
  "  author  = {Asiaee, Amir and Long, James P. and Pal, Samhita and Pua, Heather H. and Coombes, Kevin R.},",
  "  year    = {2026},",
  "  journal = {bioRxiv},",
  "  doi     = {10.1101/2026.02.05.704019}",
  "}",
  "",
  "@article{liu_drugcombdb_2020,",
  "  title   = {DrugCombDB: a comprehensive database of drug combinations toward the discovery of combinatorial therapy},",
  "  author  = {Liu, Hui and Zhang, Wenhao and Zou, Bo and Wang, Jinxian and Deng, Yuanyuan and Deng, Lei},",
  "  journal = {Nucleic Acids Research},",
  "  year    = {2020},",
  "  volume  = {48},",
  "  number  = {D1},",
  "  pages   = {D871--D881},",
  "  doi     = {10.1093/nar/gkz1007}",
  "}",
  "```",
  "",
  "## Upstream Sources",
  "",
  "The processed DrugCombDB cache includes constituent source tags such as O'Neil and other DrugCombDB-provided studies. Users should cite DrugCombDB (Liu et al., Nucleic Acids Research 2020, doi:10.1093/nar/gkz1007) and the relevant original screens when focusing on a specific source study."
)
atomic_write_text(readme_lines, readme_path)

citation_lines <- c(
  "cff-version: 1.2.0",
  "message: \"If you use this dataset, please cite the Zenodo record and the SIR method paper.\"",
  "type: dataset",
  "title: \"SIR-derived synergy calls for DrugCombDB\"",
  paste0("version: \"", sir_version, "\""),
  paste0("date-released: \"", analysis_date, "\""),
  "license: CC-BY-4.0",
  "doi: 10.5281/zenodo.TBD",
  "url: \"https://doi.org/10.5281/zenodo.TBD\"",
  "authors:",
  "  - family-names: Asiaee",
  "    given-names: Amir",
  "  - family-names: Long",
  "    given-names: James P.",
  "  - family-names: Pal",
  "    given-names: Samhita",
  "  - family-names: Pua",
  "    given-names: Heather H.",
  "  - family-names: Coombes",
  "    given-names: Kevin R.",
  "preferred-citation:",
  "  type: article",
  "  title: \"A shape-constrained regression and wild bootstrap framework for reproducible drug synergy testing\"",
  "  doi: 10.1101/2026.02.05.704019",
  "  year: 2026",
  "  authors:",
  "    - family-names: Asiaee",
  "      given-names: Amir",
  "    - family-names: Long",
  "      given-names: James P.",
  "    - family-names: Pal",
  "      given-names: Samhita",
  "    - family-names: Pua",
  "      given-names: Heather H.",
  "    - family-names: Coombes",
  "      given-names: Kevin R."
)
atomic_write_text(citation_lines, citation_path)

validated <- validate_release(main_dataset_path, schema_path, expected_n)
append_progress("validate", "completed", sprintf("schema.json validates %s; rows=%d", main_dataset_path, nrow(validated)))

phase1_wall <- as.numeric(sidecar$cumulative_wall_clock_s %||% NA_real_)
phase1_core <- as.numeric(sidecar$cumulative_core_runtime_s %||% sum(sir$runtime_s, na.rm = TRUE))
hit_counts <- validated[, .N, by = sir_hit_call][order(sir_hit_call)]
baseline_missing <- validated[bliss_unavailable | hsa_unavailable | loewe_unavailable | zip_unavailable, .N]
release_size_before_checksums <- sum(file.info(list.files(release_dir, recursive = TRUE, full.names = TRUE))$size, na.rm = TRUE)
append_progress(
  "validate", "completed",
  sprintf("final summary before checksums: phase1_wall=%0.2fh; phase1_core_hours=%0.2f; hit_counts=%s; baseline_unavailable_rows=%d; dataset_release_size_before_checksums=%s bytes; decisions are logged as event=decision above",
          phase1_wall / 3600, phase1_core / 3600,
          paste(sprintf("%s:%d", hit_counts$sir_hit_call, hit_counts$N), collapse = ", "),
          baseline_missing, format(release_size_before_checksums, scientific = FALSE))
)

append_progress("validate", "completed", "writing checksums.txt next; no later PROGRESS.md entries are appended so its checksum remains valid")
write_checksums(release_dir)
