standardize_nci_drug_names <- function(drug_map_path) {
  if (!requireNamespace("data.table", quietly = TRUE)) {
    stop("Package 'data.table' is required. Install with: install.packages('data.table')")
  }
  if (is.null(drug_map_path) || !file.exists(drug_map_path)) {
    warning("Drug map file not found: ", drug_map_path)
    return(NULL)
  }
  dt <- data.table::fread(drug_map_path, sep = "\t", check.names = TRUE, na.strings = c("", "NA"))
  if (!"NSC" %in% names(dt)) {
    warning("Drug map missing NSC column: ", drug_map_path)
    return(NULL)
  }
  dt[, nsc := as.integer(NSC)]
  # Prefer generic name, then preferred name, and fall back to NSC id.
  name_cols <- intersect(c("Generic.Name", "Preffered.Name", "Preferred.Name", "GenericName", "PreferredName"), names(dt))
  dt[, drug_name := NA_character_]
  for (col in name_cols) {
    dt[is.na(drug_name) & !is.na(get(col)) & nzchar(get(col)), drug_name := trimws(get(col))]
  }
  dt[is.na(drug_name), drug_name := paste0("NSC-", nsc)]
  unique(dt[, .(nsc, drug_name)])
}

standardize_nci_cell_lines <- function(cell_map_path) {
  if (is.null(cell_map_path) || !file.exists(cell_map_path)) return(NULL)
  if (!requireNamespace("data.table", quietly = TRUE)) {
    stop("Package 'data.table' is required. Install with: install.packages('data.table')")
  }
  dt <- data.table::fread(cell_map_path, sep = "\t", check.names = TRUE, na.strings = c("", "NA"))
  old <- intersect(c("CELLNAME", "cellname", "CellLine"), names(dt))
  if (length(old) == 0) return(NULL)
  new <- intersect(c("Name", "NAME", "cell_line", "CleanName"), names(dt))
  dt[, cell_line_raw := dt[[old[1]]]]
  dt[, cell_line_clean := if (length(new) > 0) dt[[new[1]]] else cell_line_raw]
  dt[, cell_line_clean := ifelse(!is.na(cell_line_clean) & nzchar(cell_line_clean), cell_line_clean, cell_line_raw)]
  unique(dt[, .(cell_line_raw, cell_line_clean)])
}

standardize_nci_almanac <- function(raw_csv,
                                    cell_map_path = NULL,
                                    drug_map_path = NULL,
                                    clamp_01 = TRUE,
                                    min_grid = c(3, 3)) {
  if (!requireNamespace("data.table", quietly = TRUE)) {
    stop("Package 'data.table' is required. Install with: install.packages('data.table')")
  }
  sel <- c(
    "STUDY", "PLATE", "TESTDATE", "PANEL", "CELLNAME",
    "NSC1", "SAMPLE1", "CONCINDEX1", "CONC1",
    "NSC2", "SAMPLE2", "CONCINDEX2", "CONC2",
    "PERCENTGROWTH", "VALID"
  )
  dt <- data.table::fread(raw_csv, select = sel, showProgress = TRUE)
  dt <- dt[VALID == "Y"]

  dt[, response_raw := PERCENTGROWTH / 100]
  dt[, response := response_raw]
  if (isTRUE(clamp_01)) {
    dt[, response := pmin(pmax(response, 0), 1)]
  }

  dt[, `:=`(
    study = STUDY,
    plate = PLATE,
    test_date = TESTDATE,
    panel = PANEL,
    cell_line_raw = CELLNAME,
    drugA_id = as.integer(NSC1),
    drugB_id = as.integer(NSC2),
    doseA = as.numeric(CONC1),
    doseB = as.numeric(CONC2),
    concindex1 = as.integer(CONCINDEX1),
    concindex2 = as.integer(CONCINDEX2)
  )]

  cell_map <- standardize_nci_cell_lines(cell_map_path)
  if (!is.null(cell_map)) {
    dt <- merge(dt, cell_map, by = "cell_line_raw", all.x = TRUE)
    dt[, cell_line := ifelse(!is.na(cell_line_clean), cell_line_clean, cell_line_raw)]
  } else {
    dt[, cell_line := cell_line_raw]
  }

  drug_map <- standardize_nci_drug_names(drug_map_path)

  combo <- dt[!is.na(drugB_id), .(
    study, plate, test_date, panel, cell_line,
    drugA_id, drugB_id,
    doseA, doseB, concindex1, concindex2,
    response, response_raw,
    source_detail = "combo"
  )]

  monotherapy <- dt[is.na(drugB_id), .(
    study, plate, test_date, panel, cell_line,
    drug_id = drugA_id,
    dose = doseA,
    concindex = concindex1,
    response, response_raw
  )]

  experiments <- unique(combo[, .(study, plate, test_date, panel, cell_line, drugA_id, drugB_id)])

  monoA <- merge(
    experiments,
    monotherapy,
    by.x = c("study", "plate", "test_date", "panel", "cell_line", "drugA_id"),
    by.y = c("study", "plate", "test_date", "panel", "cell_line", "drug_id"),
    all.x = TRUE,
    allow.cartesian = TRUE
  )
  monoA <- monoA[!is.na(dose)]
  monoA[, `:=`(
    doseA = dose,
    doseB = 0,
    concindex1 = concindex,
    concindex2 = 0L,
    source_detail = "monotherapy_A"
  )]
  monoA <- monoA[, .(
    study, plate, test_date, panel, cell_line,
    drugA_id, drugB_id,
    doseA, doseB, concindex1, concindex2,
    response, response_raw,
    source_detail
  )]

  monoB <- merge(
    experiments,
    monotherapy,
    by.x = c("study", "plate", "test_date", "panel", "cell_line", "drugB_id"),
    by.y = c("study", "plate", "test_date", "panel", "cell_line", "drug_id"),
    all.x = TRUE,
    allow.cartesian = TRUE
  )
  monoB <- monoB[!is.na(dose)]
  monoB[, `:=`(
    doseA = 0,
    doseB = dose,
    concindex1 = 0L,
    concindex2 = concindex,
    source_detail = "monotherapy_B"
  )]
  monoB <- monoB[, .(
    study, plate, test_date, panel, cell_line,
    drugA_id, drugB_id,
    doseA, doseB, concindex1, concindex2,
    response, response_raw,
    source_detail
  )]

  long <- data.table::rbindlist(list(combo, monoA, monoB), use.names = TRUE, fill = TRUE)
  long <- long[!is.na(drugA_id) & !is.na(drugB_id)]
  long[, experiment_id := sprintf("nci_almanac__%s__%s__%s__%s_%s", study, plate, cell_line, drugA_id, drugB_id)]
  long[, `:=`(
    source = "nci_almanac",
    response_mode = "viability",
    replicate = "1"
  )]

  if (!is.null(drug_map)) {
    long <- merge(long, drug_map, by.x = "drugA_id", by.y = "nsc", all.x = TRUE)
    data.table::setnames(long, "drug_name", "drugA")
    long <- merge(long, drug_map, by.x = "drugB_id", by.y = "nsc", all.x = TRUE)
    data.table::setnames(long, "drug_name", "drugB")
  }
  if (!"drugA" %in% names(long)) long[, drugA := paste0("NSC-", drugA_id)]
  if (!"drugB" %in% names(long)) long[, drugB := paste0("NSC-", drugB_id)]
  long[is.na(drugA) | !nzchar(drugA), drugA := paste0("NSC-", drugA_id)]
  long[is.na(drugB) | !nzchar(drugB), drugB := paste0("NSC-", drugB_id)]

  # Require at least a modest grid to keep the experiment.
  grid <- long[, .(
    n = .N,
    n_i = data.table::uniqueN(doseA),
    n_j = data.table::uniqueN(doseB),
    has_monoA = any(source_detail == "monotherapy_A"),
    has_monoB = any(source_detail == "monotherapy_B")
  ), by = experiment_id]
  grid <- grid[n_i >= min_grid[1] & n_j >= min_grid[2]]
  keep_ids <- grid$experiment_id
  long <- long[experiment_id %in% keep_ids]
  grid <- merge(grid, unique(long[, .(experiment_id, source, study, plate, cell_line, drugA, drugB)]), by = "experiment_id", all.x = TRUE)

  list(long = long, grid = grid)
}

write_nci_almanac_standard <- function(raw_csv,
                                       out_path,
                                       cell_map_path = NULL,
                                       drug_map_path = NULL,
                                       clamp_01 = TRUE,
                                       min_grid = c(3, 3),
                                       summary_path = NULL) {
  if (!requireNamespace("arrow", quietly = TRUE)) {
    stop("Package 'arrow' is required. Install with: install.packages('arrow')")
  }
  std <- standardize_nci_almanac(
    raw_csv = raw_csv,
    cell_map_path = cell_map_path,
    drug_map_path = drug_map_path,
    clamp_01 = clamp_01,
    min_grid = min_grid
  )
  arrow::write_parquet(std$long, out_path)
  if (!is.null(summary_path)) {
    arrow::write_parquet(std$grid, summary_path)
  }
  invisible(std)
}

drugcombdb_unit_to_uM <- function(x, unit) {
  unit <- trimws(as.character(unit))
  unit <- tolower(unit)
  mult <- data.table::fifelse(unit %in% c("m", "mol", "molar"), 1e6,
    data.table::fifelse(unit %in% c("mm", "mmol"), 1e3,
      data.table::fifelse(unit %in% c("um", "\u00b5m"), 1,
        data.table::fifelse(unit %in% c("nm"), 1e-3,
          data.table::fifelse(unit %in% c("pm"), 1e-6, NA_real_)
        )
      )
    )
  )
  as.numeric(x) * mult
}

standardize_drugcombdb <- function(response_csv,
                                   scored_csv,
                                   clamp_01 = TRUE,
                                   min_grid = c(3, 3),
                                   normalize_units = TRUE) {
  if (!requireNamespace("data.table", quietly = TRUE)) {
    stop("Package 'data.table' is required. Install with: install.packages('data.table')")
  }

  if (is.null(response_csv) || !file.exists(response_csv)) stop("Missing response CSV: ", response_csv)
  if (is.null(scored_csv) || !file.exists(scored_csv)) stop("Missing scored CSV: ", scored_csv)

  sel_resp <- c(
    "BlockID", "Row", "Col",
    "DrugRow", "DrugCol",
    "ConcRow", "ConcCol",
    "Response",
    "ConcRowUnit", "ConcColUnit",
    "source"
  )
  resp <- data.table::fread(
    response_csv,
    select = sel_resp,
    showProgress = TRUE,
    fill = TRUE,
    na.strings = c("", "NA")
  )
  data.table::setnames(resp, c("Row", "Col", "source"), c("row_index", "col_index", "study_source"))

  scored <- data.table::fread(scored_csv, showProgress = TRUE, fill = TRUE, na.strings = c("", "NA"))
  scored <- data.table::as.data.table(scored)
  if (!"ID" %in% names(scored)) stop("Scored CSV missing 'ID' column: ", scored_csv)
  if (!"Cell line" %in% names(scored)) stop("Scored CSV missing 'Cell line' column: ", scored_csv)
  scored <- scored[, .(
    BlockID = ID,
    drug1 = as.character(Drug1),
    drug2 = as.character(Drug2),
    cell_line = as.character(`Cell line`),
    ZIP = as.numeric(ZIP),
    Bliss = as.numeric(Bliss),
    Loewe = as.numeric(Loewe),
    HSA = as.numeric(HSA)
  )]

  dt <- merge(resp, scored, by = "BlockID", all.x = TRUE)

  dt[, `:=`(
    drugA = trimws(as.character(DrugRow)),
    drugB = trimws(as.character(DrugCol)),
    doseA = as.numeric(ConcRow),
    doseB = as.numeric(ConcCol),
    response_raw = as.numeric(Response) / 100
  )]

  dt[, response := response_raw]
  if (isTRUE(clamp_01)) dt[, response := pmin(pmax(response, 0), 1)]

  dt[, `:=`(
    doseA_unit = as.character(ConcRowUnit),
    doseB_unit = as.character(ConcColUnit)
  )]

  if (isTRUE(normalize_units)) {
    okA <- !is.na(dt$doseA) & !is.na(dt$doseA_unit) & nzchar(dt$doseA_unit)
    okB <- !is.na(dt$doseB) & !is.na(dt$doseB_unit) & nzchar(dt$doseB_unit)
    if (any(okA)) dt[okA, doseA := drugcombdb_unit_to_uM(doseA, doseA_unit)]
    if (any(okB)) dt[okB, doseB := drugcombdb_unit_to_uM(doseB, doseB_unit)]
    dt[, `:=`(doseA_unit = "uM", doseB_unit = "uM")]
  }

  dt[, `:=`(
    source = "drugcombdb",
    response_mode = "viability",
    replicate = "1",
    experiment_id = sprintf("drugcombdb__block%07d", as.integer(BlockID))
  )]

  long <- dt[, .(
    experiment_id, source, drugA, drugB, cell_line,
    doseA, doseB, response, response_mode, replicate,
    block_id = as.integer(BlockID),
    row_index = as.integer(row_index),
    col_index = as.integer(col_index),
    doseA_unit, doseB_unit,
    study_source,
    baseline_ZIP = ZIP,
    baseline_Bliss = Bliss,
    baseline_Loewe = Loewe,
    baseline_HSA = HSA
  )]

  grid <- long[, .(
    n = .N,
    n_i = data.table::uniqueN(doseA),
    n_j = data.table::uniqueN(doseB),
    has_monoA = any(doseB == 0, na.rm = TRUE),
    has_monoB = any(doseA == 0, na.rm = TRUE),
    mono_activity = mean(1 - response[(doseA == 0 | doseB == 0)], na.rm = TRUE),
    any_missing_meta = any(is.na(cell_line) | !nzchar(cell_line))
  ), by = experiment_id]

  grid_keep <- grid[n_i >= min_grid[1] & n_j >= min_grid[2]]
  keep_ids <- grid_keep$experiment_id
  long <- long[experiment_id %in% keep_ids]
  grid_keep <- merge(
    grid_keep,
    unique(long[, .(experiment_id, source, block_id, study_source, cell_line, drugA, drugB)]),
    by = "experiment_id",
    all.x = TRUE
  )

  scores <- unique(scored[, .(
    BlockID,
    experiment_id = sprintf("drugcombdb__block%07d", as.integer(BlockID)),
    drug1, drug2, cell_line,
    ZIP, Bliss, Loewe, HSA
  )])

  list(long = long, grid = grid_keep, scores = scores)
}

write_drugcombdb_standard <- function(response_csv,
                                      scored_csv,
                                      out_path,
                                      clamp_01 = TRUE,
                                      min_grid = c(3, 3),
                                      summary_path = NULL,
                                      scores_path = NULL,
                                      normalize_units = TRUE) {
  if (!requireNamespace("arrow", quietly = TRUE)) {
    stop("Package 'arrow' is required. Install with: install.packages('arrow')")
  }
  std <- standardize_drugcombdb(
    response_csv = response_csv,
    scored_csv = scored_csv,
    clamp_01 = clamp_01,
    min_grid = min_grid,
    normalize_units = normalize_units
  )
  arrow::write_parquet(std$long, out_path)
  if (!is.null(summary_path)) arrow::write_parquet(std$grid, summary_path)
  if (!is.null(scores_path)) arrow::write_parquet(std$scores, scores_path)
  invisible(std)
}
