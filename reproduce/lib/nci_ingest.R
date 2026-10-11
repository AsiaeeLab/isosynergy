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

