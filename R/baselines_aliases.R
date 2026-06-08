#' Bliss Independence Synergy Score
#'
#' Computes the Bliss independence synergy score for a long-format
#' dose-response table. This is a thin alias for [bliss_synergy()].
#'
#' @inheritParams bliss_synergy
#' @return A list with components `synergy` (matrix), `expected`
#'   (independence baseline), `observed` (normalised mean response),
#'   `doseA_levels`, and `doseB_levels`.
#' @seealso [bliss_synergy()], [hsa_score()], [loewe_score()],
#'   [zip_score()].
#' @export
bliss_score <- function(df_long, response_mode) {
  bliss_synergy(df_long = df_long, response_mode = response_mode)
}

#' Highest Single Agent (HSA) Synergy Score
#'
#' Computes the HSA synergy score. Alias for [hsa_synergy()].
#'
#' @inheritParams hsa_synergy
#' @return A list with components `synergy`, `expected`, `observed`,
#'   `doseA_levels`, `doseB_levels`.
#' @seealso [hsa_synergy()], [bliss_score()], [loewe_score()],
#'   [zip_score()].
#' @export
hsa_score <- function(df_long, response_mode) {
  hsa_synergy(df_long = df_long, response_mode = response_mode)
}

#' Loewe Additivity Synergy Score
#'
#' Computes the Loewe additivity synergy score using monotone single-agent
#' curves. Alias for [loewe_synergy()].
#'
#' @inheritParams loewe_synergy
#' @return A list with components `synergy`, `combo_index`, `observed`,
#'   `doseA_levels`, `doseB_levels`.
#' @seealso [loewe_synergy()], [bliss_score()], [hsa_score()],
#'   [zip_score()].
#' @export
loewe_score <- function(df_long, response_mode,
                        direction = c("decreasing", "increasing")) {
  loewe_synergy(df_long = df_long, response_mode = response_mode,
                direction = direction)
}

#' ZIP (Zero Interaction Potency) Synergy Score
#'
#' Computes the ZIP synergy score, optionally with 4-parameter logistic
#' single-agent curve fits. Alias for [zip_synergy()].
#'
#' @inheritParams zip_synergy
#' @return A list with components `synergy`, `expected`, `observed`,
#'   `doseA_levels`, `doseB_levels`.
#' @seealso [zip_synergy()], [bliss_score()], [hsa_score()],
#'   [loewe_score()].
#' @export
zip_score <- function(df_long, response_mode, use_curvefit = TRUE) {
  zip_synergy(df_long = df_long, response_mode = response_mode,
              use_curvefit = use_curvefit)
}
