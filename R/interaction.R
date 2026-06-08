infer_synergy_sign <- function(response_mode, direction) {
  if (!is.null(response_mode)) {
    if (response_mode == "viability") return("negative")
    if (response_mode == "inhibition") return("positive")
  }
  if (!is.null(direction)) {
    if (direction == "decreasing") return("negative")
    if (direction == "increasing") return("positive")
  }
  "negative"
}

#' Interaction Surface Summary Statistics
#'
#' Computes weighted summary statistics from an interaction surface
#' `delta = theta_iso - theta_add`, including the squared-norm energy
#' `S2`, signed-positive and signed-negative parts, max synergy and
#' antagonism, area-of-synergy counts, and a synergy index.
#'
#' @param delta Numeric matrix of interaction values (isotonic minus
#'   additive surface).
#' @param w Numeric matrix of nonnegative weights with the same
#'   dimensions as `delta`.
#' @param synergy_sign Direction of synergy: `"negative"` (default) for
#'   viability data, `"positive"` for inhibition data.
#' @param threshold Threshold beyond which a cell is counted as
#'   `area_synergy`. Defaults to `0`.
#'
#' @return A named list of weighted summaries (`S2`, `S2_mean`, `Splus`,
#'   `Sminus`, `S2_plus`, `S2_minus`, `mean_delta_w`, `max_synergy`,
#'   `max_antagonism`, `area_synergy`, `area_synergy_weighted`,
#'   `synergy_energy`, `antagonism_energy`, `synergy_index`,
#'   `synergy_sign`, `threshold`).
#'
#' @seealso [interaction_fit()], [sir_test()].
#' @export
interaction_summaries <- function(delta, w, synergy_sign = c("negative", "positive"), threshold = 0) {
  synergy_sign <- match.arg(synergy_sign)
  d <- as.numeric(delta)
  wv <- as.numeric(w)
  ok <- is.finite(d) & is.finite(wv) & (wv > 0)
  d <- d[ok]
  wv <- wv[ok]

  w_sum <- sum(wv)
  s2 <- sum(wv * d^2)
  splus <- sum(wv * pmax(d, 0))
  sminus <- sum(wv * pmax(-d, 0))
  s2_plus <- sum(wv * pmax(d, 0)^2)
  s2_minus <- sum(wv * pmax(-d, 0)^2)
  max_abs <- if (length(d) > 0) max(abs(d)) else NA_real_
  mean_delta <- if (length(d) > 0) mean(d) else NA_real_
  mean_delta_w <- if (w_sum > 0) sum(wv * d) / w_sum else NA_real_
  mean_abs_delta_w <- if (w_sum > 0) sum(wv * abs(d)) / w_sum else NA_real_

  if (synergy_sign == "negative") {
    max_synergy <- if (length(d) > 0) max(-d) else NA_real_
    mask <- d < -threshold
  } else {
    max_synergy <- if (length(d) > 0) max(d) else NA_real_
    mask <- d > threshold
  }
  max_antagonism <- if (synergy_sign == "negative") {
    if (length(d) > 0) max(d) else NA_real_
  } else {
    if (length(d) > 0) max(-d) else NA_real_
  }
  area_synergy <- sum(mask)
  area_synergy_weighted <- sum(wv[mask])

  eps <- 1e-12
  # Direction is not identifiable from the signed *mean* of δ:
  # both θ_iso and θ_add are translation-invariant projections, so they preserve the
  # weighted mean of Z, implying sum(w * δ) = 0. This makes Splus and Sminus equal.
  # To summarize direction, compare the *energy* in each sign:
  #   S2_minus = sum w * max(-δ,0)^2,  S2_plus = sum w * max(δ,0)^2.
  if (synergy_sign == "negative") {
    synergy_energy <- s2_minus
    antagonism_energy <- s2_plus
  } else {
    synergy_energy <- s2_plus
    antagonism_energy <- s2_minus
  }
  synergy_index <- (synergy_energy - antagonism_energy) / max(eps, synergy_energy + antagonism_energy)

  list(
    weight_sum = w_sum,
    S2 = s2,
    S2_mean = if (w_sum > 0) s2 / w_sum else NA_real_,
    Splus = splus,
    Splus_mean = if (w_sum > 0) splus / w_sum else NA_real_,
    Sminus = sminus,
    Sminus_mean = if (w_sum > 0) sminus / w_sum else NA_real_,
    S2_plus = s2_plus,
    S2_plus_mean = if (w_sum > 0) s2_plus / w_sum else NA_real_,
    S2_minus = s2_minus,
    S2_minus_mean = if (w_sum > 0) s2_minus / w_sum else NA_real_,
    mean_delta = mean_delta,
    mean_delta_w = mean_delta_w,
    mean_abs_delta_w = mean_abs_delta_w,
    max_abs_delta = max_abs,
    max_synergy = max_synergy,
    max_antagonism = max_antagonism,
    area_synergy = area_synergy,
    area_synergy_weighted = area_synergy_weighted,
    synergy_energy = synergy_energy,
    antagonism_energy = antagonism_energy,
    synergy_index = synergy_index,
    synergy_sign = synergy_sign,
    threshold = threshold
  )
}

interaction_stat <- function(stat, delta, w, sse_add, sse_iso, synergy_sign, threshold) {
  summaries <- interaction_summaries(delta, w, synergy_sign = synergy_sign, threshold = threshold)
  if (stat == "t_int") return(sse_add - sse_iso)
  if (stat == "t_int_norm") return((sse_add - sse_iso) / max(1e-12, sse_add))
  if (stat == "S2") return(summaries$S2)
  if (stat == "S2_norm") return(summaries$S2 / max(1e-12, sse_add))
  if (stat == "Splus") return(summaries$Splus)
  if (stat == "Sminus") return(summaries$Sminus)
  if (stat == "S2_plus") return(summaries$S2_plus)
  if (stat == "S2_minus") return(summaries$S2_minus)
  if (stat == "max_synergy") return(summaries$max_synergy)
  if (stat == "area_synergy") return(summaries$area_synergy)
  if (stat == "area_synergy_weighted") return(summaries$area_synergy_weighted)
  stop("Unknown stat: ", stat)
}

#' Fit Interaction Surface from a Dose-Response Matrix
#'
#' Fits the monotone-additive null and the fully monotone isotonic
#' surface to a weighted dose-response matrix and returns both surfaces,
#' their difference (the interaction surface), and a bundle of
#' weighted summary statistics from [interaction_summaries()].
#'
#' @param barZ Numeric matrix of (transformed) mean responses.
#' @param w Numeric matrix of nonnegative weights with the same
#'   dimensions as `barZ`.
#' @param direction Direction of monotonicity (`"decreasing"` or
#'   `"increasing"`).
#' @param osqp_pars Named list of OSQP solver parameters.
#' @param synergy_sign Direction of synergy: `"negative"` or `"positive"`.
#' @param threshold Threshold for `area_synergy` statistics.
#'
#' @return A named list combining `theta_iso`, `theta_add`, `delta`,
#'   `sse_iso`, `sse_add`, `t_int`, and the weighted summaries returned
#'   by [interaction_summaries()].
#'
#' @seealso [sir_test()], [wild_bootstrap()],
#'   [interaction_summaries()].
#' @export
interaction_fit <- function(barZ, w, direction, osqp_pars = list(),
                            synergy_sign = c("negative", "positive"), threshold = 0) {
  synergy_sign <- match.arg(synergy_sign)
  iso <- isotonic_2d_fit(barZ, w, direction = direction, osqp_pars = osqp_pars)
  add <- additive_ordered_fit(barZ, w, direction = direction, osqp_pars = osqp_pars)

  theta_iso <- iso$theta
  theta_add <- add$theta
  delta <- theta_iso - theta_add

  y <- as.numeric(barZ)
  wv <- as.numeric(w)
  theta_iso_v <- as.numeric(theta_iso)
  theta_add_v <- as.numeric(theta_add)

  sse_iso <- sse_weighted(y, theta_iso_v, wv)
  sse_add <- sse_weighted(y, theta_add_v, wv)
  t_int <- sse_add - sse_iso

  summaries <- interaction_summaries(delta, w, synergy_sign = synergy_sign, threshold = threshold)

  list(
    theta_iso = theta_iso,
    theta_add = theta_add,
    delta = delta,
    sse_iso = sse_iso,
    sse_add = sse_add,
    t_int = t_int,
    S2 = summaries$S2,
    Splus = summaries$Splus,
    Sminus = summaries$Sminus,
    S2_mean = summaries$S2_mean,
    Splus_mean = summaries$Splus_mean,
    Sminus_mean = summaries$Sminus_mean,
    S2_plus = summaries$S2_plus,
    S2_plus_mean = summaries$S2_plus_mean,
    S2_minus = summaries$S2_minus,
    S2_minus_mean = summaries$S2_minus_mean,
    max_abs_delta = summaries$max_abs_delta,
    mean_delta = summaries$mean_delta,
    mean_delta_w = summaries$mean_delta_w,
    max_synergy = summaries$max_synergy,
    max_antagonism = summaries$max_antagonism,
    area_synergy = summaries$area_synergy,
    area_synergy_weighted = summaries$area_synergy_weighted,
    synergy_energy = summaries$synergy_energy,
    antagonism_energy = summaries$antagonism_energy,
    synergy_index = summaries$synergy_index,
    synergy_sign = summaries$synergy_sign,
    threshold = summaries$threshold
  )
}
