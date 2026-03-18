compute_proposed_metrics <- function(df_long, cfg, transform, boot_override = NULL, skip_boot = FALSE) {
  ms <- summarise_matrix(
    df_long = df_long,
    transform = transform,
    response_mode = cfg$responses$mode,
    method_direction = cfg$method$monotone_direction,
    tau = cfg$weights$tau,
    winsor_q = cfg$weights$winsor_q
  )
  osqp_pars <- cfg$method$osqp %||% list()

  synergy_sign <- cfg$interaction$synergy_sign %||% "auto"
  if (synergy_sign == "auto") {
    synergy_sign <- infer_synergy_sign(cfg$responses$mode, ms$direction)
  }
  threshold <- cfg$interaction$threshold %||% 0
  boot_stat <- cfg$bootstrap$stat %||% "t_int"
  boot_residual_scale <- cfg$bootstrap$residual_scale %||% "none"
  boot_df_null <- cfg$bootstrap$df_null %||% NULL
  boot_df_tol <- cfg$bootstrap$df_tol %||% 1e-8

  fit <- interaction_fit(ms$barZ, ms$w, direction = ms$direction, osqp_pars = osqp_pars,
                         synergy_sign = synergy_sign, threshold = threshold)
  boot_B <- boot_override %||% cfg$bootstrap$B
  if (isTRUE(skip_boot) || is.null(boot_B) || boot_B <= 0) {
    boot <- list(p_value = NA_real_, t_star = NA_real_, stat = boot_stat)
  } else {
    boot <- wild_bootstrap(ms$barZ, ms$w, direction = ms$direction,
                           B = boot_B, seed = cfg$bootstrap$seed,
                           osqp_pars = osqp_pars,
                           n_cores = cfg$project$n_cores %||% 1,
                           use_parallel = TRUE,
                           stat = boot_stat,
                           synergy_sign = synergy_sign,
                           threshold = threshold,
                           residual_scale = boot_residual_scale,
                           df_null = boot_df_null,
                           df_tol = boot_df_tol)
  }

  # Synergy-aligned surfaces.
  delta_synergy_Z <- if (synergy_sign == "negative") -fit$delta else fit$delta
  add_hat <- transform$inverse(fit$theta_add)
  iso_hat <- transform$inverse(fit$theta_iso)
  if (cfg$responses$mode == "viability") {
    S_proposed <- add_hat - iso_hat
  } else {
    # inhibition scale
    S_proposed <- iso_hat - add_hat
  }

  list(
    direction = ms$direction,
    barZ = ms$barZ,
    w = ms$w,
    theta_iso = fit$theta_iso,
    theta_add = fit$theta_add,
    delta = fit$delta,
    delta_synergy_Z = delta_synergy_Z,
    S_proposed = S_proposed,
    S2 = fit$S2,
    Splus = fit$Splus,
    Sminus = fit$Sminus,
    S2_mean = fit$S2_mean,
    Splus_mean = fit$Splus_mean,
    Sminus_mean = fit$Sminus_mean,
    S2_plus = fit$S2_plus,
    S2_plus_mean = fit$S2_plus_mean,
    S2_minus = fit$S2_minus,
    S2_minus_mean = fit$S2_minus_mean,
    max_abs_delta = fit$max_abs_delta,
    mean_delta = fit$mean_delta,
    mean_delta_w = fit$mean_delta_w,
    max_synergy = fit$max_synergy,
    max_antagonism = fit$max_antagonism,
    area_synergy = fit$area_synergy,
    area_synergy_weighted = fit$area_synergy_weighted,
    synergy_energy = fit$synergy_energy,
    antagonism_energy = fit$antagonism_energy,
    synergy_index = fit$synergy_index,
    t_int = fit$t_int,
    p_value = boot$p_value,
    t_star = boot$t_star,
    boot_stat = boot$stat,
    boot_residual_scale = boot_residual_scale,
    boot_df_null = boot_df_null,
    boot_df_tol = boot_df_tol,
    synergy_sign = synergy_sign,
    threshold = threshold,
    doseA_levels = ms$doseA_levels,
    doseB_levels = ms$doseB_levels
  )
}

compute_baselines <- function(df_long, cfg) {
  response_mode <- cfg$responses$mode
  direction <- infer_monotone_direction(response_mode, cfg$method$monotone_direction)

  bliss <- bliss_synergy(df_long, response_mode = response_mode)
  hsa <- hsa_synergy(df_long, response_mode = response_mode)

  loewe <- loewe_synergy(df_long, response_mode = response_mode, direction = direction)
  zip <- zip_synergy(df_long, response_mode = response_mode, use_curvefit = isTRUE(cfg$baselines$zip_curvefit))

  list(
    bliss = bliss,
    hsa = hsa,
    loewe = loewe,
    zip = zip
  )
}

surface_to_long <- function(mat, dosesA, dosesB, value_name) {
  if (!requireNamespace("data.table", quietly = TRUE)) {
    stop("Package 'data.table' is required. Install with: install.packages('data.table')")
  }
  I <- length(dosesA)
  J <- length(dosesB)
  data.table::data.table(
    doseA = rep(dosesA, times = J),
    doseB = rep(dosesB, each = I),
    value = as.numeric(mat)
  )[, metric := value_name][]
}
