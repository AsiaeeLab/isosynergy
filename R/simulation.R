simulate_matrices <- function(cfg, transform) {
  if (!requireNamespace("data.table", quietly = TRUE)) {
    stop("Package 'data.table' is required. Install with: install.packages('data.table')")
  }
  p <- cfg$experiments$simulation
  n <- p$n_matrices
  I <- p$grid_I
  J <- p$grid_J

  strengths <- p$interaction_strength
  if (length(strengths) == 0) strengths <- 0

  interaction_mode <- p$interaction_mode %||% "projected"
  interaction_sign <- p$interaction_sign %||% "auto"
  bump_sigma <- p$bump_sigma %||% NULL
  bump_center <- p$bump_center %||% NULL

  with_seed(cfg$project$seed %||% 1, {
    mats <- vector("list", n)
    for (k in seq_len(n)) {
      s <- strengths[(k - 1L) %% length(strengths) + 1L]
      mats[[k]] <- simulate_one_matrix(
        experiment_id = sprintf("sim_%05d", k),
        I = I, J = J,
        interaction_strength = s,
        interaction_mode = interaction_mode,
        interaction_sign = interaction_sign,
        bump_sigma = bump_sigma,
        bump_center = bump_center,
        noise_sigma = p$noise_sigma,
        heavy_tail_df = p$heavy_tail_df,
        missing_rate = p$missing_rate,
        outlier_rate = p$outlier_rate,
        transform = transform,
        response_mode = cfg$responses$mode
      )
    }
    mats
  })
}

make_bump_surface <- function(I, J, center = NULL, sigma = NULL) {
  if (is.null(center)) {
    cx <- (I + 1) / 2
    cy <- (J + 1) / 2
  } else {
    cx <- center[1]
    cy <- center[2]
  }
  if (is.null(sigma)) {
    sigma_x <- I / 5
    sigma_y <- J / 5
  } else if (length(sigma) == 1) {
    sigma_x <- sigma
    sigma_y <- sigma
  } else {
    sigma_x <- sigma[1]
    sigma_y <- sigma[2]
  }

  ii <- seq_len(I)
  jj <- seq_len(J)
  outer(ii, jj, function(i, j) exp(-((i - cx)^2 / (2 * sigma_x^2) + (j - cy)^2 / (2 * sigma_y^2))))
}

simulate_true_surfaces <- function(I, J, interaction_strength, response_mode,
                                   interaction_mode = c("projected", "raw", "bump"),
                                   interaction_sign = c("auto", "negative", "positive"),
                                   bump_sigma = NULL, bump_center = NULL,
                                   osqp_pars = list(eps_abs = 1e-7, eps_rel = 1e-7, max_iter = 20000)) {
  interaction_mode <- match.arg(interaction_mode)
  interaction_sign <- match.arg(interaction_sign)
  if (interaction_mode == "bump") interaction_mode <- "raw"

  direction <- infer_monotone_direction(response_mode, "auto")
  synergy_sign <- if (interaction_sign == "auto") infer_synergy_sign(response_mode, direction) else interaction_sign
  bump_sign <- if (synergy_sign == "negative") -1 else 1

  if (direction == "decreasing") {
    u_raw <- -cumsum(abs(stats::rnorm(I, mean = 0.3, sd = 0.15)))
    v_raw <- -cumsum(abs(stats::rnorm(J, mean = 0.3, sd = 0.15)))
  } else {
    u_raw <- cumsum(abs(stats::rnorm(I, mean = 0.3, sd = 0.15)))
    v_raw <- cumsum(abs(stats::rnorm(J, mean = 0.3, sd = 0.15)))
  }
  u <- u_raw - mean(u_raw)
  v <- v_raw - mean(v_raw)
  alpha <- stats::rnorm(1, 0, 0.2)

  Z_add <- matrix(alpha, nrow = I, ncol = J) +
    matrix(u, nrow = I, ncol = J) +
    matrix(v, nrow = I, ncol = J, byrow = TRUE)

  if (interaction_strength > 0) {
    bump <- make_bump_surface(I, J, center = bump_center, sigma = bump_sigma)
    Delta_raw <- bump_sign * interaction_strength * bump
  } else {
    bump <- matrix(0, nrow = I, ncol = J)
    Delta_raw <- matrix(0, nrow = I, ncol = J)
  }

  candidate <- Z_add + Delta_raw

  if (interaction_mode == "projected") {
    theta_iso_true <- isotonic_2d_fit(candidate, w = matrix(1, I, J),
                                      direction = direction, osqp_pars = osqp_pars)$theta
  } else {
    theta_iso_true <- candidate
  }
  theta_add_true <- additive_ordered_fit(theta_iso_true, w = matrix(1, I, J),
                                         direction = direction, osqp_pars = osqp_pars)$theta
  delta_true <- theta_iso_true - theta_add_true

  list(
    direction = direction,
    synergy_sign = synergy_sign,
    u = u,
    v = v,
    alpha = alpha,
    Z_add = Z_add,
    bump = bump,
    Delta_raw = Delta_raw,
    theta_iso_true = theta_iso_true,
    theta_add_true = theta_add_true,
    delta_true = delta_true
  )
}

simulate_one_matrix <- function(experiment_id, I, J, interaction_strength,
                                interaction_mode = "projected",
                                interaction_sign = "auto",
                                bump_sigma = NULL, bump_center = NULL,
                                noise_sigma, heavy_tail_df, missing_rate, outlier_rate,
                                transform, response_mode) {
  dosesA <- 10^seq(-2, 1, length.out = I)
  dosesB <- 10^seq(-2, 1, length.out = J)

  surfaces <- simulate_true_surfaces(
    I = I, J = J,
    interaction_strength = interaction_strength,
    response_mode = response_mode,
    interaction_mode = interaction_mode,
    interaction_sign = interaction_sign,
    bump_sigma = bump_sigma,
    bump_center = bump_center
  )
  Z_true <- surfaces$theta_iso_true

  n_cells <- I * J
  eps <- stats::rt(n_cells, df = heavy_tail_df)
  eps <- eps / stats::sd(eps)
  Z_obs <- as.numeric(Z_true) + noise_sigma * eps

  # Inject outliers.
  if (outlier_rate > 0) {
    n_out <- max(0L, round(outlier_rate * n_cells))
    if (n_out > 0) {
      idx <- sample(seq_len(n_cells), n_out, replace = FALSE)
      Z_obs[idx] <- Z_obs[idx] + stats::rnorm(n_out, mean = 0, sd = 6 * noise_sigma)
    }
  }

  # Missingness.
  keep <- rep(TRUE, n_cells)
  if (missing_rate > 0) {
    n_miss <- max(0L, round(missing_rate * n_cells))
    if (n_miss > 0) {
      miss <- sample(seq_len(n_cells), n_miss, replace = FALSE)
      keep[miss] <- FALSE
    }
  }

  y_obs <- transform$inverse(Z_obs)
  if (response_mode == "inhibition") y_obs <- 1 - y_obs
  y_obs <- clamp01(y_obs, eps = 0)

  dt <- data.table::data.table(
    experiment_id = experiment_id,
    source = "simulation",
    sim_interaction_strength = interaction_strength,
    sim_interaction_mode = interaction_mode,
    sim_synergy_sign = surfaces$synergy_sign,
    drugA = "DrugA",
    drugB = "DrugB",
    cell_line = "CL",
    doseA = rep(dosesA, times = J),
    doseB = rep(dosesB, each = I),
    response = y_obs,
    replicate = "1"
  )
  dt[keep]
}
