# =============================================================================
#  oc_catalogue.R  -  shared simulation catalogue
#
#  Defines, for every implemented variance, the data-generating process used by
#  operating_characteristics.R, small_sample_all.R and exact_oc.R:
#    CATALOGUE[[metric]]  list(family, sim | make(), args)
#    sim$V, sim$truth, sim$skew, sim$p (multinomial families)
#    sim$est_se(n, reps)  matrix with one row per replicate: (estimate, plug-in SE)
#  Also MM_CONST (Miller-Madow constants) and SMALL_N_OC.
#
#  Sourced by the three scripts; not meant to be run on its own.  The RNG is
#  seeded here (set.seed(2026)) so that the pools and the U-statistic
#  reference samples are identical across scripts and across runs.
#  Requires the package to be loaded (library(metricpower) or devtools::load_all()).
# =============================================================================

set.seed(2026)
if (!exists("N_POOL"))      N_POOL      <- 10000   # size of the per-case bootstrap population
if (!exists("USTAT_EXACT")) USTAT_EXACT <- FALSE   # TRUE: exact population V for the U-statistics
                                                   # (default FALSE keeps the Table S4 reference-sample values)

# -----------------------------------------------------------------------------
# Simulators. Each returns V, truth and est_se(n, reps) -> matrix with columns
# (estimate, plug-in SE), one row per replicate.
# -----------------------------------------------------------------------------

sim_multinom <- function(p, g, grad = NULL) {
  list(V = delta_var(p, g, grad = grad), truth = g(p), p = p, skew = NA,
       est_se = function(n, reps) {
         D <- stats::rmultinom(reps, n, p) / n
         e <- apply(D, 2, g)
         s <- sqrt(apply(D, 2, function(q) delta_var(q, g, grad = grad)) / n)
         cbind(e, s)
       })
}

sim_rate <- function(lambda, theta = Inf) {
  list(V = var_rate(lambda, theta), truth = lambda, skew = NA,
       est_se = function(n, reps) {
         chunked(reps, n, function(r) {
           M <- matrix(if (is.infinite(theta)) stats::rpois(n * r, lambda)
                       else stats::rnbinom(n * r, size = theta, mu = lambda), n, r)
           cbind(colMeans(M), sqrt(col_var(M) / n)) })
       })
}

# column variances without apply(); est_se work is split into chunks of at most
# CHUNK_CELLS draws so that n = 5,000 x 20,000 replicates never allocates 800 MB at once
col_var <- function(M) { n <- nrow(M); (colSums(M^2) - n * colMeans(M)^2) / (n - 1) }
if (!exists("CHUNK_CELLS")) CHUNK_CELLS <- 2e7
chunked <- function(reps, n, f) {
  per <- max(1, floor(CHUNK_CELLS / n)); out <- vector("list", ceiling(reps / per)); done <- 0; i <- 0
  while (done < reps) { r <- min(per, reps - done); i <- i + 1; out[[i]] <- f(r); done <- done + r }
  do.call(rbind, out)
}

sim_percase <- function(x, transform = "identity") {
  r <- var_percase_transformed(x, transform)
  h  <- switch(transform, identity = function(m) m,
               sqrt = function(m) sqrt(pmax(m, 1e-12)), exp = exp)
  dh <- switch(transform, identity = function(m) 1,
               sqrt = function(m) 1 / (2 * sqrt(pmax(m, 1e-12))), exp = exp)
  list(V = r$V, truth = r$estimate, skew = r$skew,
       est_se = function(n, reps) chunked(reps, n, function(r) {
         M <- matrix(sample(x, n * r, replace = TRUE), n, r)
         mu <- colMeans(M); v <- col_var(M)
         cbind(h(mu), sqrt(v * dh(mu)^2 / n)) }))
}

# est_se_fn(n) returns c(estimate, plug-in se) from ONE sample
sim_parametric <- function(V, truth, est_se_fn) {
  list(V = V, truth = truth, skew = NA,
       est_se = function(n, reps) t(vapply(seq_len(reps), function(i) est_se_fn(n), numeric(2))))
}

sim_ustat <- function(J, index) {
  k <- nrow(J)
  drawpair <- function(n) {
    idx <- sample(length(J), n, TRUE, prob = as.vector(J))
    list(a = ((idx - 1) %% k) + 1, b = ((idx - 1) %/% k) + 1)
  }
  big <- drawpair(3000); est <- cluster_index_variance(big$a, big$b, index)
  # population value of the index from the joint table J (see validate_all.py)
  sab <- sum(J^2); sa <- sum(rowSums(J)^2); sb <- sum(colSums(J)^2)
  truth <- switch(index, rand = 2 * sab + 1 - sa - sb, fowlkes_mallows = sab / sqrt(sa * sb))
  list(V = est$V, truth = truth, skew = NA,
       est_se = function(n, reps) t(vapply(seq_len(reps), function(i) {
         d <- drawpair(n); r <- cluster_index_variance(d$a, d$b, index)
         c(r$estimate, sqrt(r$V / n)) }, numeric(2))))
}

# -----------------------------------------------------------------------------
# The catalogue - identical data-generating processes to validate_all.R
# -----------------------------------------------------------------------------

pool <- local({
  y <- stats::rbinom(N_POOL, 1, 0.3)
  ph <- stats::plogis(stats::rnorm(N_POOL, -0.8 + 1.8 * y, 1))
  list(brier = (ph - y)^2, nll = -log(ifelse(y == 1, ph, 1 - ph)))
})
PC <- list(
  brier = pool$brier, root_brier = pool$brier, nll = pool$nll,
  cross_entropy = pool$nll, perplexity = pool$nll,
  mae = abs(stats::rnorm(N_POOL, 0, 2)), mse = stats::rnorm(N_POOL, 0, 2)^2,
  rmse = stats::rnorm(N_POOL, 0, 2)^2,
  mrr = 1 / (1 + stats::rgeom(N_POOL, 0.45)),
  mape = abs(stats::rnorm(N_POOL, 0, 1)) / (2 + abs(stats::rnorm(N_POOL, 0, .3))),
  smape = stats::rbeta(N_POOL, 2, 6), mase = stats::rgamma(N_POOL, 2, 2),
  rouge_n = stats::rbeta(N_POOL, 5, 4), rouge_1 = stats::rbeta(N_POOL, 6, 3),
  rouge_2 = stats::rbeta(N_POOL, 3, 6), rouge_3 = stats::rbeta(N_POOL, 2, 8),
  rouge_l = stats::rbeta(N_POOL, 5, 3), rouge_w = stats::rbeta(N_POOL, 4, 4),
  rouge_s = stats::rbeta(N_POOL, 3, 5), rouge_su = stats::rbeta(N_POOL, 4, 5),
  bertscore = stats::rbeta(N_POOL, 20, 3), meteor = stats::rbeta(N_POOL, 4, 4),
  flesch = stats::rnorm(N_POOL, 45, 12), flesch_kincaid = stats::rnorm(N_POOL, 11, 2.5),
  gunning_fog = stats::rnorm(N_POOL, 13, 3), smog = stats::rnorm(N_POOL, 12, 2),
  ari = stats::rnorm(N_POOL, 11, 3), forcast = stats::rnorm(N_POOL, 10, 1.5),
  levenshtein = stats::rpois(N_POOL, 12), damerau = stats::rpois(N_POOL, 11),
  hamming = stats::rpois(N_POOL, 9), jaro = stats::rbeta(N_POOL, 8, 3),
  lcs = stats::rpois(N_POOL, 20), snr = stats::rnorm(N_POOL, 22, 4),
  psnr = stats::rnorm(N_POOL, 31, 5), cnr = stats::rnorm(N_POOL, 6, 2),
  ssim = stats::rbeta(N_POOL, 9, 2), ensemble_mse = stats::rgamma(N_POOL, 3, 6),
  macro_mean_prob = stats::rbeta(N_POOL, 5, 5),
  dice_percase = stats::rbeta(N_POOL, 9, 2), iou_percase = stats::rbeta(N_POOL, 7, 3),
  nsd_percase = stats::rbeta(N_POOL, 8, 2),
  assd_percase = stats::rgamma(N_POOL, 2, 1.2),
  hd95_percase = stats::rgamma(N_POOL, 1.4, 0.25)
)

p4 <- cell_prob_from_operating_point(0.70, 0.80, 0.50)
P3 <- matrix(c(.30,.04,.02, .05,.22,.05, .03,.06,.23), 3, 3, byrow = TRUE); P3 <- P3 / sum(P3)
Jab <- matrix(c(.22,.04,.02, .03,.25,.05, .02,.05,.32), 3, 3, byrow = TRUE); Jab <- Jab / sum(Jab)
cm_args <- list(expected_cost = list(c_fp = 1, c_fn = 5),
                net_benefit = list(threshold = 0.2), fbeta = list(beta = 2))

CATALOGUE <- list()
for (nm in L3_FAMILY$confusion_matrix) {
  a <- cm_args[[nm]]; if (is.null(a)) a <- list()
  CATALOGUE[[nm]] <- list(family = "confusion", args = a,
    sim = sim_multinom(p4, do.call(get_metric, c(list(nm), a)),
                       do.call(get_gradient, c(list(nm), a))))
}
for (nm in L3_FAMILY$multiclass) {
  parts <- strsplit(nm, "_")[[1]]
  g <- if (parts[1] == "micro") metric_micro(parts[2]) else metric_macro(parts[2])
  CATALOGUE[[nm]] <- list(family = "multiclass", sim = sim_multinom(as.vector(P3), g))
}
CATALOGUE[["fppi"]] <- list(family = "rate", sim = sim_rate(0.5, Inf))
CATALOGUE[["fppi_overdispersed"]] <- list(family = "rate", sim = sim_rate(0.5, 2))
for (nm in names(PERCASE_REGISTRY)) {
  CATALOGUE[[nm]] <- list(family = "per-case",
    make = local({ k <- nm; function() sim_percase(PC[[k]], PERCASE_REGISTRY[[k]]$transform) }))
}
# --- U-statistics ---
# Exact population values of V = 4 zeta_1 (first-order projection variance of a
# degree-two U-statistic).  Kendall's tau for a bivariate normal with correlation
# rho: h1(x, y) = 4 Phi2(x, y) - 2 Phi(x) - 2 Phi(y) + 1, integrated on a fine grid.
# Rand / Fowlkes-Mallows: h1 depends only on the cell (a, b) of the joint table J,
# so zeta_1 is a finite sum (same projection as cluster_index_variance()).
kendall_exact_V <- function(rho, m = 1600, L = 7.5) {
  x <- (seq_len(m) - 0.5) / m * 2 * L - L; h <- x[2] - x[1]
  X <- matrix(x, m, m); Y <- t(X)
  dens <- exp(-(X^2 - 2 * rho * X * Y + Y^2) / (2 * (1 - rho^2))) / (2 * pi * sqrt(1 - rho^2))
  F2 <- t(apply(apply(dens, 2, cumsum), 1, cumsum)) * h^2       # Phi2 at the cell corners
  Phi <- stats::pnorm(x + h / 2)
  h1 <- 4 * F2 - 2 * matrix(Phi, m, m) - 2 * t(matrix(Phi, m, m)) + 1
  w <- dens * h^2; tau <- 2 / pi * asin(rho)
  4 * (sum(w * h1^2) - tau^2)
}
cluster_index_exact <- function(J, index) {
  R <- rowSums(J); C <- colSums(J)
  if (index == "rand") {
    h1 <- 2 * J + 1 - outer(R, C, "+")
  } else {
    sab <- sum(J^2); sa <- sum(R^2); sb <- sum(C^2)
    n11 <- sab; n10 <- sa - sab; n01 <- sb - sab
    d1 <- 1 / sqrt((n11 + n10) * (n11 + n01))
    dd <- -0.5 * n11 * ((n11 + n01) + (n11 + n10)) / ((n11 + n10) * (n11 + n01))^1.5
    g11 <- J; g10 <- R - J; g01 <- t(C - t(J))
    h1 <- d1 * g11 + dd * (g11 + g10 / 2 + g01 / 2)
  }
  m <- sum(J * h1); 4 * (sum(J * h1^2) - m^2)
}
kgen <- function(n) { z <- stats::rnorm(n); list(x = z + stats::rnorm(n), y = z + stats::rnorm(n)) }
kbig <- kgen(4000); kest <- kendall_tau_variance(kbig$x, kbig$y)     # reference sample (Table S4)
V_kendall <- if (USTAT_EXACT) kendall_exact_V(0.5) else kest$V
CATALOGUE[["kendall_tau"]] <- list(family = "U-statistic",
  sim = sim_parametric(V_kendall, 2 / pi * asin(0.5),          # population tau for corr 0.5
    function(n) { d <- kgen(n); r <- kendall_tau_variance(d$x, d$y)
                  c(r$estimate, sqrt(r$V / n)) }))
CATALOGUE[["rand_index"]] <- list(family = "U-statistic", sim = sim_ustat(Jab, "rand"))
CATALOGUE[["fowlkes_mallows"]] <- list(family = "U-statistic", sim = sim_ustat(Jab, "fowlkes_mallows"))
if (USTAT_EXACT) {
  CATALOGUE[["rand_index"]]$sim$V      <- cluster_index_exact(Jab, "rand")
  CATALOGUE[["fowlkes_mallows"]]$sim$V <- cluster_index_exact(Jab, "fowlkes_mallows")
}
# --- other ---
CATALOGUE[["entropy"]] <- list(family = "other",
  sim = sim_multinom(c(.5, .3, .15, .05), metric_entropy, grad_entropy))
CATALOGUE[["conditional_entropy"]] <- list(family = "other",
  sim = sim_multinom(as.vector(P3), function(q) metric_conditional_entropy(q, 3)))
CATALOGUE[["weighted_kappa"]] <- list(family = "other",
  sim = sim_multinom(as.vector(P3), function(q) kappa_w(matrix(q, 3, 3), weights_linear(3))))
CATALOGUE[["r_squared"]] <- list(family = "other",
  sim = sim_parametric(r_squared_variance(0.49)$V, 0.49,
    function(n) { x <- stats::rnorm(n)
      r2 <- stats::cor(x, 0.7 * x + sqrt(1 - 0.49) * stats::rnorm(n))^2
      c(r2, sqrt(4 * r2 * (1 - r2)^2 / n)) }))
CATALOGUE[["coefficient_of_variation"]] <- list(family = "other",
  sim = sim_parametric(0.25^2 * (0.5 + 0.25^2), 0.25,
    function(n) { b <- stats::rnorm(n, 4, 1); cv <- stats::sd(b) / mean(b)
      c(cv, sqrt(cv^2 * (0.5 + cv^2) / n)) }))
CATALOGUE[["repeatability_coefficient"]] <- list(family = "other",
  sim = sim_parametric(repeatability_variance(0.5, replicates = 2)$V, 1.96 * sqrt(2) * 0.5,
    function(n) { d <- matrix(stats::rnorm(n * 2, 0, 0.5), n, 2)
      sw <- sqrt(mean(apply(d, 1, stats::var)))
      c(1.96 * sqrt(2) * sw, sqrt(repeatability_variance(sw, 2)$V / n)) }))
CATALOGUE[["discrimination_slope"]] <- list(family = "other",
  sim = sim_parametric(discrimination_slope_variance(.7, .3, .2, .2, .4)$V, 0.4,
    function(n) { y <- stats::rbinom(n, 1, .4)
      a <- stats::rnorm(sum(y), .7, .2); b <- stats::rnorm(sum(1 - y), .3, .2)
      c(mean(a) - mean(b), sqrt(stats::var(a) / length(a) + stats::var(b) / length(b))) }))
CATALOGUE[["risk_difference"]] <- list(family = "other",
  sim = sim_parametric(risk_difference_variance(.6, .45, .5)$V, 0.15,
    function(n) { m <- n %/% 2
      pa <- mean(stats::rbinom(m, 1, .6)); pb <- mean(stats::rbinom(m, 1, .45))
      c(pa - pb, sqrt(risk_difference_variance(pa, pb, .5)$V / n)) }))
CATALOGUE[["paired_proportion"]] <- list(family = "other",
  sim = sim_parametric(paired_proportion_variance(.05, .12)$V, 0.07,
    function(n) { q <- stats::rmultinom(1, n, c(.05, .12, .5, .33)) / n
      c(q[2] - q[1], sqrt(paired_proportion_variance(q[1], q[2])$V / n)) }))

# Miller-Madow constants c (correction c / n): entropy (k - 1) / 2; conditional entropy (k_xy - k_x) / 2.
# Plug-in entropies are biased toward the null (H is concave), which costs power; the package tells the
# user to analyse with the corrected estimate, so the test here does the same. V is unchanged.
MM_CONST <- c(entropy = (sum(CATALOGUE[["entropy"]]$sim$p > 0) - 1) / 2,
              conditional_entropy = (sum(P3 > 0) - sum(rowSums(P3) > 0)) / 2)
SMALL_N_OC <- if (exists("SMALL_N")) SMALL_N else 30
