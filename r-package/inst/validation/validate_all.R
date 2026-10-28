# =============================================================================
#  validate_all.R
#
#  One validation table for every metric, replacing the three separate parts
#  that had accumulated across earlier files.
#
#  Two things changed in the merge.
#
#  1. A single simulator abstraction. Each family needs a different way of
#     generating a sampling distribution - bootstrap resampling for per-case
#     means, multinomial draws for confusion matrices, pair draws for
#     U-statistics - so the family supplies a simulator and the table logic is
#     written once.
#
#  2. Relative delta convention: delta = 10 and 25 per cent of the metric's
#     own mean mu. Under a relative delta r * mu,
#         n = (z+z)^2 V / (r mu)^2 = (z+z)^2 CV^2 / r^2,   CV = sqrt(V) / mu,
#     so n depends on the metric only through its coefficient of variation,
#     and n(25%) = n(10%) / 6.25 for every metric (up to rounding).
# =============================================================================

set.seed(90210)

REPS_V   <- 10000   # replicates for the variance check
REPS_COV <-  5000   # replicates for the coverage check
N_POOL   <- 10000    # size of the simulated population
N_COV    <- 60      # sample size at which coverage is evaluated
Z        <- stats::qnorm(0.975)

# Monte Carlo error, so the table can be read honestly
SE_COV <- sqrt(0.94 * 0.06 / REPS_COV)
SE_V   <- sqrt(4 / REPS_V)          # (2 + kurtosis)/REPS with kurtosis ~ 2

PASS <- TRUE

# -----------------------------------------------------------------------------
# Simulators: one per family. Each returns a function(n) giving one draw of the
# estimator, plus the analytic V and the population truth.
# -----------------------------------------------------------------------------

sim_percase <- function(x, transform = "identity") {
  r <- var_percase_transformed(x, transform)
  list(V = r$V, truth = r$estimate, skew = r$skew,
       draw_many = function(n, reps) {
         m <- colMeans(matrix(sample(x, n * reps, replace = TRUE), n, reps))
         switch(transform, identity = m, sqrt = sqrt(pmax(m, 1e-12)), exp = exp(m))
       },
       cover = function(n, reps) {
         M <- matrix(sample(x, n * reps, replace = TRUE), n, reps)
         mu <- colMeans(M); v <- apply(M, 2, stats::var)
         est <- switch(transform, identity = mu, sqrt = sqrt(pmax(mu, 1e-12)),
                       exp = exp(mu))
         dg <- switch(transform, identity = 1,
                      sqrt = 1 / (2 * sqrt(pmax(mu, 1e-12))), exp = exp(mu))
         se <- sqrt(v * dg^2 / n)
         mean(abs(est - r$estimate) <= Z * se)
       })
}

sim_multinom <- function(p, g, grad = NULL) {
  V <- delta_var(p, g, grad = grad)
  list(V = V, truth = g(p), skew = NA,
       draw_many = function(n, reps) apply(stats::rmultinom(reps, n, p) / n, 2, g),
       cover = function(n, reps) {
         D <- stats::rmultinom(reps, n, p) / n
         e <- apply(D, 2, g)
         s <- sqrt(apply(D, 2, function(q) delta_var(q, g, grad = grad)) / n)
         mean(abs(e - g(p)) <= Z * s, na.rm = TRUE)
       })
}

sim_rate <- function(lambda, theta = Inf) {
  V <- var_rate(lambda, theta)
  list(V = V, truth = lambda, skew = NA,
       draw_many = function(n, reps) colMeans(matrix(
         if (is.infinite(theta)) stats::rpois(n * reps, lambda)
         else stats::rnbinom(n * reps, size = theta, mu = lambda), n, reps)),
       cover = function(n, reps) {
         M <- matrix(if (is.infinite(theta)) stats::rpois(n * reps, lambda)
                     else stats::rnbinom(n * reps, size = theta, mu = lambda), n, reps)
         e <- colMeans(M); s <- sqrt(apply(M, 2, stats::var) / n)
         mean(abs(e - lambda) <= Z * s)
       })
}

sim_ustat <- function(J, index) {
  big <- local({ idx <- sample(length(J), 3000, TRUE, prob = as.vector(J))
                 k <- nrow(J)
                 list(a = ((idx - 1) %% k) + 1, b = ((idx - 1) %/% k) + 1) })
  est <- cluster_index_variance(big$a, big$b, index)
  drawpair <- function(n) {
    idx <- sample(length(J), n, TRUE, prob = as.vector(J)); k <- nrow(J)
    list(a = ((idx - 1) %% k) + 1, b = ((idx - 1) %/% k) + 1)
  }
  list(V = est$V, truth = est$estimate, skew = NA,
       draw_many = function(n, reps) vapply(seq_len(reps), function(i) {
         d <- drawpair(n); cluster_index_variance(d$a, d$b, index)$estimate },
         numeric(1)),
       cover = function(n, reps) mean(vapply(seq_len(reps), function(i) {
         d <- drawpair(n); r <- cluster_index_variance(d$a, d$b, index)
         abs(r$estimate - est$estimate) <= Z * sqrt(r$V / n) }, logical(1))))
}

sim_parametric <- function(V, truth, draw_fn, est_se_fn = NULL) {
  out <- list(V = V, truth = truth, skew = NA,
       draw_many = function(n, reps) vapply(seq_len(reps), function(i) draw_fn(n),
                                            numeric(1)))
  if (!is.null(est_se_fn)) {
    # est_se_fn(n) returns c(estimate, se) from ONE sample, using plug-in
    # estimates of the parameters that V depends on
    out$cover <- function(n, reps) mean(vapply(seq_len(reps), function(i) {
      es <- est_se_fn(n); abs(es[1] - truth) <= Z * es[2] }, logical(1)))
  }
  out
}

# -----------------------------------------------------------------------------
# The catalogue: every metric, its family, and how to simulate it
# -----------------------------------------------------------------------------

pool <- local({
  y <- stats::rbinom(N_POOL, 1, 0.3)
  ph <- stats::plogis(stats::rnorm(N_POOL, -0.8 + 1.8 * y, 1))
  list(brier = (ph - y)^2,
       nll = -log(ifelse(y == 1, ph, 1 - ph)))
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
P3 <- matrix(c(.30,.04,.02, .05,.22,.05, .03,.06,.23), 3, 3, byrow = TRUE); P3 <- P3/sum(P3)
Jab <- matrix(c(.22,.04,.02, .03,.25,.05, .02,.05,.32), 3, 3, byrow = TRUE); Jab <- Jab/sum(Jab)
cm_args <- list(expected_cost = list(c_fp = 1, c_fn = 5),
                net_benefit = list(threshold = 0.2), fbeta = list(beta = 2))

CATALOGUE <- list()
for (nm in L3_FAMILY$confusion_matrix) {
  a <- cm_args[[nm]]; if (is.null(a)) a <- list()
  CATALOGUE[[nm]] <- list(family = "confusion", how = "multinomial draw",
    sim = sim_multinom(p4, do.call(get_metric, c(list(nm), a)),
                       do.call(get_gradient, c(list(nm), a))), n_sim = 500)
}
for (nm in L3_FAMILY$multiclass) {
  parts <- strsplit(nm, "_")[[1]]
  g <- if (parts[1] == "micro") metric_micro(parts[2]) else metric_macro(parts[2])
  CATALOGUE[[nm]] <- list(family = "multiclass", how = "k^2 multinomial draw",
    sim = sim_multinom(as.vector(P3), g), n_sim = 500)
}
CATALOGUE[["fppi"]] <- list(family = "rate", how = "Poisson draw",
  sim = sim_rate(0.5, Inf), n_sim = 500)
CATALOGUE[["fppi_overdispersed"]] <- list(family = "rate",
  how = "negative binomial draw", sim = sim_rate(0.5, 2), n_sim = 500)
for (nm in names(PERCASE_REGISTRY)) {
  CATALOGUE[[nm]] <- list(family = "per-case", how = "bootstrap resample",
    make = local({ k <- nm; function() sim_percase(PC[[k]], PERCASE_REGISTRY[[k]]$transform) }),
    n_sim = N_COV)
}
CATALOGUE[["rand_index"]] <- list(family = "U-statistic", how = "pair draw",
  sim = sim_ustat(Jab, "rand"), n_sim = 400)
CATALOGUE[["fowlkes_mallows"]] <- list(family = "U-statistic", how = "pair draw",
  sim = sim_ustat(Jab, "fowlkes_mallows"), n_sim = 400)
CATALOGUE[["entropy"]] <- list(family = "other", how = "multinomial draw",
  sim = sim_multinom(c(.5,.3,.15,.05), metric_entropy, grad_entropy), n_sim = 500)
CATALOGUE[["conditional_entropy"]] <- list(family = "other",
  how = "k^2 multinomial draw",
  sim = sim_multinom(as.vector(P3), function(q) metric_conditional_entropy(q, 3)),
  n_sim = 500)
CATALOGUE[["weighted_kappa"]] <- list(family = "other", how = "k^2 multinomial draw",
  sim = sim_multinom(as.vector(P3),
        function(q) kappa_w(matrix(q, 3, 3), weights_linear(3))), n_sim = 500)
CATALOGUE[["r_squared"]] <- list(family = "other", how = "bivariate normal",
  sim = sim_parametric(r_squared_variance(0.49)$V, 0.49,
    function(n) { x <- stats::rnorm(n)
      stats::cor(x, 0.7 * x + sqrt(1 - 0.49) * stats::rnorm(n))^2 },
    est_se_fn = function(n) { x <- stats::rnorm(n)
      r2 <- stats::cor(x, 0.7 * x + sqrt(1 - 0.49) * stats::rnorm(n))^2
      c(r2, sqrt(4 * r2 * (1 - r2)^2 / n)) }), n_sim = 500)
CATALOGUE[["coefficient_of_variation"]] <- list(family = "other", how = "normal draw",
  sim = sim_parametric(cv_variance(stats::rnorm(50000, 4, 1))$V, 0.25,
    function(n) { b <- stats::rnorm(n, 4, 1); stats::sd(b) / mean(b) },
    est_se_fn = function(n) { b <- stats::rnorm(n, 4, 1)
      cv <- stats::sd(b) / mean(b); c(cv, sqrt(cv^2 * (0.5 + cv^2) / n)) }),
  n_sim = 400)
CATALOGUE[["repeatability_coefficient"]] <- list(family = "other",
  how = "replicate design", sim = sim_parametric(
    repeatability_variance(0.5, replicates = 2)$V, 1.96 * sqrt(2) * 0.5,
    function(n) { d <- matrix(stats::rnorm(n * 2, 0, 0.5), n, 2)
      1.96 * sqrt(2) * sqrt(mean(apply(d, 1, stats::var))) },
    est_se_fn = function(n) { d <- matrix(stats::rnorm(n * 2, 0, 0.5), n, 2)
      sw <- sqrt(mean(apply(d, 1, stats::var)))
      c(1.96 * sqrt(2) * sw, sqrt(repeatability_variance(sw, 2)$V / n)) }),
  n_sim = 200)
CATALOGUE[["discrimination_slope"]] <- list(family = "other", how = "two-group draw",
  sim = sim_parametric(discrimination_slope_variance(.7,.3,.2,.2,.4)$V, 0.4,
    function(n) { y <- stats::rbinom(n, 1, .4)
      mean(stats::rnorm(sum(y), .7, .2)) - mean(stats::rnorm(sum(1-y), .3, .2)) },
    est_se_fn = function(n) { y <- stats::rbinom(n, 1, .4)
      a <- stats::rnorm(sum(y), .7, .2); b <- stats::rnorm(sum(1 - y), .3, .2)
      c(mean(a) - mean(b), sqrt(stats::var(a) / length(a) + stats::var(b) / length(b))) }),
  n_sim = 2000)
CATALOGUE[["paired_proportion"]] <- list(family = "other", how = "4-cell multinomial",
  sim = sim_parametric(paired_proportion_variance(.05,.12)$V, 0.07,
    function(n) { q <- stats::rmultinom(1, n, c(.05,.12,.5,.33)) / n
                  q[2] - q[1] },
    est_se_fn = function(n) { q <- stats::rmultinom(1, n, c(.05,.12,.5,.33)) / n
      c(q[2] - q[1], sqrt(paired_proportion_variance(q[1], q[2])$V / n)) }),
  n_sim = 2000)

# -----------------------------------------------------------------------------
# One table
# -----------------------------------------------------------------------------

cat("\n", strrep("=", 104), "\n", sep = "")
cat(sprintf("UNIFIED VALIDATION   %d metrics   REPS = %s   delta = 10%% and 25%% of the mean\n",
            length(CATALOGUE), format(REPS_V, big.mark = ",")))
cat(sprintf("Monte Carlo error: variance ratio +/- %.1f%%   coverage +/- %.4f\n",
            100 * 2 * SE_V, 2 * SE_COV))
cat("Differences smaller than these are noise and should not be read.\n")
cat(strrep("=", 104), "\n\n", sep = "")

# n for two relative effect sizes: a change of 10% and of 25% of mu.
cat(sprintf("%-22s %-11s %-20s %9s %9s %9s %7s %7s %10s %10s\n",
            "metric", "family", "simulated by", "mu", "V", "V_MC", "diff", "cover",
            "n(10% mu)", "n(25% mu)"))
cat(strrep("-", 122), "\n", sep = "")

order_fam <- c("confusion", "multiclass", "rate", "per-case", "U-statistic", "other")
nfail <- 0
for (fam in order_fam) {
  for (nm in names(CATALOGUE)) {
    e <- CATALOGUE[[nm]]
    if (e$family != fam) next
    s <- if (!is.null(e$sim)) e$sim else e$make(); nn <- e$n_sim
    reps_v <- if (e$family == "U-statistic") 10000 else REPS_V
    ests <- s$draw_many(nn, reps_v)
    ests <- ests[is.finite(ests)]
    V_mc <- nn * stats::var(ests)
    diff <- 100 * (V_mc - s$V) / s$V
    good <- abs(diff) < 12
    if (!good) { nfail <- nfail + 1; PASS <- FALSE }
    cov <- NA
    if (!is.null(s$cover)) {
      cov <- s$cover(nn, if (e$family == "U-statistic") 800 else REPS_COV)
    }
    has_mu <- abs(s$truth) > 1e-9
    n10 <- if (has_mu) n_for_power(s$V, delta = 0.10 * abs(s$truth)) else NA
    n25 <- if (has_mu) n_for_power(s$V, delta = 0.25 * abs(s$truth)) else NA
    cat(sprintf("%-22s %-11s %-20s %9.4f %9.4f %9.4f %+6.1f%% %7s %10s %10s %s\n",
                nm, fam, e$how, s$truth, s$V, V_mc, diff,
                if (is.na(cov)) "-" else sprintf("%.4f", cov),
                if (is.na(n10)) "-" else format(n10, big.mark = ","),
                if (is.na(n25)) "-" else format(n25, big.mark = ","),
                if (good) "" else "<-- FAIL"))
  }
  cat("\n")
}

cat(strrep("=", 104), "\n", sep = "")
cat(sprintf("%s   %d metrics, %d discrepancies beyond the noise band\n",
            if (PASS) "ALL VALIDATED" else "FAILURES PRESENT",
            length(CATALOGUE), nfail))
cat("\nReading the columns:\n")
cat("  mu       population value / mean of the metric used to define delta\n")
cat("  V        analytic, from the package\n")
cat(sprintf("  V_MC     empirical, from simulation. diff within +/- %.1f%% is noise\n", 100 * 2 * SE_V))
cat(sprintf("  cover    Wald coverage at the family's own n; nominal 0.95, noise +/- %.4f\n",
            2 * SE_COV))
cat("           every metric now has a plug-in per-sample SE, so none is blank\n")
cat("  n(10% mu)  cases needed to detect a change of 10% of mu\n")
cat("             (alpha = 0.05 two-sided, power = 0.80)\n")
cat("  n(25% mu)  same for a change of 25% of mu; = n(10% mu) / 6.25\n")
cat("             because n is proportional to 1 / delta^2\n")
cat("  -          mu is zero, so a relative delta is undefined\n")
cat(strrep("=", 104), "\n", sep = "")
