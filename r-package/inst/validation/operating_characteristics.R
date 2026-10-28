# =============================================================================
#  operating_characteristics.R
#
#  Empirical type I error and power of the Wald test (and, for confusion-matrix
#  metrics, the score test) at the sample size metricpower calculates, for
#  every implemented variance.  Companion to validate_all.R: same
#  data-generating processes, one extra question - does the test behave?
#
#  For each metric, data are drawn at the planning operating point
#  (truth theta_1, analytic per-unit variance V_1) and
#    dV%        100 * (V_MC / V_1 - 1), simulated vs analytic variance at n_W
#    type I     rejection rate of H0: theta = theta_1 at nominal alpha
#               (data drawn at theta_1; = 1 - coverage of the CI). Per-case means
#               use the one-sample t-test (n from power.t.test, critical value
#               qt(1 - alpha/2, n - 1)); everything else the z-based Wald test
#               (needs metricpower >= 0.1.2 for n_for_power_t)
#    power      rejection rate of H0: theta = theta_0 = theta_1 - delta at
#               n_W = n_for_power(V_1, delta); target 0.80
#    power_S    confusion-matrix metrics only: score statistic standardised by
#               sqrt(V_0 / n_S) at n_S, V_0 from the null operating point
#  delta is DELTA_FRAC * |theta_1| (default 10% of the metric value); set
#  DELTA_ABS to use an absolute delta instead.
#
#  Usage (package loaded with library(metricpower) or devtools::load_all()):
#    source("operating_characteristics.R")          # 10,000 reps, ~15-30 min
#    REPS <- 1000; source("operating_characteristics.R")   # quick look
#  Writes operating_characteristics.csv and prints the Results sentence.
# =============================================================================

SCRIPT_VERSION <- "2026-10-28c"
if (!exists("REPS"))       REPS       <- 10000
if (!exists("DELTA_FRAC")) DELTA_FRAC <- 0.10
if (!exists("DELTA_ABS"))  DELTA_ABS  <- NA        # e.g. 0.05 for an absolute delta
if (!exists("ALPHA"))      ALPHA      <- 0.05
if (!exists("POWER"))      POWER      <- 0.80
if (!exists("MAX_N"))      MAX_N      <- 60000     # skip a metric whose n is larger
set.seed(2026)

N_POOL <- 10000
Z_A <- stats::qnorm(1 - ALPHA / 2)
SE_RATE <- sqrt(0.05 * 0.95 / REPS)   # MC standard error of a rejection rate near .05
SE_POW  <- sqrt(0.8 * 0.2 / REPS)
SE_V    <- sqrt(2 / REPS)             # MC standard error of a variance ratio

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
         M <- matrix(if (is.infinite(theta)) stats::rpois(n * reps, lambda)
                     else stats::rnbinom(n * reps, size = theta, mu = lambda), n, reps)
         cbind(colMeans(M), sqrt(apply(M, 2, stats::var) / n))
       })
}

sim_percase <- function(x, transform = "identity") {
  r <- var_percase_transformed(x, transform)
  h  <- switch(transform, identity = function(m) m,
               sqrt = function(m) sqrt(pmax(m, 1e-12)), exp = exp)
  dh <- switch(transform, identity = function(m) 1,
               sqrt = function(m) 1 / (2 * sqrt(pmax(m, 1e-12))), exp = exp)
  list(V = r$V, truth = r$estimate, skew = r$skew,
       est_se = function(n, reps) {
         M <- matrix(sample(x, n * reps, replace = TRUE), n, reps)
         mu <- colMeans(M); v <- apply(M, 2, stats::var)
         cbind(h(mu), sqrt(v * dh(mu)^2 / n))
       })
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
kgen <- function(n) { z <- stats::rnorm(n); list(x = z + stats::rnorm(n), y = z + stats::rnorm(n)) }
kbig <- kgen(4000); kest <- kendall_tau_variance(kbig$x, kbig$y)
CATALOGUE[["kendall_tau"]] <- list(family = "U-statistic",
  sim = sim_parametric(kest$V, 2 / pi * asin(0.5),          # population tau for corr 0.5
    function(n) { d <- kgen(n); r <- kendall_tau_variance(d$x, d$y)
                  c(r$estimate, sqrt(r$V / n)) }))
CATALOGUE[["rand_index"]] <- list(family = "U-statistic", sim = sim_ustat(Jab, "rand"))
CATALOGUE[["fowlkes_mallows"]] <- list(family = "U-statistic", sim = sim_ustat(Jab, "fowlkes_mallows"))
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

# -----------------------------------------------------------------------------
# The simulation
# -----------------------------------------------------------------------------

rows <- list(); t0 <- Sys.time()
for (fam in c("confusion", "multiclass", "rate", "per-case", "U-statistic", "other")) {
  for (nm in names(CATALOGUE)) {
    e <- CATALOGUE[[nm]]; if (e$family != fam) next
    s <- if (!is.null(e$sim)) e$sim else e$make()
    th1 <- s$truth; V1 <- s$V
    delta <- if (!is.na(DELTA_ABS)) DELTA_ABS else DELTA_FRAC * abs(th1)
    row <- list(metric = nm, family = fam, theta = th1, V = V1, skew = s$skew)
    if (delta <= 0) { row$note <- "theta = 0: relative delta undefined"; rows[[nm]] <- row; next }
    # per-case means are analysed with a one-sample t-test (sample SD as SE): size with the
    # noncentral t (power.t.test) and test with qt(., n - 1); everything else is a z-based Wald test
    t_based <- fam == "per-case"
    folded  <- nm == "volumetric_similarity"       # kink at fp = fn: folded-normal test, not Wald
    n_w <- if (folded) n_for_power_vs_folded(s$p, delta, alpha = ALPHA, power = POWER)
           else if (t_based) n_for_power_t(V1, delta, alpha = ALPHA, power = POWER)
           else n_for_power(V1, delta, alpha = ALPHA, power = POWER)
    if (n_w > MAX_N) { row$n_wald <- n_w; row$note <- "n above MAX_N, skipped"; rows[[nm]] <- row; next }
    th0 <- th1 - delta
    crit <- if (t_based) stats::qt(1 - ALPHA / 2, n_w - 1) else Z_A
    es <- s$est_se(n_w, REPS)
    ok <- is.finite(es[, 1]) & is.finite(es[, 2]) & es[, 2] > 0
    est <- es[ok, 1]; se <- es[ok, 2]
    row$dV_pct      <- 100 * (n_w * stats::var(est) / V1 - 1)        # simulated vs analytic (delta-method) V
    if (folded)     # VS: also vs the folded-normal V, which is the relevant one near the kink
      row$dV_folded_pct <- 100 * (n_w * stats::var(est) / vs_variance_folded(s$p, n_w)$V - 1)
    row$n_wald      <- n_w
    row$test        <- if (folded) "folded" else if (t_based) "t" else "z"
    if (folded) {
      # equal-tailed folded-normal test with plug-in s and D from each sample
      a <- c(0, 1, -1, 0); b <- c(2, 1, 1, 0)
      Q <- stats::rmultinom(REPS, n_w, s$p) / n_w
      rej <- c(t1 = 0, pw = 0)
      for (j in seq_len(REPS)) {
        q <- Q[, j]; Sig <- diag(q) - tcrossprod(q)
        sq <- sqrt(max(drop(t(a) %*% Sig %*% a), 1e-12) / n_w); Dq <- sum(b * q); A <- abs(sum(a * q))
        for (k in 1:2) {
          mu0 <- Dq * (1 - c(th1, th0)[k])
          lo <- vs_folded_quantile(ALPHA / 2, mu0, sq); hi <- vs_folded_quantile(1 - ALPHA / 2, mu0, sq)
          rej[k] <- rej[k] + ((A < lo) || (A > hi))
        }
      }
      row$type1_wald <- rej[["t1"]] / REPS; row$power_wald <- rej[["pw"]] / REPS
    } else {
      row$type1_wald  <- mean(abs(est - th1) > crit * se)    # H0 at the truth
      row$power_wald  <- mean(abs(est - th0) > crit * se)    # H0 at theta_0, n = n_W
    }
    row$undefined_pct <- 100 * mean(!ok)
    # score test for confusion-matrix metrics: null variance from the null operating point
    if (fam == "confusion" && !folded) {
      r <- tryCatch(suppressWarnings(do.call(n_for_power_metric,
             c(list(p = p4, name = nm, delta = delta, alpha = ALPHA, power = POWER,
                    test = "score", null_path = "auto"), e$args))), error = function(err) NULL)
      if (!is.null(r) && !is.na(r$n_score)) {
        es_s <- s$est_se(r$n_score, REPS); est_s <- es_s[is.finite(es_s[, 1]), 1]
        row$V_null <- r$V_null; row$n_score <- r$n_score
        row$power_score <- mean(abs(est_s - th0) > Z_A * sqrt(r$V_null / r$n_score))
      } else row$note <- "score: null value unreachable"
    }
    rows[[nm]] <- row
    cat(sprintf("%-26s%-12s n_W=%6d  dV=%+6.1f%%%s  typeI=%.3f  power=%.3f%s\n", nm, fam, n_w,
                row$dV_pct,
                if (folded) sprintf(" (vs delta V; vs folded V %+.1f%%)", row$dV_folded_pct) else "",
                row$type1_wald, row$power_wald,
                if (!is.null(row$power_score)) sprintf("  | n_S=%6d power_S=%.3f", row$n_score, row$power_score) else ""))
    flush.console()
  }
}

# -----------------------------------------------------------------------------
# Table, summary, Results sentence
# -----------------------------------------------------------------------------

cols <- c("metric", "family", "test", "theta", "V", "dV_pct", "dV_folded_pct", "n_wald", "type1_wald", "power_wald",
          "V_null", "n_score", "power_score", "undefined_pct", "skew", "note")
tab <- do.call(rbind, lapply(rows, function(r) {
  as.data.frame(lapply(setNames(cols, cols), function(k) if (is.null(r[[k]])) NA else r[[k]]),
                stringsAsFactors = FALSE) }))
utils::write.csv(tab, "operating_characteristics.csv", row.names = FALSE)

done <- tab[!is.na(tab$power_wald), ]
dv <- done$dV_pct; t1 <- done$type1_wald; pw <- done$power_wald
off_pow <- done[abs(done$power_wald - POWER) > 0.02, ]
off_t1  <- done[abs(done$type1_wald - ALPHA) > 0.01, ]
cat("\n", strrep("=", 118), "\n", sep = "")
cat(sprintf("OPERATING CHARACTERISTICS   %d metrics   REPS = %s   script %s   delta = %s   alpha = %s   target power = %s\n",
            nrow(done), format(REPS, big.mark = ","), SCRIPT_VERSION,
            if (!is.na(DELTA_ABS)) sprintf("%s (absolute)", DELTA_ABS) else sprintf("%.0f%% of the metric value", 100 * DELTA_FRAC),
            ALPHA, POWER))
cat(sprintf("Monte Carlo error (2 SE): variance ratio +/- %.1f%%   type I error +/- %.3f   power +/- %.3f\n",
            200 * SE_V, 2 * SE_RATE, 2 * SE_POW))
cat(strrep("=", 118), "\n", sep = "")
print(tab[, c("metric", "family", "test", "theta", "V", "dV_pct", "n_wald", "type1_wald", "power_wald", "n_score", "power_score", "skew")],
      digits = 3, row.names = FALSE)
cat(strrep("=", 118), "\nSUMMARY\n", sep = "")
nf <- is.na(done$dV_folded_pct)
cat(sprintf("  simulated vs analytic (delta-method) variance, %d metrics: %+.1f%% to %+.1f%%; %d of %d within the MC noise band\n",
            sum(nf), min(dv[nf]), max(dv[nf]), sum(abs(dv[nf]) <= 200 * SE_V), sum(nf)))
for (i in which(!nf))
  cat(sprintf("  %s: %+.1f%% vs the delta-method V (expected: the distribution is folded at the kink), %+.1f%% vs the folded-normal V; sized and tested with the folded-normal test\n",
              done$metric[i], done$dV_pct[i], done$dV_folded_pct[i]))
cat(sprintf("  type I error (nominal %s): %.3f to %.3f; %d outside %s +/- 0.01: %s\n", ALPHA, min(t1), max(t1),
            nrow(off_t1), ALPHA, paste(sprintf("%s (%.3f)", off_t1$metric, off_t1$type1_wald), collapse = ", ")))
cat(sprintf("  power at n_Wald (target %s): %.3f to %.3f; %d outside +/- 0.02: %s\n", POWER, min(pw), max(pw),
            nrow(off_pow), paste(sprintf("%s (%.3f)", off_pow$metric, off_pow$power_wald), collapse = ", ")))
sc <- done[!is.na(done$power_score), ]
if (nrow(sc)) cat(sprintf("  score test (confusion-matrix metrics, n = %d): power %.3f to %.3f\n",
                          nrow(sc), min(sc$power_score), max(sc$power_score)))
cat("\nResults sentence (fill into the manuscript):\n")
cat(sprintf(paste0("  For all %d implemented variances the Monte Carlo variance was within %.0f%% of the analytic value ",
                   "(range %+.1f%% to %+.1f%%). At nominal alpha = .05 the empirical type I error of the Wald test ranged from ",
                   "%.3f to %.3f, and empirical power at the calculated sample size ranged from %.2f to %.2f against a target of %.2f; ",
                   "the %d metrics outside +/- 0.02 of the target were %s.\n"),
            nrow(done), max(abs(dv)), min(dv), max(dv), min(t1), max(t1), min(pw), max(pw), POWER,
            nrow(off_pow), if (nrow(off_pow)) paste(off_pow$metric, collapse = ", ") else "none"))
cat(sprintf("(%.0f s)\n", as.numeric(difftime(Sys.time(), t0, units = "secs"))))
