# =============================================================================
#  variance_core.R
#
#  Sampling variance and sample size for AI performance metrics.
#  Layer 1: metrics that are smooth functions of confusion-matrix cell
#  probabilities, handled by a single multinomial delta-method engine.
#
#  Design note
#  -----------
#  Every confusion-matrix metric in the Metrics Reloaded library is a smooth
#  function g() of the multinomial cell probabilities. So rather than deriving
#  and hard-coding a separate variance per metric, we implement the engine once
#  and register each metric as a plug-in g(). Adding a metric is then one line,
#  and paired two-model comparison and clustering corrections are obtained for
#  free rather than re-derived per metric.
#
#  Convention: cell probability vectors are ordered (tp, fp, fn, tn).
#  V denotes the PER-UNIT variance, i.e. V = n * Var(theta_hat).
# =============================================================================


# -----------------------------------------------------------------------------
# 1. Constructing cell probabilities
# -----------------------------------------------------------------------------

#' Cell probabilities from counts or probabilities
#' @param tp,fp,fn,tn non-negative counts or probabilities; rescaled to sum to 1
cell_prob <- function(tp, fp, fn, tn) {
  v <- c(tp = tp, fp = fp, fn = fn, tn = tn)
  if (any(v < 0)) stop("cells must be non-negative")
  s <- sum(v)
  if (s <= 0) stop("cells sum to zero")
  v / s
}

#' Cell probabilities from an assumed operating point
#'
#' The practical planning entry point: at design time you rarely have a
#' confusion matrix, you have assumed sensitivity, specificity and prevalence.
cell_prob_from_operating_point <- function(sensitivity, specificity, prevalence) {
  if (any(c(sensitivity, specificity, prevalence) < 0) ||
      any(c(sensitivity, specificity, prevalence) > 1)) {
    stop("sensitivity, specificity and prevalence must lie in [0, 1]")
  }
  cell_prob(
    tp = sensitivity * prevalence,
    fp = (1 - specificity) * (1 - prevalence),
    fn = (1 - sensitivity) * prevalence,
    tn = specificity * (1 - prevalence)
  )
}


# -----------------------------------------------------------------------------
# 2. The delta-method engine
# -----------------------------------------------------------------------------

#' Central-difference gradient with a relative step
numeric_gradient <- function(g, p, h = 1e-6) {
  p <- as.numeric(p)
  vapply(seq_along(p), function(i) {
    step <- h * max(abs(p[i]), 1e-4)
    up <- p; up[i] <- up[i] + step
    dn <- p; dn[i] <- dn[i] - step
    (as.numeric(g(up)) - as.numeric(g(dn))) / (2 * step)
  }, numeric(1))
}

#' Per-unit variance of a smooth function of multinomial cell probabilities
#'
#' Returns V = n * Var(g(p_hat)) using Sigma = diag(p) - p p'.
#'
#' Any smooth extension of g off the simplex gives the same answer: two
#' extensions agreeing on the simplex have gradients differing by a multiple of
#' the all-ones vector, and Sigma annihilates it. So the numeric gradient is
#' safe here and no renormalisation is needed inside metric definitions.
#'
#' @param p cell probability vector (any length; must sum to 1)
#' @param g function of p returning a scalar
#' @param grad optional analytic gradient function, used in place of the
#'   numeric one when supplied
delta_var <- function(p, g, grad = NULL, h = 1e-6) {
  p <- as.numeric(p)
  if (any(p < 0)) stop("p must be non-negative")
  if (abs(sum(p) - 1) > 1e-8) stop("p must sum to 1")
  gr <- if (is.null(grad)) numeric_gradient(g, p, h) else as.numeric(grad(p))
  Sigma <- diag(p) - tcrossprod(p)
  v <- as.numeric(crossprod(gr, Sigma %*% gr))
  max(v, 0)
}


# -----------------------------------------------------------------------------
# 3. Metric definitions on (tp, fp, fn, tn)
#
#    Each is a plain function of the cell probability vector. Metrics that take
#    a parameter (beta, cost, threshold) are factories returning such a
#    function, so they slot into the engine identically.
# -----------------------------------------------------------------------------

metric_accuracy    <- function(p) unname(p[1] + p[4])
metric_sensitivity <- function(p) unname(p[1] / (p[1] + p[3]))
metric_specificity <- function(p) unname(p[4] / (p[4] + p[2]))
metric_ppv         <- function(p) unname(p[1] / (p[1] + p[2]))
metric_npv         <- function(p) unname(p[4] / (p[4] + p[3]))

metric_balanced_accuracy <- function(p) {
  unname(0.5 * (p[1] / (p[1] + p[3]) + p[4] / (p[4] + p[2])))
}

metric_youden <- function(p) {
  unname(p[1] / (p[1] + p[3]) + p[4] / (p[4] + p[2]) - 1)
}

#' F-beta score. metric_fbeta(1) is F1 / DSC on a pooled confusion matrix.
metric_fbeta <- function(beta = 1) {
  b2 <- beta^2
  function(p) unname(((1 + b2) * p[1]) / ((1 + b2) * p[1] + b2 * p[3] + p[2]))
}
metric_f1 <- metric_fbeta(1)

#' IoU / Jaccard. Equals F1 / (2 - F1); see test suite.
metric_iou <- function(p) unname(p[1] / (p[1] + p[2] + p[3]))

metric_mcc <- function(p) {
  num <- p[1] * p[4] - p[2] * p[3]
  den <- sqrt((p[1] + p[2]) * (p[1] + p[3]) * (p[4] + p[2]) * (p[4] + p[3]))
  unname(num / den)
}

#' Expected Cost. Linear in the cells, so the delta method is exact-ish here.
metric_expected_cost <- function(c_fp, c_fn, c_tp = 0, c_tn = 0) {
  function(p) unname(c_tp * p[1] + c_fp * p[2] + c_fn * p[3] + c_tn * p[4])
}

#' Net Benefit at a risk threshold.
metric_net_benefit <- function(threshold) {
  if (threshold <= 0 || threshold >= 1) stop("threshold must be in (0, 1)")
  w <- threshold / (1 - threshold)
  function(p) unname(p[1] - w * p[2])
}


# -----------------------------------------------------------------------------
# 3b. Analytic gradients
#
#     Closed-form partial derivatives for every metric above, so that no
#     numerical differentiation enters the default computation path. The
#     numeric gradient in delta_var() remains only as a fallback for
#     user-supplied metrics with no registered gradient, and the test suite
#     checks the two agree.
#
#     Writing p = (a, b, c, d) = (tp, fp, fn, tn).
# -----------------------------------------------------------------------------

grad_accuracy <- function(p) c(1, 0, 0, 1)

grad_sensitivity <- function(p) {
  S <- p[1] + p[3]
  unname(c(p[3] / S^2, 0, -p[1] / S^2, 0))
}

grad_specificity <- function(p) {
  T_ <- p[4] + p[2]
  unname(c(0, -p[4] / T_^2, 0, p[2] / T_^2))
}

grad_ppv <- function(p) {
  U <- p[1] + p[2]
  unname(c(p[2] / U^2, -p[1] / U^2, 0, 0))
}

grad_npv <- function(p) {
  W <- p[4] + p[3]
  unname(c(0, 0, -p[4] / W^2, p[3] / W^2))
}

grad_balanced_accuracy <- function(p) 0.5 * (grad_sensitivity(p) + grad_specificity(p))

grad_youden <- function(p) grad_sensitivity(p) + grad_specificity(p)

#' Gradient of F-beta.  F = A/B with A = (1+b2) a,  B = (1+b2) a + b2 c + b.
grad_fbeta <- function(beta = 1) {
  b2 <- beta^2
  function(p) {
    A  <- (1 + b2) * p[1]
    B_ <- (1 + b2) * p[1] + b2 * p[3] + p[2]
    unname(c((1 + b2) * (B_ - A) / B_^2, -A / B_^2, -A * b2 / B_^2, 0))
  }
}
grad_f1 <- grad_fbeta(1)

grad_iou <- function(p) {
  R <- p[1] + p[2] + p[3]
  unname(c((p[2] + p[3]) / R^2, -p[1] / R^2, -p[1] / R^2, 0))
}

#' Gradient of MCC.  With N = ad - bc and D = sqrt(U S T W),
#'   d(N/D)/dx = (dN/dx)/D - (N/D) * dlog(D)/dx.
grad_mcc <- function(p) {
  a <- p[1]; b <- p[2]; cc <- p[3]; d <- p[4]
  U <- a + b; S <- a + cc; T_ <- d + b; W <- d + cc
  N <- a * d - b * cc
  D <- sqrt(U * S * T_ * W)
  m <- N / D
  dN     <- c(d, -cc, -b, a)
  dlogD  <- 0.5 * c(1 / U + 1 / S, 1 / U + 1 / T_, 1 / S + 1 / W, 1 / T_ + 1 / W)
  unname(dN / D - m * dlogD)
}

grad_expected_cost <- function(c_fp, c_fn, c_tp = 0, c_tn = 0) {
  function(p) c(c_tp, c_fp, c_fn, c_tn)
}

grad_net_benefit <- function(threshold) {
  w <- threshold / (1 - threshold)
  function(p) c(1, -w, 0, 0)
}

#
#    The numbering matches the gap-analysis document. This is the R-side
#    counterpart of the variance card: it records which estimator family a
#    metric belongs to and at which aggregation level the variance is valid.
# -----------------------------------------------------------------------------

METRIC_REGISTRY <- list(
  accuracy = list(
    id = 1, family = "confusion_matrix", aggregation = "pooled",
    fun = function(...) metric_accuracy,
    grad = function(...) grad_accuracy),
  balanced_accuracy = list(
    id = 5, family = "confusion_matrix", aggregation = "pooled",
    fun = function(...) metric_balanced_accuracy,
    grad = function(...) grad_balanced_accuracy),
  iou = list(
    id = 8, family = "confusion_matrix", aggregation = "pooled",
    fun = function(...) metric_iou,
    grad = function(...) grad_iou),
  dsc_f1 = list(
    id = 13, family = "confusion_matrix", aggregation = "pooled",
    fun = function(...) metric_f1,
    grad = function(...) grad_f1),
  expected_cost = list(
    id = 16, family = "confusion_matrix", aggregation = "pooled",
    fun = function(c_fp, c_fn, c_tp = 0, c_tn = 0, ...)
      metric_expected_cost(c_fp, c_fn, c_tp, c_tn),
    grad = function(c_fp, c_fn, c_tp = 0, c_tn = 0, ...) grad_expected_cost(c_fp, c_fn, c_tp, c_tn)),
  fbeta = list(
    id = 18, family = "confusion_matrix", aggregation = "pooled",
    fun = function(beta = 1, ...) metric_fbeta(beta),
    grad = function(beta = 1, ...) grad_fbeta(beta)),
  mcc = list(
    id = 28, family = "confusion_matrix", aggregation = "pooled",
    fun = function(...) metric_mcc,
    grad = function(...) grad_mcc),
  npv = list(
    id = 31, family = "confusion_matrix", aggregation = "pooled",
    fun = function(...) metric_npv,
    grad = function(...) grad_npv),
  net_benefit = list(
    id = 32, family = "confusion_matrix", aggregation = "pooled",
    fun = function(threshold, ...) metric_net_benefit(threshold),
    grad = function(threshold, ...) grad_net_benefit(threshold)),
  ppv = list(
    id = 38, family = "confusion_matrix", aggregation = "pooled",
    fun = function(...) metric_ppv,
    grad = function(...) grad_ppv),
  sensitivity = list(
    id = 40, family = "confusion_matrix", aggregation = "pooled",
    fun = function(...) metric_sensitivity,
    grad = function(...) grad_sensitivity),
  specificity = list(
    id = 41, family = "confusion_matrix", aggregation = "pooled",
    fun = function(...) metric_specificity,
    grad = function(...) grad_specificity),
  youden = list(
    id = NA, family = "confusion_matrix", aggregation = "pooled",
    fun = function(...) metric_youden,
    grad = function(...) grad_youden)
)

get_metric <- function(name, ...) {
  entry <- METRIC_REGISTRY[[name]]
  if (is.null(entry)) {
    stop("unknown metric '", name, "'. Available: ",
         paste(names(METRIC_REGISTRY), collapse = ", "))
  }
  entry$fun(...)
}

#' Registered analytic gradient for a metric, or NULL if none
get_gradient <- function(name, ...) {
  entry <- METRIC_REGISTRY[[name]]
  if (is.null(entry) || is.null(entry$grad)) return(NULL)
  entry$grad(...)
}

#' Point estimate and per-unit variance for a registered metric
#'
#' Uses the closed-form gradient. No numerical differentiation, no resampling:
#' V = grad' [diag(p) - p p'] grad, evaluated exactly.
#'
#' @return list(estimate, V, method) where V = n * Var(estimate)
metric_variance <- function(p, name, ...) {
  g  <- get_metric(name, ...)
  gr <- get_gradient(name, ...)
  list(estimate = g(p),
       V = delta_var(p, g, grad = gr),
       method = if (is.null(gr)) "delta (numeric gradient)" else "delta (analytic gradient)")
}


# -----------------------------------------------------------------------------
# 5. Paired two-model comparison on the same test set
#
#    Joint cell probabilities over (truth, prediction_A, prediction_B), 8 cells.
#    Feeding the difference through the same engine yields
#    V_A + V_B - 2 Cov(A, B) without deriving the covariance per metric.
# -----------------------------------------------------------------------------

PAIRED_CELLS <- c("y1a1b1", "y1a1b0", "y1a0b1", "y1a0b0",
                  "y0a1b1", "y0a1b0", "y0a0b1", "y0a0b0")

#' Collapse the 8-cell joint distribution to one model's 2x2 cells
paired_marginal <- function(p8, model = c("A", "B")) {
  model <- match.arg(model)
  p8 <- as.numeric(p8)
  y <- c(1, 1, 1, 1, 0, 0, 0, 0)
  a <- c(1, 1, 0, 0, 1, 1, 0, 0)
  b <- c(1, 0, 1, 0, 1, 0, 1, 0)
  pr <- if (model == "A") a else b
  c(sum(p8[y == 1 & pr == 1]),   # tp
    sum(p8[y == 0 & pr == 1]),   # fp
    sum(p8[y == 1 & pr == 0]),   # fn
    sum(p8[y == 0 & pr == 0]))   # tn
}

#' Marginalisation matrices: the linear map from the 8 joint cells to one
#' model's (tp, fp, fn, tn). Used for the chain rule in delta_var_paired().
PAIRED_M <- list(
  A = rbind(tp = c(1, 1, 0, 0, 0, 0, 0, 0),
            fp = c(0, 0, 0, 0, 1, 1, 0, 0),
            fn = c(0, 0, 1, 1, 0, 0, 0, 0),
            tn = c(0, 0, 0, 0, 0, 0, 1, 1)),
  B = rbind(tp = c(1, 0, 1, 0, 0, 0, 0, 0),
            fp = c(0, 0, 0, 0, 1, 0, 1, 0),
            fn = c(0, 1, 0, 1, 0, 0, 0, 0),
            tn = c(0, 0, 0, 0, 0, 1, 0, 1))
)

#' Per-unit variance of the paired difference metric(A) - metric(B)
#'
#' Yields V_A + V_B - 2 Cov(A, B) in closed form. The gradient with respect to
#' the 8 joint cells follows from the chain rule through the linear
#' marginalisation M: grad = M_A' grad_g(p_A) - M_B' grad_g(p_B).
delta_var_paired <- function(p8, name, ...) {
  g  <- get_metric(name, ...)
  gr <- get_gradient(name, ...)
  d  <- function(p) g(paired_marginal(p, "A")) - g(paired_marginal(p, "B"))
  dgrad <- if (is.null(gr)) NULL else function(p) {
    as.numeric(crossprod(PAIRED_M$A, gr(paired_marginal(p, "A"))) -
               crossprod(PAIRED_M$B, gr(paired_marginal(p, "B"))))
  }
  list(estimate = d(p8),
       V = delta_var(p8, d, grad = dgrad),
       method = if (is.null(dgrad)) "delta (numeric gradient)" else "delta (analytic gradient)")
}


# -----------------------------------------------------------------------------
# 6. Clustering
# -----------------------------------------------------------------------------

#' Design effect for clustered units (m_bar units per cluster, ICC rho)
#'
#' Multiply the required n by this when the analysis unit is a lesion, slice or
#' object but the sampling unit is a patient. Applies to any metric.
design_effect <- function(m_bar, icc) {
  if (m_bar < 1) stop("m_bar must be >= 1")
  if (icc < 0 || icc > 1) stop("icc must lie in [0, 1]")
  1 + (m_bar - 1) * icc
}


# -----------------------------------------------------------------------------
# 7. Sample size and power
# -----------------------------------------------------------------------------

#' Required n for a target power
#'
#' Uses separate null and alternative variances where available:
#'   n = (z_{1-alpha/2} sqrt(V0) + z_{1-beta} sqrt(V1))^2 / delta^2
#' which reduces to the familiar (z + z)^2 V / delta^2 when V0 = V1.
#'
#' @param V_alt per-unit variance under the alternative
#' @param delta effect size to detect, on the metric's own scale
#' @param V_null per-unit variance under the null; defaults to V_alt
#' @param deff design effect from clustering; see design_effect()
n_for_power <- function(V_alt, delta, V_null = V_alt, alpha = 0.05,
                        power = 0.80, two_sided = TRUE, deff = 1) {
  if (delta == 0) stop("delta must be non-zero")
  za <- stats::qnorm(1 - alpha / (if (two_sided) 2 else 1))
  zb <- stats::qnorm(power)
  n <- (za * sqrt(V_null) + zb * sqrt(V_alt))^2 / delta^2
  ceiling(n * deff)
}

#' Sample size for a one-sample t-test on a per-case mean (noncentral t, as power.t.test)
#'
#' The z formula treats sigma as known; with the sample SD as the SE the z critical
#' value inflates the type I error at small n (0.12 at n = 5, 0.07 at n = 15).
#' This returns the smallest n whose t-test power reaches the target; it equals the
#' z-based n for large n. deff multiplies the result.
n_for_power_t <- function(V_alt, delta, alpha = 0.05, power = 0.80,
                          two_sided = TRUE, deff = 1) {
  if (delta == 0) stop("delta must be non-zero")
  r <- stats::power.t.test(delta = abs(delta), sd = sqrt(V_alt), sig.level = alpha,
                           power = power, type = "one.sample",
                           alternative = if (two_sided) "two.sided" else "one.sided")
  ceiling(r$n * deff)
}

#' Power of the one-sample t-test at a given n (noncentral t)
power_for_n_t <- function(n, V_alt, delta, alpha = 0.05, two_sided = TRUE, deff = 1) {
  stats::power.t.test(n = n / deff, delta = abs(delta), sd = sqrt(V_alt), sig.level = alpha,
                      type = "one.sample",
                      alternative = if (two_sided) "two.sided" else "one.sided")$power
}

#' Folded-normal test for volumetric similarity (VS = 1 - |fp - fn| / (2tp + fp + fn))
#'
#' Y = fp - fn is a linear function of the cell proportions, so Y ~ N(mu, s^2) with
#' s^2 = Var(Y)/n, and |Y| is folded normal. Near the kink at fp = fn the sampling
#' distribution of VS is therefore folded, not normal: the Wald test with the
#' delta-method SE is conservative (type I 0.02 at 1.6 SE) and the Wald test with the
#' folded-normal SE is anti-conservative (0.08-0.10). This test uses the folded
#' distribution itself and holds the nominal level; D is taken at the operating point.
vs_folded_cdf <- function(cc, mu, s) stats::pnorm((cc - mu) / s) - stats::pnorm((-cc - mu) / s)
vs_folded_quantile <- function(q, mu, s)
  stats::uniroot(function(cc) vs_folded_cdf(cc, mu, s) - q, c(0, abs(mu) + 12 * s), tol = 1e-10)$root

#' Power of the equal-tailed folded-normal test of H0: VS = vs0 when the truth is p1
vs_folded_power <- function(n, p1, vs0, alpha = 0.05, two_sided = TRUE) {
  a <- c(0, 1, -1, 0); b <- c(2, 1, 1, 0)
  Sig <- diag(p1) - tcrossprod(p1); D <- sum(b * p1); mu1 <- abs(sum(a * p1))
  s <- sqrt(max(drop(t(a) %*% Sig %*% a) / n, 1e-300)); mu0 <- D * (1 - vs0)
  if (two_sided) {
    lo <- vs_folded_quantile(alpha / 2, mu0, s); hi <- vs_folded_quantile(1 - alpha / 2, mu0, s)
    return(vs_folded_cdf(lo, mu1, s) + 1 - vs_folded_cdf(hi, mu1, s))
  }
  if (mu1 > mu0) 1 - vs_folded_cdf(vs_folded_quantile(1 - alpha, mu0, s), mu1, s)
  else vs_folded_cdf(vs_folded_quantile(alpha, mu0, s), mu1, s)
}

#' Smallest n at which the folded-normal test for volumetric similarity reaches the target power
n_for_power_vs_folded <- function(p1, delta, alpha = 0.05, power = 0.80, two_sided = TRUE, deff = 1) {
  vs0 <- metric_volumetric_similarity(p1) - delta
  lo <- 4
  hi <- max(8, 2 * n_for_power(delta_var(p1, metric_volumetric_similarity,
                                         grad = grad_volumetric_similarity), delta,
                               alpha = alpha, power = power, two_sided = two_sided))
  while (vs_folded_power(hi, p1, vs0, alpha, two_sided) < power && hi < 1e7) hi <- hi * 2
  while (hi - lo > 1) {
    mid <- (lo + hi) %/% 2
    if (vs_folded_power(mid, p1, vs0, alpha, two_sided) >= power) hi <- mid else lo <- mid
  }
  ceiling(hi * deff)
}

#' Volumetric similarity: distance of |fp - fn| from the kink at fp = fn, in SEs
#'
#' The metric is not differentiable at fp = fn. Below about 3 SE a share of samples
#' crosses the kink, the absolute value folds the sampling distribution and the
#' delta-method variance is too large (18% at 1.6 SE; the test is conservative).
kink_distance <- function(p, n) {
  fp <- p[2]; fn <- p[3]
  abs(fp - fn) / sqrt(max((fp + fn - (fp - fn)^2) / n, 1e-300))
}

#' Power attained at a given n
power_for_n <- function(n, V_alt, delta, V_null = V_alt, alpha = 0.05,
                        two_sided = TRUE, deff = 1) {
  za <- stats::qnorm(1 - alpha / (if (two_sided) 2 else 1))
  n_eff <- n / deff
  stats::pnorm((sqrt(n_eff) * abs(delta) - za * sqrt(V_null)) / sqrt(V_alt))
}

#' Required n for a confidence interval of given half-width
n_for_precision <- function(V, halfwidth, conf.level = 0.95, deff = 1) {
  if (halfwidth <= 0) stop("halfwidth must be positive")
  z <- stats::qnorm(1 - (1 - conf.level) / 2)
  ceiling(z^2 * V / halfwidth^2 * deff)
}

#' Wald confidence interval from a point estimate and per-unit variance
metric_ci <- function(estimate, V, n, conf.level = 0.95) {
  z <- stats::qnorm(1 - (1 - conf.level) / 2)
  se <- sqrt(V / n)
  c(estimate = estimate, se = se,
    lower = estimate - z * se, upper = estimate + z * se)
}


#' Cell probabilities achieving a target metric value
#'
#' For a one-sample test of H0: theta = theta0, the variance under the null is
#' V(p0), not V(p1). The two differ substantially for most metrics, so the null
#' distribution has to be constructed rather than assumed equal.
#'
#' The map from a metric value back to a confusion matrix is not unique, so a
#' path must be chosen. The default degrades sensitivity, holding specificity
#' and prevalence fixed: mass moves from tp to fn. This corresponds to "the
#' model misses more positives", which is usually the scenario of interest.
#' path = "specificity" moves mass from tn to fp instead.
#'
#' @return cell probability vector with metric value equal to target
cell_prob_at_metric <- function(p, name, target,
                                path = c("sensitivity", "specificity"), ...) {
  path <- match.arg(path)
  g <- get_metric(name, ...)
  shift <- if (path == "sensitivity") {
    function(t) cell_prob(p[1] - t, p[2], p[3] + t, p[4])
  } else {
    function(t) cell_prob(p[1], p[2] + t, p[3], p[4] - t)
  }
  hi <- if (path == "sensitivity") p[1] else p[4]
  f <- function(t) g(shift(t)) - target
  lo_val <- f(0)
  hi_val <- f(hi * (1 - 1e-8))
  if (lo_val * hi_val > 0) {
    stop("target metric value ", signif(target, 6),
         " is not reachable along the '", path, "' path from this operating point")
  }
  t0 <- stats::uniroot(f, c(0, hi * (1 - 1e-8)), tol = 1e-12)$root
  shift(t0)
}

#' Sample size for a registered metric
#'
#' The sample-size formula must match the test statistic that will actually be
#' computed. Two conventions are in use and they are NOT interchangeable:
#'
#'   test = "wald"   reject when |theta_hat - theta_0| / se_hat > z, where
#'                   se_hat is the plug-in standard error estimated from the
#'                   data. Under the alternative se_hat converges to
#'                   sqrt(V1 / n), so
#'                       n = (z_{1-alpha/2} + z_{1-beta})^2 V1 / delta^2.
#'                   This matches metric_ci() and is what most people compute.
#'
#'   test = "score"  reject when |theta_hat - theta_0| / sqrt(V0 / n) > z,
#'                   standardising by the variance under the null, giving
#'                       n = (z_{1-alpha/2} sqrt(V0) + z_{1-beta} sqrt(V1))^2
#'                           / delta^2.
#'                   This is the convention used by classical one-sample
#'                   proportion sample-size formulas.
#'
#' Applying the score-test n to a Wald test over-powers the study, and applying
#' the Wald n to a score test under-powers it. For F1 at a typical operating
#' point the two differ by about 14%.
#'
#' Both variances are always returned so the choice is visible.
#'
#' Note on finite samples: the Wald statistic's plug-in standard error is
#' strongly negatively correlated with the point estimate, which costs roughly
#' one to two percentage points of power at moderate n. The analytic n is
#' asymptotically correct; treat it as a floor rather than a guarantee.
#'
#' @param p cell probabilities under the alternative
#' @param delta reduction in the metric to detect (positive)
#' @param test which test statistic the analysis will use
#' @return list with n, the test convention, both variances, the null cell
#'   distribution, and the n the other convention would have given
n_for_power_metric <- function(p, name, delta, alpha = 0.05, power = 0.80,
                               two_sided = TRUE, deff = 1,
                               test = c("wald", "score"),
                               null_path = c("auto", "sensitivity", "specificity"), ...) {
  test      <- match.arg(test)
  null_path <- match.arg(null_path)
  g  <- get_metric(name, ...)
  gr <- get_gradient(name, ...)
  theta1 <- g(p)
  V1 <- delta_var(p, g, grad = gr)

  # The null cell distribution is only needed for the score n. Some metrics do
  # not move along a given path (specificity is unchanged when mass moves from
  # tp to fn), so "auto" tries the sensitivity path and falls back to the
  # specificity path. If neither reaches the target, the Wald n is still valid
  # and only the score n is unavailable.
  paths <- if (null_path == "auto") c("sensitivity", "specificity") else null_path
  p0 <- NULL; used_path <- NA_character_
  for (pth in paths) {
    p0 <- tryCatch(cell_prob_at_metric(p, name, theta1 - delta, path = pth, ...),
                   error = function(e) NULL)
    if (!is.null(p0)) { used_path <- pth; break }
  }
  if (is.null(p0)) {
    if (test == "score") stop("null value ", signif(theta1 - delta, 6),
      " is not reachable from this operating point along any path; ",
      "use test = \"wald\" or a smaller delta", call. = FALSE)
    warning("null value not reachable along any path; n_score is NA, ",
            "the Wald n is unaffected", call. = FALSE)
    V0 <- NA_real_
  } else {
    V0 <- delta_var(p0, g, grad = gr)
  }

  n_wald  <- n_for_power(V_alt = V1, delta = delta, V_null = V1,
                         alpha = alpha, power = power,
                         two_sided = two_sided, deff = deff)
  n_score <- if (is.na(V0)) NA_integer_ else
    n_for_power(V_alt = V1, delta = delta, V_null = V0,
                alpha = alpha, power = power,
                two_sided = two_sided, deff = deff)

  n_folded <- NA_integer_
  if (name == "volumetric_similarity") {
    # kink at fp = fn: the sampling distribution of VS is folded normal there, so the
    # sample size comes from the folded-normal test (vs_folded_power), not from Wald
    n_folded <- n_for_power_vs_folded(p, delta, alpha, power, two_sided, deff)
    kd <- kink_distance(p, n_folded)
    message(sprintf(paste0("volumetric similarity: |fp - fn| is %.1f SE from the kink at fp = fn at ",
                           "n = %d. n is the sample size of the folded-normal test (the test to run); ",
                           "the delta-method Wald n would be %d and its test is conservative near the kink."),
                    kd, n_folded, n_wald))
  }

  list(n = if (!is.na(n_folded)) n_folded else if (test == "wald") n_wald else n_score,
       n_folded = n_folded,
       test = test,
       estimate = theta1, null_value = theta1 - delta,
       V_alt = V1, V_null = V0, p_null = p0, null_path = used_path,
       n_wald = n_wald, n_score = n_score,
       min_expected_cell = check_cell_counts(p, if (test == "wald") n_wald else n_score))
}

#' Warn when the asymptotic approximation is unreliable
#'
#' The delta method is a large-sample result. When the smallest expected cell
#' count is small the approximation degrades and, worse, degrades in the
#' optimistic direction: at prevalence 0.02 with n = 200 the delta variance for
#' F1 understates the truth by about 6%, and a noticeable fraction of samples
#' contain no predicted positives at all, leaving the metric undefined.
#'
#' @param p cell probabilities
#' @param n sample size
#' @param min_expected threshold on the smallest expected cell count
#' @return invisibly, the smallest expected cell count
check_cell_counts <- function(p, n, min_expected = 10) {
  m <- min(p) * n
  if (m < min_expected) {
    warning(sprintf(
      paste0("smallest expected cell count is %.1f (threshold %g). The delta ",
             "method is unreliable here and errs optimistically; the metric ",
             "may also be undefined in some samples. Prefer a simulation- or ",
             "exact-based sample size at this operating point."),
      m, min_expected), call. = FALSE)
  }
  invisible(m)
}

# -----------------------------------------------------------------------------
# 11. Finite-sample sensitivity analysis
#
#     Everything above is asymptotic. In practice the available test set is
#     often small, and the delta method errs in the OPTIMISTIC direction: it
#     understates the variance, so the computed n is too small and the achieved
#     power falls below nominal. These functions quantify that gap at the n you
#     actually have, rather than assuming it away.
#
#     Simulation is used here deliberately. This is the one place in the package
#     where the asymptotic answer is the thing being questioned, so it cannot
#     also be the thing doing the checking.
# -----------------------------------------------------------------------------

#' How far off is the asymptotic variance at the sample sizes you can afford?
#'
#' @param p cell probabilities
#' @param name registered metric name
#' @param n_grid sample sizes to examine
#' @param delta optional effect size; if supplied, achieved power is reported
#' @param reps simulation replicates per grid point
#' @return data frame with V_analytic, V_finite, their ratio, and the sample
#'   size inflation that would restore the nominal variance
sensitivity_by_n <- function(p, name, n_grid = c(50, 100, 250, 500, 1000, 2500),
                             delta = NULL, reps = 20000, alpha = 0.05,
                             power = 0.80, ...) {
  g  <- get_metric(name, ...)
  gr <- get_gradient(name, ...)
  V  <- delta_var(p, g, grad = gr)
  z  <- stats::qnorm(1 - alpha / 2)
  theta1 <- g(p)
  out <- lapply(n_grid, function(nn) {
    dr  <- stats::rmultinom(reps, size = nn, prob = p) / nn
    est <- apply(dr, 2, g)
    ok  <- is.finite(est)
    V_emp <- nn * stats::var(est[ok])
    row <- data.frame(
      n = nn,
      V_analytic = V,
      V_finite = V_emp,
      ratio = V_emp / V,
      n_inflation = V_emp / V,
      undefined_pct = 100 * mean(!ok),
      min_expected_cell = min(p) * nn
    )
    if (!is.null(delta)) {
      se  <- sqrt(apply(dr, 2, function(q) delta_var(q, g, grad = gr)) / nn)
      row$power_actual <- mean(
        abs(est - (theta1 - delta)) / se > z, na.rm = TRUE)
      row$power_nominal <- power_for_n(nn, V_alt = V, delta = delta,
                                       alpha = alpha)
    }
    row
  })
  do.call(rbind, out)
}

#' The same question for the per-case mean family (Brier, RBS, NLL, per-case DSC)
#'
#' Note carefully what is and is not at issue here. Var(mean) = sigma^2 / n
#' holds EXACTLY for any distribution, so there is nothing to check about the
#' variance itself. What the CLT governs is whether the sampling distribution
#' is close enough to normal for a Wald interval to cover at its nominal rate,
#' and that is what degrades when per-case scores are skewed — as Brier scores
#' are, being bounded below by zero with most mass near it.
#'
#' Coverage is therefore computed with the ESTIMATED standard error, as it
#' would be in practice, not the true one. Using the true standard error hides
#' the problem, because the estimated variance is correlated with the mean.
#'
#' Resamples from the observed per-case scores, so no distributional
#' assumption is needed.
#'
#' @return data frame with empirical coverage and the two one-sided miss rates;
#'   asymmetric misses are the signature of skew
sensitivity_by_n_percase <- function(x, n_grid = c(25, 50, 100, 250, 500),
                                     reps = 20000, conf.level = 0.95) {
  mu <- mean(x); z <- stats::qnorm(1 - (1 - conf.level) / 2)
  out <- lapply(n_grid, function(nn) {
    s <- replicate(reps, {
      b  <- sample(x, nn, replace = TRUE)
      se <- sqrt(stats::var(b) / nn)
      c(mean(b) - z * se, mean(b) + z * se)
    })
    lo <- s[1, ]; hi <- s[2, ]
    data.frame(n = nn,
               coverage_wald = mean(lo <= mu & mu <= hi),
               miss_low  = mean(mu < lo),   # interval entirely below the truth
               miss_high = mean(mu > hi),
               skew = mean(((x - mu) / stats::sd(x))^3),
               nominal = conf.level)
  })
  do.call(rbind, out)
}

#' Monte Carlo noise band for a simulated variance ratio
#'
#' The ratio V_finite / V_analytic is itself estimated, and a sample variance
#' has relative standard error sqrt((2 + kurtosis) / B). Any observed ratio
#' inside two of these of 1 carries no information: it is simulation noise, not
#' a finite-sample effect. Reporting a "correction" from inside this band is
#' how a diagnostic turns into a random number generator.
#'
#' @param est simulated metric estimates
#' @param reps number of replicates used
#' @return list with the standard deviation and the two-sided band around 1
ratio_noise_band <- function(est, reps) {
  z <- (est - mean(est)) / stats::sd(est)
  ku <- mean(z^4) - 3
  sd_ratio <- sqrt((2 + max(ku, -2)) / reps)
  list(sd = sd_ratio, lower = 1 - 2 * sd_ratio, upper = 1 + 2 * sd_ratio,
       kurtosis = ku,
       reps_for = function(bias) ceiling((2 + max(ku, -2)) / (bias / 2)^2))
}

#' Inflate a sample size to compensate for the finite-sample shortfall
#'
#' Refuses to correct when the observed ratio lies inside the Monte Carlo noise
#' band, because at moderate n the true bias is well under one per cent while
#' the noise at a typical number of replicates is around two per cent. Without
#' this guard the function happily returns corrections that go the wrong way -
#' a smaller n than the analytic one - which is impossible, since the delta
#' method can only understate the variance.
#'
#' @return list with the analytic n, the corrected n, whether a correction was
#'   applied, and the replicates that would be needed to resolve the observed gap
n_finite_sample_corrected <- function(p, name, delta, alpha = 0.05,
                                      power = 0.80, reps = 20000, ...) {
  n0 <- n_for_power_metric(p, name, delta = delta, alpha = alpha,
                           power = power, ...)$n
  g  <- get_metric(name, ...); gr <- get_gradient(name, ...)
  V  <- delta_var(p, g, grad = gr)
  dr  <- stats::rmultinom(reps, size = n0, prob = p) / n0
  est <- apply(dr, 2, g); est <- est[is.finite(est)]
  ratio <- (n0 * stats::var(est)) / V
  band  <- ratio_noise_band(est, length(est))
  inside <- ratio > band$lower && ratio < band$upper
  list(n_asymptotic = n0,
       n_corrected  = if (inside) n0 else ceiling(n0 * ratio),
       ratio = ratio, applied = !inside,
       noise_band = c(lower = band$lower, upper = band$upper),
       reps_to_resolve = band$reps_for(max(abs(ratio - 1), 1e-4)),
       note = if (inside)
         sprintf(paste0("ratio %.3f is inside the noise band [%.3f, %.3f]; no ",
                        "correction applied. %s replicates would be needed to ",
                        "resolve a gap this size."),
                 ratio, band$lower, band$upper,
                 format(band$reps_for(max(abs(ratio - 1), 1e-4)), big.mark = ","))
       else sprintf("ratio %.3f exceeds the noise band [%.3f, %.3f]; correction applied.",
                    ratio, band$lower, band$upper))
}

# -----------------------------------------------------------------------------
# 8. Weighted Cohen's kappa (metric 42)
#
#    Not a 2x2 metric, so it gets its own entry point. Two independent
#    implementations are provided and cross-checked in the test suite: the
#    closed-form Fleiss-Cohen-Everitt (1969) variance, and the generic engine
#    applied to the vectorised k x k table.
# -----------------------------------------------------------------------------

weights_identity   <- function(k) diag(k)
weights_linear     <- function(k) 1 - abs(outer(1:k, 1:k, "-")) / (k - 1)
weights_quadratic  <- function(k) 1 - (outer(1:k, 1:k, "-") / (k - 1))^2

#' Weighted kappa from a joint probability matrix
kappa_w <- function(P, W) {
  P <- P / sum(P)
  pi_ <- rowSums(P); p_j <- colSums(P)
  po <- sum(W * P)
  pe <- sum(W * outer(pi_, p_j))
  unname((po - pe) / (1 - pe))
}

#' Fleiss-Cohen-Everitt (1969) asymptotic per-unit variance of weighted kappa
kappa_w_var_fce <- function(P, W) {
  P <- P / sum(P)
  k <- nrow(P)
  pi_ <- rowSums(P); p_j <- colSums(P)
  po <- sum(W * P)
  pe <- sum(W * outer(pi_, p_j))
  wi <- as.vector(W %*% p_j)    # wbar_{i.}
  wj <- as.vector(t(W) %*% pi_) # wbar_{.j}
  inner <- outer(wi, wj, "+")
  term <- sum(P * (W * (1 - pe) - inner * (1 - po))^2)
  V <- (term - (po * pe - 2 * pe + po)^2) / (1 - pe)^4
  max(V, 0)
}

#' Weighted kappa variance via the generic multinomial engine
kappa_w_var_delta <- function(P, W) {
  k <- nrow(P)
  P <- P / sum(P)
  delta_var(as.vector(P), function(p) kappa_w(matrix(p, k, k), W))
}


# -----------------------------------------------------------------------------
# 9. Per-case mean metrics (metric 9 Brier, 39 RBS, 30 NLL)
#
#    A different estimator family: the mean of a per-case score. The per-unit
#    variance is just the between-case variance, so the same n_for_power()
#    applies. This is also the family that per-case DSC, NSD and HD95 belong
#    to, which is why it is kept general rather than named after Brier.
# -----------------------------------------------------------------------------

#' Per-unit variance of a per-case mean metric
var_percase <- function(x) stats::var(x)

#' Per-unit variance of a paired per-case difference (Diebold-Mariano style)
var_percase_paired <- function(x_a, x_b) {
  if (length(x_a) != length(x_b)) stop("x_a and x_b must have equal length")
  stats::var(x_a - x_b)
}

#' Per-case Brier scores from labels and predicted probabilities
brier_scores <- function(y, phat) (phat - y)^2

#' Variance of sqrt(mean) via the delta method, for Root Brier Score
var_sqrt_transform <- function(V, mean_value) {
  if (mean_value <= 0) stop("mean_value must be positive")
  V / (4 * mean_value)
}


# -----------------------------------------------------------------------------
# 10. Guard: calibration error metrics (14, 15, 12, 24)
# -----------------------------------------------------------------------------

#' Deliberately refuses. ECE, ECE-KDE, CWCE and KCE have non-regular
#' asymptotics: the estimators are biased or degenerate under the null of
#' perfect calibration, and the convergence rate and asymptotic variance differ
#' between calibrated and miscalibrated models. Inverting a plug-in sigma^2
#' through n = (z + z)^2 V / delta^2 gives a number, but not a valid one.
n_for_power_calibration_error <- function(...) {
  stop(
    "Sample size for calibration-error metrics is not supported.\n",
    "  ECE / ECE-KDE / CWCE / KCE have non-regular asymptotics: the estimator\n",
    "  is degenerate under the null of perfect calibration, the convergence\n",
    "  rate differs between calibrated and miscalibrated models, and Wald\n",
    "  intervals cover negative values for a non-negative quantity.\n",
    "  Use a simulation-based or permutation-test route instead.",
    call. = FALSE
  )
}

# -----------------------------------------------------------------------------
# 12. Is the delta method reliable at the n you are planning?
#
#     The delta method is a large-sample result, so it needs a minimum n. That
#     is the only reason a sensitivity analysis exists in this package. One
#     function, one question, one answer.
# -----------------------------------------------------------------------------

#' Check whether the delta method can be trusted at a planned sample size
#'
#' @param p cell probabilities
#' @param name registered metric
#' @param n the sample size you are planning
#' @param reps simulation replicates used for the check
#' @return list with the analytic variance, the simulated variance, their ratio
#'   and a plain verdict
check_delta_reliable <- function(p, name, n, reps = 20000, ...) {
  g  <- get_metric(name, ...); gr <- get_gradient(name, ...)
  V  <- delta_var(p, g, grad = gr)
  dr <- stats::rmultinom(reps, size = n, prob = p) / n
  est <- apply(dr, 2, g); est <- est[is.finite(est)]
  ratio <- (n * stats::var(est)) / V
  reliable <- n >= 250 || ratio < 1.02
  list(metric = name, n = n, V_analytic = V,
       V_finite = n * stats::var(est), ratio = ratio,
       reliable = reliable,
       verdict = if (reliable)
         sprintf("OK. The delta method is reliable at n = %d.", n)
       else sprintf(paste0("Caution. At n = %d the analytic variance understates ",
                           "the truth by about %.0f%%, so the required sample size ",
                           "is optimistic. The delta method becomes reliable from ",
                           "about n = 250."), n, 100 * (ratio - 1)))
}
