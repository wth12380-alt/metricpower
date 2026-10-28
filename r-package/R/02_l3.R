# =============================================================================
#  variance_l3.R
#
#  L3 layer: metrics whose asymptotic variance is published or elementary, but
#  which no package inverts to a sample size.
#
#  Four estimator families are handled here, and the split matters because the
#  small-sample behaviour differs sharply between them:
#
#    (a) confusion-matrix functions  - delta method on a multinomial
#    (b) multiclass micro/macro      - delta method on a k x k multinomial
#    (c) count rates                 - Poisson or negative binomial
#    (d) per-case means              - CLT for a mean, optionally transformed
#
#  For (a), (b) and (c) the delta method is doing work and the finite-sample
#  error shows up in the variance. For (d) the variance is exact at every n,
#  Var(mean) = sigma^2 / n holding for any distribution, and what degrades
#  instead is the normal approximation - which is why coverage, not variance,
#  is the diagnostic reported for that family.
#
#  Requires variance_core.R, which is loaded automatically below if it has not
#  been sourced already.
# =============================================================================

# -----------------------------------------------------------------------------
# (a) Additional confusion-matrix metrics
#     Each is a one-line plug-in with a closed-form gradient. p = (tp, fp, fn, tn).
# -----------------------------------------------------------------------------

metric_markedness <- function(p) unname(p[1] / (p[1] + p[2]) + p[4] / (p[4] + p[3]) - 1)
grad_markedness <- function(p) {
  U <- p[1] + p[2]; W <- p[4] + p[3]
  unname(c(p[2] / U^2, -p[1] / U^2, -p[4] / W^2, p[3] / W^2))
}

metric_balanced_error_rate <- function(p) unname(1 - metric_balanced_accuracy(p))
grad_balanced_error_rate <- function(p) -grad_balanced_accuracy(p)

metric_fdr <- function(p) unname(p[2] / (p[1] + p[2]))
grad_fdr <- function(p) { U <- p[1] + p[2]; unname(c(-p[2] / U^2, p[1] / U^2, 0, 0)) }

metric_for <- function(p) unname(p[3] / (p[4] + p[3]))
grad_for <- function(p) { W <- p[4] + p[3]; unname(c(0, 0, p[4] / W^2, -p[3] / W^2)) }

#' Class error: the misclassification rate within the positive class
metric_class_error <- function(p) unname(p[3] / (p[1] + p[3]))
grad_class_error <- function(p) { S <- p[1] + p[3]; unname(c(-p[3] / S^2, 0, p[1] / S^2, 0)) }

#' False referral rate: of those referred on, the fraction that did not need it
metric_false_referral_rate <- metric_fdr
grad_false_referral_rate <- grad_fdr

L3_CM <- list(
  markedness           = list(fun = metric_markedness,           grad = grad_markedness),
  balanced_error_rate  = list(fun = metric_balanced_error_rate,  grad = grad_balanced_error_rate),
  false_discovery_rate = list(fun = metric_fdr,                  grad = grad_fdr),
  false_omission_rate  = list(fun = metric_for,                  grad = grad_for),
  class_error          = list(fun = metric_class_error,          grad = grad_class_error),
  false_referral_rate  = list(fun = metric_false_referral_rate,  grad = grad_false_referral_rate)
)

#' Locate the environment that actually holds METRIC_REGISTRY
#'
#' Using `<<-` for this was a mistake: it searches enclosing environments from
#' the function's closure outwards, so the result depends on how the files were
#' sourced (top level, local = TRUE, or inside a package). Resolving the
#' environment explicitly from where get_metric() was defined works in all three.
.registry_env <- function() {
  if (!exists("get_metric", mode = "function")) {
    stop("variance_core.R is not loaded", call. = FALSE)
  }
  e <- environment(get("get_metric", mode = "function"))
  if (is.null(e) || identical(e, emptyenv())) globalenv() else e
}

#' Add the L3 confusion-matrix metrics to METRIC_REGISTRY
#'
#' Idempotent: calling it twice is harmless. Runs automatically when this file
#' is sourced, so get_metric("markedness") works immediately.
register_l3_cm <- function(envir = NULL) {
  if (is.null(envir)) envir <- .registry_env()
  reg <- get("METRIC_REGISTRY", envir = envir)
  for (nm in names(L3_CM)) {
    e <- L3_CM[[nm]]
    reg[[nm]] <- list(
      id = NA, family = "confusion_matrix", aggregation = "pooled",
      fun  = local({ f <- e$fun;  function(...) f }),
      grad = local({ g <- e$grad; function(...) g }))
  }
  assign("METRIC_REGISTRY", reg, envir = envir)
  invisible(names(L3_CM))
}

# -----------------------------------------------------------------------------
# (b) Multiclass micro and macro averaging
#
#     The engine takes any length of probability vector, so a k x k joint table
#     vectorised to k^2 cells goes through unchanged. Micro-averaging pools the
#     cells first; macro-averaging computes per class and then averages. The two
#     have materially different variances under class imbalance, because macro
#     gives a rare class the same weight as a common one and therefore inherits
#     its instability.
# -----------------------------------------------------------------------------

.cm_class <- function(P, j) {
  tp <- P[j, j]; fp <- sum(P[, j]) - tp; fn <- sum(P[j, ]) - tp
  c(tp = tp, fp = fp, fn = fn, tn = 1 - tp - fp - fn)
}

metric_micro <- function(what = c("precision", "recall", "f1")) {
  what <- match.arg(what)
  function(p) {
    k <- sqrt(length(p)); P <- matrix(p, k, k)
    cells <- rowSums(vapply(seq_len(k), function(j) .cm_class(P, j), numeric(4)))
    g <- switch(what, precision = metric_ppv, recall = metric_sensitivity, f1 = metric_f1)
    g(cells)
  }
}

metric_macro <- function(what = c("precision", "recall", "f1")) {
  what <- match.arg(what)
  function(p) {
    k <- sqrt(length(p)); P <- matrix(p, k, k)
    g <- switch(what, precision = metric_ppv, recall = metric_sensitivity, f1 = metric_f1)
    mean(vapply(seq_len(k), function(j) g(.cm_class(P, j)), numeric(1)))
  }
}

#' Variance and sample size for a multiclass averaged metric
#'
#' @param P k x k joint probability matrix, rows = truth, columns = prediction
#' @param what "precision", "recall" or "f1"
#' @param average "micro" or "macro"
metric_variance_multiclass <- function(P, what = "f1", average = c("micro", "macro")) {
  average <- match.arg(average)
  P <- P / sum(P)
  g <- if (average == "micro") metric_micro(what) else metric_macro(what)
  list(estimate = g(as.vector(P)), V = delta_var(as.vector(P), g),
       method = paste0("delta (numeric gradient), ", average, "-", what))
}

# -----------------------------------------------------------------------------
# (c) Count-rate metrics: false positives per image
#
#     Not a function of a confusion matrix at all, which is why Metrics Reloaded
#     excludes it from that pool. It is a rate, and the textbook Poisson variance
#     has simply never been applied to it - the literature uses bootstrap.
#     Real detection data are almost always overdispersed, so the negative
#     binomial form should be the default and the Poisson form the optimistic
#     bound.
# -----------------------------------------------------------------------------

#' Per-unit variance of a count rate
#'
#' @param lambda expected false positives per image
#' @param theta negative-binomial dispersion; Inf gives the Poisson case
var_rate <- function(lambda, theta = Inf) {
  if (lambda <= 0) stop("lambda must be positive")
  if (is.infinite(theta)) lambda else lambda + lambda^2 / theta
}

#' Sample size for a count rate, on the log scale
#'
#' Working on the log scale keeps the interval positive and is the usual
#' convention for rate ratios.
n_for_power_rate <- function(lambda, ratio, theta = Inf, alpha = 0.05,
                             power = 0.80, two_sided = TRUE, deff = 1) {
  if (ratio <= 0 || ratio == 1) stop("ratio must be positive and not 1")
  V_log <- var_rate(lambda, theta) / lambda^2      # delta on log(lambda)
  n_for_power(V_alt = V_log, delta = abs(log(ratio)),
              alpha = alpha, power = power, two_sided = two_sided, deff = deff)
}

# -----------------------------------------------------------------------------
# (d) Per-case mean metrics
#
#     One registry covering everything computed as an average over cases:
#     Brier, NLL, MAE, MSE, MRR, ROUGE, BERTScore, METEOR, the readability
#     indices, the edit distances, SNR, PSNR, CNR, SSIM, and eventually the
#     per-case segmentation metrics.
#
#     transform is applied to the mean, with the delta method supplying the
#     variance: "identity", "sqrt" (root Brier, RMSE) or "exp" (perplexity).
# -----------------------------------------------------------------------------

PERCASE_METRICS <- c(
  "brier", "root_brier", "nll", "cross_entropy", "perplexity",
  "mae", "mse", "rmse", "mrr",
  "rouge_1", "rouge_2", "rouge_l", "rouge_w", "rouge_s", "bertscore", "meteor",
  "flesch", "flesch_kincaid", "smog", "gunning_fog", "ari", "forcast",
  "levenshtein", "hamming", "jaro", "lcs",
  "snr", "psnr", "cnr", "ssim",
  "dice_percase", "iou_percase", "nsd_percase", "assd_percase", "hd95_percase")

#' Per-unit variance of a transformed per-case mean
#'
#' @param x per-case scores
#' @param transform "identity", "sqrt" or "exp"
var_percase_transformed <- function(x, transform = c("identity", "sqrt", "exp")) {
  transform <- match.arg(transform)
  mu <- mean(x); V <- stats::var(x)
  est <- switch(transform, identity = mu, sqrt = sqrt(mu), exp = exp(mu))
  dg  <- switch(transform, identity = 1,
                sqrt = { if (mu <= 0) stop("sqrt transform needs a positive mean"); 1 / (2 * sqrt(mu)) },
                exp  = exp(mu))
  list(estimate = est, V = V * dg^2, mean = mu, V_mean = V,
       skew = mean(((x - mu) / stats::sd(x))^3), transform = transform)
}

#' Sample size for a per-case mean metric
#'
#' The variance here is exact for any distribution, so the returned n is not
#' limited by the delta method. What limits it is the normal approximation used
#' by the Wald test, which degrades under skew. skew_warn triggers a warning
#' when the per-case scores are skewed enough that the interval will sit
#' systematically on one side of the truth at this n.
n_for_power_percase <- function(x, delta, transform = "identity", alpha = 0.05,
                                power = 0.80, two_sided = TRUE, deff = 1,
                                skew_warn = 1.0) {
  r <- var_percase_transformed(x, transform)
  n <- n_for_power(V_alt = r$V, delta = delta, alpha = alpha, power = power,
                   two_sided = two_sided, deff = deff)
  if (abs(r$skew) > skew_warn) {
    warning(sprintf(
      paste0("per-case scores have skew %.2f. The variance is exact, but Wald ",
             "coverage is below nominal at small n and the misses are one-sided. ",
             "Run sensitivity_by_n_percase() before committing to n = %d."),
      r$skew, n), call. = FALSE)
  }
  c(r, list(n = n, delta = delta))
}

# -----------------------------------------------------------------------------
# (e) Two further families, included to show the engine is not confined to
#     confusion matrices
# -----------------------------------------------------------------------------

#' Shannon entropy of a discrete distribution, through the same engine
#'
#' H = -sum p log p is a smooth function of a multinomial, so delta_var()
#' applies unchanged and reproduces Basharin's (1959) classical variance. The
#' plug-in estimator is biased downwards by roughly (k-1)/(2n); Miller-Madow
#' corrects the point estimate but not the variance.
metric_entropy <- function(p) -sum(p[p > 0] * log(p[p > 0]))
grad_entropy   <- function(p) ifelse(p > 0, -(log(pmax(p, 1e-300)) + 1), 0)

#' Entropy variance, with a degeneracy guard
#'
#' Entropy is degenerate at the uniform distribution. There the gradient is
#' constant across cells, the multinomial covariance annihilates constant
#' vectors, and the delta variance is exactly zero - the estimator converges at
#' rate 1/n with a chi-square limit rather than at root-n with a normal one.
#' This is the same non-regular failure as mutual information under
#' independence and the calibration errors under perfect calibration, so the
#' standard sample-size formula is invalid in a neighbourhood of uniformity.
entropy_variance <- function(p, degenerate_tol = 1e-3) {
  p <- p / sum(p)
  V <- delta_var(p, metric_entropy, grad = grad_entropy)
  k <- sum(p > 0)
  degenerate <- V < degenerate_tol
  if (degenerate) {
    warning(sprintf(
      paste0("entropy is at or near its maximum (V = %.2e). The gradient is ",
             "constant there, so the delta variance vanishes and the limit law ",
             "is chi-square, not normal. Sample size from V is invalid in this ",
             "neighbourhood."), V), call. = FALSE)
  }
  list(estimate = metric_entropy(p), V = V,
       V_basharin = sum(p * log(p)^2) - (sum(p * log(p)))^2,
       bias_plugin = -(k - 1) / 2,        # divide by n for the bias itself
       degenerate = degenerate)
}

#' Coefficient of variation
#'
#' A ratio in which both parts are random. Under normality the per-unit
#' variance of the sample CV is cv^2 (0.5 + cv^2), which is the delta method
#' applied jointly to the mean and the second moment.
cv_variance <- function(x) {
  mu <- mean(x); s <- stats::sd(x); cv <- s / mu
  list(estimate = cv, V = cv^2 * (0.5 + cv^2),
       note = "normal-theory result; check it against sensitivity_by_n_percase() on a skewed sample")
}
