# =============================================================================
#  variance_l3_all.R
#
#  Sample size given power, for every L3 metric in the inventory.
#
#  Five estimator families, one dispatcher. The families are not cosmetic: the
#  variance has a different form in each, and the small-sample behaviour differs
#  too, so the family a metric belongs to determines both the formula and the
#  caveat.
#
#    A  confusion matrix     V = grad' Sigma grad on one multinomial
#    B  multiclass           the same, on a k x k table vectorised to k^2 cells
#    C  count rate           Poisson or negative binomial
#    D  per-case mean        V = Var(x_i), exact for any distribution
#    E  other structures     U-statistics, variance components, two-group means
#
#  Entry point:  n_for_power_l3(metric, ..., delta, power)
#  Manifest:     l3_manifest()   lists all 47 with family and status
#
#  Requires variance_core.R and variance_l3.R.
# =============================================================================

# =============================================================================
# FAMILY A - additional confusion-matrix metrics
# =============================================================================

#' Volumetric similarity
#'
#' VS = 1 - |fp - fn| / (2tp + fp + fn). A confusion-matrix function, so the
#' engine applies - but it is NOT differentiable at fp = fn, which is exactly
#' where it attains its maximum. The delta variance vanishes there and the
#' sample-size formula is invalid in that neighbourhood, the same failure class
#' as entropy at the uniform distribution.
metric_volumetric_similarity <- function(p) {
  D <- 2 * p[1] + p[2] + p[3]
  unname(1 - abs(p[2] - p[3]) / D)
}

grad_volumetric_similarity <- function(p) {
  D <- 2 * p[1] + p[2] + p[3]
  N <- p[2] - p[3]
  s <- sign(N)
  if (s == 0) return(c(0, 0, 0, 0))     # stationary point; guarded downstream
  unname(s / D^2 * c(2 * N, -(D - N), (D + N), 0))
}

#' Intersection over reference, i.e. pixel-level sensitivity
#'
#' Algebraically identical to sensitivity. Kept as its own entry because the
#' analysis unit is a pixel, so the variance is only valid after a clustering
#' correction: pixels within an image are not independent.
metric_ior <- function(p) metric_sensitivity(p)
grad_ior   <- function(p) grad_sensitivity(p)

L3_CM_EXTRA <- list(
  volumetric_similarity = list(fun = metric_volumetric_similarity,
                               grad = grad_volumetric_similarity),
  ior                   = list(fun = metric_ior, grad = grad_ior)
)

register_l3_cm_extra <- function(envir = NULL) {
  if (is.null(envir)) envir <- .registry_env()
  reg <- get("METRIC_REGISTRY", envir = envir)
  for (nm in names(L3_CM_EXTRA)) {
    e <- L3_CM_EXTRA[[nm]]
    reg[[nm]] <- list(id = NA, family = "confusion_matrix", aggregation = "pooled",
                      fun  = local({ f <- e$fun;  function(...) f }),
                      grad = local({ g <- e$grad; function(...) g }))
  }
  assign("METRIC_REGISTRY", reg, envir = envir)
  invisible(names(L3_CM_EXTRA))
}

# =============================================================================
# FAMILY D - named per-case metrics
#
#   Var(mean) = sigma^2 / n holds exactly for any distribution, so the variance
#   needs no approximation. The caveat is coverage, not variance: a Wald
#   interval under-covers when the per-case scores are skewed, and the direction
#   of the skew sets the direction of the failure.
# =============================================================================

PERCASE_REGISTRY <- list(
  brier            = list(transform = "identity", note = "(phat - y)^2 per case"),
  root_brier       = list(transform = "sqrt",     note = "sqrt of the mean Brier"),
  nll              = list(transform = "identity", note = "-log phat(y) per case"),
  cross_entropy    = list(transform = "identity", note = "same as nll"),
  perplexity       = list(transform = "exp",      note = "exp of the mean NLL"),
  mae              = list(transform = "identity", note = "|yhat - y|"),
  mse              = list(transform = "identity", note = "(yhat - y)^2; heavily right-skewed"),
  rmse             = list(transform = "sqrt",     note = "sqrt of the mean MSE"),
  mrr              = list(transform = "identity", note = "1 / rank"),
  mape             = list(transform = "identity", note = "|yhat-y|/|y|; undefined at y=0, heavy tails"),
  smape            = list(transform = "identity", note = "symmetric MAPE; bounded but heavy-tailed"),
  mase             = list(transform = "identity", note = "scaled by a benchmark whose value is also random"),
  rouge_n          = list(transform = "identity", note = "per-example ROUGE-N"),
  rouge_1          = list(transform = "identity", note = "per-example ROUGE-1"),
  rouge_2          = list(transform = "identity", note = "per-example ROUGE-2"),
  rouge_3          = list(transform = "identity", note = "per-example ROUGE-3"),
  rouge_l          = list(transform = "identity", note = "per-example ROUGE-L"),
  rouge_w          = list(transform = "identity", note = "per-example ROUGE-W"),
  rouge_s          = list(transform = "identity", note = "per-example ROUGE-S"),
  rouge_su         = list(transform = "identity", note = "per-example ROUGE-SU"),
  bertscore        = list(transform = "identity", note = "per-example embedding similarity"),
  meteor           = list(transform = "identity", note = "per-example METEOR"),
  flesch           = list(transform = "identity", note = "per-document Flesch reading ease"),
  flesch_kincaid   = list(transform = "identity", note = "per-document grade level"),
  gunning_fog      = list(transform = "identity", note = "per-document Gunning Fog"),
  smog             = list(transform = "identity", note = "per-document SMOG"),
  ari              = list(transform = "identity", note = "per-document ARI"),
  forcast          = list(transform = "identity", note = "per-document FORCAST"),
  levenshtein      = list(transform = "identity", note = "per-pair edit distance"),
  damerau          = list(transform = "identity", note = "per-pair Damerau-Levenshtein"),
  hamming          = list(transform = "identity", note = "per-pair Hamming"),
  jaro             = list(transform = "identity", note = "per-pair Jaro"),
  lcs              = list(transform = "identity", note = "per-pair longest common subsequence"),
  snr              = list(transform = "identity", note = "per-image signal-to-noise"),
  psnr             = list(transform = "identity", note = "per-image peak SNR"),
  cnr              = list(transform = "identity", note = "per-image contrast-to-noise"),
  ssim             = list(transform = "identity", note = "per-image structural similarity"),
  ensemble_mse     = list(transform = "identity", note = "correlated across models; not an i.i.d. mean"),
  macro_mean_prob  = list(transform = "identity", note = "correlated across classes"),
  dice_percase     = list(transform = "identity", note = "per-image Dice; LEFT-skewed, so intervals sit above the truth"),
  iou_percase      = list(transform = "identity", note = "per-image IoU; left-skewed"),
  nsd_percase      = list(transform = "identity", note = "per-image NSD"),
  assd_percase     = list(transform = "identity", note = "per-image ASSD; heavy-tailed"),
  hd95_percase     = list(transform = "identity", note = "per-image HD95; heavy-tailed, near-extreme-value")
)

#' Sample size for a named per-case metric
#'
#' @param name key in PERCASE_REGISTRY
#' @param x observed per-case scores (pilot data)
#' @param delta difference to detect, on the metric's own scale
n_for_power_percase_named <- function(name, x, delta, alpha = 0.05,
                                      power = 0.80, two_sided = TRUE, deff = 1,
                                      skew_warn = 1.0) {
  e <- PERCASE_REGISTRY[[name]]
  if (is.null(e)) {
    stop("unknown per-case metric '", name, "'. Available: ",
         paste(names(PERCASE_REGISTRY), collapse = ", "), call. = FALSE)
  }
  r <- n_for_power_percase(x, delta = delta, transform = e$transform,
                           alpha = alpha, power = power,
                           two_sided = two_sided, deff = deff,
                           skew_warn = skew_warn)
  c(list(metric = name, note = e$note), r)
}

# =============================================================================
# FAMILY E - other estimator structures
# =============================================================================

# --- E1. Conditional entropy -------------------------------------------------

#' H(Y | X) = H(X,Y) - H(X), a smooth function of the k x k joint table
metric_conditional_entropy <- function(p, k = NULL) {
  if (is.null(k)) k <- sqrt(length(p))
  P <- matrix(p, k, k); P <- P / sum(P)
  pi_ <- rowSums(P)
  ok <- P > 0
  -sum(P[ok] * log(P[ok] / pi_[row(P)[ok]]))
}

grad_conditional_entropy <- function(p, k = NULL) {
  if (is.null(k)) k <- sqrt(length(p))
  P <- matrix(p, k, k); P <- P / sum(P)
  pi_ <- rowSums(P)
  G <- -log(pmax(P, 1e-300) / pi_[row(P)])
  G[P <= 0] <- 0
  as.vector(G)
}

conditional_entropy_variance <- function(P) {
  P <- P / sum(P); k <- nrow(P)
  g  <- function(q) metric_conditional_entropy(q, k)
  gr <- function(q) grad_conditional_entropy(q, k)
  list(estimate = g(as.vector(P)),
       V = delta_var(as.vector(P), g, grad = gr))
}

# --- E2. Degree-two U-statistics: Kendall's tau, Rand, Fowlkes-Mallows -------

#' Per-unit variance of a degree-two U-statistic via its Hoeffding projection
#'
#' For U = mean over pairs of h(Z_i, Z_j), sqrt(n)(U - theta) -> N(0, 4*zeta1)
#' with zeta1 = Var(E[h(Z,Z')|Z]). The projection is estimated by the leave-one-
#' out average h1(Z_i) = mean_{j != i} h(Z_i, Z_j).
#'
#' Pair counts are NOT a multinomial - that is the tempting shortcut and it is
#' wrong, because pairs sharing an item are dependent.
#'
#' @param h1 the n leave-one-out projections
ustat_variance <- function(h1) 4 * stats::var(h1)

#' Kendall's tau with its asymptotic variance
#'
#' Under independence zeta1 = 1/9 so V = 4/9, which the test suite checks.
kendall_tau_variance <- function(x, y) {
  n <- length(x)
  if (n != length(y)) stop("x and y must have equal length")
  h1 <- vapply(seq_len(n), function(i)
    mean(sign((x[i] - x[-i]) * (y[i] - y[-i]))), numeric(1))
  list(estimate = stats::cor(x, y, method = "kendall"),
       V = ustat_variance(h1), zeta1 = stats::var(h1))
}

#' Rand index and Fowlkes-Mallows index, both degree-two U-statistics
#'
#' @param a,b cluster or class labels for the same n items
cluster_index_variance <- function(a, b, index = c("rand", "fowlkes_mallows")) {
  index <- match.arg(index)
  n <- length(a)
  sa <- outer(a, a, "==") ; sb <- outer(b, b, "==")
  diag(sa) <- NA; diag(sb) <- NA
  if (index == "rand") {
    H  <- (sa == sb) * 1
    est <- mean(H, na.rm = TRUE)
    h1 <- rowMeans(H, na.rm = TRUE)
  } else {
    # FM = a11 / sqrt((a11+a10)(a11+a01)); a ratio of pair counts, so the
    # projection is taken on the numerator and denominator jointly
    A <- (sa & sb) * 1; B <- (sa & !sb) * 1; C <- (!sa & sb) * 1
    n11 <- mean(A, na.rm = TRUE); n10 <- mean(B, na.rm = TRUE); n01 <- mean(C, na.rm = TRUE)
    est <- n11 / sqrt((n11 + n10) * (n11 + n01))
    d1 <- 1 / sqrt((n11 + n10) * (n11 + n01))
    dd <- -0.5 * n11 * ((n11 + n01) + (n11 + n10)) /
           ((n11 + n10) * (n11 + n01))^1.5
    h1 <- d1 * rowMeans(A, na.rm = TRUE) +
          dd * (rowMeans(A, na.rm = TRUE) + rowMeans(B, na.rm = TRUE) / 2 +
                rowMeans(C, na.rm = TRUE) / 2)
  }
  list(estimate = est, V = ustat_variance(h1), zeta1 = stats::var(h1))
}

# --- E3. Squared multiple correlation ---------------------------------------

#' R-squared
#'
#' V = 4 rho^2 (1 - rho^2)^2, which vanishes at rho = 0. That is a stationary
#' point of the metric, so the limit law there is chi-square rather than normal
#' and the sample-size formula does not apply.
r_squared_variance <- function(r2, degenerate_tol = 1e-3) {
  if (r2 < 0 || r2 > 1) stop("r2 must lie in [0, 1]")
  V <- 4 * r2 * (1 - r2)^2
  if (V < degenerate_tol) {
    warning(sprintf(paste0("R-squared is near zero (V = %.2e). The gradient ",
                           "vanishes there, so the limit is chi-square, not ",
                           "normal, and sample size from V is invalid."), V),
            call. = FALSE)
  }
  list(estimate = r2, V = V, degenerate = V < degenerate_tol)
}

# --- E4. Variance-component metrics -----------------------------------------

#' Repeatability or reproducibility coefficient
#'
#' RC = 1.96 * sqrt(2) * sigma_w. With m replicates per subject, sigma_w^2 has
#' n(m-1) degrees of freedom, Var(sigma_w^2) = 2 sigma_w^4 / df, and the delta
#' method on the square root gives Var(sigma_w) = sigma_w^2 / (2 df).
#'
#' @param sigma_w within-subject standard deviation
#' @param m replicates per subject
repeatability_variance <- function(sigma_w, replicates = 2) {
  if (replicates < 2) stop("replicates must be at least 2")
  k <- 1.96 * sqrt(2)
  list(estimate = k * sigma_w,
       V = k^2 * sigma_w^2 / (2 * (replicates - 1)))
}

# --- E5. Two-group and paired differences -----------------------------------

#' Discrimination slope: the gap in mean predicted risk between cases and controls
discrimination_slope_variance <- function(mu1, mu0, sd1, sd0, prevalence) {
  if (prevalence <= 0 || prevalence >= 1) stop("prevalence must be in (0, 1)")
  list(estimate = mu1 - mu0,
       V = sd1^2 / prevalence + sd0^2 / (1 - prevalence))
}

#' Difference of two independent proportions
#'
#' Covers demographic parity, equalized odds, equality of opportunity and the
#' unpaired form of the enhanced detection rate.
#'
#' @param kappa fraction of the sample allocated to group 1
risk_difference_variance <- function(p1, p2, kappa = 0.5) {
  list(estimate = p1 - p2,
       V = p1 * (1 - p1) / kappa + p2 * (1 - p2) / (1 - kappa))
}

#' Difference of two PAIRED proportions (McNemar structure)
#'
#' This is the right form for the enhanced detection rate as usually measured:
#' the same cases read with and without AI assistance. Only the discordant
#' cells carry information, which is why the paired design is so much more
#' efficient.
#'
#' @param p01,p10 discordant cell probabilities
paired_proportion_variance <- function(p01, p10) {
  d <- p10 - p01
  list(estimate = d, V = p01 + p10 - d^2,
       note = "only discordant pairs contribute; concordant pairs carry no information")
}

# --- E6. Metrics that are refused, with the reason --------------------------

#' Mahalanobis distance
#'
#' The squared distance is chi-square distributed under normality, not
#' asymptotically normal, so it does not belong in this framework. Returning a
#' plausible number here would be worse than refusing.
mahalanobis_sample_size <- function(...) {
  stop("Mahalanobis distance is not supported.\n",
       "  The squared distance is chi-square under normality rather than\n",
       "  asymptotically normal, so n = (z+z)^2 V / delta^2 does not apply.\n",
       "  Use a non-central chi-square power calculation instead.",
       call. = FALSE)
}

#' Relative standard error
#'
#' RSE = SE / estimate is not a quantity to power on: it IS the precision. A
#' target RSE is a precision requirement, so it maps onto n_for_precision()
#' rather than onto a hypothesis test.
n_for_target_rse <- function(V, estimate, target_rse, deff = 1) {
  if (target_rse <= 0) stop("target_rse must be positive")
  n <- V / (target_rse * estimate)^2
  list(n = ceiling(n * deff), target_rse = target_rse,
       note = "a target RSE is a precision requirement, not a hypothesis test")
}

#' Survival and time-dependent AUC variants
#'
#' Graded L3 in the inventory because published variances exist, but each needs
#' censoring machinery from a separate literature. Routed to the packages that
#' already implement them rather than approximated here.
timedep_auc_sample_size <- function(...) {
  stop("Time-dependent, integrated and multiclass AUC are not implemented here.\n",
       "  Variances are published but require censoring machinery from a\n",
       "  separate literature. Use timeROC or survivalROC for the interval, and\n",
       "  pmvalsampsize or SSAML to size a study on the c-statistic.",
       call. = FALSE)
}

# =============================================================================
# UNIFIED DISPATCHER
# =============================================================================

L3_FAMILY <- local({
  cm <- c("accuracy", "sensitivity", "specificity", "ppv", "npv",
          "balanced_accuracy", "youden", "dsc_f1", "fbeta", "iou", "mcc",
          "expected_cost", "net_benefit", "markedness", "balanced_error_rate",
          "false_discovery_rate", "false_omission_rate", "class_error",
          "false_referral_rate", "volumetric_similarity", "ior")
  list(
    confusion_matrix = cm,
    multiclass = c("micro_precision", "micro_recall", "micro_f1",
                   "macro_precision", "macro_recall", "macro_f1"),
    rate = c("fppi"),
    percase = names(PERCASE_REGISTRY),
    other = c("weighted_kappa", "entropy", "conditional_entropy",
              "coefficient_of_variation", "kendall_tau", "r_squared",
              "rand_index", "fowlkes_mallows", "repeatability_coefficient",
              "reproducibility_coefficient", "discrimination_slope",
              "risk_difference", "paired_proportion", "enhanced_detection_rate"),
    refused = c("mahalanobis", "relative_standard_error", "timedep_auc")
  )
})

#' Which family does a metric belong to?
l3_family_of <- function(metric) {
  for (f in names(L3_FAMILY)) if (metric %in% L3_FAMILY[[f]]) return(f)
  NA_character_
}

#' Sample size given power, for any L3 metric
#'
#' A single entry point over five estimator families. The arguments differ by
#' family because the information a variance needs differs by family, and
#' pretending otherwise would hide the distinction that matters most.
#'
#'   confusion_matrix : p (cell probabilities), delta
#'   multiclass       : P (k x k joint table), delta
#'   rate             : lambda, ratio, theta
#'   percase          : x (pilot per-case scores), delta
#'   other            : arguments specific to the estimator, see the functions
#'
#' @return list with n and the ingredients that produced it
n_for_power_l3 <- function(metric, ..., delta = NULL, alpha = 0.05,
                           power = 0.80, two_sided = TRUE, deff = 1) {
  # R partially matches arguments that appear BEFORE `...`, so an argument
  # named m, me or met would be silently captured by `metric` and the real
  # metric name lost. Catch that here rather than reporting it as an unknown
  # metric several frames later.
  if (!is.character(metric) || length(metric) != 1) {
    stop("`metric` must be a single metric name. If you passed an argument ",
         "whose name begins with 'm', R partially matched it to `metric`; ",
         "spell that argument out in full.", call. = FALSE)
  }
  fam <- l3_family_of(metric)
  if (is.na(fam)) {
    stop("unknown L3 metric '", metric, "'. See l3_manifest() for the list.",
         call. = FALSE)
  }
  args <- list(...)

  if (fam == "refused") {
    switch(metric,
      mahalanobis = mahalanobis_sample_size(),
      relative_standard_error = stop(
        "Relative standard error is a precision requirement, not a testable ",
        "quantity. Use n_for_target_rse(V, estimate, target_rse).", call. = FALSE),
      timedep_auc = timedep_auc_sample_size())
  }

  if (fam == "confusion_matrix") {
    p <- args$p
    if (is.null(p)) stop("family 'confusion_matrix' needs p = cell_prob(...)", call. = FALSE)
    extra <- args[setdiff(names(args), "p")]
    r <- do.call(n_for_power_metric,
                 c(list(p, metric, delta = delta, alpha = alpha, power = power,
                        two_sided = two_sided, deff = deff), extra))
    return(c(list(metric = metric, family = fam), r))
  }

  if (fam == "multiclass") {
    P <- args$P
    if (is.null(P)) stop("family 'multiclass' needs P = <k x k joint table>", call. = FALSE)
    parts <- strsplit(metric, "_")[[1]]
    r <- metric_variance_multiclass(P, what = parts[2], average = parts[1])
    return(list(metric = metric, family = fam, estimate = r$estimate, V = r$V,
                n = n_for_power(r$V, delta = delta, alpha = alpha, power = power,
                                two_sided = two_sided, deff = deff)))
  }

  if (fam == "rate") {
    lam <- args$lambda; ratio <- args$ratio
    th <- if (is.null(args$theta)) Inf else args$theta
    if (is.null(lam) || is.null(ratio))
      stop("family 'rate' needs lambda = <FP per image> and ratio = <rate ratio>",
           call. = FALSE)
    return(list(metric = metric, family = fam, estimate = lam,
                V = var_rate(lam, th) / lam^2,
                n = n_for_power_rate(lam, ratio, theta = th, alpha = alpha,
                                     power = power, two_sided = two_sided,
                                     deff = deff)))
  }

  if (fam == "percase") {
    x <- args$x
    if (is.null(x)) stop("family 'percase' needs x = <pilot per-case scores>", call. = FALSE)
    r <- n_for_power_percase_named(metric, x, delta = delta, alpha = alpha,
                                   power = power, two_sided = two_sided,
                                   deff = deff)
    return(c(list(family = fam), r))
  }

  # ---- other ----
  r <- switch(metric,
    weighted_kappa = {
      P <- args$P; W <- if (is.null(args$W)) weights_linear(nrow(P)) else args$W
      list(estimate = kappa_w(P, W), V = kappa_w_var_fce(P, W)) },
    entropy = { e <- suppressWarnings(entropy_variance(args$p)); e },
    conditional_entropy = conditional_entropy_variance(args$P),
    coefficient_of_variation = cv_variance(args$x),
    kendall_tau = kendall_tau_variance(args$x, args$y),
    r_squared = suppressWarnings(r_squared_variance(args$r2)),
    rand_index = cluster_index_variance(args$a, args$b, "rand"),
    fowlkes_mallows = cluster_index_variance(args$a, args$b, "fowlkes_mallows"),
    repeatability_coefficient = repeatability_variance(
      args$sigma_w, args$replicates %||% 2),
    reproducibility_coefficient = repeatability_variance(
      args$sigma_w, args$replicates %||% 2),
    discrimination_slope = discrimination_slope_variance(
      args$mu1, args$mu0, args$sd1, args$sd0, args$prevalence),
    risk_difference = risk_difference_variance(args$p1, args$p2, args$kappa %||% 0.5),
    paired_proportion = paired_proportion_variance(args$p01, args$p10),
    enhanced_detection_rate = paired_proportion_variance(args$p01, args$p10))

  list(metric = metric, family = fam, estimate = r$estimate, V = r$V,
       n = n_for_power(r$V, delta = delta, alpha = alpha, power = power,
                       two_sided = two_sided, deff = deff),
       degenerate = isTRUE(r$degenerate))
}

`%||%` <- function(a, b) if (is.null(a)) b else a

#' The full L3 manifest: every metric, its family, and how to call it
l3_manifest <- function(print = TRUE) {
  rows <- do.call(rbind, lapply(names(L3_FAMILY), function(f)
    data.frame(family = f, metric = L3_FAMILY[[f]], stringsAsFactors = FALSE)))
  needs <- c(confusion_matrix = "p, delta", multiclass = "P, delta",
             rate = "lambda, ratio, theta", percase = "x, delta",
             other = "estimator-specific", refused = "n/a - refused with reason")
  rows$arguments <- needs[rows$family]
  if (print) {
    for (f in names(L3_FAMILY)) {
      cat(sprintf("\n%s  [%d]   arguments: %s\n", toupper(f),
                  length(L3_FAMILY[[f]]), needs[f]))
      cat(paste0("   ", paste(L3_FAMILY[[f]], collapse = ", ")), "\n")
    }
    cat(sprintf("\nTotal: %d metrics across %d families\n",
                nrow(rows), length(L3_FAMILY)))
  }
  invisible(rows)
}
