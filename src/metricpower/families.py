"""
Group B estimator families beyond the binary confusion matrix:

  multiclass   k x k multinomial, micro- and macro-averaged precision/recall/F1
  rate         count rates (false positives per image), Poisson / negative binomial
  percase      means of one score per case (Brier, per-case Dice, ROUGE, ...)
  other        entropy, conditional entropy, CV, U-statistics, R^2, repeatability,
               discrimination slope, risk difference, paired proportions

Port of R/02_l3.R and R/03_l3_all.R.
"""
from __future__ import annotations

import math
import warnings
from typing import Callable, Dict, Optional

import numpy as np
from scipy.stats import kendalltau

from .core import (delta_var, metric_f1, metric_ppv, metric_sensitivity, n_for_power,
                   var_percase, var_sqrt_transform)


# -----------------------------------------------------------------------------
# (b) multiclass micro / macro
# -----------------------------------------------------------------------------

def _cm_class(P: np.ndarray, j: int) -> np.ndarray:
    tp = P[j, j]; fp = P[:, j].sum() - tp; fn = P[j, :].sum() - tp
    return np.array([tp, fp, fn, 1 - tp - fp - fn])


_WHAT = {"precision": metric_ppv, "recall": metric_sensitivity, "f1": metric_f1}


def metric_micro(what: str = "f1") -> Callable:
    g = _WHAT[what]
    def f(p):
        p = np.asarray(p, dtype=float); k = int(round(math.sqrt(len(p)))); P = p.reshape(k, k)
        cells = sum(_cm_class(P, j) for j in range(k))
        return g(cells)
    return f


def metric_macro(what: str = "f1") -> Callable:
    g = _WHAT[what]
    def f(p):
        p = np.asarray(p, dtype=float); k = int(round(math.sqrt(len(p)))); P = p.reshape(k, k)
        return float(np.mean([g(_cm_class(P, j)) for j in range(k)]))
    return f


def metric_variance_multiclass(P, what: str = "f1", average: str = "micro") -> dict:
    """P: k x k joint probability matrix, rows = truth, columns = prediction."""
    P = np.asarray(P, dtype=float); P = P / P.sum()
    g = metric_micro(what) if average == "micro" else metric_macro(what)
    return dict(estimate=float(g(P.ravel())), V=delta_var(P.ravel(), g),
                method=f"delta (numeric gradient), {average}-{what}")


# -----------------------------------------------------------------------------
# (c) count rates
# -----------------------------------------------------------------------------

def var_rate(lam: float, theta: float = math.inf) -> float:
    if lam <= 0:
        raise ValueError("lambda must be positive")
    return lam if math.isinf(theta) else lam + lam ** 2 / theta


def n_for_power_rate(lam: float, ratio: float, theta: float = math.inf, alpha: float = 0.05,
                     power: float = 0.80, two_sided: bool = True, deff: float = 1.0) -> int:
    """Sample size (images) to detect a rate ratio, on the log scale."""
    if ratio <= 0 or ratio == 1:
        raise ValueError("ratio must be positive and not 1")
    V_log = var_rate(lam, theta) / lam ** 2
    return n_for_power(V_log, abs(math.log(ratio)), alpha=alpha, power=power,
                       two_sided=two_sided, deff=deff)


# -----------------------------------------------------------------------------
# (d) per-case means
# -----------------------------------------------------------------------------

PERCASE_REGISTRY: Dict[str, dict] = {
    "brier":          dict(transform="identity", note="(phat - y)^2 per case"),
    "root_brier":     dict(transform="sqrt",     note="sqrt of the mean Brier"),
    "nll":            dict(transform="identity", note="-log phat(y) per case"),
    "cross_entropy":  dict(transform="identity", note="same as nll"),
    "perplexity":     dict(transform="exp",      note="exp of the mean NLL"),
    "mae":            dict(transform="identity", note="|yhat - y|"),
    "mse":            dict(transform="identity", note="(yhat - y)^2; heavily right-skewed"),
    "rmse":           dict(transform="sqrt",     note="sqrt of the mean MSE"),
    "mrr":            dict(transform="identity", note="1 / rank"),
    "mape":           dict(transform="identity", note="|yhat-y|/|y|; undefined at y=0, heavy tails"),
    "smape":          dict(transform="identity", note="symmetric MAPE; bounded but heavy-tailed"),
    "mase":           dict(transform="identity", note="scaled by a benchmark whose value is also random"),
    "rouge_n":        dict(transform="identity", note="per-example ROUGE-N"),
    "rouge_1":        dict(transform="identity", note="per-example ROUGE-1"),
    "rouge_2":        dict(transform="identity", note="per-example ROUGE-2"),
    "rouge_3":        dict(transform="identity", note="per-example ROUGE-3"),
    "rouge_l":        dict(transform="identity", note="per-example ROUGE-L"),
    "rouge_w":        dict(transform="identity", note="per-example ROUGE-W"),
    "rouge_s":        dict(transform="identity", note="per-example ROUGE-S"),
    "rouge_su":       dict(transform="identity", note="per-example ROUGE-SU"),
    "bertscore":      dict(transform="identity", note="per-example embedding similarity"),
    "meteor":         dict(transform="identity", note="per-example METEOR"),
    "flesch":         dict(transform="identity", note="per-document Flesch reading ease"),
    "flesch_kincaid": dict(transform="identity", note="per-document grade level"),
    "gunning_fog":    dict(transform="identity", note="per-document Gunning Fog"),
    "smog":           dict(transform="identity", note="per-document SMOG"),
    "ari":            dict(transform="identity", note="per-document ARI"),
    "forcast":        dict(transform="identity", note="per-document FORCAST"),
    "levenshtein":    dict(transform="identity", note="per-pair edit distance"),
    "damerau":        dict(transform="identity", note="per-pair Damerau-Levenshtein"),
    "hamming":        dict(transform="identity", note="per-pair Hamming"),
    "jaro":           dict(transform="identity", note="per-pair Jaro"),
    "lcs":            dict(transform="identity", note="per-pair longest common subsequence"),
    "snr":            dict(transform="identity", note="per-image signal-to-noise"),
    "psnr":           dict(transform="identity", note="per-image peak SNR"),
    "cnr":            dict(transform="identity", note="per-image contrast-to-noise"),
    "ssim":           dict(transform="identity", note="per-image structural similarity"),
    "ensemble_mse":   dict(transform="identity", note="correlated across models; not an i.i.d. mean"),
    "macro_mean_prob": dict(transform="identity", note="correlated across classes"),
    "dice_percase":   dict(transform="identity", note="per-image Dice; LEFT-skewed, so intervals sit above the truth"),
    "iou_percase":    dict(transform="identity", note="per-image IoU; left-skewed"),
    "nsd_percase":    dict(transform="identity", note="per-image NSD"),
    "assd_percase":   dict(transform="identity", note="per-image ASSD; heavy-tailed"),
    "hd95_percase":   dict(transform="identity", note="per-image HD95; heavy-tailed, near-extreme-value"),
}


def var_percase_transformed(x=None, transform: str = "identity", mean: Optional[float] = None,
                            sd: Optional[float] = None) -> dict:
    """Per-unit variance of a (transformed) per-case mean.

    Supply either the pilot scores x, or (mean, sd) when only summaries are known.
    """
    if x is not None:
        x = np.asarray(x, dtype=float); mu = float(x.mean()); V = var_percase(x)
        skew = float(np.mean(((x - mu) / x.std(ddof=1)) ** 3))
    else:
        if mean is None or sd is None:
            raise ValueError("supply x, or mean and sd")
        mu, V, skew = float(mean), float(sd) ** 2, float("nan")
    if transform == "identity":
        est, dg = mu, 1.0
    elif transform == "sqrt":
        if mu <= 0:
            raise ValueError("sqrt transform needs a positive mean")
        est, dg = math.sqrt(mu), 1 / (2 * math.sqrt(mu))
    elif transform == "exp":
        est, dg = math.exp(mu), math.exp(mu)
    else:
        raise ValueError("transform must be identity, sqrt or exp")
    return dict(estimate=est, V=V * dg ** 2, mean=mu, V_mean=V, skew=skew, transform=transform, dg=dg)


def n_for_power_percase(x=None, delta: float = None, transform: str = "identity", alpha: float = 0.05,
                        power: float = 0.80, two_sided: bool = True, deff: float = 1.0,
                        skew_warn: float = 1.0, mean: Optional[float] = None,
                        sd: Optional[float] = None) -> dict:
    r = var_percase_transformed(x, transform, mean=mean, sd=sd)
    n = n_for_power(r["V"], delta, alpha=alpha, power=power, two_sided=two_sided, deff=deff)
    if not math.isnan(r["skew"]) and abs(r["skew"]) > skew_warn:
        warnings.warn(
            f"per-case scores have skew {r['skew']:.2f}. The variance is exact, but Wald coverage is "
            f"below nominal at small n and the misses are one-sided. Run sensitivity_by_n_percase() "
            f"before committing to n = {n}.", stacklevel=2)
    r.update(n=n, delta=delta)
    return r


# -----------------------------------------------------------------------------
# (e) other estimator structures
# -----------------------------------------------------------------------------

def metric_entropy(p):
    p = np.asarray(p, dtype=float); q = p[p > 0]
    return float(-(q * np.log(q)).sum())


def grad_entropy(p):
    p = np.asarray(p, dtype=float)
    return np.where(p > 0, -(np.log(np.maximum(p, 1e-300)) + 1), 0.0)


def entropy_variance(p, degenerate_tol: float = 1e-3) -> dict:
    p = np.asarray(p, dtype=float); p = p / p.sum()
    V = delta_var(p, metric_entropy, grad=grad_entropy)
    k = int((p > 0).sum())
    degenerate = V < degenerate_tol
    if degenerate:
        warnings.warn(f"entropy is at or near its maximum (V = {V:.2e}). The gradient is constant there, "
                      "so the delta variance vanishes and the limit law is chi-square, not normal. "
                      "Sample size from V is invalid in this neighbourhood.", stacklevel=2)
    q = p[p > 0]
    return dict(estimate=metric_entropy(p), V=V,
                V_basharin=float((q * np.log(q) ** 2).sum() - (q * np.log(q)).sum() ** 2),
                bias_plugin=-(k - 1) / 2, degenerate=degenerate)


def metric_conditional_entropy(p, k: Optional[int] = None) -> float:
    p = np.asarray(p, dtype=float)
    if k is None:
        k = int(round(math.sqrt(len(p))))
    P = p.reshape(k, k); P = P / P.sum(); pi_ = P.sum(1)
    ok = P > 0
    rows = np.repeat(pi_[:, None], k, axis=1)
    return float(-(P[ok] * np.log(P[ok] / rows[ok])).sum())


def grad_conditional_entropy(p, k: Optional[int] = None):
    p = np.asarray(p, dtype=float)
    if k is None:
        k = int(round(math.sqrt(len(p))))
    P = p.reshape(k, k); P = P / P.sum(); pi_ = P.sum(1)
    rows = np.repeat(pi_[:, None], k, axis=1)
    G = -np.log(np.maximum(P, 1e-300) / rows)
    G[P <= 0] = 0
    return G.ravel()


def conditional_entropy_variance(P) -> dict:
    P = np.asarray(P, dtype=float); P = P / P.sum(); k = P.shape[0]
    g = lambda q: metric_conditional_entropy(q, k); gr = lambda q: grad_conditional_entropy(q, k)
    return dict(estimate=g(P.ravel()), V=delta_var(P.ravel(), g, grad=gr))


def cv_variance(x=None, cv: Optional[float] = None) -> dict:
    """Var(cv_hat) * n = cv^2 (0.5 + cv^2) under normality."""
    if x is not None:
        x = np.asarray(x, dtype=float); cv = float(x.std(ddof=1) / x.mean())
    if cv is None:
        raise ValueError("supply x or cv")
    return dict(estimate=cv, V=cv ** 2 * (0.5 + cv ** 2),
                note="normal-theory result; check it against sensitivity_by_n_percase() on a skewed sample")


def ustat_variance(h1) -> float:
    return 4 * float(np.var(np.asarray(h1, dtype=float), ddof=1))


def kendall_tau_variance(x, y) -> dict:
    """Kendall's tau with its Hoeffding-projection variance (O(n) per projection via outer products)."""
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    n = len(x)
    if n != len(y):
        raise ValueError("x and y must have equal length")
    S = np.sign(np.subtract.outer(x, x)) * np.sign(np.subtract.outer(y, y))
    np.fill_diagonal(S, 0.0)
    h1 = S.sum(1) / (n - 1)
    return dict(estimate=float(kendalltau(x, y).statistic), V=ustat_variance(h1),
                zeta1=float(np.var(h1, ddof=1)))


def cluster_index_variance(a, b, index: str = "rand") -> dict:
    """Rand or Fowlkes-Mallows index as a degree-2 U-statistic; O(n) via the contingency table."""
    a = np.asarray(a); b = np.asarray(b); n = len(a)
    if n != len(b):
        raise ValueError("a and b must have equal length")
    ua, ia = np.unique(a, return_inverse=True); ub, ib = np.unique(b, return_inverse=True)
    N = np.zeros((len(ua), len(ub))); np.add.at(N, (ia, ib), 1)
    ra, cb = N.sum(1), N.sum(0)
    # per item: number of other items in same a-cluster (sa), same b-cluster (sb), both (sab)
    sa = ra[ia] - 1; sb = cb[ib] - 1; sab = N[ia, ib] - 1
    m = n - 1
    if index == "rand":
        # agree if (same a & same b) or (diff a & diff b)
        h1 = (sab + (m - sa - sb + sab)) / m
        est = float(h1.mean())  # mean over items of row means == mean over ordered pairs
        return dict(estimate=est, V=ustat_variance(h1), zeta1=float(np.var(h1, ddof=1)))
    elif index == "fowlkes_mallows":
        A = sab / m; B = (sa - sab) / m; C = (sb - sab) / m
        n11, n10, n01 = float(A.mean()), float(B.mean()), float(C.mean())
        est = n11 / math.sqrt((n11 + n10) * (n11 + n01))
        d1 = 1 / math.sqrt((n11 + n10) * (n11 + n01))
        dd = -0.5 * n11 * ((n11 + n01) + (n11 + n10)) / ((n11 + n10) * (n11 + n01)) ** 1.5
        h1 = d1 * A + dd * (A + B / 2 + C / 2)
        return dict(estimate=est, V=ustat_variance(h1), zeta1=float(np.var(h1, ddof=1)))
    raise ValueError("index must be 'rand' or 'fowlkes_mallows'")


def r_squared_variance(r2: float, degenerate_tol: float = 1e-3) -> dict:
    if not 0 <= r2 <= 1:
        raise ValueError("r2 must lie in [0, 1]")
    V = 4 * r2 * (1 - r2) ** 2
    degenerate = V < degenerate_tol
    if degenerate:
        warnings.warn(f"R-squared is near zero (V = {V:.2e}). The gradient vanishes there, so the limit is "
                      "chi-square, not normal, and sample size from V is invalid.", stacklevel=2)
    return dict(estimate=r2, V=V, degenerate=degenerate)


def repeatability_variance(sigma_w: float, replicates: int = 2) -> dict:
    """RC = 1.96 sqrt(2) sigma_w; Var(sigma_w_hat) = sigma_w^2 / (2 df), df = n (m - 1)."""
    if replicates < 2:
        raise ValueError("replicates must be at least 2")
    k = 1.96 * math.sqrt(2)
    return dict(estimate=k * sigma_w, V=k ** 2 * sigma_w ** 2 / (2 * (replicates - 1)))


def discrimination_slope_variance(mu1, mu0, sd1, sd0, prevalence) -> dict:
    if not 0 < prevalence < 1:
        raise ValueError("prevalence must be in (0, 1)")
    return dict(estimate=mu1 - mu0, V=sd1 ** 2 / prevalence + sd0 ** 2 / (1 - prevalence))


def risk_difference_variance(p1, p2, kappa: float = 0.5) -> dict:
    """Difference of two independent proportions; kappa = fraction allocated to group 1."""
    return dict(estimate=p1 - p2, V=p1 * (1 - p1) / kappa + p2 * (1 - p2) / (1 - kappa))


def paired_proportion_variance(p01, p10) -> dict:
    d = p10 - p01
    return dict(estimate=d, V=p01 + p10 - d ** 2,
                note="only discordant pairs contribute; concordant pairs carry no information")


# --- refusals ----------------------------------------------------------------

def mahalanobis_sample_size(*a, **k):
    raise NotImplementedError(
        "Mahalanobis distance is not supported. The squared distance is chi-square under normality "
        "rather than asymptotically normal, so n = (z+z)^2 V / delta^2 does not apply. "
        "Use a non-central chi-square power calculation instead.")


def n_for_target_rse(V: float, estimate: float, target_rse: float, deff: float = 1.0) -> dict:
    if target_rse <= 0:
        raise ValueError("target_rse must be positive")
    n = V / (target_rse * estimate) ** 2
    return dict(n=int(math.ceil(n * deff)), target_rse=target_rse,
                note="a target RSE is a precision requirement, not a hypothesis test")


def timedep_auc_sample_size(*a, **k):
    raise NotImplementedError(
        "Time-dependent, integrated and multiclass AUC are not implemented here. Variances are published "
        "but require censoring machinery from a separate literature. Use timeROC or survivalROC (R) for "
        "the interval, and pmvalsampsize or SSAML (R) to size a study on the c-statistic.")
