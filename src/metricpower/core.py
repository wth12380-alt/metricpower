"""
Core engine: sampling variance and sample size for metrics that are smooth
functions of multinomial cell probabilities.

Convention: binary cell probability vectors are ordered (tp, fp, fn, tn).
V denotes the PER-UNIT variance, V = n * Var(theta_hat).

This file is a line-for-line port of R/01_core.R in the R package, with
one addition: every sample-size function returns a `Result` carrying the
formula, the substituted numbers and the reference, so that a user interface
can show *how* the number was obtained.
"""
from __future__ import annotations

import math
import warnings
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence

import numpy as np
from scipy.optimize import brentq
from scipy.stats import norm

from .references import REFERENCES

Vector = Sequence[float]


# -----------------------------------------------------------------------------
# Result container
# -----------------------------------------------------------------------------

@dataclass
class Result:
    """What a sample-size calculation returns.

    n            required sample size (ceiling, after the design effect)
    metric       registry name
    group        "A" (existing software), "B" (implemented here)
    estimate     metric value under the alternative
    null_value   metric value under the null (if a test)
    V_alt, V_null  per-unit variances
    inputs       the user-facing inputs that produced the answer
    steps        ordered (label, value) pairs for display
    formula_latex   the general formula(s), LaTeX
    formula_numeric the same formula(s) with numbers substituted, LaTeX
    reference    key(s) into REFERENCES
    notes        caveats the user should read
    """
    n: Optional[int]
    metric: str
    group: str
    estimate: Optional[float] = None
    null_value: Optional[float] = None
    V_alt: Optional[float] = None
    V_null: Optional[float] = None
    inputs: Dict[str, object] = field(default_factory=dict)
    steps: List[tuple] = field(default_factory=list)
    formula_latex: List[str] = field(default_factory=list)
    formula_numeric: List[str] = field(default_factory=list)
    reference: List[str] = field(default_factory=list)
    notes: List[str] = field(default_factory=list)
    extra: Dict[str, object] = field(default_factory=dict)

    def references_full(self) -> List[str]:
        return [REFERENCES.get(k, k) for k in self.reference]

    def __repr__(self) -> str:  # compact, for the console
        head = f"{self.metric} [Group {self.group}]  n = {self.n}"
        if self.estimate is not None:
            head += f"   estimate = {self.estimate:.4f}"
        if self.V_alt is not None:
            head += f"   V = {self.V_alt:.4f}"
        return head


# -----------------------------------------------------------------------------
# 1. Cell probabilities
# -----------------------------------------------------------------------------

def cell_prob(tp: float, fp: float, fn: float, tn: float) -> np.ndarray:
    """Cell probabilities from counts or probabilities; rescaled to sum to 1."""
    v = np.array([tp, fp, fn, tn], dtype=float)
    if np.any(v < 0):
        raise ValueError("cells must be non-negative")
    s = v.sum()
    if s <= 0:
        raise ValueError("cells sum to zero")
    return v / s


def cell_prob_from_operating_point(sensitivity: float, specificity: float,
                                   prevalence: float) -> np.ndarray:
    """The planning entry point: assumed sensitivity, specificity, prevalence."""
    for name, x in (("sensitivity", sensitivity), ("specificity", specificity),
                    ("prevalence", prevalence)):
        if not 0 <= x <= 1:
            raise ValueError(f"{name} must lie in [0, 1]")
    return cell_prob(tp=sensitivity * prevalence,
                     fp=(1 - specificity) * (1 - prevalence),
                     fn=(1 - sensitivity) * prevalence,
                     tn=specificity * (1 - prevalence))


# -----------------------------------------------------------------------------
# 2. The delta-method engine
# -----------------------------------------------------------------------------

def numeric_gradient(g: Callable, p: Vector, h: float = 1e-6) -> np.ndarray:
    p = np.asarray(p, dtype=float)
    out = np.zeros_like(p)
    for i in range(len(p)):
        step = h * max(abs(p[i]), 1e-4)
        up = p.copy(); up[i] += step
        dn = p.copy(); dn[i] -= step
        out[i] = (float(g(up)) - float(g(dn))) / (2 * step)
    return out


def delta_var(p: Vector, g: Callable, grad: Optional[Callable] = None,
              h: float = 1e-6) -> float:
    """Per-unit variance V = grad' [diag(p) - p p'] grad of a smooth g(p).

    Exact covariance of multinomial proportions, so the only approximation is
    the linearisation of g.
    """
    p = np.asarray(p, dtype=float)
    if np.any(p < 0):
        raise ValueError("p must be non-negative")
    if abs(p.sum() - 1) > 1e-8:
        raise ValueError("p must sum to 1")
    gr = numeric_gradient(g, p, h) if grad is None else np.asarray(grad(p), dtype=float)
    Sigma = np.diag(p) - np.outer(p, p)
    v = float(gr @ Sigma @ gr)
    return max(v, 0.0)


# -----------------------------------------------------------------------------
# 3. Metric definitions and closed-form gradients on (tp, fp, fn, tn)
# -----------------------------------------------------------------------------

def metric_accuracy(p):    return p[0] + p[3]
def metric_sensitivity(p): return p[0] / (p[0] + p[2])
def metric_specificity(p): return p[3] / (p[3] + p[1])
def metric_ppv(p):         return p[0] / (p[0] + p[1])
def metric_npv(p):         return p[3] / (p[3] + p[2])
def metric_balanced_accuracy(p): return 0.5 * (metric_sensitivity(p) + metric_specificity(p))
def metric_youden(p):      return metric_sensitivity(p) + metric_specificity(p) - 1
def metric_iou(p):         return p[0] / (p[0] + p[1] + p[2])


def metric_fbeta(beta: float = 1.0):
    b2 = beta ** 2
    return lambda p: ((1 + b2) * p[0]) / ((1 + b2) * p[0] + b2 * p[2] + p[1])


metric_f1 = metric_fbeta(1.0)


def metric_mcc(p):
    num = p[0] * p[3] - p[1] * p[2]
    den = math.sqrt((p[0] + p[1]) * (p[0] + p[2]) * (p[3] + p[1]) * (p[3] + p[2]))
    return num / den


def metric_expected_cost(c_fp, c_fn, c_tp=0.0, c_tn=0.0):
    return lambda p: c_tp * p[0] + c_fp * p[1] + c_fn * p[2] + c_tn * p[3]


def metric_net_benefit(threshold: float):
    if not 0 < threshold < 1:
        raise ValueError("threshold must be in (0, 1)")
    w = threshold / (1 - threshold)
    return lambda p: p[0] - w * p[1]


def metric_markedness(p):            return p[0] / (p[0] + p[1]) + p[3] / (p[3] + p[2]) - 1
def metric_balanced_error_rate(p):   return 1 - metric_balanced_accuracy(p)
def metric_fdr(p):                   return p[1] / (p[0] + p[1])
def metric_for(p):                   return p[2] / (p[3] + p[2])
def metric_class_error(p):           return p[2] / (p[0] + p[2])
metric_false_referral_rate = metric_fdr


def metric_volumetric_similarity(p):
    D = 2 * p[0] + p[1] + p[2]
    return 1 - abs(p[1] - p[2]) / D


metric_ior = metric_sensitivity


# gradients, p = (a, b, c, d) = (tp, fp, fn, tn)
def grad_accuracy(p):    return np.array([1.0, 0.0, 0.0, 1.0])
def grad_sensitivity(p):
    S = p[0] + p[2]; return np.array([p[2] / S**2, 0.0, -p[0] / S**2, 0.0])
def grad_specificity(p):
    T = p[3] + p[1]; return np.array([0.0, -p[3] / T**2, 0.0, p[1] / T**2])
def grad_ppv(p):
    U = p[0] + p[1]; return np.array([p[1] / U**2, -p[0] / U**2, 0.0, 0.0])
def grad_npv(p):
    W = p[3] + p[2]; return np.array([0.0, 0.0, -p[3] / W**2, p[2] / W**2])
def grad_balanced_accuracy(p): return 0.5 * (grad_sensitivity(p) + grad_specificity(p))
def grad_youden(p):            return grad_sensitivity(p) + grad_specificity(p)


def grad_fbeta(beta: float = 1.0):
    b2 = beta ** 2
    def g(p):
        A = (1 + b2) * p[0]
        B = (1 + b2) * p[0] + b2 * p[2] + p[1]
        return np.array([(1 + b2) * (B - A) / B**2, -A / B**2, -A * b2 / B**2, 0.0])
    return g


grad_f1 = grad_fbeta(1.0)


def grad_iou(p):
    R = p[0] + p[1] + p[2]
    return np.array([(p[1] + p[2]) / R**2, -p[0] / R**2, -p[0] / R**2, 0.0])


def grad_mcc(p):
    a, b, c, d = p[0], p[1], p[2], p[3]
    U, S, T, W = a + b, a + c, d + b, d + c
    N = a * d - b * c
    D = math.sqrt(U * S * T * W)
    m = N / D
    dN = np.array([d, -c, -b, a])
    dlogD = 0.5 * np.array([1 / U + 1 / S, 1 / U + 1 / T, 1 / S + 1 / W, 1 / T + 1 / W])
    return dN / D - m * dlogD


def grad_expected_cost(c_fp, c_fn, c_tp=0.0, c_tn=0.0):
    return lambda p: np.array([c_tp, c_fp, c_fn, c_tn], dtype=float)


def grad_net_benefit(threshold):
    w = threshold / (1 - threshold)
    return lambda p: np.array([1.0, -w, 0.0, 0.0])


def grad_markedness(p):
    U = p[0] + p[1]; W = p[3] + p[2]
    return np.array([p[1] / U**2, -p[0] / U**2, -p[3] / W**2, p[2] / W**2])
def grad_balanced_error_rate(p): return -grad_balanced_accuracy(p)
def grad_fdr(p):
    U = p[0] + p[1]; return np.array([-p[1] / U**2, p[0] / U**2, 0.0, 0.0])
def grad_for(p):
    W = p[3] + p[2]; return np.array([0.0, 0.0, p[3] / W**2, -p[2] / W**2])
def grad_class_error(p):
    S = p[0] + p[2]; return np.array([-p[2] / S**2, 0.0, p[0] / S**2, 0.0])
grad_false_referral_rate = grad_fdr


def grad_volumetric_similarity(p):
    D = 2 * p[0] + p[1] + p[2]
    N = p[1] - p[2]
    s = np.sign(N)
    if s == 0:
        return np.zeros(4)
    return s / D**2 * np.array([2 * N, -(D - N), (D + N), 0.0])


grad_ior = grad_sensitivity


# -----------------------------------------------------------------------------
# 4. Registry
# -----------------------------------------------------------------------------

def _const(f):
    return lambda **kw: f


METRIC_REGISTRY: Dict[str, dict] = {
    # name: fun(**params) -> g, grad(**params) -> grad, label, latex definition
    "accuracy":            dict(fun=_const(metric_accuracy), grad=_const(grad_accuracy),
                                label="Accuracy", latex=r"\frac{TP+TN}{N}"),
    "sensitivity":         dict(fun=_const(metric_sensitivity), grad=_const(grad_sensitivity),
                                label="Sensitivity / recall / TPR", latex=r"\frac{TP}{TP+FN}"),
    "specificity":         dict(fun=_const(metric_specificity), grad=_const(grad_specificity),
                                label="Specificity / TNR", latex=r"\frac{TN}{TN+FP}"),
    "ppv":                 dict(fun=_const(metric_ppv), grad=_const(grad_ppv),
                                label="Positive predictive value / precision", latex=r"\frac{TP}{TP+FP}"),
    "npv":                 dict(fun=_const(metric_npv), grad=_const(grad_npv),
                                label="Negative predictive value", latex=r"\frac{TN}{TN+FN}"),
    "balanced_accuracy":   dict(fun=_const(metric_balanced_accuracy), grad=_const(grad_balanced_accuracy),
                                label="Balanced accuracy", latex=r"\tfrac12\left(\frac{TP}{TP+FN}+\frac{TN}{TN+FP}\right)"),
    "youden":              dict(fun=_const(metric_youden), grad=_const(grad_youden),
                                label="Youden index", latex=r"\frac{TP}{TP+FN}+\frac{TN}{TN+FP}-1"),
    "dsc_f1":              dict(fun=_const(metric_f1), grad=_const(grad_f1),
                                label="Dice / F1 (pooled)", latex=r"\frac{2TP}{2TP+FP+FN}"),
    "fbeta":               dict(fun=lambda beta=1.0, **kw: metric_fbeta(beta),
                                grad=lambda beta=1.0, **kw: grad_fbeta(beta),
                                label="F-beta (pooled)", latex=r"\frac{(1+\beta^2)TP}{(1+\beta^2)TP+\beta^2 FN+FP}",
                                params=dict(beta=1.0)),
    "iou":                 dict(fun=_const(metric_iou), grad=_const(grad_iou),
                                label="IoU / Jaccard (pooled)", latex=r"\frac{TP}{TP+FP+FN}"),
    "mcc":                 dict(fun=_const(metric_mcc), grad=_const(grad_mcc),
                                label="Matthews correlation coefficient",
                                latex=r"\frac{TP\cdot TN-FP\cdot FN}{\sqrt{(TP+FP)(TP+FN)(TN+FP)(TN+FN)}}"),
    "expected_cost":       dict(fun=lambda c_fp=1.0, c_fn=1.0, c_tp=0.0, c_tn=0.0, **kw: metric_expected_cost(c_fp, c_fn, c_tp, c_tn),
                                grad=lambda c_fp=1.0, c_fn=1.0, c_tp=0.0, c_tn=0.0, **kw: grad_expected_cost(c_fp, c_fn, c_tp, c_tn),
                                label="Expected cost", latex=r"c_{TP}\,p_{TP}+c_{FP}\,p_{FP}+c_{FN}\,p_{FN}+c_{TN}\,p_{TN}",
                                params=dict(c_fp=1.0, c_fn=1.0, c_tp=0.0, c_tn=0.0)),
    "net_benefit":         dict(fun=lambda threshold=0.2, **kw: metric_net_benefit(threshold),
                                grad=lambda threshold=0.2, **kw: grad_net_benefit(threshold),
                                label="Net benefit", latex=r"p_{TP}-\frac{p_t}{1-p_t}\,p_{FP}",
                                params=dict(threshold=0.2)),
    "markedness":          dict(fun=_const(metric_markedness), grad=_const(grad_markedness),
                                label="Markedness", latex=r"PPV+NPV-1"),
    "balanced_error_rate": dict(fun=_const(metric_balanced_error_rate), grad=_const(grad_balanced_error_rate),
                                label="Balanced error rate", latex=r"1-\text{balanced accuracy}"),
    "false_discovery_rate": dict(fun=_const(metric_fdr), grad=_const(grad_fdr),
                                label="False discovery rate", latex=r"\frac{FP}{TP+FP}"),
    "false_omission_rate": dict(fun=_const(metric_for), grad=_const(grad_for),
                                label="False omission rate", latex=r"\frac{FN}{TN+FN}"),
    "class_error":         dict(fun=_const(metric_class_error), grad=_const(grad_class_error),
                                label="Class error", latex=r"\frac{FN}{TP+FN}"),
    "false_referral_rate": dict(fun=_const(metric_false_referral_rate), grad=_const(grad_false_referral_rate),
                                label="False referral rate", latex=r"\frac{FP}{TP+FP}"),
    "volumetric_similarity": dict(fun=_const(metric_volumetric_similarity), grad=_const(grad_volumetric_similarity),
                                label="Volumetric similarity (pooled)", latex=r"1-\frac{|FP-FN|}{2TP+FP+FN}"),
    "ior":                 dict(fun=_const(metric_ior), grad=_const(grad_ior),
                                label="Intersection over reference", latex=r"\frac{TP}{TP+FN}"),
}

CM_METRICS = list(METRIC_REGISTRY.keys())


def get_metric(name: str, **params) -> Callable:
    if name not in METRIC_REGISTRY:
        raise KeyError(f"unknown metric '{name}'. Available: {', '.join(METRIC_REGISTRY)}")
    return METRIC_REGISTRY[name]["fun"](**params)


def get_gradient(name: str, **params) -> Optional[Callable]:
    e = METRIC_REGISTRY.get(name)
    if e is None or e.get("grad") is None:
        return None
    return e["grad"](**params)


def metric_variance(p: Vector, name: str, **params) -> dict:
    """Point estimate and per-unit variance V for a registered metric."""
    g = get_metric(name, **params)
    gr = get_gradient(name, **params)
    p = np.asarray(p, dtype=float)
    return dict(estimate=float(g(p)), V=delta_var(p, g, grad=gr),
                method="delta (analytic gradient)" if gr else "delta (numeric gradient)")


# -----------------------------------------------------------------------------
# 5. Paired two-model comparison (8-cell joint distribution)
# -----------------------------------------------------------------------------

PAIRED_CELLS = ["y1a1b1", "y1a1b0", "y1a0b1", "y1a0b0",
                "y0a1b1", "y0a1b0", "y0a0b1", "y0a0b0"]
_Y = np.array([1, 1, 1, 1, 0, 0, 0, 0]); _A = np.array([1, 1, 0, 0, 1, 1, 0, 0]); _B = np.array([1, 0, 1, 0, 1, 0, 1, 0])
PAIRED_M = {
    "A": np.array([(_Y == 1) & (_A == 1), (_Y == 0) & (_A == 1), (_Y == 1) & (_A == 0), (_Y == 0) & (_A == 0)], dtype=float),
    "B": np.array([(_Y == 1) & (_B == 1), (_Y == 0) & (_B == 1), (_Y == 1) & (_B == 0), (_Y == 0) & (_B == 0)], dtype=float),
}


def paired_marginal(p8: Vector, model: str = "A") -> np.ndarray:
    return PAIRED_M[model] @ np.asarray(p8, dtype=float)


def delta_var_paired(p8: Vector, name: str, **params) -> dict:
    """V of metric(A) - metric(B) on the same cases: V_A + V_B - 2 Cov(A, B)."""
    g = get_metric(name, **params)
    gr = get_gradient(name, **params)
    p8 = np.asarray(p8, dtype=float)
    d = lambda q: g(paired_marginal(q, "A")) - g(paired_marginal(q, "B"))
    dgrad = None
    if gr is not None:
        dgrad = lambda q: PAIRED_M["A"].T @ gr(paired_marginal(q, "A")) - PAIRED_M["B"].T @ gr(paired_marginal(q, "B"))
    return dict(estimate=float(d(p8)), V=delta_var(p8, d, grad=dgrad))


# -----------------------------------------------------------------------------
# 6. Clustering
# -----------------------------------------------------------------------------

def design_effect(m_bar: float, icc: float) -> float:
    """1 + (m_bar - 1) * icc for m_bar units per cluster and intra-cluster correlation icc."""
    if m_bar < 1:
        raise ValueError("m_bar must be >= 1")
    if not 0 <= icc <= 1:
        raise ValueError("icc must lie in [0, 1]")
    return 1 + (m_bar - 1) * icc


# -----------------------------------------------------------------------------
# 7. Sample size and power
# -----------------------------------------------------------------------------

def _z(alpha: float, two_sided: bool) -> float:
    return float(norm.ppf(1 - alpha / (2 if two_sided else 1)))


def n_for_power(V_alt: float, delta: float, V_null: Optional[float] = None,
                alpha: float = 0.05, power: float = 0.80, two_sided: bool = True,
                deff: float = 1.0) -> int:
    """n = (z_{1-alpha/2} sqrt(V0) + z_{1-beta} sqrt(V1))^2 / delta^2, times deff, ceiling."""
    if delta == 0:
        raise ValueError("delta must be non-zero")
    if V_null is None:
        V_null = V_alt
    za = _z(alpha, two_sided)
    zb = float(norm.ppf(power))
    n = (za * math.sqrt(V_null) + zb * math.sqrt(V_alt)) ** 2 / delta ** 2
    return int(math.ceil(n * deff - 1e-12))


def n_for_power_t(V_alt: float, delta: float, alpha: float = 0.05, power: float = 0.80,
                  two_sided: bool = True, deff: float = 1.0, n_max: int = 10_000_000) -> int:
    """Sample size for a one-sample t-test on a per-case mean (noncentral-t power, as power.t.test).

    The z formula uses sigma as if known; at n below ~50 the plug-in SE inflates the Wald type I
    error (0.12 at n = 5, 0.07 at n = 15).  This returns the smallest n whose t-test power reaches
    the target: power(n) = P(|T_{n-1, ncp}| > t_{n-1, 1-alpha/2}), ncp = |delta| sqrt(n / V_alt).
    Equals the z-based n for large n.  deff multiplies the result."""
    if delta == 0:
        raise ValueError("delta must be non-zero")
    from scipy.stats import nct, t as tdist
    a = alpha / (2 if two_sided else 1)
    n = max(2, n_for_power(V_alt, delta, None, alpha, power, two_sided, 1.0))   # z-based start (a lower bound)
    while n < n_max:
        df = n - 1; crit = tdist.ppf(1 - a, df); ncp = abs(delta) * math.sqrt(n / V_alt)
        pw = 1 - nct.cdf(crit, df, ncp) + (nct.cdf(-crit, df, ncp) if two_sided else 0.0)
        if pw >= power:
            break
        n += 1
    return int(math.ceil(n * deff - 1e-12))


def power_for_n_t(n: float, V_alt: float, delta: float, alpha: float = 0.05, two_sided: bool = True,
                  deff: float = 1.0) -> float:
    """Power of the one-sample t-test on a per-case mean at n (noncentral t)."""
    from scipy.stats import nct, t as tdist
    a = alpha / (2 if two_sided else 1); n_eff = n / deff; df = n_eff - 1
    crit = tdist.ppf(1 - a, df); ncp = abs(delta) * math.sqrt(n_eff / V_alt)
    return float(1 - nct.cdf(crit, df, ncp) + (nct.cdf(-crit, df, ncp) if two_sided else 0.0))


def kink_distance(p: Vector, n: float) -> float:
    """For volumetric similarity: |fp - fn| in standard errors at sample size n.  The metric is not
    differentiable at fp = fn; below about 3 SE a share of samples crosses the kink, the absolute
    value folds the sampling distribution and the delta-method variance is too large (18% at 1.6 SE)."""
    p = np.asarray(p, dtype=float); fp, fn = p[1], p[2]
    se = math.sqrt(max((fp + fn - (fp - fn) ** 2) / n, 1e-300))
    return abs(fp - fn) / se


def power_for_n(n: float, V_alt: float, delta: float, V_null: Optional[float] = None,
                alpha: float = 0.05, two_sided: bool = True, deff: float = 1.0) -> float:
    if V_null is None:
        V_null = V_alt
    za = _z(alpha, two_sided)
    n_eff = n / deff
    return float(norm.cdf((math.sqrt(n_eff) * abs(delta) - za * math.sqrt(V_null)) / math.sqrt(V_alt)))


def n_for_precision(V: float, halfwidth: float, conf_level: float = 0.95, deff: float = 1.0) -> int:
    """n = z^2 V / w^2 for a CI of half-width w."""
    if halfwidth <= 0:
        raise ValueError("halfwidth must be positive")
    z = float(norm.ppf(1 - (1 - conf_level) / 2))
    return int(math.ceil(z ** 2 * V / halfwidth ** 2 * deff - 1e-12))


def metric_ci(estimate: float, V: float, n: float, conf_level: float = 0.95) -> dict:
    z = float(norm.ppf(1 - (1 - conf_level) / 2))
    se = math.sqrt(V / n)
    return dict(estimate=estimate, se=se, lower=estimate - z * se, upper=estimate + z * se)


def cell_prob_at_metric(p: Vector, name: str, target: float, path: str = "sensitivity",
                        **params) -> np.ndarray:
    """Cell probabilities with metric value == target, reached by moving mass
    tp->fn (path 'sensitivity') or tn->fp (path 'specificity')."""
    p = np.asarray(p, dtype=float)
    g = get_metric(name, **params)
    if path == "sensitivity":
        shift = lambda t: cell_prob(p[0] - t, p[1], p[2] + t, p[3]); hi = p[0]
    elif path == "specificity":
        shift = lambda t: cell_prob(p[0], p[1] + t, p[2], p[3] - t); hi = p[3]
    else:
        raise ValueError("path must be 'sensitivity' or 'specificity'")
    f = lambda t: float(g(shift(t))) - target
    lo_val, hi_val = f(0.0), f(hi * (1 - 1e-8))
    if lo_val * hi_val > 0:
        raise ValueError(f"target metric value {target:.6g} is not reachable along the "
                         f"'{path}' path from this operating point")
    t0 = brentq(f, 0.0, hi * (1 - 1e-8), xtol=1e-12)
    return shift(t0)


def check_cell_counts(p: Vector, n: float, min_expected: float = 10) -> float:
    m = float(np.min(p)) * n
    if m < min_expected:
        warnings.warn(
            f"smallest expected cell count is {m:.1f} (threshold {min_expected:g}). The delta "
            "method is unreliable here and errs optimistically; the metric may also be undefined "
            "in some samples. Prefer a simulation- or exact-based sample size at this operating point.",
            stacklevel=2)
    return m


def vs_variance_folded(p: Vector, n: float) -> dict:
    """Per-unit variance of volumetric similarity, VS = 1 - |Y|/D with Y = fp - fn and D = 2tp + fp + fn,
    using the folded-normal distribution of |Y| instead of the delta method's linearisation.

    Y is a linear function of the cell proportions, so Y ~ N(mu, sigma^2/n) by the CLT; |Y| is then
    folded normal with E|Y| = 2 s phi(mu/s) + mu (2 Phi(mu/s) - 1) and Var|Y| = mu^2 + s^2 - (E|Y|)^2,
    s = sigma/sqrt(n).  The ratio |Y|/D is expanded to first order in D (which is smooth), with
    Cov(|Y|, D) from the joint normality of (Y, D).  The delta-method variance is the limit
    |mu|/s -> infinity; at 1.6 SE from the kink it overstates V by 16%.  The variance depends on n,
    so the result is returned per unit (multiply by 1/n for Var(VS-hat)) together with the small-n
    bias of the point estimate, E[VS-hat] - VS = -(E|Y| - |mu|)/D."""
    p = np.asarray(p, dtype=float)
    a = np.array([0.0, 1.0, -1.0, 0.0]); b = np.array([2.0, 1.0, 1.0, 0.0])
    Sig = np.diag(p) - np.outer(p, p)
    mu = float(a @ p); D = float(b @ p)
    var_y, var_d, cov_yd = float(a @ Sig @ a) / n, float(b @ Sig @ b) / n, float(a @ Sig @ b) / n
    s = math.sqrt(max(var_y, 1e-300)); r = mu / s
    EA = 2 * s * norm.pdf(r) + mu * (2 * norm.cdf(r) - 1)                   # E|Y|
    VarA = mu ** 2 + s ** 2 - EA ** 2                                       # Var|Y|
    EY2sgn = (mu ** 2 + s ** 2) * (2 * norm.cdf(r) - 1) + 2 * mu * s * norm.pdf(r)   # E[Y^2 sign(Y)]
    cov_ay = EY2sgn - EA * mu                                               # Cov(|Y|, Y)
    cov_ad = cov_yd / var_y * cov_ay                                        # Cov(|Y|, D) via D | Y regression
    var_vs = VarA / D ** 2 + EA ** 2 * var_d / D ** 4 - 2 * EA * cov_ad / D ** 3
    return dict(V=var_vs * n, bias=-(EA - abs(mu)) / D, kink_se=abs(mu) / s, estimate=1 - abs(mu) / D)


def vs_folded_cdf(c: float, mu: float, s: float) -> float:
    """P(|Y| <= c) for Y ~ N(mu, s^2) (folded normal)."""
    return float(norm.cdf((c - mu) / s) - norm.cdf((-c - mu) / s))


def vs_folded_quantile(q: float, mu: float, s: float) -> float:
    from scipy.optimize import brentq
    return float(brentq(lambda c: vs_folded_cdf(c, mu, s) - q, 0.0, abs(mu) + 12 * s))


def vs_folded_power(n: float, p1: Vector, vs0: float, alpha: float = 0.05, two_sided: bool = True) -> float:
    """Power of the folded-normal test of H0: VS = vs0 for volumetric similarity when the truth is p1.

    VS = 1 - |Y|/D with Y = fp - fn ~ N(mu, s^2), s^2 = Var(Y)/n, so |Y| is folded normal and VS is a
    monotone function of |Y|.  Under H0, |mu| = mu0 = D (1 - vs0); the equal-tailed test rejects when
    |Y-hat| falls below the alpha/2 or above the 1 - alpha/2 quantile of the folded normal with mean mu0
    (one-sided: only the tail toward the alternative).  D is taken at the operating point; its sampling
    variability is second order.  The Wald test with the delta-method SE is conservative near the kink
    (type I 0.02 at 1.6 SE), the Wald test with the folded-normal SE is anti-conservative (0.08-0.10),
    because the sampling distribution there is folded, not normal; this test uses the folded
    distribution itself and holds the nominal level."""
    p1 = np.asarray(p1, dtype=float)
    a = np.array([0.0, 1.0, -1.0, 0.0]); b = np.array([2.0, 1.0, 1.0, 0.0])
    Sig = np.diag(p1) - np.outer(p1, p1); D = float(b @ p1); mu1 = abs(float(a @ p1))
    s = math.sqrt(max(float(a @ Sig @ a) / n, 1e-300)); mu0 = D * (1 - vs0)
    if two_sided:
        lo, hi = vs_folded_quantile(alpha / 2, mu0, s), vs_folded_quantile(1 - alpha / 2, mu0, s)
        return vs_folded_cdf(lo, mu1, s) + 1 - vs_folded_cdf(hi, mu1, s)
    if mu1 > mu0:                      # alternative: VS lower, |Y| larger
        return 1 - vs_folded_cdf(vs_folded_quantile(1 - alpha, mu0, s), mu1, s)
    return vs_folded_cdf(vs_folded_quantile(alpha, mu0, s), mu1, s)


def n_for_power_vs_folded(p1: Vector, delta: float, alpha: float = 0.05, power: float = 0.80,
                          two_sided: bool = True, deff: float = 1.0, n_max: int = 10_000_000) -> int:
    """Smallest n at which the folded-normal test for volumetric similarity reaches the target power."""
    p1 = np.asarray(p1, dtype=float); vs1 = float(metric_volumetric_similarity(p1)); vs0 = vs1 - delta
    lo = 4
    hi = max(8, 2 * n_for_power(delta_var(p1, metric_volumetric_similarity, grad=grad_volumetric_similarity), delta,
                                None, alpha, power, two_sided, 1.0))
    while vs_folded_power(hi, p1, vs0, alpha, two_sided) < power and hi < n_max:
        hi *= 2
    while hi - lo > 1:                                        # bisection on the (monotone) power curve
        mid = (lo + hi) // 2
        if vs_folded_power(mid, p1, vs0, alpha, two_sided) >= power: hi = mid
        else: lo = mid
    return int(math.ceil(hi * deff - 1e-12))


def _n_for_power_vs(p1: Vector, p0, delta, alpha, power, two_sided, deff) -> dict:
    """Fixed-point iteration for the n-dependent folded-normal variance of volumetric similarity."""
    za, zb = _z(alpha, two_sided), float(norm.ppf(power))
    n = max(n_for_power(delta_var(p1, metric_volumetric_similarity, grad=grad_volumetric_similarity), delta,
                        None, alpha, power, two_sided, 1.0), 4)
    for _ in range(100):
        V1 = vs_variance_folded(p1, n)["V"]
        V0 = vs_variance_folded(p0, n)["V"] if p0 is not None else V1
        n_new = int(math.ceil((za * math.sqrt(V0) + zb * math.sqrt(V1)) ** 2 / delta ** 2 - 1e-12))
        if n_new == n:
            break
        n = n_new
    return dict(n=int(math.ceil(n * deff - 1e-12)), V1=V1, V0=V0)


def n_for_power_metric(p: Vector, name: str, delta: float, alpha: float = 0.05,
                       power: float = 0.80, two_sided: bool = True, deff: float = 1.0,
                       test: str = "wald", null_path: str = "auto", **params) -> Result:
    """Sample size for a registered confusion-matrix metric.

    test = "wald"  : n = (z_a + z_b)^2 V1 / delta^2  (plug-in SE)
    test = "score" : n = (z_a sqrt(V0) + z_b sqrt(V1))^2 / delta^2 (null-standardised)
    Both are returned in `extra`; `n` is the one matching `test`.
    """
    if test not in ("wald", "score"):
        raise ValueError("test must be 'wald' or 'score'")
    p = np.asarray(p, dtype=float)
    g = get_metric(name, **params)
    gr = get_gradient(name, **params)
    theta1 = float(g(p))
    V1 = delta_var(p, g, grad=gr)

    paths = ["sensitivity", "specificity"] if null_path == "auto" else [null_path]
    p0, used = None, None
    for pth in paths:
        try:
            p0 = cell_prob_at_metric(p, name, theta1 - delta, path=pth, **params); used = pth
            break
        except ValueError:
            continue
    notes = []
    if p0 is None:
        if test == "score":
            raise ValueError(f"null value {theta1 - delta:.6g} is not reachable from this operating "
                             "point along any path; use test='wald' or a smaller delta")
        notes.append("null value not reachable along any path; n_score is unavailable, the Wald n is unaffected")
        V0 = float("nan")
    else:
        V0 = delta_var(p0, g, grad=gr)

    za, zb = _z(alpha, two_sided), float(norm.ppf(power))
    n_wald = n_for_power(V1, delta, V1, alpha, power, two_sided, deff)
    n_score = None if math.isnan(V0) else n_for_power(V1, delta, V0, alpha, power, two_sided, deff)
    n = n_wald if test == "wald" else n_score
    m = float(np.min(p)) * n
    if m < 10:
        notes.append(f"smallest expected cell count is {m:.1f} (< 10): the delta method errs "
                     "optimistically here; the metric may be undefined in some samples")
    if n < 250:
        notes.append("n < 250: the delta-method variance is optimistic by a few per cent at this "
                     "size (up to ~10% at n = 25 for Dice); treat n as a floor")
    if V1 < 1e-6:
        notes.append("V is essentially zero: the null sits at a stationary point of the metric, "
                     "the limit is chi-square rather than normal, and this n is not valid")
    if name == "volumetric_similarity":
        # |fp - fn| has a kink at fp = fn.  Near it the sampling distribution of VS is folded normal, not
        # normal: the Wald test with the delta-method SE is conservative and the one with the folded-normal
        # SE is anti-conservative.  The sample size is therefore taken from the folded-normal test itself.
        n_f = n_for_power_vs_folded(p, delta, alpha, power, two_sided, deff)
        f = vs_variance_folded(p, n_f)
        extra_vs = dict(n_folded=n_f, n_wald_delta=n_wald, n_score_delta=n_score, V_folded=f["V"], bias_folded=f["bias"], kink_se=f["kink_se"])
        n = n_f
        notes.append(f"volumetric similarity: |fp - fn| is {f['kink_se']:.1f} SE from the kink at fp = fn at n = {n_f}. "
                     f"n = {n_f} is the sample size of the folded-normal test (vs_folded_power), which is the test to run: "
                     f"the Wald test with the delta-method SE is conservative there (delta-method n = {n_wald}) and the point "
                     f"estimate is biased by {f['bias']:+.4f}. The folded-normal variance is {f['V']:.4f} against the "
                     f"delta-method {V1:.4f}")

    e = METRIC_REGISTRY[name]
    latex_def = e["latex"]
    steps = [("p = (tp, fp, fn, tn)", np.round(p, 4).tolist()),
             ("metric value under H1", round(theta1, 4)),
             ("metric value under H0", round(theta1 - delta, 4)),
             ("gradient at p", np.round(gr(p) if gr else numeric_gradient(g, p), 4).tolist()),
             ("V_alt = grad' Sigma grad", round(V1, 5)),
             ("V_null (along %s path)" % used if used else "V_null", None if math.isnan(V0) else round(V0, 5)),
             ("z_{1-alpha/2}" if two_sided else "z_{1-alpha}", round(za, 6)),
             ("z_{1-beta}", round(zb, 6)),
             ("n (Wald)", n_wald), ("n (score)", n_score)]
    if deff != 1:
        steps.append(("design effect", deff))
    formula_latex = [
        r"\theta = " + latex_def,
        r"\Sigma = \operatorname{diag}(p) - pp^{\prime},\qquad V = \nabla g(p)^{\prime}\,\Sigma\,\nabla g(p)",
        (r"n_{\text{Wald}} = \frac{(z_{1-\alpha/2}+z_{1-\beta})^2\,V_1}{\Delta^2}" if test == "wald"
         else r"n_{\text{score}} = \frac{\left(z_{1-\alpha/2}\sqrt{V_0}+z_{1-\beta}\sqrt{V_1}\right)^2}{\Delta^2}"),
    ]
    if deff != 1:
        formula_latex.append(r"n_{\text{clustered}} = n \times \text{deff},\quad \text{deff}=1+(\bar m-1)\rho")
    if test == "wald":
        num = (f"n = \\frac{{({za:.6f}+{zb:.6f})^2 \\times {V1:.5f}}}{{{delta:.4f}^2}} "
               f"= {(za + zb) ** 2 * V1 / delta ** 2:.2f} \\to {n_wald if deff == 1 else n_wald}")
    else:
        num = (f"n = \\frac{{({za:.6f}\\sqrt{{{V0:.5f}}}+{zb:.6f}\\sqrt{{{V1:.5f}}})^2}}{{{delta:.4f}^2}} "
               f"= {(za * math.sqrt(V0) + zb * math.sqrt(V1)) ** 2 / delta ** 2:.2f} \\to {n_score}")
    formula_numeric = [f"\\theta_1 = {theta1:.4f},\\quad \\theta_0 = {theta1 - delta:.4f},\\quad \\Delta = {delta:.4f}",
                       f"V_1 = {V1:.5f}" + ("" if math.isnan(V0) else f",\\quad V_0 = {V0:.5f}"),
                       num]
    if deff != 1:
        formula_numeric.append(f"n \\times \\text{{deff}} = {n / deff:.0f} \\times {deff:.3f} \\to {n}")
    refs = ["delta_method", "takahashi2022"] if name in ("dsc_f1", "fbeta", "iou") else \
           ["delta_method", "itaya2024"] if name == "mcc" else ["delta_method"]
    return Result(n=n, metric=name, group="B", estimate=theta1, null_value=theta1 - delta,
                  V_alt=V1, V_null=None if math.isnan(V0) else V0,
                  inputs=dict(p=p.tolist(), delta=delta, alpha=alpha, power=power, two_sided=two_sided,
                              deff=deff, test=test, **params),
                  steps=steps, formula_latex=formula_latex, formula_numeric=formula_numeric,
                  reference=refs, notes=notes,
                  extra=dict(n_wald=n_wald, n_score=n_score, p_null=None if p0 is None else p0.tolist(), **(extra_vs if name == 'volumetric_similarity' else {}),
                             null_path=used, min_expected_cell=m))


# -----------------------------------------------------------------------------
# 8. Weighted Cohen's kappa
# -----------------------------------------------------------------------------

def weights_identity(k):  return np.eye(k)
def weights_linear(k):
    i = np.arange(k); return 1 - np.abs(np.subtract.outer(i, i)) / (k - 1)
def weights_quadratic(k):
    i = np.arange(k); return 1 - (np.subtract.outer(i, i) / (k - 1)) ** 2


def kappa_w(P, W) -> float:
    P = np.asarray(P, dtype=float); P = P / P.sum()
    pi_, p_j = P.sum(1), P.sum(0)
    po = float((W * P).sum()); pe = float((W * np.outer(pi_, p_j)).sum())
    return (po - pe) / (1 - pe)


def kappa_w_var_fce(P, W) -> float:
    """Fleiss, Cohen & Everitt (1969) per-unit variance of weighted kappa."""
    P = np.asarray(P, dtype=float); P = P / P.sum(); W = np.asarray(W, dtype=float)
    pi_, p_j = P.sum(1), P.sum(0)
    po = float((W * P).sum()); pe = float((W * np.outer(pi_, p_j)).sum())
    wi = W @ p_j; wj = W.T @ pi_
    inner = np.add.outer(wi, wj)
    term = float((P * (W * (1 - pe) - inner * (1 - po)) ** 2).sum())
    V = (term - (po * pe - 2 * pe + po) ** 2) / (1 - pe) ** 4
    return max(V, 0.0)


def kappa_w_var_delta(P, W) -> float:
    P = np.asarray(P, dtype=float); k = P.shape[0]; P = P / P.sum()
    return delta_var(P.ravel(), lambda q: kappa_w(np.asarray(q).reshape(k, k), W))


# -----------------------------------------------------------------------------
# 9. Per-case means
# -----------------------------------------------------------------------------

def var_percase(x) -> float:
    return float(np.var(np.asarray(x, dtype=float), ddof=1))


def var_percase_paired(x_a, x_b) -> float:
    x_a, x_b = np.asarray(x_a, dtype=float), np.asarray(x_b, dtype=float)
    if len(x_a) != len(x_b):
        raise ValueError("x_a and x_b must have equal length")
    return float(np.var(x_a - x_b, ddof=1))


def brier_scores(y, phat):
    return (np.asarray(phat, dtype=float) - np.asarray(y, dtype=float)) ** 2


def var_sqrt_transform(V: float, mean_value: float) -> float:
    if mean_value <= 0:
        raise ValueError("mean_value must be positive")
    return V / (4 * mean_value)


# -----------------------------------------------------------------------------
# 10. Guard: calibration errors
# -----------------------------------------------------------------------------

def n_for_power_calibration_error(*args, **kwargs):
    raise NotImplementedError(
        "Sample size for calibration-error metrics is not supported.\n"
        "  ECE / ECE-KDE / CWCE / KCE have non-regular asymptotics: the estimator is degenerate\n"
        "  under the null of perfect calibration, the convergence rate differs between calibrated\n"
        "  and miscalibrated models, and Wald intervals cover negative values for a non-negative\n"
        "  quantity. Use a simulation-based or permutation-test route instead.")


# -----------------------------------------------------------------------------
# 11. Finite-sample checks (simulation; the only place the asymptotics are questioned)
# -----------------------------------------------------------------------------

def sensitivity_by_n(p: Vector, name: str, n_grid=(50, 100, 250, 500, 1000, 2500),
                     delta: Optional[float] = None, reps: int = 20000, alpha: float = 0.05,
                     power: float = 0.80, seed: Optional[int] = None, **params) -> List[dict]:
    rng = np.random.default_rng(seed)
    p = np.asarray(p, dtype=float)
    g = get_metric(name, **params); gr = get_gradient(name, **params)
    V = delta_var(p, g, grad=gr); theta1 = float(g(p)); z = _z(alpha, True)
    out = []
    for nn in n_grid:
        dr = rng.multinomial(nn, p, size=reps) / nn
        with np.errstate(divide="ignore", invalid="ignore"):
            est = np.array([g(q) for q in dr], dtype=float)
        ok = np.isfinite(est)
        V_emp = nn * float(np.var(est[ok], ddof=1))
        row = dict(n=nn, V_analytic=V, V_finite=V_emp, ratio=V_emp / V,
                   undefined_pct=100 * float(np.mean(~ok)), min_expected_cell=float(p.min()) * nn)
        if delta is not None:
            with np.errstate(divide="ignore", invalid="ignore"):
                se = np.sqrt(np.array([delta_var(q, g, grad=gr) if np.isfinite(g(q)) else np.nan for q in dr]) / nn)
            row["power_actual"] = float(np.nanmean(np.abs(est - (theta1 - delta)) / se > z))
            row["power_nominal"] = power_for_n(nn, V, delta, alpha=alpha)
        out.append(row)
    return out


def sensitivity_by_n_percase(x, n_grid=(25, 50, 100, 250, 500), reps: int = 20000,
                             conf_level: float = 0.95, seed: Optional[int] = None) -> List[dict]:
    rng = np.random.default_rng(seed)
    x = np.asarray(x, dtype=float); mu = x.mean(); sd = x.std(ddof=1)
    z = float(norm.ppf(1 - (1 - conf_level) / 2))
    skew = float(np.mean(((x - mu) / sd) ** 3))
    out = []
    for nn in n_grid:
        B = rng.choice(x, size=(reps, nn), replace=True)
        m = B.mean(1); se = np.sqrt(B.var(1, ddof=1) / nn)
        lo, hi = m - z * se, m + z * se
        out.append(dict(n=nn, coverage_wald=float(np.mean((lo <= mu) & (mu <= hi))),
                        miss_low=float(np.mean(mu < lo)), miss_high=float(np.mean(mu > hi)),
                        skew=skew, nominal=conf_level))
    return out


def ratio_noise_band(est, reps: int) -> dict:
    est = np.asarray(est, dtype=float)
    z = (est - est.mean()) / est.std(ddof=1)
    ku = float(np.mean(z ** 4) - 3)
    sd_ratio = math.sqrt((2 + max(ku, -2)) / reps)
    return dict(sd=sd_ratio, lower=1 - 2 * sd_ratio, upper=1 + 2 * sd_ratio, kurtosis=ku,
                reps_for=lambda bias: math.ceil((2 + max(ku, -2)) / (bias / 2) ** 2))


def check_delta_reliable(p: Vector, name: str, n: int, reps: int = 20000,
                         seed: Optional[int] = None, **params) -> dict:
    rng = np.random.default_rng(seed)
    p = np.asarray(p, dtype=float)
    g = get_metric(name, **params); gr = get_gradient(name, **params)
    V = delta_var(p, g, grad=gr)
    dr = rng.multinomial(n, p, size=reps) / n
    with np.errstate(divide="ignore", invalid="ignore"):
        est = np.array([g(q) for q in dr], dtype=float)
    est = est[np.isfinite(est)]
    ratio = n * float(np.var(est, ddof=1)) / V
    reliable = n >= 250 or ratio < 1.02
    verdict = (f"OK. The delta method is reliable at n = {n}." if reliable else
               f"Caution. At n = {n} the analytic variance understates the truth by about "
               f"{100 * (ratio - 1):.0f}%, so the required sample size is optimistic. The delta "
               "method becomes reliable from about n = 250.")
    return dict(metric=name, n=n, V_analytic=V, V_finite=n * float(np.var(est, ddof=1)),
                ratio=ratio, reliable=reliable, verdict=verdict)
