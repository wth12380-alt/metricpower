"""
Group A: metrics for which established software already computes a sample
size. In R the package only *points* to that software (presize, MKpower,
pROC, kappaSize, ICC.Sample.Size, pmvalsampsize, powerSurvEpi, RJafroc).
Those packages have no Python equivalents, so here the underlying published
formulas are implemented natively. Every function names the formula and the
primary reference, and says when it differs from the R package it replaces.

Two things are NOT ported, because they need pilot variance components or
machinery that no closed form captures:
    froc / afroc (RJafroc, multi-reader multi-case)   -> pointer + single-reader approximation
    partial_auc (pROC with partial.auc)               -> pointer only
"""
from __future__ import annotations

import math
from typing import Dict, Optional

import numpy as np
from scipy.optimize import brentq
from scipy.special import expit
from scipy.stats import norm

from .core import Result, _z, delta_var, cell_prob_from_operating_point, kappa_w, kappa_w_var_fce, weights_identity
from . import families as F


def _zz(alpha, power, two_sided):
    return _z(alpha, two_sided), float(norm.ppf(power))


def _ceil(x):
    return int(math.ceil(x - 1e-12))


# -----------------------------------------------------------------------------
# A1. Binomial proportion: accuracy, PPV, NPV, top-n accuracy, agreement, yield
# -----------------------------------------------------------------------------

def proportion(p1: float, p0: Optional[float] = None, halfwidth: Optional[float] = None,
               alpha: float = 0.05, power: float = 0.80, two_sided: bool = True, deff: float = 1.0,
               metric: str = "accuracy", method: str = "score") -> Result:
    """One binomial proportion. Power (p1 vs p0) or precision (CI half-width)."""
    for nm, v in (("p1", p1), ("p0", p0)):
        if v is not None and not 0 < v < 1:
            raise ValueError(f"{nm} must be in (0, 1)")
    za, zb = _zz(alpha, power, two_sided)
    inputs = dict(p1=p1, p0=p0, halfwidth=halfwidth, alpha=alpha, power=power, deff=deff, method=method)
    if halfwidth is not None:
        V = p1 * (1 - p1)
        if method == "wilson":
            f = lambda n: za * math.sqrt(V / n + za ** 2 / (4 * n ** 2)) / (1 + za ** 2 / n) - halfwidth
            n_raw = brentq(f, 1, 1e9)
            fl = [r"\text{Wilson half-width } w = \frac{z\sqrt{p(1-p)/n + z^2/(4n^2)}}{1+z^2/n}\ \text{solved for } n"]
            fn = [f"p = {p1:.4f},\\ w = {halfwidth:.4f}\\ \\Rightarrow\\ n = {n_raw:.2f}"]
            refs = ["wald_proportion"]
        else:
            n_raw = za ** 2 * V / halfwidth ** 2
            fl = [r"n = \frac{z_{1-\alpha/2}^2\,p(1-p)}{w^2}"]
            fn = [f"n = \\frac{{{za:.6f}^2\\times{p1:.4f}(1-{p1:.4f})}}{{{halfwidth:.4f}^2}} = {n_raw:.2f}"]
            refs = ["wald_proportion"]
        n = _ceil(n_raw * deff)
        return Result(n=n, metric=metric, group="A", estimate=p1, V_alt=V, inputs=inputs,
                      steps=[("p", p1), ("V = p(1-p)", round(V, 5)), ("z", round(za, 6)), ("n", n)],
                      formula_latex=fl, formula_numeric=fn, reference=refs,
                      notes=["precision target: a CI of half-width w around the proportion"])
    if p0 is None:
        raise ValueError("give p0 (null) for a power calculation or halfwidth for precision")
    V1, V0 = p1 * (1 - p1), p0 * (1 - p0)
    d = p1 - p0
    if method == "wald":
        n_raw = (za + zb) ** 2 * V1 / d ** 2
        fl = [r"n = \frac{(z_{1-\alpha/2}+z_{1-\beta})^2\,p_1(1-p_1)}{(p_1-p_0)^2}"]
        fn = [f"n = \\frac{{({za:.6f}+{zb:.6f})^2\\times{V1:.5f}}}{{({d:.4f})^2}} = {n_raw:.2f}"]
    else:
        n_raw = (za * math.sqrt(V0) + zb * math.sqrt(V1)) ** 2 / d ** 2
        fl = [r"n = \frac{\left(z_{1-\alpha/2}\sqrt{p_0(1-p_0)}+z_{1-\beta}\sqrt{p_1(1-p_1)}\right)^2}{(p_1-p_0)^2}"]
        fn = [f"n = \\frac{{({za:.6f}\\sqrt{{{V0:.5f}}}+{zb:.6f}\\sqrt{{{V1:.5f}}})^2}}{{({d:.4f})^2}} = {n_raw:.2f}"]
    n = _ceil(n_raw * deff)
    if deff != 1:
        fn.append(f"{_ceil(n_raw)} \\times \\text{{deff}}\\ {deff:.3f} \\to {n}")
    return Result(n=n, metric=metric, group="A", estimate=p1, null_value=p0, V_alt=V1, V_null=V0, inputs=inputs,
                  steps=[("p1 (H1)", p1), ("p0 (H0)", p0), ("V1 = p1(1-p1)", round(V1, 5)), ("V0 = p0(1-p0)", round(V0, 5)),
                         ("z_{1-alpha/2}", round(za, 6)), ("z_{1-beta}", round(zb, 6)), ("n", n)],
                  formula_latex=fl, formula_numeric=fn, reference=["wald_proportion"],
                  notes=["one-sample test of a proportion; 'score' standardises by the null variance (classical), "
                         "'wald' by the alternative (matches a plug-in-SE interval)"])


# -----------------------------------------------------------------------------
# A2. Sensitivity / specificity with prevalence (Buderer 1996)
# -----------------------------------------------------------------------------

def sensitivity_specificity(value1: float, prevalence: float, value0: Optional[float] = None,
                            halfwidth: Optional[float] = None, which: str = "sensitivity",
                            alpha: float = 0.05, power: float = 0.80, two_sided: bool = True,
                            deff: float = 1.0, method: str = "score") -> Result:
    """Buderer: size the diseased (or non-diseased) subgroup as a binomial proportion,
    then divide by prevalence (or 1 - prevalence) to get the total."""
    if not 0 < prevalence < 1:
        raise ValueError("prevalence must be in (0, 1)")
    share = prevalence if which == "sensitivity" else 1 - prevalence
    r = proportion(value1, value0, halfwidth, alpha, power, two_sided, 1.0, metric=which, method=method)
    n_sub = r.n
    n_total = _ceil(n_sub / share * deff)
    r.n = n_total
    r.extra = dict(n_subgroup=n_sub, share=share)
    r.inputs.update(prevalence=prevalence, which=which)
    r.steps.insert(-1, (f"n in the {'diseased' if which == 'sensitivity' else 'non-diseased'} subgroup", n_sub))
    r.steps[-1] = ("n total = n_subgroup / share", n_total)
    r.formula_latex.append(r"n_{\text{total}} = \frac{n_{\text{subgroup}}}{\phi}" if which == "sensitivity"
                           else r"n_{\text{total}} = \frac{n_{\text{subgroup}}}{1-\phi}")
    r.formula_numeric.append(f"n_{{\\text{{total}}}} = {n_sub}/{share:.3f} = {n_sub / share:.1f} \\to {n_total}")
    r.reference = ["buderer1996", "wald_proportion"]
    r.notes = [f"n is the TOTAL number of cases at prevalence {prevalence:.3f}; {n_sub} must be "
               f"{'diseased' if which == 'sensitivity' else 'non-diseased'}",
               "MKpower::power.diagnostic.test (R) uses the exact-binomial method of Flahault et al. 2005; "
               "numbers differ slightly from this normal approximation"]
    return r


# -----------------------------------------------------------------------------
# A3. Likelihood ratios (Simel 1991), odds ratio (Woolf), risk difference
# -----------------------------------------------------------------------------

def likelihood_ratio(sens: float, spec: float, prevalence: float, which: str = "positive",
                     lr0: Optional[float] = None, ci_ratio: Optional[float] = None,
                     alpha: float = 0.05, power: float = 0.80, two_sided: bool = True, deff: float = 1.0) -> Result:
    """LR+ = sens/(1-spec), LR- = (1-sens)/spec, on the log scale.
    Power: detect LR1 (from sens, spec) against lr0.  Precision: CI for LR of the form (LR/ci_ratio, LR*ci_ratio)."""
    for nm, v in (("sens", sens), ("spec", spec), ("prevalence", prevalence)):
        if not 0 < v < 1:
            raise ValueError(f"{nm} must be in (0, 1)")
    if which == "positive":
        lr = sens / (1 - spec)
        V = (1 - sens) / (sens * prevalence) + spec / ((1 - spec) * (1 - prevalence))
        fl = [r"LR^+ = \frac{Se}{1-Sp},\qquad n\,\operatorname{Var}(\log LR^+) = \frac{1-Se}{Se\,\phi}+\frac{Sp}{(1-Sp)(1-\phi)}"]
    else:
        lr = (1 - sens) / spec
        V = sens / ((1 - sens) * prevalence) + (1 - spec) / (spec * (1 - prevalence))
        fl = [r"LR^- = \frac{1-Se}{Sp},\qquad n\,\operatorname{Var}(\log LR^-) = \frac{Se}{(1-Se)\,\phi}+\frac{1-Sp}{Sp\,(1-\phi)}"]
    za, zb = _zz(alpha, power, two_sided)
    fn = [f"LR = {lr:.4f},\\quad V = {V:.5f}"]
    if ci_ratio is not None:
        w = math.log(ci_ratio); n_raw = za ** 2 * V / w ** 2
        fl.append(r"n = \frac{z^2\,V}{(\log r)^2}\quad\text{for a CI } (LR/r,\ LR\cdot r)")
        fn.append(f"n = {za:.6f}^2\\times{V:.5f}/(\\log {ci_ratio})^2 = {n_raw:.2f}")
        null = None
    else:
        if lr0 is None:
            raise ValueError("give lr0 for power or ci_ratio for precision")
        d = abs(math.log(lr) - math.log(lr0)); n_raw = (za + zb) ** 2 * V / d ** 2
        fl.append(r"n = \frac{(z_{1-\alpha/2}+z_{1-\beta})^2\,V}{(\log LR_1-\log LR_0)^2}")
        fn.append(f"n = ({za:.6f}+{zb:.6f})^2\\times{V:.5f}/({d:.4f})^2 = {n_raw:.2f}")
        null = lr0
    n = _ceil(n_raw * deff)
    return Result(n=n, metric=f"lr_{which}", group="A", estimate=lr, null_value=null, V_alt=V,
                  inputs=dict(sens=sens, spec=spec, prevalence=prevalence, lr0=lr0, ci_ratio=ci_ratio),
                  steps=[("LR", round(lr, 4)), ("V (log scale)", round(V, 5)), ("n", n)],
                  formula_latex=fl, formula_numeric=fn, reference=["simel1991"],
                  notes=["n is the TOTAL at the given prevalence; cf. presize::prec_pos_lr / prec_neg_lr (R)"])


def odds_ratio(p1: float, p2: float, or0: float = 1.0, kappa: float = 0.5, ci_ratio: Optional[float] = None,
               alpha: float = 0.05, power: float = 0.80, two_sided: bool = True, deff: float = 1.0) -> Result:
    """Woolf: n Var(log OR) = 1/(kappa p1(1-p1)) + 1/((1-kappa) p2(1-p2)); n is the TOTAL."""
    OR = (p1 / (1 - p1)) / (p2 / (1 - p2))
    V = 1 / (kappa * p1 * (1 - p1)) + 1 / ((1 - kappa) * p2 * (1 - p2))
    za, zb = _zz(alpha, power, two_sided)
    fl = [r"OR = \frac{p_1/(1-p_1)}{p_2/(1-p_2)},\qquad n\,\operatorname{Var}(\log OR) = \frac{1}{\kappa p_1(1-p_1)}+\frac{1}{(1-\kappa)p_2(1-p_2)}"]
    fn = [f"OR = {OR:.4f},\\quad V = {V:.5f}"]
    if ci_ratio is not None:
        n_raw = za ** 2 * V / math.log(ci_ratio) ** 2
        fl.append(r"n = z^2 V/(\log r)^2"); fn.append(f"n = {n_raw:.2f}"); null = None
    else:
        d = abs(math.log(OR) - math.log(or0)); n_raw = (za + zb) ** 2 * V / d ** 2
        fl.append(r"n = \frac{(z_{1-\alpha/2}+z_{1-\beta})^2 V}{(\log OR_1-\log OR_0)^2}")
        fn.append(f"n = ({za:.6f}+{zb:.6f})^2\\times{V:.5f}/({d:.4f})^2 = {n_raw:.2f}"); null = or0
    n = _ceil(n_raw * deff)
    return Result(n=n, metric="odds_ratio", group="A", estimate=OR, null_value=null, V_alt=V,
                  inputs=dict(p1=p1, p2=p2, or0=or0, kappa=kappa, ci_ratio=ci_ratio),
                  steps=[("OR", round(OR, 4)), ("V (log scale)", round(V, 5)), ("n", n)],
                  formula_latex=fl, formula_numeric=fn, reference=["woolf1955"],
                  notes=["n is the TOTAL across both groups; cf. presize::prec_or (R)"])


# -----------------------------------------------------------------------------
# A4. AUC (Hanley-McNeil 1982; Obuchowski 1994 binormal)
# -----------------------------------------------------------------------------

def _hanley_var(A: float, n1: float, n0: float) -> float:
    Q1 = A / (2 - A); Q2 = 2 * A ** 2 / (1 + A)
    return (A * (1 - A) + (n1 - 1) * (Q1 - A ** 2) + (n0 - 1) * (Q2 - A ** 2)) / (n1 * n0)


def _obuchowski_V(A: float, R: float) -> float:
    a = norm.ppf(A) * math.sqrt(2)
    return 0.0099 * math.exp(-a ** 2 / 2) * (5 * a ** 2 + 8 + (a ** 2 + 8) / R)


def auc(auc1: float, auc0: float = 0.5, ratio: float = 1.0, halfwidth: Optional[float] = None,
        method: str = "hanley", alpha: float = 0.05, power: float = 0.80, two_sided: bool = True,
        deff: float = 1.0) -> Result:
    """Sample size for one AUC. ratio = n_negative / n_positive.

    method 'hanley'     : Hanley & McNeil (1982) nonparametric variance, exact in n (solved numerically)
    method 'obuchowski' : Obuchowski (1994) binormal approximation, the route taken by pROC::power.roc.test
    """
    for nm, v in (("auc1", auc1), ("auc0", auc0)):
        if not 0.5 <= v < 1:
            raise ValueError(f"{nm} must lie in [0.5, 1)")
    za, zb = _zz(alpha, power, two_sided)
    if method == "hanley":
        if halfwidth is not None:
            f = lambda n1: za * math.sqrt(_hanley_var(auc1, n1, ratio * n1)) - halfwidth
            fl = [r"\operatorname{Var}(\hat A) = \frac{A(1-A)+(n_1-1)(Q_1-A^2)+(n_0-1)(Q_2-A^2)}{n_1 n_0},\quad Q_1=\frac{A}{2-A},\ Q_2=\frac{2A^2}{1+A}",
                  r"z\sqrt{\operatorname{Var}(\hat A)} = w\ \text{solved for } n_1,\qquad n_0 = \kappa n_1"]
        else:
            if auc1 == auc0:
                raise ValueError("auc1 must differ from auc0")
            f = lambda n1: (za * math.sqrt(_hanley_var(auc0, n1, ratio * n1)) + zb * math.sqrt(_hanley_var(auc1, n1, ratio * n1))) - abs(auc1 - auc0)
            fl = [r"\operatorname{Var}(\hat A) = \frac{A(1-A)+(n_1-1)(Q_1-A^2)+(n_0-1)(Q_2-A^2)}{n_1 n_0},\quad Q_1=\frac{A}{2-A},\ Q_2=\frac{2A^2}{1+A}",
                  r"z_{1-\alpha/2}\sqrt{\operatorname{Var}_0}+z_{1-\beta}\sqrt{\operatorname{Var}_1} = |A_1-A_0|\ \text{solved for } n_1,\qquad n_0=\kappa n_1"]
        n1 = brentq(f, 2, 1e8)
        n1c = _ceil(n1 * deff); n0c = _ceil(ratio * n1 * deff)
        V1 = _hanley_var(auc1, n1, ratio * n1) * n1
        fn = [f"A_1 = {auc1:.3f},\\ A_0 = {auc0:.3f},\\ \\kappa = {ratio}",
              f"n_1 = {n1:.2f} \\to {n1c},\\quad n_0 = {ratio}\\times n_1 \\to {n0c},\\quad n = {n1c + n0c}"]
        refs = ["hanley1982"]
        notes = ["n_1 positives, n_0 negatives; n is their sum. Hanley-McNeil assumes the exponential-score model"]
    elif method == "obuchowski":
        V1 = _obuchowski_V(auc1, ratio); V0 = _obuchowski_V(auc0, ratio)
        if halfwidth is not None:
            n1 = za ** 2 * V1 / halfwidth ** 2
            fl = [r"V(A) = 0.0099\,e^{-a^2/2}\left(5a^2+8+\frac{a^2+8}{\kappa}\right),\quad a=\sqrt2\,\Phi^{-1}(A)",
                  r"n_1 = z^2 V(A)/w^2"]
        else:
            n1 = (za * math.sqrt(V0) + zb * math.sqrt(V1)) ** 2 / (auc1 - auc0) ** 2
            fl = [r"V(A) = 0.0099\,e^{-a^2/2}\left(5a^2+8+\frac{a^2+8}{\kappa}\right),\quad a=\sqrt2\,\Phi^{-1}(A)",
                  r"n_1 = \frac{\left(z_{1-\alpha/2}\sqrt{V(A_0)}+z_{1-\beta}\sqrt{V(A_1)}\right)^2}{(A_1-A_0)^2},\qquad n_0=\kappa n_1"]
        n1c = _ceil(n1 * deff); n0c = _ceil(ratio * n1 * deff)
        fn = [f"V(A_1) = {V1:.5f},\\ V(A_0) = {V0:.5f}", f"n_1 = {n1:.2f} \\to {n1c},\\quad n_0 \\to {n0c},\\quad n = {n1c + n0c}"]
        refs = ["obuchowski1994", "obuchowski1997"]
        notes = ["binormal (b = 1) approximation; this is the route pROC::power.roc.test (R) takes [verify against pROC]"]
    else:
        raise ValueError("method must be 'hanley' or 'obuchowski'")
    return Result(n=n1c + n0c, metric="auroc", group="A", estimate=auc1, null_value=None if halfwidth else auc0,
                  V_alt=V1, inputs=dict(auc1=auc1, auc0=auc0, ratio=ratio, halfwidth=halfwidth, method=method),
                  steps=[("n positives", n1c), ("n negatives", n0c), ("n total", n1c + n0c)],
                  formula_latex=fl, formula_numeric=fn, reference=refs,
                  notes=notes + ["for comparing two correlated AUCs use DeLong (1988); not implemented here"],
                  extra=dict(n1=n1c, n0=n0c))


# -----------------------------------------------------------------------------
# A5. Cohen's kappa, unweighted (common-marginal agreement model + FCE variance)
# -----------------------------------------------------------------------------

def _kappa_table(kappa: float, marg: np.ndarray) -> np.ndarray:
    return np.outer(marg, marg) + kappa * (np.diag(marg) - np.outer(marg, marg))


def kappa_unweighted(kappa1: float, kappa0: float, marginals, alpha: float = 0.05, power: float = 0.80,
                     two_sided: bool = True, deff: float = 1.0) -> Result:
    """Two raters, k categories with common marginal probabilities.

    The joint table implied by kappa under common marginals is
        P_ij = pi_i pi_j + kappa (delta_ij pi_i - pi_i pi_j),
    and the Fleiss-Cohen-Everitt (1969) variance with identity weights gives V at
    kappa0 and kappa1.  n = (z_a sqrt(V0) + z_b sqrt(V1))^2 / (kappa1 - kappa0)^2.
    """
    marg = np.asarray(marginals, dtype=float); marg = marg / marg.sum(); k = len(marg)
    if k < 2:
        raise ValueError("need at least two categories")
    W = weights_identity(k)
    P1, P0 = _kappa_table(kappa1, marg), _kappa_table(kappa0, marg)
    if np.any(P1 < -1e-12) or np.any(P0 < -1e-12):
        raise ValueError("kappa not attainable with these marginals (negative cell)")
    V1, V0 = kappa_w_var_fce(P1, W), kappa_w_var_fce(P0, W)
    k1, k0 = kappa_w(P1, W), kappa_w(P0, W)
    za, zb = _zz(alpha, power, two_sided)
    d = abs(kappa1 - kappa0)
    n_raw = (za * math.sqrt(V0) + zb * math.sqrt(V1)) ** 2 / d ** 2
    n = _ceil(n_raw * deff)
    return Result(n=n, metric="kappa_unweighted", group="A", estimate=k1, null_value=k0, V_alt=V1, V_null=V0,
                  inputs=dict(kappa1=kappa1, kappa0=kappa0, marginals=marg.tolist()),
                  steps=[("V1 (FCE at kappa1)", round(V1, 5)), ("V0 (FCE at kappa0)", round(V0, 5)), ("n", n)],
                  formula_latex=[r"P_{ij} = \pi_i\pi_j+\kappa(\delta_{ij}\pi_i-\pi_i\pi_j)\ \text{(common marginals)}",
                                 r"V = \text{Fleiss–Cohen–Everitt variance with } w_{ij}=\delta_{ij}",
                                 r"n = \frac{\left(z_{1-\alpha/2}\sqrt{V_0}+z_{1-\beta}\sqrt{V_1}\right)^2}{(\kappa_1-\kappa_0)^2}"],
                  formula_numeric=[f"V_1 = {V1:.5f},\\ V_0 = {V0:.5f}",
                                   f"n = ({za:.6f}\\sqrt{{{V0:.5f}}}+{zb:.6f}\\sqrt{{{V1:.5f}}})^2/({d:.4f})^2 = {n_raw:.2f} \\to {n}"],
                  reference=["fleiss1969"],
                  notes=["two raters; kappaSize::PowerBinary (R, Rotondi & Donner 2012) uses a goodness-of-fit "
                         "statistic rather than this Wald/score form, so its numbers differ somewhat",
                         "for weighted kappa use Group B 'weighted_kappa'"])


# -----------------------------------------------------------------------------
# A6. ICC (Fisher-type transform for power; Bonett 2002 for precision)
# -----------------------------------------------------------------------------

def icc(icc1: float, icc0: Optional[float] = None, k: int = 2, halfwidth: Optional[float] = None,
        alpha: float = 0.05, power: float = 0.80, two_sided: bool = True, deff: float = 1.0) -> Result:
    for nm, v in (("icc1", icc1), ("icc0", icc0)):
        if v is not None and not 0 <= v < 1:
            raise ValueError(f"{nm} must lie in [0, 1)")
    za, zb = _zz(alpha, power, two_sided)
    if halfwidth is not None:
        w = 2 * halfwidth
        n_raw = 8 * za ** 2 * (1 - icc1) ** 2 * (1 + (k - 1) * icc1) ** 2 / (k * (k - 1) * w ** 2) + 1
        n = _ceil(n_raw * deff)
        return Result(n=n, metric="icc", group="A", estimate=icc1, inputs=dict(icc1=icc1, k=k, halfwidth=halfwidth),
                      steps=[("n", n)],
                      formula_latex=[r"n = \frac{8 z_{1-\alpha/2}^2 (1-\rho)^2 (1+(k-1)\rho)^2}{k(k-1)\,w^2}+1\quad(w=\text{full CI width})"],
                      formula_numeric=[f"n = \\frac{{8\\times{za:.6f}^2 (1-{icc1})^2 (1+({k}-1){icc1})^2}}{{{k}({k}-1)\\,{w:.4f}^2}}+1 = {n_raw:.2f} \\to {n}"],
                      reference=["bonett2002"], notes=["precision form; cf. presize::prec_icc (R)"])
    if icc0 is None:
        raise ValueError("give icc0 for power or halfwidth for precision")
    zt = lambda r: 0.5 * math.log((1 + (k - 1) * r) / (1 - r))
    z1, z0 = zt(icc1), zt(icc0)
    Vz = k / (2 * (k - 1))                      # n Var(z) with n - 2 in place of n
    n_raw = 2 + (za + zb) ** 2 * Vz / (z1 - z0) ** 2
    n = _ceil(n_raw * deff)
    return Result(n=n, metric="icc", group="A", estimate=icc1, null_value=icc0, V_alt=Vz, V_null=Vz,
                  inputs=dict(icc1=icc1, icc0=icc0, k=k),
                  steps=[("z(icc1)", round(z1, 4)), ("z(icc0)", round(z0, 4)), ("V_z = k/(2(k-1))", round(Vz, 4)), ("n", n)],
                  formula_latex=[r"z(\rho) = \tfrac12\log\frac{1+(k-1)\rho}{1-\rho},\qquad \operatorname{Var}(z)\approx\frac{k}{2(k-1)(n-2)}",
                                 r"n = 2+\frac{(z_{1-\alpha/2}+z_{1-\beta})^2\,k}{2(k-1)\,(z_1-z_0)^2}"],
                  formula_numeric=[f"z_1 = {z1:.4f},\\ z_0 = {z0:.4f}",
                                   f"n = 2+\\frac{{({za:.6f}+{zb:.6f})^2\\times{k}}}{{2({k}-1)({z1 - z0:.4f})^2}} = {n_raw:.2f} \\to {n}"],
                  reference=["zou2012", "fisher_z"],
                  notes=["n = number of subjects, each rated k times; Fisher-type transform of the one-way ICC. "
                         "ICC.Sample.Size::calculateIccSampleSize (R, Zou 2012) uses this transform with an assurance "
                         "term; without assurance the numbers agree closely"])


# -----------------------------------------------------------------------------
# A7. Pearson correlation and Lin's concordance correlation (Fisher z)
# -----------------------------------------------------------------------------

def pearson_correlation(r1: float, r0: float = 0.0, halfwidth: Optional[float] = None, alpha: float = 0.05,
                        power: float = 0.80, two_sided: bool = True, deff: float = 1.0) -> Result:
    za, zb = _zz(alpha, power, two_sided)
    if halfwidth is not None:
        wz = 0.5 * (math.atanh(min(r1 + halfwidth, 0.999999)) - math.atanh(max(r1 - halfwidth, -0.999999)))
        n_raw = 3 + za ** 2 / wz ** 2
        n = _ceil(n_raw * deff)
        return Result(n=n, metric="pearson_correlation", group="A", estimate=r1, inputs=dict(r1=r1, halfwidth=halfwidth),
                      steps=[("half-width on z scale", round(wz, 4)), ("n", n)],
                      formula_latex=[r"z=\operatorname{atanh}(r),\ \operatorname{Var}(z)=\frac{1}{n-3},\qquad n = 3+\frac{z_{1-\alpha/2}^2}{w_z^2}"],
                      formula_numeric=[f"w_z = {wz:.4f},\\quad n = 3+{za:.6f}^2/{wz:.4f}^2 = {n_raw:.2f} \\to {n}"],
                      reference=["fisher_z"], notes=["cf. presize::prec_cor (R)"])
    z1, z0 = math.atanh(r1), math.atanh(r0)
    n_raw = 3 + (za + zb) ** 2 / (z1 - z0) ** 2
    n = _ceil(n_raw * deff)
    return Result(n=n, metric="pearson_correlation", group="A", estimate=r1, null_value=r0, V_alt=1.0, V_null=1.0,
                  inputs=dict(r1=r1, r0=r0),
                  steps=[("z1 = atanh(r1)", round(z1, 4)), ("z0 = atanh(r0)", round(z0, 4)), ("n", n)],
                  formula_latex=[r"z=\operatorname{atanh}(r),\ \operatorname{Var}(z)=\frac{1}{n-3},\qquad n = 3+\frac{(z_{1-\alpha/2}+z_{1-\beta})^2}{(z_1-z_0)^2}"],
                  formula_numeric=[f"n = 3+\\frac{{({za:.6f}+{zb:.6f})^2}}{{({z1:.4f}-{z0:.4f})^2}} = {n_raw:.2f} \\to {n}"],
                  reference=["fisher_z"], notes=["cf. pwr::pwr.r.test (R), which uses the same transform"])


def ccc(ccc1: float, ccc0: float, r: float, u: float = 0.0, alpha: float = 0.05, power: float = 0.80,
        two_sided: bool = True, deff: float = 1.0) -> Result:
    """Lin (1989): Var(rho_c) = (1/(n-2)) [ (1-r^2) rho_c^2 (1-rho_c^2)/r^2 + 2 rho_c^3 (1-rho_c) u^2/r - rho_c^4 u^4/(2 r^2) ],
    with u = (mu1 - mu2)/sqrt(sigma1 sigma2) the location shift; power on the z = atanh scale."""
    def bracket(rc):
        return (1 - r ** 2) * rc ** 2 * (1 - rc ** 2) / r ** 2 + 2 * rc ** 3 * (1 - rc) * u ** 2 / r - rc ** 4 * u ** 4 / (2 * r ** 2)
    V1, V0 = bracket(ccc1) / (1 - ccc1 ** 2) ** 2, bracket(ccc0) / (1 - ccc0 ** 2) ** 2   # on the z scale
    z1, z0 = math.atanh(ccc1), math.atanh(ccc0)
    za, zb = _zz(alpha, power, two_sided)
    n_raw = 2 + (za * math.sqrt(V0) + zb * math.sqrt(V1)) ** 2 / (z1 - z0) ** 2
    n = _ceil(n_raw * deff)
    return Result(n=n, metric="ccc", group="A", estimate=ccc1, null_value=ccc0, V_alt=V1, V_null=V0,
                  inputs=dict(ccc1=ccc1, ccc0=ccc0, r=r, u=u),
                  steps=[("V1 (z scale)", round(V1, 5)), ("V0 (z scale)", round(V0, 5)), ("n", n)],
                  formula_latex=[r"(n-2)\operatorname{Var}(\hat\rho_c) = \frac{(1-r^2)\rho_c^2(1-\rho_c^2)}{r^2}+\frac{2\rho_c^3(1-\rho_c)u^2}{r}-\frac{\rho_c^4u^4}{2r^2}",
                                 r"Z=\operatorname{atanh}(\rho_c),\ \operatorname{Var}(Z)=\operatorname{Var}(\hat\rho_c)/(1-\rho_c^2)^2,\qquad n = 2+\frac{(z_{1-\alpha/2}\sqrt{V_0}+z_{1-\beta}\sqrt{V_1})^2}{(Z_1-Z_0)^2}"],
                  formula_numeric=[f"V_1 = {V1:.5f},\\ V_0 = {V0:.5f},\\ Z_1 = {z1:.4f},\\ Z_0 = {z0:.4f}", f"n = {n_raw:.2f} \\to {n}"],
                  reference=["lin1989"], notes=["needs the Pearson r between the two methods and the standardised mean shift u; cf. cccPower (R)"])


# -----------------------------------------------------------------------------
# A8. Prediction-model validation criteria (Riley et al. 2021)
# -----------------------------------------------------------------------------

def _gh_expect(fun, mean, sd, m=80):
    x, w = np.polynomial.hermite_e.hermegauss(m)
    lp = mean + sd * x
    return float((w * fun(lp)).sum() / math.sqrt(2 * math.pi))


def calibration_slope(lp_mean: float, lp_sd: float, halfwidth: float = 0.1, conf_level: float = 0.95,
                      deff: float = 1.0) -> Result:
    """Riley 2021 criterion 3. LP (linear predictor) assumed normal; calibration assumed (alpha=0, beta=1).
    n = I_aa / (SE^2 (I_aa I_bb - I_ab^2)), with I the expected Fisher information elements."""
    z = float(norm.ppf(1 - (1 - conf_level) / 2))
    SE = halfwidth / z
    pq = lambda lp: expit(lp) * (1 - expit(lp))
    Iaa = _gh_expect(pq, lp_mean, lp_sd)
    Iab = _gh_expect(lambda lp: lp * pq(lp), lp_mean, lp_sd)
    Ibb = _gh_expect(lambda lp: lp ** 2 * pq(lp), lp_mean, lp_sd)
    n_raw = Iaa / (SE ** 2 * (Iaa * Ibb - Iab ** 2))
    n = _ceil(n_raw * deff)
    return Result(n=n, metric="calibration_slope", group="A", estimate=1.0, inputs=dict(lp_mean=lp_mean, lp_sd=lp_sd, halfwidth=halfwidth),
                  steps=[("I_aa = E[p(1-p)]", round(Iaa, 5)), ("I_ab = E[LP p(1-p)]", round(Iab, 5)),
                         ("I_bb = E[LP^2 p(1-p)]", round(Ibb, 5)), ("target SE(slope)", round(SE, 5)), ("n", n)],
                  formula_latex=[r"\text{LP}\sim N(\mu,\sigma^2),\ p=\operatorname{expit}(\text{LP}),\quad I_{\alpha\alpha}=E[p(1-p)],\ I_{\alpha\beta}=E[\text{LP}\,p(1-p)],\ I_{\beta\beta}=E[\text{LP}^2p(1-p)]",
                                 r"n = \frac{I_{\alpha\alpha}}{SE(\hat\beta)^2\,(I_{\alpha\alpha}I_{\beta\beta}-I_{\alpha\beta}^2)},\qquad SE(\hat\beta)=\frac{w}{z_{1-\alpha/2}}"],
                  formula_numeric=[f"I_{{\\alpha\\alpha}}={Iaa:.5f},\\ I_{{\\alpha\\beta}}={Iab:.5f},\\ I_{{\\beta\\beta}}={Ibb:.5f},\\ SE={SE:.5f}",
                                   f"n = {n_raw:.2f} \\to {n}"],
                  reference=["riley2021"], notes=["precision criterion for the calibration slope; the same computation as "
                                                 "pmvalsampsize (R) with lpnormal = c(mean, sd)"])


def oe_ratio(prevalence: float, ci_width: float = 0.2, conf_level: float = 0.95, deff: float = 1.0) -> Result:
    """Riley 2021 criterion 2: n = (1 - phi) / (phi SE(ln OE)^2), SE from a target CI width on the O/E scale (O/E = 1)."""
    z = float(norm.ppf(1 - (1 - conf_level) / 2))
    SE = math.asinh(ci_width / 2) / z
    n_raw = (1 - prevalence) / (prevalence * SE ** 2)
    n = _ceil(n_raw * deff)
    return Result(n=n, metric="oe_ratio", group="A", estimate=1.0, inputs=dict(prevalence=prevalence, ci_width=ci_width),
                  steps=[("target SE(ln O/E)", round(SE, 5)), ("n", n)],
                  formula_latex=[r"\operatorname{Var}(\ln O/E)\approx\frac{1-\phi}{n\phi},\qquad n = \frac{1-\phi}{\phi\,SE^2},\quad SE=\frac{\operatorname{asinh}(w/2)}{z}"],
                  formula_numeric=[f"SE = {SE:.5f},\\quad n = (1-{prevalence})/({prevalence}\\times{SE:.5f}^2) = {n_raw:.2f} \\to {n}"],
                  reference=["riley2021"], notes=["precision criterion; cf. pmvalsampsize oeciwidth (R)"])


def c_statistic(c: float, prevalence: float, halfwidth: float = 0.05, conf_level: float = 0.95, deff: float = 1.0) -> Result:
    """Riley 2021 criterion 4 (Hanley-McNeil form): solve n so that z SE(C) = halfwidth."""
    z = float(norm.ppf(1 - (1 - conf_level) / 2))
    def se2(n):
        return c * (1 - c) * (1 + (n / 2 - 1) * (1 - c) / (2 - c) + (n / 2 - 1) * c / (1 + c)) / (n ** 2 * prevalence * (1 - prevalence))
    f = lambda n: z * math.sqrt(se2(n)) - halfwidth
    n_raw = brentq(f, 4, 1e9)
    n = _ceil(n_raw * deff)
    return Result(n=n, metric="c_statistic", group="A", estimate=c, inputs=dict(c=c, prevalence=prevalence, halfwidth=halfwidth),
                  steps=[("n", n)],
                  formula_latex=[r"SE(\hat C)^2 = \frac{C(1-C)\left[1+(\tfrac n2-1)\frac{1-C}{2-C}+(\tfrac n2-1)\frac{C}{1+C}\right]}{n^2\phi(1-\phi)}",
                                 r"z_{1-\alpha/2}\,SE(\hat C) = w\ \text{solved for } n"],
                  formula_numeric=[f"C = {c},\\ \\phi = {prevalence},\\ w = {halfwidth}\\ \\Rightarrow\\ n = {n_raw:.2f} \\to {n}"],
                  reference=["riley2021", "hanley1982"], notes=["precision criterion; cf. pmvalsampsize cstatciwidth (R)"])


def standardised_net_benefit(sens: float, spec: float, prevalence: float, threshold: float,
                             delta: Optional[float] = None, halfwidth: Optional[float] = None,
                             alpha: float = 0.05, power: float = 0.80, two_sided: bool = True, deff: float = 1.0) -> Result:
    """sNB = NB / prevalence with NB = p_tp - w p_fp, w = pt/(1-pt); variance by the delta method on the cells."""
    p = cell_prob_from_operating_point(sens, spec, prevalence)
    w = threshold / (1 - threshold)
    g = lambda q: (q[0] - w * q[1]) / (q[0] + q[2])
    est = float(g(p)); V = delta_var(p, g)
    za, zb = _zz(alpha, power, two_sided)
    if halfwidth is not None:
        n_raw = za ** 2 * V / halfwidth ** 2; null = None
        fl2 = r"n = z_{1-\alpha/2}^2 V/w^2"; fn2 = f"n = {za:.6f}^2\\times{V:.5f}/{halfwidth}^2 = {n_raw:.2f}"
    else:
        if delta is None:
            raise ValueError("give delta or halfwidth")
        n_raw = (za + zb) ** 2 * V / delta ** 2; null = est - delta
        fl2 = r"n = (z_{1-\alpha/2}+z_{1-\beta})^2 V/\Delta^2"; fn2 = f"n = ({za:.6f}+{zb:.6f})^2\\times{V:.5f}/{delta}^2 = {n_raw:.2f}"
    n = _ceil(n_raw * deff)
    return Result(n=n, metric="net_benefit_standardised", group="A", estimate=est, null_value=null, V_alt=V,
                  inputs=dict(sens=sens, spec=spec, prevalence=prevalence, threshold=threshold, delta=delta, halfwidth=halfwidth),
                  steps=[("sNB", round(est, 4)), ("V (delta method on cells)", round(V, 5)), ("n", n)],
                  formula_latex=[r"sNB = \frac{p_{TP}-\frac{p_t}{1-p_t}p_{FP}}{p_{TP}+p_{FN}} = Se-\frac{p_t}{1-p_t}\frac{1-\phi}{\phi}(1-Sp),\qquad V=\nabla g^{\prime}\Sigma\nabla g", fl2],
                  formula_numeric=[f"sNB = {est:.4f},\\ V = {V:.5f}", fn2 + f" \\to {n}"],
                  reference=["riley2021", "delta_method"],
                  notes=["Riley 2021 criterion 5 treats Se, Sp and phi as independent binomials; the multinomial delta method gives the same leading term"])


# -----------------------------------------------------------------------------
# A9. Time-to-event: hazard ratio (Schoenfeld), survival probability
# -----------------------------------------------------------------------------

def hazard_ratio(hr: float, p_event_control: float, p_event_treat: Optional[float] = None, allocation: float = 0.5,
                 hr0: float = 1.0, alpha: float = 0.05, power: float = 0.80, two_sided: bool = True) -> Result:
    """Schoenfeld: events d = (z_a + z_b)^2 / (k(1-k) (log HR - log HR0)^2); n = d / overall event probability."""
    if p_event_treat is None:
        p_event_treat = 1 - (1 - p_event_control) ** hr    # proportional hazards
    za, zb = _zz(alpha, power, two_sided)
    k = allocation
    d = (za + zb) ** 2 / (k * (1 - k) * (math.log(hr) - math.log(hr0)) ** 2)
    pbar = k * p_event_treat + (1 - k) * p_event_control
    n_raw = d / pbar
    n = _ceil(n_raw)
    return Result(n=n, metric="hazard_ratio", group="A", estimate=hr, null_value=hr0, V_alt=1 / (k * (1 - k)),
                  inputs=dict(hr=hr, p_event_control=p_event_control, p_event_treat=p_event_treat, allocation=k),
                  steps=[("events required d", _ceil(d)), ("overall event probability", round(pbar, 4)), ("n", n)],
                  formula_latex=[r"d = \frac{(z_{1-\alpha/2}+z_{1-\beta})^2}{k(1-k)\,(\log HR-\log HR_0)^2},\qquad n = \frac{d}{k\,p_E+(1-k)\,p_C}"],
                  formula_numeric=[f"d = \\frac{{({za:.6f}+{zb:.6f})^2}}{{{k}(1-{k})(\\log {hr}-\\log {hr0})^2}} = {d:.2f}",
                                   f"n = {d:.2f}/{pbar:.4f} = {n_raw:.2f} \\to {n}"],
                  reference=["schoenfeld1983"], notes=["n is the TOTAL; cf. powerSurvEpi::ssizeCT.default (R)"])


def survival_probability(S: float, halfwidth: float, censoring: float = 0.0, conf_level: float = 0.95) -> Result:
    """Kaplan-Meier survival probability at a fixed time: with no censoring Greenwood reduces to S(1-S)/n.
    Censoring is handled crudely by inflating n by 1/(1 - censoring)."""
    z = float(norm.ppf(1 - (1 - conf_level) / 2))
    n_raw = z ** 2 * S * (1 - S) / halfwidth ** 2 / (1 - censoring)
    n = _ceil(n_raw)
    return Result(n=n, metric="survival_probability", group="A", estimate=S, V_alt=S * (1 - S),
                  inputs=dict(S=S, halfwidth=halfwidth, censoring=censoring),
                  steps=[("n", n)],
                  formula_latex=[r"\operatorname{Var}(\hat S)\approx\frac{S(1-S)}{n}\ \text{(Greenwood, no censoring)},\qquad n = \frac{z^2 S(1-S)}{w^2(1-c)}"],
                  formula_numeric=[f"n = {z:.6f}^2\\times{S}(1-{S})/({halfwidth}^2(1-{censoring})) = {n_raw:.2f} \\to {n}"],
                  reference=["greenwood1926"], notes=["crude: censoring inflates n by 1/(1-c); for a proper design use a log-rank / Schoenfeld calculation"])


# -----------------------------------------------------------------------------
# A10. Not ported: pointers
# -----------------------------------------------------------------------------

def froc(**kw) -> Result:
    return Result(n=None, metric="froc", group="A",
                  notes=["Not computable in closed form: FROC/AFROC sample size needs multi-reader multi-case "
                         "variance components from a pilot study (Obuchowski-Rockette / Dorfman-Berbaum-Metz).",
                         "R: RJafroc::SsSampleSizeKGivenJ(dataset = <pilot>, J = <readers>, effectSize = <delta>); "
                         "MRMCsamplesize plans without pilot data.",
                         "For a STANDALONE algorithm (no readers) the AFROC figure of merit is a Wilcoxon-type statistic; "
                         "auc(method='hanley') with n_1 = abnormal cases and n_0 = normal cases is a first approximation."],
                  reference=["chakraborty"])


def partial_auc(**kw) -> Result:
    return Result(n=None, metric="partial_auc", group="A",
                  notes=["Not ported. The partial-area variance (Obuchowski & McClish 1997) depends on the binormal "
                         "parameters and the FPR range; R: pROC::power.roc.test(<roc object with partial.auc set>)."],
                  reference=["obuchowski1997"])


# -----------------------------------------------------------------------------
# Registry (mirrors R/04_l4_wrap.R, with native functions attached)
# -----------------------------------------------------------------------------

GROUP_A_REGISTRY: Dict[str, dict] = {
    "accuracy":            dict(label="Accuracy", fn="proportion", r="presize::prec_prop", inputs="p1, p0 (or halfwidth)"),
    "ppv":                 dict(label="Positive predictive value", fn="proportion", r="presize::prec_prop", inputs="p1, p0"),
    "npv":                 dict(label="Negative predictive value", fn="proportion", r="presize::prec_prop", inputs="p1, p0"),
    "top_n_accuracy":      dict(label="Top-n accuracy", fn="proportion", r="presize::prec_prop", inputs="p1, p0"),
    "percent_agreement":   dict(label="Percent agreement / PPA / NPA", fn="proportion", r="presize::prec_prop", inputs="p1, p0"),
    "diagnostic_yield":    dict(label="Diagnostic yield", fn="proportion", r="presize::prec_prop", inputs="p1, p0"),
    "sensitivity":         dict(label="Sensitivity (with prevalence)", fn="sensitivity_specificity", r="MKpower::power.diagnostic.test; presize::prec_sens", inputs="value1, value0, prevalence"),
    "specificity":         dict(label="Specificity (with prevalence)", fn="sensitivity_specificity", r="MKpower::power.diagnostic.test; presize::prec_spec", inputs="value1, value0, prevalence"),
    "lr_positive":         dict(label="Positive likelihood ratio", fn="likelihood_ratio", r="presize::prec_pos_lr", inputs="sens, spec, prevalence, lr0"),
    "lr_negative":         dict(label="Negative likelihood ratio", fn="likelihood_ratio", r="presize::prec_neg_lr", inputs="sens, spec, prevalence, lr0"),
    "odds_ratio":          dict(label="Diagnostic odds ratio", fn="odds_ratio", r="presize::prec_or", inputs="p1, p2, or0"),
    "risk_difference":     dict(label="Risk difference / demographic parity / equalized odds", fn="risk_difference", r="presize::prec_riskdiff; pwr::pwr.2p.test", inputs="p1, p2, delta"),
    "auroc":               dict(label="AUROC / C-statistic (independent cases)", fn="auc", r="pROC::power.roc.test", inputs="auc1, auc0, ratio"),
    "partial_auc":         dict(label="Partial AUC", fn="partial_auc", r="pROC::power.roc.test (partial.auc)", inputs="not ported"),
    "kappa_unweighted":    dict(label="Cohen's kappa (unweighted)", fn="kappa_unweighted", r="kappaSize::PowerBinary", inputs="kappa1, kappa0, marginals"),
    "icc":                 dict(label="Intraclass correlation", fn="icc", r="ICC.Sample.Size::calculateIccSampleSize; presize::prec_icc", inputs="icc1, icc0, k"),
    "ccc":                 dict(label="Concordance correlation coefficient", fn="ccc", r="cccPower", inputs="ccc1, ccc0, r, u"),
    "pearson_correlation": dict(label="Pearson correlation", fn="pearson_correlation", r="pwr::pwr.r.test", inputs="r1, r0"),
    "calibration_slope":   dict(label="Calibration slope", fn="calibration_slope", r="pmvalsampsize", inputs="lp_mean, lp_sd, halfwidth"),
    "oe_ratio":            dict(label="Observed/expected ratio", fn="oe_ratio", r="pmvalsampsize", inputs="prevalence, ci_width"),
    "c_statistic":         dict(label="C-statistic (validation precision)", fn="c_statistic", r="pmvalsampsize", inputs="c, prevalence, halfwidth"),
    "net_benefit":         dict(label="Standardised net benefit", fn="standardised_net_benefit", r="pmvalsampsize", inputs="sens, spec, prevalence, threshold, delta"),
    "froc":                dict(label="FROC score", fn="froc", r="RJafroc", inputs="not ported (pointer)"),
    "afroc":               dict(label="AFROC AUC", fn="froc", r="RJafroc", inputs="not ported (pointer)"),
    "hazard_ratio":        dict(label="Hazard ratio", fn="hazard_ratio", r="powerSurvEpi::ssizeCT.default", inputs="hr, p_event_control, allocation"),
    "survival_probability": dict(label="Survival probability", fn="survival_probability", r="powerSurvEpi; survival", inputs="S, halfwidth"),
}


def risk_difference(p1: float, p2: float, delta: Optional[float] = None, kappa: float = 0.5, alpha: float = 0.05,
                    power: float = 0.80, two_sided: bool = True, deff: float = 1.0) -> Result:
    from .groupb import n_for_power_group_b
    r = n_for_power_group_b("risk_difference", delta=delta if delta is not None else abs(p1 - p2),
                            alpha=alpha, power=power, two_sided=two_sided, deff=deff, p1=p1, p2=p2, kappa=kappa)
    r.group = "A"; r.notes.append("cf. presize::prec_riskdiff / pwr::pwr.2p.test (R)")
    return r


_FN = dict(proportion=proportion, sensitivity_specificity=sensitivity_specificity, likelihood_ratio=likelihood_ratio,
           odds_ratio=odds_ratio, risk_difference=risk_difference, auc=auc, partial_auc=partial_auc,
           kappa_unweighted=kappa_unweighted, icc=icc, ccc=ccc, pearson_correlation=pearson_correlation,
           calibration_slope=calibration_slope, oe_ratio=oe_ratio, c_statistic=c_statistic,
           standardised_net_benefit=standardised_net_benefit, froc=froc, hazard_ratio=hazard_ratio,
           survival_probability=survival_probability)


def n_for_power_group_a(metric: str, **kw) -> Result:
    """Dispatcher. Fills `which`/`metric` for the shared binomial and LR functions."""
    e = GROUP_A_REGISTRY.get(metric)
    if e is None:
        raise KeyError(f"unknown Group A metric '{metric}'. Available: {', '.join(GROUP_A_REGISTRY)}")
    fn = _FN[e["fn"]]
    if e["fn"] == "proportion":
        kw.setdefault("metric", metric)
    if e["fn"] == "sensitivity_specificity":
        kw.setdefault("which", metric)
    if e["fn"] == "likelihood_ratio":
        kw.setdefault("which", "positive" if metric == "lr_positive" else "negative")
    r = fn(**kw)
    r.extra = dict(r.extra, r_package=e["r"])
    return r
