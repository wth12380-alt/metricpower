"""
Group B: metrics with a published or elementary variance that no package
inverts to a sample size. One dispatcher over five estimator families.

    n_for_power_group_b(metric, delta=..., power=..., **family_inputs)

Family inputs
    confusion_matrix : p (cell probabilities) [+ beta / c_fp,c_fn / threshold]
    multiclass       : P (k x k joint table)
    rate             : lam, ratio [, theta]
    percase          : x (pilot scores)  or  mean, sd
    other            : estimator-specific, see GROUP_B_INPUTS
"""
from __future__ import annotations

import math
from typing import Dict, List, Optional

import numpy as np
from scipy.stats import norm

from . import families as F
from .core import (n_for_power_t, CM_METRICS, Result, _z, kappa_w, kappa_w_var_fce, n_for_power,
                   n_for_power_metric, weights_identity, weights_linear, weights_quadratic)

GROUP_B_FAMILY: Dict[str, List[str]] = {
    "confusion_matrix": CM_METRICS,
    "multiclass": ["micro_precision", "micro_recall", "micro_f1",
                   "macro_precision", "macro_recall", "macro_f1"],
    "rate": ["fppi"],
    "percase": list(F.PERCASE_REGISTRY.keys()),
    "other": ["weighted_kappa", "entropy", "conditional_entropy", "coefficient_of_variation",
              "kendall_tau", "r_squared", "rand_index", "fowlkes_mallows",
              "repeatability_coefficient", "reproducibility_coefficient",
              "discrimination_slope", "risk_difference", "paired_proportion",
              "enhanced_detection_rate"],
    "refused": ["mahalanobis", "relative_standard_error", "timedep_auc"],
}

GROUP_B_INPUTS = {
    "confusion_matrix": "p (or sensitivity, specificity, prevalence), delta",
    "multiclass": "P (k x k joint table), delta",
    "rate": "lam, ratio [, theta]",
    "percase": "x (pilot per-case scores) or mean, sd; delta",
    "other": "estimator-specific",
    "refused": "refused with reason",
}


def family_of(metric: str) -> Optional[str]:
    for f, names in GROUP_B_FAMILY.items():
        if metric in names:
            return f
    return None


def group_b_manifest() -> List[dict]:
    return [dict(family=f, metric=m, inputs=GROUP_B_INPUTS[f])
            for f, names in GROUP_B_FAMILY.items() for m in names]


def _finish(metric, fam, est, V, delta, alpha, power, two_sided, deff, formula_latex, formula_numeric,
            refs, notes=None, inputs=None, steps=None, null_value=None, t_test=False):
    za, zb = _z(alpha, two_sided), float(norm.ppf(power))
    n_z = n_for_power(V, delta, alpha=alpha, power=power, two_sided=two_sided, deff=deff)
    raw = (za + zb) ** 2 * V / delta ** 2
    steps = list(steps or []) + [("V (per-unit variance)", round(V, 6)),
                                 ("z_{1-alpha/2}" if two_sided else "z_{1-alpha}", round(za, 6)),
                                 ("z_{1-beta}", round(zb, 6))]
    fl = list(formula_latex) + [r"n_z = \frac{(z_{1-\alpha/2}+z_{1-\beta})^2\,V}{\Delta^2}"]
    fn = list(formula_numeric) + [f"n_z = \\frac{{({za:.6f}+{zb:.6f})^2 \\times {V:.5f}}}{{{delta:.4f}^2}} = {raw:.2f} \\to {n_z if deff == 1 else int(math.ceil(raw - 1e-12))}"]
    extra = {"n_z": n_z}
    if t_test:
        # per-case mean: the analysis is a one-sample t-test with the sample SD, so size it with the
        # noncentral t (as power.t.test); matters below n ~ 50, where the z formula under-sizes
        n = n_for_power_t(V, delta, alpha=alpha, power=power, two_sided=two_sided, deff=deff)
        from scipy.stats import t as tdist
        ta, tb = float(tdist.ppf(1 - alpha / (2 if two_sided else 1), n - 1)), float(tdist.ppf(power, n - 1))
        steps += [("t_{n-1,1-alpha/2}", round(ta, 4)), ("t_{n-1,1-beta}", round(tb, 4)), ("n (z formula)", n_z), ("n (t-test, noncentral t)", n)]
        fl.append(r"n = \min\{n : P(|T_{n-1,\,\lambda}| > t_{n-1,1-\alpha/2}) \ge 1-\beta\},\quad \lambda = \Delta\sqrt{n/V}")
        fn.append(f"t_{{{n-1},0.975}} = {ta:.4f},\\quad n_z = {n_z} \\to n_t = {n}")
        if n > n_z:
            notes = list(notes or []) + [f"n uses the t-test (noncentral t) rather than the z formula ({n_z}): with the sample SD "
                                         "as SE the z critical value inflates the type I error at small n"]
    else:
        n = n_z
        steps.append(("n", n))
    if deff != 1:
        fl.append(r"n_{\text{clustered}} = n \times \text{deff}")
        fn.append(f"{math.ceil(raw - 1e-12)} \\times {deff:.3f} \\to {n}")
    return Result(n=n, metric=metric, group="B", estimate=est, null_value=null_value, V_alt=V, V_null=V,
                  inputs=inputs or {}, steps=steps, formula_latex=fl, formula_numeric=fn,
                  reference=refs, notes=notes or [], extra=extra)


def n_for_power_group_b(metric: str, *, delta: Optional[float] = None, alpha: float = 0.05,
                        power: float = 0.80, two_sided: bool = True, deff: float = 1.0,
                        **kw) -> Result:
    fam = family_of(metric)
    if fam is None:
        raise KeyError(f"unknown Group B metric '{metric}'. See group_b_manifest().")

    if fam == "refused":
        if metric == "mahalanobis":
            F.mahalanobis_sample_size()
        if metric == "timedep_auc":
            F.timedep_auc_sample_size()
        raise NotImplementedError("Relative standard error is a precision requirement, not a testable "
                                  "quantity. Use families.n_for_target_rse(V, estimate, target_rse).")

    if fam == "confusion_matrix":
        p = kw.pop("p", None)
        if p is None:
            try:
                from .core import cell_prob_from_operating_point
                p = cell_prob_from_operating_point(kw.pop("sensitivity"), kw.pop("specificity"), kw.pop("prevalence"))
            except KeyError:
                raise ValueError("family 'confusion_matrix' needs p = cell_prob(...) or sensitivity, specificity, prevalence")
        if delta is None:
            raise ValueError("delta is required")
        test = kw.pop("test", "wald"); null_path = kw.pop("null_path", "auto")
        return n_for_power_metric(p, metric, delta, alpha=alpha, power=power, two_sided=two_sided,
                                  deff=deff, test=test, null_path=null_path, **kw)

    if delta is None and fam != "rate":
        raise ValueError("delta is required")

    if fam == "multiclass":
        P = kw.get("P")
        if P is None:
            raise ValueError("family 'multiclass' needs P = <k x k joint table>")
        average, what = metric.split("_", 1)
        r = F.metric_variance_multiclass(P, what=what, average=average)
        return _finish(metric, fam, r["estimate"], r["V"], delta, alpha, power, two_sided, deff,
                       [r"\text{%s-%s on the } k\times k \text{ table},\quad V=\nabla g^{\prime}\Sigma\nabla g" % (average, what)],
                       [f"\\theta_1 = {r['estimate']:.4f},\\quad V = {r['V']:.5f}"],
                       ["delta_method", "takahashi2022"], inputs=dict(P=np.asarray(P).tolist(), delta=delta),
                       null_value=r["estimate"] - delta)

    if fam == "rate":
        lam = kw.get("lam"); ratio = kw.get("ratio"); theta = kw.get("theta", math.inf)
        if lam is None or ratio is None:
            raise ValueError("family 'rate' needs lam = <FP per image> and ratio = <rate ratio>")
        V_log = F.var_rate(lam, theta) / lam ** 2
        d = abs(math.log(ratio))
        vr = F.var_rate(lam, theta)
        r = _finish(metric, fam, lam, V_log, d, alpha, power, two_sided, deff,
                    [r"\operatorname{Var}(\hat\lambda)\,n = \lambda + \lambda^2/\theta\quad(\theta=\infty:\ \text{Poisson})",
                     r"V_{\log} = \frac{\lambda + \lambda^2/\theta}{\lambda^2},\qquad \Delta = |\log(\text{ratio})|"],
                    [f"\\lambda = {lam:.4f},\\ \\theta = {theta},\\quad \\lambda+\\lambda^2/\\theta = {vr:.4f}",
                     f"V_{{\\log}} = {vr:.4f}/{lam:.4f}^2 = {V_log:.5f},\\quad \\Delta = |\\log {ratio}| = {d:.4f}"],
                    ["poisson_nb"], inputs=dict(lam=lam, ratio=ratio, theta=theta), null_value=lam * ratio)
        r.notes.append("n is the number of images; the test is on log(lambda)")
        return r

    if fam == "percase":
        e = F.PERCASE_REGISTRY[metric]
        x = kw.get("x"); mean = kw.get("mean"); sd = kw.get("sd")
        r = F.var_percase_transformed(x, e["transform"], mean=mean, sd=sd)
        notes = [e["note"]]
        if not math.isnan(r["skew"]) and abs(r["skew"]) > 1.0:
            notes.append(f"pilot scores have skew {r['skew']:.2f}: Wald coverage is below nominal at small n "
                         "and the misses are one-sided; see sensitivity_by_n_percase()")
        fl = [r"\operatorname{Var}(\bar x) = \sigma^2/n\ \text{(exact for any distribution)}"]
        fn = [f"\\mu = {r['mean']:.4f},\\quad \\sigma^2 = {r['V_mean']:.5f}"]
        if e["transform"] != "identity":
            fl.append(r"V = \sigma^2\,[h'(\mu)]^2,\quad h = " + (r"\sqrt{\cdot}" if e["transform"] == "sqrt" else r"\exp(\cdot)"))
            fn.append(f"V = {r['V_mean']:.5f} \\times {r['dg']:.5f}^2 = {r['V']:.5f}")
        return _finish(metric, fam, r["estimate"], r["V"], delta, alpha, power, two_sided, deff, fl, fn,
                       ["clt_mean", "bradley2008"] if metric in ("brier", "root_brier") else ["clt_mean"],
                       notes=notes, inputs=dict(mean=r["mean"], sd=math.sqrt(r["V_mean"]), delta=delta, transform=e["transform"]),
                       null_value=r["estimate"] - delta, t_test=True)

    # ---- other ----
    if metric == "weighted_kappa":
        P = np.asarray(kw["P"], dtype=float); k = P.shape[0]
        wname = kw.get("weights", "linear")
        W = {"identity": weights_identity, "linear": weights_linear, "quadratic": weights_quadratic}[wname](k) \
            if isinstance(wname, str) else np.asarray(wname, dtype=float)
        est = kappa_w(P, W); V = kappa_w_var_fce(P, W)
        return _finish(metric, fam, est, V, delta, alpha, power, two_sided, deff,
                       [r"\kappa_w = \frac{p_o - p_e}{1 - p_e},\quad p_o=\sum w_{ij}p_{ij},\ p_e=\sum w_{ij}p_{i\cdot}p_{\cdot j}",
                        r"V = \frac{\sum_{ij} p_{ij}\left[w_{ij}(1-p_e)-(\bar w_{i\cdot}+\bar w_{\cdot j})(1-p_o)\right]^2-(p_o p_e-2p_e+p_o)^2}{(1-p_e)^4}"],
                       [f"\\kappa_w = {est:.4f},\\quad V = {V:.5f}"], ["fleiss1969"],
                       inputs=dict(P=P.tolist(), weights=wname if isinstance(wname, str) else "custom", delta=delta),
                       null_value=est - delta)
    if metric == "entropy":
        r = F.entropy_variance(kw["p"])
        res = _finish(metric, fam, r["estimate"], r["V"], delta, alpha, power, two_sided, deff,
                      [r"H = -\sum p_i\log p_i,\quad V = \sum p_i(\log p_i)^2-\left(\sum p_i\log p_i\right)^2"],
                      [f"H = {r['estimate']:.4f},\\quad V = {r['V']:.5f}"], ["basharin1959", "delta_method"],
                      inputs=dict(p=list(kw["p"]), delta=delta), null_value=r["estimate"] - delta)
        if r["degenerate"]:
            res.notes.append("entropy is at its maximum (uniform): V = 0, the limit is chi-square, n is invalid")
        return res
    if metric == "conditional_entropy":
        r = F.conditional_entropy_variance(kw["P"])
        return _finish(metric, fam, r["estimate"], r["V"], delta, alpha, power, two_sided, deff,
                       [r"H(Y|X) = -\sum_{ij} p_{ij}\log\frac{p_{ij}}{p_{i\cdot}},\quad V=\nabla g^{\prime}\Sigma\nabla g"],
                       [f"H(Y|X) = {r['estimate']:.4f},\\quad V = {r['V']:.5f}"], ["delta_method", "basharin1959"],
                       inputs=dict(P=np.asarray(kw["P"]).tolist(), delta=delta), null_value=r["estimate"] - delta)
    if metric == "coefficient_of_variation":
        r = F.cv_variance(kw.get("x"), kw.get("cv"))
        return _finish(metric, fam, r["estimate"], r["V"], delta, alpha, power, two_sided, deff,
                       [r"V = cv^2\,(0.5 + cv^2)\quad\text{(normal theory)}"],
                       [f"V = {r['estimate']:.4f}^2\\,(0.5+{r['estimate']:.4f}^2) = {r['V']:.5f}"], ["cv_normal"],
                       notes=[r["note"]], inputs=dict(cv=r["estimate"], delta=delta), null_value=r["estimate"] - delta)
    if metric == "kendall_tau":
        if "x" in kw:
            r = F.kendall_tau_variance(kw["x"], kw["y"]); est = r["estimate"]; V = r["V"]
            fn = [f"\\hat\\tau = {est:.4f},\\quad \\hat\\zeta_1 = {r['zeta1']:.5f},\\quad V = 4\\hat\\zeta_1 = {V:.5f}"]
            notes = []
        else:
            est = kw.get("tau", 0.0); V = 4 / 9
            fn = [f"\\tau = {est:.4f},\\quad V = 4/9 = {V:.5f}\\ \\text{{(independence value)}}"]
            notes = ["without pilot data V is taken at its value under independence, 4/9; "
                     "with dependence V is smaller, so this n is conservative"]
        return _finish(metric, fam, est, V, delta, alpha, power, two_sided, deff,
                       [r"\sqrt n(\hat\tau-\tau)\to N(0,4\zeta_1),\quad \zeta_1=\operatorname{Var}\,E[h(Z,Z')\mid Z]"],
                       fn, ["kendall_hoeffding", "hoeffding1948"], notes=notes,
                       inputs=dict(tau=est, delta=delta), null_value=est - delta)
    if metric in ("rand_index", "fowlkes_mallows"):
        r = F.cluster_index_variance(kw["a"], kw["b"], "rand" if metric == "rand_index" else "fowlkes_mallows")
        return _finish(metric, fam, r["estimate"], r["V"], delta, alpha, power, two_sided, deff,
                       [r"\text{degree-2 U-statistic over item pairs},\quad V = 4\zeta_1"],
                       [f"\\hat\\theta = {r['estimate']:.4f},\\quad \\hat\\zeta_1 = {r['zeta1']:.5f},\\quad V = {r['V']:.5f}"],
                       ["rand_ustat", "hoeffding1948"], inputs=dict(delta=delta), null_value=r["estimate"] - delta)
    if metric == "r_squared":
        r = F.r_squared_variance(kw["r2"])
        res = _finish(metric, fam, r["estimate"], r["V"], delta, alpha, power, two_sided, deff,
                      [r"V = 4\rho^2(1-\rho^2)^2"], [f"V = 4\\times{r['estimate']:.4f}\\times(1-{r['estimate']:.4f})^2 = {r['V']:.5f}"],
                      ["r_squared"], inputs=dict(r2=kw["r2"], delta=delta), null_value=r["estimate"] - delta)
        if r["degenerate"]:
            res.notes.append("R-squared near zero: V vanishes, the limit is chi-square, n is invalid")
        return res
    if metric in ("repeatability_coefficient", "reproducibility_coefficient"):
        m = kw.get("replicates", 2); r = F.repeatability_variance(kw["sigma_w"], m)
        return _finish(metric, fam, r["estimate"], r["V"], delta, alpha, power, two_sided, deff,
                       [r"RC = 1.96\sqrt2\,\sigma_w,\qquad V = \frac{(1.96\sqrt2)^2\sigma_w^2}{2(m-1)}"],
                       [f"RC = {r['estimate']:.4f},\\quad V = \\frac{{(1.96\\sqrt2)^2\\times{kw['sigma_w']:.4f}^2}}{{2({m}-1)}} = {r['V']:.5f}"],
                       ["repeatability"], inputs=dict(sigma_w=kw["sigma_w"], replicates=m, delta=delta),
                       null_value=r["estimate"] - delta)
    if metric == "discrimination_slope":
        r = F.discrimination_slope_variance(kw["mu1"], kw["mu0"], kw["sd1"], kw["sd0"], kw["prevalence"])
        return _finish(metric, fam, r["estimate"], r["V"], delta, alpha, power, two_sided, deff,
                       [r"DS = \mu_1-\mu_0,\qquad V = \frac{\sigma_1^2}{\phi}+\frac{\sigma_0^2}{1-\phi}"],
                       [f"V = {kw['sd1']:.4f}^2/{kw['prevalence']:.3f} + {kw['sd0']:.4f}^2/{1 - kw['prevalence']:.3f} = {r['V']:.5f}"],
                       ["two_proportions"], inputs=dict(delta=delta, **{k: kw[k] for k in ('mu1', 'mu0', 'sd1', 'sd0', 'prevalence')}),
                       null_value=r["estimate"] - delta)
    if metric == "risk_difference":
        kap = kw.get("kappa", 0.5); r = F.risk_difference_variance(kw["p1"], kw["p2"], kap)
        res = _finish(metric, fam, r["estimate"], r["V"], delta, alpha, power, two_sided, deff,
                      [r"V = \frac{p_1(1-p_1)}{\kappa}+\frac{p_2(1-p_2)}{1-\kappa}\quad(\kappa=\text{share in group 1})"],
                      [f"V = {kw['p1']:.3f}(1-{kw['p1']:.3f})/{kap:.2f} + {kw['p2']:.3f}(1-{kw['p2']:.3f})/{1 - kap:.2f} = {r['V']:.5f}"],
                      ["two_proportions"], inputs=dict(p1=kw["p1"], p2=kw["p2"], kappa=kap, delta=delta),
                      null_value=r["estimate"] - delta)
        res.notes.append("n is the TOTAL across both groups")
        return res
    if metric in ("paired_proportion", "enhanced_detection_rate"):
        r = F.paired_proportion_variance(kw["p01"], kw["p10"])
        return _finish(metric, fam, r["estimate"], r["V"], delta, alpha, power, two_sided, deff,
                       [r"d = p_{10}-p_{01},\qquad V = p_{01}+p_{10}-d^2"],
                       [f"V = {kw['p01']:.4f}+{kw['p10']:.4f}-{r['estimate']:.4f}^2 = {r['V']:.5f}"],
                       ["mcnemar"], notes=[r["note"]], inputs=dict(p01=kw["p01"], p10=kw["p10"], delta=delta),
                       null_value=r["estimate"] - delta)
    raise KeyError(metric)
