"""
Input specifications for the Shiny app: for every metric, which inputs to
show, their defaults, and how to turn the values into a package call.

Each spec: dict(group, label, family, fields=[(id, label, default, kwargs)], build=fn)
build(vals, common) -> Result, where common = dict(alpha, power, two_sided, deff, test)
"""
from __future__ import annotations

import math
import numpy as np

from .. import core, families as F
from ..groupa import GROUP_A_REGISTRY, n_for_power_group_a
from ..groupb import GROUP_B_FAMILY, n_for_power_group_b
from ..groupc import GROUP_C, explain_group_c

NUM = "numeric"


def _f(id_, label, default, min=None, max=None, step=None, kind=NUM, choices=None, help=None):
    return dict(id=id_, label=label, default=default, min=min, max=max, step=step, kind=kind, choices=choices, help=help)


def _parse_table(txt: str) -> np.ndarray:
    rows = [r for r in txt.replace(";", "\n").strip().splitlines() if r.strip()]
    M = np.array([[float(v) for v in r.replace(",", " ").split()] for r in rows], dtype=float)
    if M.ndim != 2 or M.shape[0] != M.shape[1]:
        raise ValueError("the table must be square, one row per line")
    return M / M.sum()


def _parse_vec(txt: str) -> np.ndarray:
    v = np.array([float(x) for x in txt.replace(",", " ").split()], dtype=float)
    return v / v.sum()


# ---------------------------------------------------------------- Group B: confusion matrix
OP_FIELDS = [_f("sens", "Sensitivity at the operating point (H1)", 0.70, 0.01, 0.99, 0.01,
                help="the sensitivity you expect the model to achieve (from a pilot or the literature)"),
             _f("spec", "Specificity at the operating point (H1)", 0.80, 0.01, 0.99, 0.01,
                help="the specificity you expect at the same threshold"),
             _f("prev", "Prevalence", 0.50, 0.01, 0.99, 0.01,
                help="fraction of positive cases in the test set you will collect, not in the population")]

EXTRA_PARAMS = {
    "fbeta": [_f("beta", "beta", 2.0, 0.1, 10, 0.1)],
    "expected_cost": [_f("c_fp", "cost of a false positive", 1.0, 0, None, 0.5), _f("c_fn", "cost of a false negative", 5.0, 0, None, 0.5)],
    "net_benefit": [_f("threshold", "risk threshold p_t", 0.2, 0.01, 0.99, 0.01)],
}


def _cm_params(m, vals):
    return {f["id"]: vals[f["id"]] for f in EXTRA_PARAMS.get(m, [])}


def cm_value(m, vals):
    p = core.cell_prob_from_operating_point(vals["sens"], vals["spec"], vals["prev"])
    return float(core.get_metric(m, **_cm_params(m, vals))(p))


def _build_cm(m):
    def build(vals, common):
        p = core.cell_prob_from_operating_point(vals["sens"], vals["spec"], vals["prev"])
        theta1 = cm_value(m, vals)
        delta = theta1 - vals["null"]
        kw = dict(alpha=common["alpha"], power=common["power"], two_sided=common["two_sided"], deff=common["deff"])
        try:
            return core.n_for_power_metric(p, m, delta, test=common["test"], **kw, **_cm_params(m, vals))
        except ValueError as e:
            if common["test"] != "score" or "not reachable" not in str(e):
                raise
            r = core.n_for_power_metric(p, m, delta, test="wald", **kw, **_cm_params(m, vals))
            r.notes.insert(0, "the score n is unavailable here (the null value cannot be reached by moving "
                              "sensitivity or specificity at this prevalence), so the Wald n is shown")
            return r
    return build


SPECS: dict = {}
for m in GROUP_B_FAMILY["confusion_matrix"]:
    SPECS[m] = dict(group="B", label=core.METRIC_REGISTRY[m]["label"] + " · from sensitivity/specificity/prevalence (delta method)", family="confusion_matrix",
                    fields=OP_FIELDS + EXTRA_PARAMS.get(m, []) + [_f("null", "Null (H0) value of the metric", None, help="the value you want to rule out, e.g. the benchmark to beat or the lowest acceptable performance; n grows with 1/Δ²")],
                    build=_build_cm(m), null_from_h1=lambda v, m=m: round(cm_value(m, v) - 0.05, 4))

# ---------------------------------------------------------------- Group B: multiclass
for m in GROUP_B_FAMILY["multiclass"]:
    avg, what = m.split("_", 1)
    SPECS[m] = dict(group="B", label=f"{avg.capitalize()}-averaged {what} (k classes)", family="multiclass",
                    fields=[_f("P", "k x k joint table (rows = truth, columns = prediction; one row per line)",
                               "0.30 0.04 0.02\n0.05 0.22 0.05\n0.03 0.06 0.23", kind="textarea"),
                            _f("delta", "Difference to detect Δ", 0.05, 0.001, None, 0.005)],
                    build=lambda vals, common, m=m: n_for_power_group_b(m, P=_parse_table(vals["P"]), delta=vals["delta"], **_c(common)))

# ---------------------------------------------------------------- Group B: rate
SPECS["fppi"] = dict(group="B", label="False positives per image (count rate)", family="rate",
                     fields=[_f("lam", "FP per image under H1 (λ)", 0.5, 0.001, None, 0.05),
                             _f("lam0", "FP per image under H0", 0.7, 0.001, None, 0.05),
                             _f("theta", "Negative-binomial dispersion θ (leave blank for Poisson)", None, 0.01, None, 0.5)],
                     build=lambda vals, common: n_for_power_group_b("fppi", lam=vals["lam"], ratio=vals["lam0"] / vals["lam"],
                                                                     theta=vals["theta"] if vals["theta"] else math.inf, **_c(common)))

# ---------------------------------------------------------------- Group B: per-case
for m in GROUP_B_FAMILY["percase"]:
    SPECS[m] = dict(group="B", label=f"{m} (per-case mean) — {F.PERCASE_REGISTRY[m]['note']}", family="percase",
                    fields=[_f("mean", "Mean per-case score under H1", 0.80, None, None, 0.01),
                            _f("sd", "SD of the per-case scores (pilot)", 0.12, 0.0001, None, 0.01),
                            _f("null", "Mean under H0", 0.77, None, None, 0.01)],
                    build=lambda vals, common, m=m: n_for_power_group_b(m, mean=vals["mean"], sd=vals["sd"],
                                                                         delta=_delta_transformed(m, vals), **_c(common)))


def _delta_transformed(m, vals):
    t = F.PERCASE_REGISTRY[m]["transform"]
    h = {"identity": lambda x: x, "sqrt": math.sqrt, "exp": math.exp}[t]
    return h(vals["mean"]) - h(vals["null"])


# ---------------------------------------------------------------- Group B: other
def _c(common):
    return dict(alpha=common["alpha"], power=common["power"], two_sided=common["two_sided"], deff=common["deff"])


SPECS["weighted_kappa"] = dict(group="B", label="Weighted Cohen's kappa", family="other",
    fields=[_f("P", "k x k joint rater table (one row per line)", "0.30 0.04 0.02\n0.05 0.22 0.05\n0.03 0.06 0.23", kind="textarea"),
            _f("weights", "weights", "linear", kind="select", choices=["identity", "linear", "quadratic"]),
            _f("delta", "Difference to detect Δ", 0.05, 0.001, None, 0.005)],
    build=lambda v, c: n_for_power_group_b("weighted_kappa", P=_parse_table(v["P"]), weights=v["weights"], delta=v["delta"], **_c(c)))
SPECS["entropy"] = dict(group="B", label="Shannon entropy", family="other",
    fields=[_f("p", "class probabilities (comma separated)", "0.5, 0.3, 0.15, 0.05", kind="text"),
            _f("delta", "Difference to detect Δ (nats)", 0.1, 0.001, None, 0.01)],
    build=lambda v, c: n_for_power_group_b("entropy", p=_parse_vec(v["p"]), delta=v["delta"], **_c(c)))
SPECS["conditional_entropy"] = dict(group="B", label="Conditional entropy H(Y|X)", family="other",
    fields=[_f("P", "k x k joint table (one row per line)", "0.30 0.04 0.02\n0.05 0.22 0.05\n0.03 0.06 0.23", kind="textarea"),
            _f("delta", "Difference to detect Δ (nats)", 0.1, 0.001, None, 0.01)],
    build=lambda v, c: n_for_power_group_b("conditional_entropy", P=_parse_table(v["P"]), delta=v["delta"], **_c(c)))
SPECS["coefficient_of_variation"] = dict(group="B", label="Coefficient of variation", family="other",
    fields=[_f("cv", "CV under H1", 0.25, 0.001, None, 0.01), _f("null", "CV under H0", 0.30, 0.001, None, 0.01)],
    build=lambda v, c: n_for_power_group_b("coefficient_of_variation", cv=v["cv"], delta=v["cv"] - v["null"], **_c(c)))
SPECS["kendall_tau"] = dict(group="B", label="Kendall's tau (no pilot data: V at independence)", family="other",
    fields=[_f("tau", "tau under H1", 0.30, -0.99, 0.99, 0.01), _f("null", "tau under H0", 0.0, -0.99, 0.99, 0.01)],
    build=lambda v, c: n_for_power_group_b("kendall_tau", tau=v["tau"], delta=v["tau"] - v["null"], **_c(c)))
SPECS["r_squared"] = dict(group="B", label="R-squared", family="other",
    fields=[_f("r2", "R² under H1", 0.49, 0.0, 0.999, 0.01), _f("null", "R² under H0", 0.39, 0.0, 0.999, 0.01)],
    build=lambda v, c: n_for_power_group_b("r_squared", r2=v["r2"], delta=v["r2"] - v["null"], **_c(c)))
SPECS["repeatability_coefficient"] = dict(group="B", label="Repeatability / reproducibility coefficient", family="other",
    fields=[_f("sigma_w", "within-subject SD σ_w (H1)", 0.5, 0.0001, None, 0.05), _f("replicates", "replicates per subject m", 2, 2, 20, 1),
            _f("null", "RC under H0", 1.6, 0.0001, None, 0.05)],
    build=lambda v, c: n_for_power_group_b("repeatability_coefficient", sigma_w=v["sigma_w"], replicates=int(v["replicates"]),
                                           delta=1.96 * math.sqrt(2) * v["sigma_w"] - v["null"], **_c(c)))
SPECS["discrimination_slope"] = dict(group="B", label="Discrimination slope", family="other",
    fields=[_f("mu1", "mean predicted risk, cases", 0.7, 0, 1, 0.01), _f("mu0", "mean predicted risk, controls", 0.3, 0, 1, 0.01),
            _f("sd1", "SD, cases", 0.2, 0.001, 1, 0.01), _f("sd0", "SD, controls", 0.2, 0.001, 1, 0.01),
            _f("prevalence", "prevalence", 0.4, 0.01, 0.99, 0.01), _f("null", "slope under H0", 0.35, None, None, 0.01)],
    build=lambda v, c: n_for_power_group_b("discrimination_slope", mu1=v["mu1"], mu0=v["mu0"], sd1=v["sd1"], sd0=v["sd0"],
                                           prevalence=v["prevalence"], delta=(v["mu1"] - v["mu0"]) - v["null"], **_c(c)))
SPECS["risk_difference"] = dict(group="B", label="Risk difference (two independent groups; also demographic parity, equalized odds)", family="other",
    fields=[_f("p1", "proportion in group 1 (H1)", 0.60, 0.01, 0.99, 0.01), _f("p2", "proportion in group 2 (H1)", 0.45, 0.01, 0.99, 0.01),
            _f("kappa", "share of the sample in group 1", 0.5, 0.05, 0.95, 0.05), _f("null", "difference under H0", 0.0, None, None, 0.01)],
    build=lambda v, c: n_for_power_group_b("risk_difference", p1=v["p1"], p2=v["p2"], kappa=v["kappa"], delta=(v["p1"] - v["p2"]) - v["null"], **_c(c)))
SPECS["paired_proportion"] = dict(group="B", label="Paired proportions / enhanced detection rate (McNemar)", family="other",
    fields=[_f("p01", "P(detected without AI only) p01", 0.05, 0.001, 0.99, 0.01), _f("p10", "P(detected with AI only) p10", 0.12, 0.001, 0.99, 0.01),
            _f("null", "difference under H0", 0.0, None, None, 0.01)],
    build=lambda v, c: n_for_power_group_b("paired_proportion", p01=v["p01"], p10=v["p10"], delta=(v["p10"] - v["p01"]) - v["null"], **_c(c)))

# ---------------------------------------------------------------- Group A
def _ga(metric, **fixed):
    return lambda v, c: n_for_power_group_a(metric, **fixed, **{k: v[k] for k in v if k not in ("_",)}, **_c(c))


for m in ("accuracy", "ppv", "npv", "top_n_accuracy", "percent_agreement", "diagnostic_yield"):
    SPECS["A_" + m] = dict(group="A", label=GROUP_A_REGISTRY[m]["label"] + " · enter the proportion directly (binomial)", family="A",
        fields=[_f("p1", "proportion under H1", 0.85, 0.01, 0.99, 0.01), _f("p0", "proportion under H0", 0.80, 0.01, 0.99, 0.01)],
        build=lambda v, c, m=m: n_for_power_group_a(m, p1=v["p1"], p0=v["p0"], method=c["test"] if c["test"] in ("wald", "score") else "score", **_c(c)))
for m in ("sensitivity", "specificity"):
    SPECS["A_" + m] = dict(group="A", label=GROUP_A_REGISTRY[m]["label"] + " · enter the value directly (Buderer, binomial ÷ prevalence)", family="A",
        fields=[_f("value1", f"{m} under H1", 0.85, 0.01, 0.99, 0.01), _f("value0", f"{m} under H0", 0.75, 0.01, 0.99, 0.01),
                _f("prevalence", "prevalence", 0.30, 0.01, 0.99, 0.01)],
        build=lambda v, c, m=m: n_for_power_group_a(m, value1=v["value1"], value0=v["value0"], prevalence=v["prevalence"], method=c["test"], **_c(c)))
for m in ("lr_positive", "lr_negative"):
    SPECS[m] = dict(group="A", label=GROUP_A_REGISTRY[m]["label"] + " (Simel)", family="A",
        fields=[_f("sens", "sensitivity", 0.80, 0.01, 0.99, 0.01), _f("spec", "specificity", 0.90, 0.01, 0.99, 0.01),
                _f("prevalence", "prevalence", 0.30, 0.01, 0.99, 0.01), _f("lr0", "LR under H0", 4.0 if m == "lr_positive" else 0.4, 0.001, None, 0.1)],
        build=lambda v, c, m=m: n_for_power_group_a(m, sens=v["sens"], spec=v["spec"], prevalence=v["prevalence"], lr0=v["lr0"], **_c(c)))
SPECS["odds_ratio"] = dict(group="A", label="Diagnostic odds ratio (Woolf)", family="A",
    fields=[_f("p1", "proportion in group 1", 0.60, 0.01, 0.99, 0.01), _f("p2", "proportion in group 2", 0.40, 0.01, 0.99, 0.01),
            _f("kappa", "share in group 1", 0.5, 0.05, 0.95, 0.05), _f("or0", "OR under H0", 1.0, 0.01, None, 0.1)],
    build=lambda v, c: n_for_power_group_a("odds_ratio", p1=v["p1"], p2=v["p2"], kappa=v["kappa"], or0=v["or0"], **_c(c)))
SPECS["auroc"] = dict(group="A", label="AUROC (Hanley–McNeil or Obuchowski binormal)", family="A",
    fields=[_f("auc1", "AUC under H1", 0.80, 0.5, 0.999, 0.01), _f("auc0", "AUC under H0", 0.70, 0.5, 0.999, 0.01),
            _f("ratio", "negatives per positive (κ)", 1.0, 0.1, 20, 0.1),
            _f("method", "variance", "hanley", kind="select", choices=["hanley", "obuchowski"])],
    build=lambda v, c: n_for_power_group_a("auroc", auc1=v["auc1"], auc0=v["auc0"], ratio=v["ratio"], method=v["method"], **_c(c)))
SPECS["kappa_unweighted"] = dict(group="A", label="Cohen's kappa, unweighted (two raters)", family="A",
    fields=[_f("kappa1", "kappa under H1", 0.60, -0.99, 0.99, 0.01), _f("kappa0", "kappa under H0", 0.40, -0.99, 0.99, 0.01),
            _f("marginals", "category probabilities (comma separated)", "0.3, 0.7", kind="text")],
    build=lambda v, c: n_for_power_group_a("kappa_unweighted", kappa1=v["kappa1"], kappa0=v["kappa0"], marginals=_parse_vec(v["marginals"]), **_c(c)))
SPECS["icc"] = dict(group="A", label="Intraclass correlation coefficient", family="A",
    fields=[_f("icc1", "ICC under H1", 0.80, 0.0, 0.999, 0.01), _f("icc0", "ICC under H0", 0.60, 0.0, 0.999, 0.01), _f("k", "ratings per subject k", 2, 2, 50, 1)],
    build=lambda v, c: n_for_power_group_a("icc", icc1=v["icc1"], icc0=v["icc0"], k=int(v["k"]), **_c(c)))
SPECS["ccc"] = dict(group="A", label="Lin's concordance correlation coefficient", family="A",
    fields=[_f("ccc1", "CCC under H1", 0.90, 0.0, 0.999, 0.01), _f("ccc0", "CCC under H0", 0.80, 0.0, 0.999, 0.01),
            _f("r", "Pearson r between methods", 0.92, 0.01, 0.999, 0.01), _f("u", "standardised mean shift u", 0.1, 0, None, 0.05)],
    build=lambda v, c: n_for_power_group_a("ccc", ccc1=v["ccc1"], ccc0=v["ccc0"], r=v["r"], u=v["u"], **_c(c)))
SPECS["pearson_correlation"] = dict(group="A", label="Pearson correlation (Fisher z)", family="A",
    fields=[_f("r1", "r under H1", 0.30, -0.999, 0.999, 0.01), _f("r0", "r under H0", 0.0, -0.999, 0.999, 0.01)],
    build=lambda v, c: n_for_power_group_a("pearson_correlation", r1=v["r1"], r0=v["r0"], **_c(c)))
SPECS["calibration_slope"] = dict(group="A", label="Calibration slope (Riley 2021, precision)", family="A",
    fields=[_f("lp_mean", "mean of the linear predictor", -1.5, None, None, 0.1), _f("lp_sd", "SD of the linear predictor", 1.2, 0.01, None, 0.1),
            _f("halfwidth", "CI half-width for the slope", 0.10, 0.005, None, 0.01)],
    build=lambda v, c: n_for_power_group_a("calibration_slope", lp_mean=v["lp_mean"], lp_sd=v["lp_sd"], halfwidth=v["halfwidth"], conf_level=1 - c["alpha"], deff=c["deff"]))
SPECS["oe_ratio"] = dict(group="A", label="Observed/expected ratio (Riley 2021, precision)", family="A",
    fields=[_f("prevalence", "outcome prevalence", 0.20, 0.001, 0.999, 0.01), _f("ci_width", "CI width on the O/E scale", 0.20, 0.01, None, 0.05)],
    build=lambda v, c: n_for_power_group_a("oe_ratio", prevalence=v["prevalence"], ci_width=v["ci_width"], conf_level=1 - c["alpha"], deff=c["deff"]))
SPECS["c_statistic"] = dict(group="A", label="C-statistic for validation (Riley 2021, precision)", family="A",
    fields=[_f("c", "anticipated C-statistic", 0.80, 0.5, 0.999, 0.01), _f("prevalence", "outcome prevalence", 0.20, 0.001, 0.999, 0.01),
            _f("halfwidth", "CI half-width", 0.05, 0.005, None, 0.005)],
    build=lambda v, c: n_for_power_group_a("c_statistic", c=v["c"], prevalence=v["prevalence"], halfwidth=v["halfwidth"], conf_level=1 - c["alpha"], deff=c["deff"]))
SPECS["net_benefit_standardised"] = dict(group="A", label="Standardised net benefit (Riley 2021)", family="A",
    fields=[_f("sens", "sensitivity at threshold", 0.80, 0.01, 0.99, 0.01), _f("spec", "specificity at threshold", 0.70, 0.01, 0.99, 0.01),
            _f("prevalence", "prevalence", 0.20, 0.01, 0.99, 0.01), _f("threshold", "risk threshold p_t", 0.10, 0.01, 0.99, 0.01),
            _f("null", "sNB under H0", 0.60, None, None, 0.01)],
    build=lambda v, c: n_for_power_group_a("net_benefit", sens=v["sens"], spec=v["spec"], prevalence=v["prevalence"], threshold=v["threshold"],
                                           delta=_snb(v) - v["null"], **_c(c)))


def _snb(v):
    w = v["threshold"] / (1 - v["threshold"])
    return v["sens"] - w * (1 - v["prevalence"]) / v["prevalence"] * (1 - v["spec"])


SPECS["hazard_ratio"] = dict(group="A", label="Hazard ratio (Schoenfeld)", family="A",
    fields=[_f("hr", "hazard ratio under H1", 0.70, 0.01, 10, 0.05), _f("hr0", "hazard ratio under H0", 1.0, 0.01, 10, 0.05),
            _f("p_event_control", "event probability, control arm", 0.50, 0.01, 0.99, 0.01), _f("allocation", "share allocated to treatment", 0.5, 0.05, 0.95, 0.05)],
    build=lambda v, c: n_for_power_group_a("hazard_ratio", hr=v["hr"], hr0=v["hr0"], p_event_control=v["p_event_control"], allocation=v["allocation"],
                                           alpha=c["alpha"], power=c["power"], two_sided=c["two_sided"]))
SPECS["survival_probability"] = dict(group="A", label="Survival probability at a fixed time (precision)", family="A",
    fields=[_f("S", "survival probability", 0.70, 0.01, 0.99, 0.01), _f("halfwidth", "CI half-width", 0.05, 0.005, None, 0.005),
            _f("censoring", "fraction censored", 0.0, 0, 0.95, 0.05)],
    build=lambda v, c: n_for_power_group_a("survival_probability", S=v["S"], halfwidth=v["halfwidth"], censoring=v["censoring"], conf_level=1 - c["alpha"]))

# ---------------------------------------------------------------- problem categories for the drop-down
# Every metric the app can size, placed in the problem category it belongs to. A metric that belongs to
# more than one category is listed under each. Category 11 is not in the source taxonomy; see CATEGORY_NOTES.
CATEGORIES = [
    (1, "1. Confusion-matrix-based classification metrics",
     ["accuracy", "A_accuracy", "A_top_n_accuracy", "sensitivity", "A_sensitivity", "specificity", "A_specificity",
      "ppv", "A_ppv", "npv", "A_npv", "balanced_accuracy", "balanced_error_rate", "youden", "markedness",
      "dsc_f1", "fbeta", "mcc", "micro_precision", "micro_recall", "micro_f1", "macro_precision", "macro_recall", "macro_f1",
      "class_error", "false_referral_rate", "false_discovery_rate", "false_omission_rate",
      "fppi", "odds_ratio", "lr_positive", "lr_negative"]),
    (2, "2. Precision-recall and ranking metrics", ["mrr"]),
    (3, "3. ROC / AUC family", ["auroc", "c_statistic"]),
    (4, "4. Calibration and probabilistic metrics",
     ["brier", "root_brier", "nll", "cross_entropy", "perplexity", "calibration_slope", "oe_ratio",
      "macro_mean_prob", "expected_cost"]),
    (5, "5. Segmentation and overlap metrics",
     ["dsc_f1", "dice_percase", "iou", "iou_percase", "volumetric_similarity", "ior"]),
    (6, "6. Distance / geometry metrics", ["assd_percase", "hd95_percase", "nsd_percase"]),
    (7, "7. Regression / error metrics", ["mae", "mse", "rmse", "mape", "smape", "mase", "ensemble_mse"]),
    (8, "8. Correlation, agreement, reliability",
     ["pearson_correlation", "r_squared", "kappa_unweighted", "weighted_kappa", "icc", "ccc", "c_statistic",
      "kendall_tau", "A_percent_agreement", "repeatability_coefficient", "coefficient_of_variation", "discrimination_slope"]),
    (9, "9. Information theory and divergence", ["entropy", "conditional_entropy"]),
    (10, "10. Fairness, survival, decision, NLP, readability",
     ["risk_difference", "paired_proportion", "hazard_ratio", "survival_probability",
      "net_benefit", "net_benefit_standardised", "A_diagnostic_yield",
      "rouge_n", "rouge_1", "rouge_2", "rouge_3", "rouge_l", "rouge_w", "rouge_s", "rouge_su", "bertscore", "meteor",
      "flesch", "flesch_kincaid", "gunning_fog", "smog", "ari", "forcast",
      "levenshtein", "damerau", "hamming", "jaro", "lcs"]),
    (11, "11. Image quality (reconstruction / denoising) — added", ["snr", "psnr", "cnr", "ssim"]),
]

CATEGORY_NOTES = {
    11: "SNR, PSNR, CNR and SSIM are not in the ten-category taxonomy; they are image-quality metrics and get a category of their own.",
    "paired_proportion": "the enhanced detection rate (paired, McNemar structure) is placed under 10 as a clinical-utility metric.",
    "c_statistic": "listed under 3 (an AUC) and 8 (a concordance index).",
    "dsc_f1": "listed under 1 (F1) and 5 (Dice).",
}


def dropdown_choices() -> dict:
    """{category title: {"<spec key>|<category id>": label}} for ui.input_select; the title becomes a greyed optgroup."""
    out = {}
    for cid, title, keys in CATEGORIES:
        out[title] = {f"{k}|{cid}": SPECS[k]["label"] for k in keys}
    return out


def spec_key(choice: str) -> str:
    return choice.split("|")[0]


# every spec must be in at least one category, and every category key must exist
_in_cat = {k for _, _, keys in CATEGORIES for k in keys}
assert _in_cat <= set(SPECS), sorted(_in_cat - set(SPECS))
assert set(SPECS) <= _in_cat, sorted(set(SPECS) - _in_cat)
