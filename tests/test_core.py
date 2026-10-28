"""Engine checks. Numbers marked (R) reproduce the validated R package output."""
import math
import numpy as np
import pytest

import metricpower as mp
from metricpower import core

P = mp.cell_prob_from_operating_point(0.70, 0.80, 0.50)


def test_cell_probabilities():
    assert np.allclose(P, [0.35, 0.10, 0.15, 0.40])
    assert np.allclose(mp.cell_prob(70, 20, 30, 80), P)
    with pytest.raises(ValueError):
        mp.cell_prob_from_operating_point(1.2, 0.8, 0.5)


def test_linear_metric_variance_is_exact():
    # accuracy = tp + tn is a proportion: V = acc(1 - acc) exactly
    acc = 0.75
    assert mp.metric_variance(P, "accuracy")["V"] == pytest.approx(acc * (1 - acc))


@pytest.mark.parametrize("name", mp.CM_METRICS)
def test_analytic_gradient_matches_numeric(name):
    g = mp.get_metric(name); gr = mp.get_gradient(name)
    assert np.allclose(gr(P), core.numeric_gradient(g, P), atol=1e-6)
    assert mp.delta_var(P, g, grad=gr) == pytest.approx(mp.delta_var(P, g), rel=1e-6)


def test_f1_reference_values_R():
    r = mp.metric_variance(P, "dsc_f1")
    assert r["estimate"] == pytest.approx(0.736842, abs=1e-6)
    assert r["V"] == pytest.approx(0.257825, abs=1e-6)          # (R)


def test_iou_equals_f1_map():
    f1 = mp.metric_variance(P, "dsc_f1")["estimate"]
    assert mp.metric_variance(P, "iou")["estimate"] == pytest.approx(f1 / (2 - f1))
    assert mp.metric_variance(P, "iou")["V"] == pytest.approx(0.40509, abs=1e-5)   # (R)


@pytest.mark.parametrize("name,n_wald,n_score", [
    ("accuracy", 589, 638), ("balanced_accuracy", 581, 614), ("dsc_f1", 810, 921),
    ("specificity", 1005, 1125), ("iou", 1272, 1294), ("sensitivity", 1319, 1396), ("mcc", 2323, 2412)])
def test_sample_sizes_R(name, n_wald, n_score):
    r = mp.n_for_power_metric(P, name, delta=0.05)
    assert r.n == n_wald and r.extra["n_wald"] == n_wald and r.extra["n_score"] == n_score   # (R)


def test_hand_computation_f1():
    za, zb = 1.959964, 0.841621
    assert math.ceil((za + zb) ** 2 * 0.257825 / 0.05 ** 2) == 810


def test_design_settings():
    f = lambda **k: mp.n_for_power_metric(P, "dsc_f1", 0.05, **k).n
    assert f(power=0.9) == 1084 and f(alpha=0.01) == 1205 and f(two_sided=False) == 638     # (R)
    assert f(deff=mp.design_effect(3, 0.30)) == 1296


def test_power_and_n_are_mutually_consistent():
    V = mp.metric_variance(P, "dsc_f1")["V"]
    n = mp.n_for_power(V, 0.05)
    assert abs(mp.power_for_n(n, V, 0.05) - 0.80) < 0.002
    assert mp.n_for_precision(V, 0.05) == 397


def test_specificity_null_path_fallback():
    # specificity does not move along the tp->fn path; 'auto' must fall back and not raise
    r = mp.n_for_power_metric(P, "specificity", 0.05)
    assert r.extra["null_path"] == "specificity" and r.n == 1005


def test_cell_prob_at_metric_round_trip():
    p0 = mp.cell_prob_at_metric(P, "dsc_f1", 0.6868)
    assert mp.get_metric("dsc_f1")(p0) == pytest.approx(0.6868, abs=1e-9)


def test_paired_variance_R():
    p8 = [0.30, 0.05, 0.03, 0.12, 0.04, 0.06, 0.02, 0.38]
    r = mp.delta_var_paired(p8, "dsc_f1")
    assert r["V"] == pytest.approx(0.2021, abs=2e-4)                   # (R)
    VA = mp.metric_variance(mp.paired_marginal(p8, "A"), "dsc_f1")["V"]
    VB = mp.metric_variance(mp.paired_marginal(p8, "B"), "dsc_f1")["V"]
    assert r["V"] < VA + VB                                            # pairing helps


def test_delta_var_against_monte_carlo():
    rng = np.random.default_rng(90210)
    n, reps = 500, 20000
    g = mp.get_metric("dsc_f1")
    est = np.array([g(q) for q in rng.multinomial(n, P, size=reps) / n])
    V_mc = n * est.var(ddof=1)
    assert abs(V_mc / 0.257825 - 1) < 0.05        # noise band at 20k reps is about +/-2.8%


def test_small_sample_check_runs():
    r = mp.check_delta_reliable(P, "dsc_f1", n=30, reps=4000, seed=1)
    assert r["ratio"] > 1.0 and not r["reliable"]
    r = mp.check_delta_reliable(P, "dsc_f1", n=500, reps=4000, seed=1)
    assert r["reliable"]


def test_calibration_error_refused():
    with pytest.raises(NotImplementedError):
        mp.n_for_power_calibration_error()


def test_result_carries_explanation():
    r = mp.n_for_power_metric(P, "dsc_f1", 0.05)
    assert r.formula_latex and r.formula_numeric and r.reference
    assert "810" in r.formula_numeric[-1]
    assert all(isinstance(s, str) for s in r.references_full())


# ---------------------------------------------------------------- t-based n for per-case means, kink guard
def test_n_for_power_t_vs_z():
    import metricpower as mp
    # small n: the t-based n exceeds the z-based n; large n: they agree within a case or two
    assert mp.n_for_power_t(0.0126, 0.0817) > mp.n_for_power(0.0126, 0.0817)
    assert abs(mp.n_for_power_t(0.0238, 0.0145) - mp.n_for_power(0.0238, 0.0145)) <= 3
    # power.t.test(delta = 1, sd = 1, power = 0.8, type = "one.sample") gives n = 9.94 -> 10
    assert mp.n_for_power_t(1.0, 1.0) == 10
    assert abs(mp.power_for_n_t(10, 1.0, 1.0) - 0.803) < 0.01


def test_percase_result_uses_t():
    import metricpower as mp
    r = mp.n_for_power_group_b("ssim", mean=0.817, sd=0.1123, delta=0.0817)
    assert r.n == 17 and r.extra["n_z"] == 15
    assert any("t-test" in n for n in r.notes)


def test_volumetric_similarity_kink_warning():
    import warnings, numpy as np, metricpower as mp
    p = mp.cell_prob_from_operating_point(0.70, 0.80, 0.50)        # fp = .10, fn = .15: 1.6 SE from the kink at n = 242
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        r = mp.n_for_power_metric(p, "volumetric_similarity", 0.0947)
    assert any("kink" in str(x.message) for x in w) and any("kink" in n for n in r.notes)
    assert mp.kink_distance(p, 242) < 2
    with warnings.catch_warnings(record=True) as w:                 # far from the kink: no warning
        warnings.simplefilter("always")
        r = mp.n_for_power_metric(np.array([.35, .05, .25, .35]), "volumetric_similarity", 0.05)
    assert not any("kink" in str(x.message) for x in w)
