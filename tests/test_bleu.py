"""Corpus BLEU: the delta-method variance must agree with Monte Carlo and the bootstrap."""
import math
import numpy as np
import pytest

import metricpower as mp
from metricpower.bleu import bleu_variance, bootstrap_bleu, corpus_bleu_from_counts

rng = np.random.default_rng(11)


def make_corpus(N, len_ratio=1.03):
    r = 5 + rng.poisson(18, N)
    c = np.maximum(1, np.round(r * rng.lognormal(math.log(len_ratio) - 0.15 ** 2 / 2, 0.15, N))).astype(int)
    q = rng.beta(0.62 * 12, 0.38 * 12, N)
    t = np.stack([np.maximum(c - n, 0) for n in range(4)], 1); m = np.zeros_like(t)
    for n in range(4):
        m[:, n] = rng.binomial(t[:, n], np.clip(q * 0.65 ** n, 0, 1))
        if n:
            m[:, n] = np.minimum(m[:, n], m[:, n - 1])
    return np.column_stack([m, t, c, r]).astype(float)


def test_corpus_bleu_definition():
    C = np.array([[8, 5, 3, 2, 10, 9, 8, 7, 10, 12]], float)       # one segment, candidate shorter than reference
    p = [8 / 10, 5 / 9, 3 / 8, 2 / 7]
    expected = math.exp(1 - 12 / 10) * math.exp(np.mean(np.log(p)))
    assert corpus_bleu_from_counts(C) == pytest.approx(expected)


def test_gradient_matches_numeric():
    C = make_corpus(300)
    r = bleu_variance(C)
    theta = C.mean(0)
    def g(th):
        m, t, c, rr = th[:4], th[4:8], th[8], th[9]
        bp = 1.0 if c >= rr else math.exp(1 - rr / c)
        return bp * math.exp(np.mean(np.log(m / t)))
    num = np.array([(g(theta + h * e) - g(theta - h * e)) / (2 * h) for e, h in ((np.eye(10)[i], 1e-5) for i in range(10))])
    assert np.allclose(num, r["gradient"], rtol=1e-4, atol=1e-7)


def test_delta_matches_bootstrap():
    C = make_corpus(300)
    d = bleu_variance(C); b = bootstrap_bleu(C, reps=600, seed=3)
    assert abs(d["se"] / b["se"] - 1) < 0.15


def test_delta_matches_monte_carlo_and_covers():
    N, B = 200, 600
    big = make_corpus(100_000); pop = bleu_variance(big)
    ests, cover = [], []
    for _ in range(B):
        C = make_corpus(N); ests.append(corpus_bleu_from_counts(C))
        r = bleu_variance(C); cover.append(abs(r["estimate"] - pop["estimate"]) <= 1.959964 * r["se"])
    V_mc = N * np.var(ests, ddof=1)
    assert abs(V_mc / pop["V"] - 1) < 0.2          # noise band at 600 reps is about +/-16%
    assert np.mean(cover) > 0.90


def test_brevity_penalty_branch_and_kink_warning():
    C = make_corpus(300, len_ratio=0.90)
    assert bleu_variance(C)["bp_active"] is True
    C = make_corpus(2000, len_ratio=1.0)
    with pytest.warns(UserWarning):
        bleu_variance(C)


def test_result_object_and_not_wired():
    from metricpower.bleu import n_for_power_bleu
    C = make_corpus(300)
    r = n_for_power_bleu(C, delta=0.01)
    assert r.n > 0 and "bleu_delta" in r.reference and r.formula_numeric
    assert "bleu" in mp.GROUP_C                      # still listed as not sized a priori
    with pytest.raises(KeyError):
        mp.n_for_power_group_b("bleu", counts=C, delta=0.01)
