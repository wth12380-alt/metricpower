"""
EXPERIMENTAL - not wired into n_for_power_group_b() or the app.

Corpus BLEU: an analytic (delta-method) variance.

Koehn (2004) states that the usual confidence-interval machinery "does not work
for the BLEU metric, since the BLEU metric is not the mean of single sentence
scores", and the MT literature has used bootstrap resampling ever since. That
sentence is true but it is not the end of the story: corpus BLEU is a SMOOTH
FUNCTION OF MEANS, and the multivariate delta method (Taylor linearisation) was
built for exactly that case. Writing, for segment i and n-gram order n,

    m_in  clipped matched n-grams      t_in  candidate n-grams
    c_i   candidate length             r_i   (effective) reference length

the corpus statistics are p_n = sum_i m_in / sum_i t_in = mbar_n / tbar_n and
BP = min(1, exp(1 - rbar / cbar)), so

    BLEU = BP * exp( sum_n w_n log(mbar_n / tbar_n) ),   w_n = 1/4.

Let theta = (mbar_1..4, tbar_1..4, cbar, rbar), a vector of ten sample means of
i.i.d. per-segment vectors. By the CLT and the delta method,

    sqrt(N) (BLEU_hat - BLEU) -> N(0, V),   V = grad g(theta)' Sigma grad g(theta),

where Sigma is the 10 x 10 covariance of the per-segment vector (estimated by
the sample covariance) and the gradient is closed-form:

    d log BLEU / d mbar_n = w_n / mbar_n,     d log BLEU / d tbar_n = -w_n / tbar_n,
    d log BLEU / d cbar   = rbar / cbar^2  if cbar < rbar else 0,
    d log BLEU / d rbar   = -1 / cbar      if cbar < rbar else 0,
    grad g = BLEU * grad log BLEU.

This is the classical ratio-estimator linearisation of survey sampling (Cochran
1977, ch. 6) applied to a product of four ratios, times the brevity penalty.
Two caveats are genuine and are reported, not hidden:

  * The brevity penalty has a kink at cbar = rbar. There the limit is not
    normal (it is a mixture), so a warning is raised when |cbar - rbar| is
    within a few standard errors of zero.
  * Segments are assumed independent. Koehn (2004) showed that neighbouring
    sentences in a document have correlated quality; use a design effect or a
    block bootstrap when the test set is a few long documents.

No published derivation of this variance for BLEU was found when this module
was written (October 2026); the result is standard but its application here is
our own. It is validated against Monte Carlo and the bootstrap in
tests/test_bleu.py and validation/validate_bleu.py.
"""
from __future__ import annotations

import math
import warnings
from typing import Optional, Sequence

import numpy as np

from .core import Result, n_for_power, _z
from scipy.stats import norm

COLUMNS = ["m1", "m2", "m3", "m4", "t1", "t2", "t3", "t4", "c", "r"]


def corpus_bleu_from_counts(counts: np.ndarray, max_order: int = 4) -> float:
    """Corpus BLEU from an N x 10 per-segment count matrix (no smoothing, as sacrebleu's corpus_bleu)."""
    C = np.asarray(counts, dtype=float)
    m = C[:, :max_order].sum(0); t = C[:, max_order:2 * max_order].sum(0)
    c, r = C[:, 2 * max_order].sum(), C[:, 2 * max_order + 1].sum()
    if np.any(m <= 0):
        return 0.0
    bp = 1.0 if c >= r else math.exp(1 - r / c)
    return bp * math.exp(np.mean(np.log(m / t)))


def bleu_variance(counts: np.ndarray, max_order: int = 4, kink_tol_se: float = 2.0) -> dict:
    """Delta-method variance of corpus BLEU from per-segment counts.

    counts: N x (2*max_order + 2) matrix, columns m1..m4, t1..t4, c, r (see COLUMNS).
    Returns estimate, V (per-segment variance, Var(BLEU) = V / N), se, gradient, and flags.
    """
    C = np.asarray(counts, dtype=float)
    N, k = C.shape
    if k != 2 * max_order + 2:
        raise ValueError(f"expected {2 * max_order + 2} columns {COLUMNS}, got {k}")
    if N < 2:
        raise ValueError("need at least two segments")
    theta = C.mean(0)
    Sigma = np.cov(C, rowvar=False, ddof=1)
    mbar, tbar = theta[:max_order], theta[max_order:2 * max_order]
    cbar, rbar = theta[2 * max_order], theta[2 * max_order + 1]
    if np.any(mbar <= 0) or np.any(tbar <= 0):
        raise ValueError("some n-gram order has zero matches in the whole corpus; BLEU is 0 and its variance is not defined")
    w = 1.0 / max_order
    bp_active = cbar < rbar
    bp = math.exp(1 - rbar / cbar) if bp_active else 1.0
    bleu = bp * math.exp(w * np.sum(np.log(mbar / tbar)))
    grad_log = np.concatenate([w / mbar, -w / tbar,
                               [rbar / cbar ** 2 if bp_active else 0.0, -1 / cbar if bp_active else 0.0]])
    grad = bleu * grad_log
    V = float(grad @ Sigma @ grad)
    # kink diagnostic: how many SEs is (cbar - rbar) from zero?
    d = cbar - rbar
    var_d = Sigma[2 * max_order, 2 * max_order] + Sigma[2 * max_order + 1, 2 * max_order + 1] - 2 * Sigma[2 * max_order, 2 * max_order + 1]
    kink_z = abs(d) / math.sqrt(var_d / N) if var_d > 0 else math.inf
    near_kink = kink_z < kink_tol_se
    if near_kink:
        warnings.warn(f"candidate and reference lengths are within {kink_z:.1f} SE of each other: the brevity penalty is at "
                      "its kink, the sampling distribution is a mixture rather than normal, and this variance is approximate",
                      stacklevel=2)
    return dict(estimate=bleu, V=V, se=math.sqrt(V / N), n_segments=N, gradient=grad, Sigma=Sigma,
                precisions=(mbar / tbar).tolist(), brevity_penalty=bp, bp_active=bool(bp_active),
                kink_z=kink_z, near_kink=bool(near_kink), method="delta method on ten per-segment means")


def bleu_ci(counts: np.ndarray, conf_level: float = 0.95, **kw) -> dict:
    r = bleu_variance(counts, **kw)
    z = float(norm.ppf(1 - (1 - conf_level) / 2))
    return dict(estimate=r["estimate"], se=r["se"], lower=r["estimate"] - z * r["se"], upper=r["estimate"] + z * r["se"])


def n_for_power_bleu(counts: np.ndarray, delta: float, alpha: float = 0.05, power: float = 0.80,
                     two_sided: bool = True, deff: float = 1.0) -> Result:
    """Segments needed to detect a change of `delta` in corpus BLEU (BLEU on the 0-1 scale)."""
    r = bleu_variance(counts)
    n = n_for_power(r["V"], delta, alpha=alpha, power=power, two_sided=two_sided, deff=deff)
    za, zb = _z(alpha, two_sided), float(norm.ppf(power))
    notes = ["segments are assumed independent; for a few long documents use a design effect or block bootstrap (Koehn 2004)",
             "no smoothing: a corpus with zero matches at some order has BLEU = 0 and no variance"]
    if r["near_kink"]:
        notes.append("brevity penalty at its kink (candidate length ≈ reference length): variance is approximate")
    return Result(n=n, metric="bleu", group="B", estimate=r["estimate"], null_value=r["estimate"] - delta, V_alt=r["V"], V_null=r["V"],
                  inputs=dict(n_segments=r["n_segments"], delta=delta, alpha=alpha, power=power, two_sided=two_sided, deff=deff),
                  steps=[("corpus BLEU", round(r["estimate"], 4)), ("n-gram precisions p1..p4", [round(x, 4) for x in r["precisions"]]),
                         ("brevity penalty", round(r["brevity_penalty"], 4)), ("V (per-segment)", round(r["V"], 6)),
                         ("z_{1-alpha/2}", round(za, 6)), ("z_{1-beta}", round(zb, 6)), ("n (segments)", n)],
                  formula_latex=[r"\text{BLEU} = BP\,\exp\!\Big(\tfrac14\sum_{n=1}^{4}\log\frac{\bar m_n}{\bar t_n}\Big),\quad BP=\min\!\big(1,e^{1-\bar r/\bar c}\big)",
                                 r"\theta=(\bar m_{1..4},\bar t_{1..4},\bar c,\bar r)\ \text{(ten per-segment means)},\qquad V=\nabla g(\theta)^{\prime}\,\Sigma\,\nabla g(\theta)",
                                 r"\frac{\partial\log\text{BLEU}}{\partial\bar m_n}=\frac{1}{4\bar m_n},\ \frac{\partial\log\text{BLEU}}{\partial\bar t_n}=-\frac{1}{4\bar t_n},\ \frac{\partial\log\text{BLEU}}{\partial\bar c}=\frac{\bar r}{\bar c^2}\mathbb 1[\bar c<\bar r],\ \frac{\partial\log\text{BLEU}}{\partial\bar r}=-\frac{1}{\bar c}\mathbb 1[\bar c<\bar r]",
                                 r"n = \frac{(z_{1-\alpha/2}+z_{1-\beta})^2\,V}{\Delta^2}"],
                  formula_numeric=[f"\\text{{BLEU}} = {r['estimate']:.4f},\\quad V = {r['V']:.6f}\\ (N = {r['n_segments']}\\ \\text{{pilot segments}})",
                                   f"n = \\frac{{({za:.6f}+{zb:.6f})^2\\times{r['V']:.6f}}}{{{delta:.4f}^2}} = {(za + zb) ** 2 * r['V'] / delta ** 2:.2f} \\to {n}"],
                  reference=["bleu_delta", "koehn2004", "cochran1977"], notes=notes)


# --- helpers -----------------------------------------------------------------

def bleu_counts_from_sacrebleu(hypotheses: Sequence[str], references: Sequence[Sequence[str]], **bleu_kwargs) -> np.ndarray:
    """Per-segment count matrix via sacrebleu (pip install sacrebleu). references: one list of reference strings per segment."""
    try:
        from sacrebleu.metrics import BLEU
    except ImportError as e:
        raise ImportError("pip install sacrebleu") from e
    bleu = BLEU(**bleu_kwargs)
    rows = []
    for hyp, refs in zip(hypotheses, references):
        s = bleu.sentence_score(hyp, list(refs))
        rows.append(list(s.counts) + list(s.totals) + [s.sys_len, s.ref_len])
    return np.asarray(rows, dtype=float)


def bootstrap_bleu(counts: np.ndarray, reps: int = 1000, seed: Optional[int] = None) -> dict:
    """Koehn-style bootstrap of corpus BLEU over segments, for comparison with bleu_variance()."""
    rng = np.random.default_rng(seed)
    C = np.asarray(counts, dtype=float); N = len(C)
    b = np.array([corpus_bleu_from_counts(C[rng.integers(0, N, N)]) for _ in range(reps)])
    return dict(estimate=corpus_bleu_from_counts(C), se=float(b.std(ddof=1)),
                lower=float(np.quantile(b, 0.025)), upper=float(np.quantile(b, 0.975)), reps=reps)
