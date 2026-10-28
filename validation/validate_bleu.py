"""
validate_bleu.py — does the delta-method variance of corpus BLEU hold?

Three checks on a synthetic corpus with realistic per-segment structure
(variable segment length, a per-segment quality effect that correlates the
four n-gram orders, candidate/reference length ratio that puts the brevity
penalty either off, on, or at its kink):

  1. Monte Carlo:  V_delta (from one large pilot) vs N * Var(BLEU) over B fresh corpora of N segments
  2. Bootstrap:    on ONE corpus, the delta SE vs Koehn-style bootstrap SE
  3. Coverage:     Wald 95% CI from the delta SE, at N = 100, 300, 1000

    python validation/validate_bleu.py            # B = 4000
"""
import argparse
import math
import time
import warnings

import numpy as np
from scipy.stats import norm

from metricpower.bleu import bleu_variance, bootstrap_bleu, corpus_bleu_from_counts

ap = argparse.ArgumentParser()
ap.add_argument("--reps", type=int, default=4000)
ap.add_argument("--seed", type=int, default=2026)
args = ap.parse_args()
rng = np.random.default_rng(args.seed)
warnings.simplefilter("ignore")
Z = norm.ppf(0.975)


def make_corpus(N, len_ratio=1.03, q_mean=0.62, q_conc=12.0, decay=0.65):
    """Per-segment counts m1..m4, t1..t4, c, r."""
    r = 5 + rng.poisson(18, N)
    c = np.maximum(1, np.round(r * rng.lognormal(math.log(len_ratio) - 0.15 ** 2 / 2, 0.15, N))).astype(int)
    q = rng.beta(q_mean * q_conc, (1 - q_mean) * q_conc, N)          # segment quality, shared across orders
    t = np.stack([np.maximum(c - n, 0) for n in range(4)], 1)       # candidate n-grams of order 1..4
    m = np.zeros_like(t)
    for n in range(4):
        m[:, n] = rng.binomial(t[:, n], np.clip(q * decay ** n, 0, 1))
        if n > 0:
            m[:, n] = np.minimum(m[:, n], m[:, n - 1])               # matched 4-grams cannot exceed matched 3-grams
    return np.column_stack([m, t, c, r]).astype(float)


def mc_check(label, N, B, **dgp):
    big = make_corpus(200_000, **dgp)                                 # population values
    pop = bleu_variance(big)
    ests = np.array([corpus_bleu_from_counts(make_corpus(N, **dgp)) for _ in range(B)])
    V_mc = N * ests.var(ddof=1)
    diff = 100 * (V_mc - pop["V"]) / pop["V"]
    # coverage with the per-sample delta SE
    cov = []
    for _ in range(B // 2):
        C = make_corpus(N, **dgp)
        r = bleu_variance(C)
        cov.append(abs(r["estimate"] - pop["estimate"]) <= Z * r["se"])
    noise = 100 * 2 * math.sqrt(4 / B)
    print(f"{label:<34} N={N:>5}  BLEU={pop['estimate']:.4f}  BP={'on ' if pop['bp_active'] else 'off'}  "
          f"V_delta={pop['V']:.5f}  V_MC={V_mc:.5f}  diff={diff:+.1f}% (noise ±{noise:.1f}%)  coverage={np.mean(cov):.3f}")
    return abs(diff) < max(noise, 5), np.mean(cov)


t0 = time.time()
print("=" * 120)
print(f"CORPUS BLEU: delta-method variance vs Monte Carlo and bootstrap   (B = {args.reps:,})")
print("=" * 120)
ok = True
print("\n[1] Monte Carlo: V_delta (population) vs N * Var(BLEU) over fresh corpora; Wald coverage of the per-sample delta CI\n")
for label, N, dgp in [
    ("BP off (candidate 3% longer)", 100, dict(len_ratio=1.03)),
    ("BP off (candidate 3% longer)", 300, dict(len_ratio=1.03)),
    ("BP off (candidate 3% longer)", 1000, dict(len_ratio=1.03)),
    ("BP on  (candidate 5% shorter)", 300, dict(len_ratio=0.95)),
    ("BP on  (candidate 5% shorter)", 1000, dict(len_ratio=0.95)),
    ("low quality corpus (BLEU ~0.1)", 300, dict(len_ratio=1.03, q_mean=0.45, decay=0.55)),
    ("AT THE KINK (lengths equal)", 300, dict(len_ratio=1.00)),
    ("AT THE KINK (lengths equal)", 1000, dict(len_ratio=1.00)),
]:
    good, cov = mc_check(label, N, args.reps, **dgp)
    if "KINK" not in label:
        ok &= good and cov > 0.93

print("\n[2] One corpus of 300 segments: delta SE vs bootstrap SE (1000 resamples), five independent corpora\n")
for i in range(5):
    C = make_corpus(300, len_ratio=1.03)
    d = bleu_variance(C); b = bootstrap_bleu(C, reps=1000, seed=i)
    ratio = d["se"] / b["se"]
    print(f"   corpus {i + 1}: BLEU={d['estimate']:.4f}   SE_delta={d['se']:.5f}   SE_boot={b['se']:.5f}   ratio={ratio:.3f}"
          f"   CI_delta=[{d['estimate'] - Z * d['se']:.4f}, {d['estimate'] + Z * d['se']:.4f}]   CI_boot=[{b['lower']:.4f}, {b['upper']:.4f}]")
    ok &= 0.85 < ratio < 1.15

print("\n[3] The naive route Koehn warned against: SD of sentence-level BLEU / sqrt(N) vs the delta SE\n")
C = make_corpus(300, len_ratio=1.03)
sent = np.array([corpus_bleu_from_counts(C[[i]]) for i in range(len(C))])
print(f"   mean of sentence BLEU = {sent.mean():.4f} (corpus BLEU = {corpus_bleu_from_counts(C):.4f});  "
      f"naive SE = {sent.std(ddof=1) / math.sqrt(len(C)):.5f};  delta SE = {bleu_variance(C)['se']:.5f};  bootstrap SE = {bootstrap_bleu(C, 1000, 7)['se']:.5f}")
print("   (the naive SE targets a different estimand - the mean of sentence scores - which is why it was never usable)")

print("\n" + "=" * 120)
print(f"{'BLEU DELTA METHOD VALIDATED' if ok else 'CHECK FAILED'}   ({time.time() - t0:.0f} s)")
print("""
Reading: V_delta is the per-segment variance from the gradient and the per-segment covariance; N * Var(BLEU) is the
truth from fresh corpora. Agreement within the noise band, and coverage near 0.95, means the delta method gives corpus
BLEU a valid analytic variance - except at the brevity-penalty kink (candidate length = reference length), where the
limit law is a mixture and the variance is only approximate; the package warns there.""")
