"""Print the Python column of the R-vs-Python cross-check table (README)."""
import numpy as np
import metricpower as mp

p = mp.cell_prob_from_operating_point(0.70, 0.80, 0.50)
P3 = [[.30, .04, .02], [.05, .22, .05], [.03, .06, .23]]
rows = []
for m in ["accuracy", "balanced_accuracy", "dsc_f1", "specificity", "iou", "sensitivity", "mcc"]:
    r = mp.n_for_power_group_b(m, p=p, delta=0.05)
    rows.append((m, f"{r.n} / {r.extra['n_score']}"))
rows.append(("macro_f1", mp.n_for_power_group_b("macro_f1", P=P3, delta=0.05).n))
rows.append(("weighted_kappa", mp.n_for_power_group_b("weighted_kappa", P=P3, delta=0.05).n))
rows.append(("fppi Poisson / NB(2)", f"{mp.n_for_power_group_b('fppi', lam=0.5, ratio=0.7).n} / {mp.n_for_power_group_b('fppi', lam=0.5, ratio=0.7, theta=2).n}"))
rows.append(("risk_difference", mp.n_for_power_group_b("risk_difference", p1=.6, p2=.45, delta=.15).n))
rows.append(("paired dsc_f1 V", round(mp.delta_var_paired([0.30, 0.05, 0.03, 0.12, 0.04, 0.06, 0.02, 0.38], "dsc_f1")["V"], 4)))
rows.append(("dsc_f1 power .9 / alpha .01 / one-sided",
             f"{mp.n_for_power_group_b('dsc_f1', p=p, delta=.05, power=.9).n} / {mp.n_for_power_group_b('dsc_f1', p=p, delta=.05, alpha=.01).n} / {mp.n_for_power_group_b('dsc_f1', p=p, delta=.05, two_sided=False).n}"))
rows.append(("dsc_f1 deff 1.6", mp.n_for_power_group_b("dsc_f1", p=p, delta=.05, deff=mp.design_effect(3, .3)).n))
print(f"{'metric':<42}{'Python (Wald / score)':>24}")
for k, v in rows:
    print(f"{k:<42}{str(v):>24}")
