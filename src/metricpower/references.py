"""Reference list. Keys are used by Result.reference; values are citations.

Entries marked [verify] were cited from background knowledge during the
project and should be checked against the primary source before publication.
"""

REFERENCES = {
    "delta_method": "Multivariate delta method on multinomial cell proportions: Cov(p̂) = (1/n)[diag(p) − pp′] exactly; "
                    "√n(g(p̂) − g(p)) → N(0, ∇g′Σ∇g). See e.g. Agresti A. Categorical Data Analysis, 3rd ed. Wiley, 2013, §16.1.",
    "takahashi2022": "Takahashi K, Yamamoto K, Kuchiba A, Koyama T. Confidence interval for micro-averaged F1 and macro-averaged F1 scores. "
                     "Appl Intell 2022;52:4961–4972. doi:10.1007/s10489-021-02635-5",
    "itaya2024": "Itaya T, Murakami R, et al. Asymptotic distribution of the Matthews correlation coefficient. Stat Med 2024. [verify]",
    "fleiss1969": "Fleiss JL, Cohen J, Everitt BS. Large sample standard errors of kappa and weighted kappa. Psychol Bull 1969;72(5):323–327. doi:10.1037/h0028106",
    "basharin1959": "Basharin GP. On a statistical estimate for the entropy of a sequence of independent random variables. Theory Probab Appl 1959;4(3):333–336.",
    "hoeffding1948": "Hoeffding W. A class of statistics with asymptotically normal distribution. Ann Math Stat 1948;19(3):293–325. doi:10.1214/aoms/1177730196",
    "bradley2008": "Bradley AA, Schwartz SS, Hashino T. Sampling uncertainty and confidence intervals for the Brier score and Brier skill score. "
                   "Weather Forecast 2008;23(5):992–1006. doi:10.1175/2007WAF2007049.1",
    "clt_mean": "Central limit theorem for a mean: Var(x̄) = σ²/n exactly for any distribution with finite variance.",
    "poisson_nb": "Poisson variance λ; negative-binomial variance λ + λ²/θ. Textbook results (e.g. Cameron AC, Trivedi PK. Regression Analysis of Count Data, 2nd ed. CUP, 2013).",
    "buderer1996": "Buderer NMF. Statistical methodology: incorporating the prevalence of disease into the sample size calculation for sensitivity and specificity. "
                   "Acad Emerg Med 1996;3(9):895–900. doi:10.1111/j.1553-2712.1996.tb03538.x",
    "simel1991": "Simel DL, Samsa GP, Matchar DB. Likelihood ratios with confidence: sample size estimation for diagnostic test studies. "
                 "J Clin Epidemiol 1991;44(8):763–770. doi:10.1016/0895-4356(91)90128-V",
    "woolf1955": "Woolf B. On estimating the relation between blood group and disease. Ann Hum Genet 1955;19(4):251–253. doi:10.1111/j.1469-1809.1955.tb01348.x",
    "hanley1982": "Hanley JA, McNeil BJ. The meaning and use of the area under a receiver operating characteristic (ROC) curve. "
                  "Radiology 1982;143(1):29–36. doi:10.1148/radiology.143.1.7063747",
    "obuchowski1994": "Obuchowski NA. Computing sample size for receiver operating characteristic studies. Invest Radiol 1994;29(2):238–243. [verify]",
    "obuchowski1997": "Obuchowski NA, McClish DK. Sample size determination for diagnostic accuracy studies involving binormal ROC curve indices. "
                      "Stat Med 1997;16(13):1529–1542.",
    "delong1988": "DeLong ER, DeLong DM, Clarke-Pearson DL. Comparing the areas under two or more correlated ROC curves: a nonparametric approach. "
                  "Biometrics 1988;44(3):837–845. doi:10.2307/2531595",
    "riley2021": "Riley RD, Debray TPA, Collins GS, et al. Minimum sample size for external validation of a clinical prediction model with a binary outcome. "
                 "Stat Med 2021;40(19):4230–4251. doi:10.1002/sim.9025",
    "rotondi2012": "Rotondi MA, Donner A. A confidence interval approach to sample size estimation for interobserver agreement studies with multiple raters and outcomes. "
                   "J Clin Epidemiol 2012;65(7):778–784. doi:10.1016/j.jclinepi.2011.10.019",
    "zou2012": "Zou GY. Sample size formulas for estimating intraclass correlation coefficients with precision and assurance. Stat Med 2012;31(29):3972–3981. doi:10.1002/sim.5466",
    "bonett2002": "Bonett DG. Sample size requirements for estimating intraclass correlations with desired precision. Stat Med 2002;21(9):1331–1335. doi:10.1002/sim.1108",
    "fisher_z": "Fisher RA. Frequency distribution of the values of the correlation coefficient in samples from an indefinitely large population. "
                "Biometrika 1915;10(4):507–521. (Fisher z transform, Var(z) ≈ 1/(n − 3).)",
    "lin1989": "Lin LI-K. A concordance correlation coefficient to evaluate reproducibility. Biometrics 1989;45(1):255–268. doi:10.2307/2532051",
    "schoenfeld1983": "Schoenfeld DA. Sample-size formula for the proportional-hazards regression model. Biometrics 1983;39(2):499–503. doi:10.2307/2531021",
    "greenwood1926": "Greenwood M. The natural duration of cancer. Reports on Public Health and Medical Subjects 1926;33:1–26. (Greenwood variance of the Kaplan–Meier estimator.)",
    "chakraborty": "Chakraborty DP. Observer Performance Methods for Diagnostic Imaging: Foundations, Modeling, and Applications with R-Based Examples. CRC Press, 2017. "
                   "(RJafroc; Obuchowski–Rockette and Dorfman–Berbaum–Metz variance components.)",
    "wald_proportion": "Normal approximation to a binomial proportion: Var(p̂) = p(1 − p)/n; one-sample test n = (z_{1−α/2}√(p₀(1−p₀)) + z_{1−β}√(p₁(1−p₁)))²/(p₁ − p₀)². "
                       "Fleiss JL, Levin B, Paik MC. Statistical Methods for Rates and Proportions, 3rd ed. Wiley, 2003.",
    "two_proportions": "Difference of two independent proportions: Var = p₁(1−p₁)/n₁ + p₂(1−p₂)/n₂. Fleiss, Levin & Paik 2003.",
    "mcnemar": "Paired proportions (McNemar structure): only discordant pairs carry information. Var(p₁₀ − p₀₁) = (p₀₁ + p₁₀ − (p₁₀ − p₀₁)²)/n. Fleiss, Levin & Paik 2003, ch. 13.",
    "kendall_hoeffding": "Kendall's tau as a degree-2 U-statistic: √n(τ̂ − τ) → N(0, 4ζ₁); under independence ζ₁ = 1/9. Hoeffding 1948.",
    "rand_ustat": "Rand and Fowlkes–Mallows indices as degree-2 U-statistics over item pairs; Hoeffding projection. "
                  "Brennan RL, Light RJ. Br J Math Stat Psychol 1974;27:154–163.",
    "r_squared": "Var(R²) ≈ 4ρ²(1 − ρ²)²/n under bivariate normality (delta method on r²); degenerate at ρ = 0.",
    "cv_normal": "Sample coefficient of variation under normality: Var(ĉv) ≈ cv²(0.5 + cv²)/n (delta method on the mean and SD).",
    "repeatability": "Repeatability coefficient RC = 1.96√2 σ_w; Var(σ̂_w) ≈ σ_w²/(2·df) with df = n(m − 1). Bland JM, Altman DG. Stat Methods Med Res 1999;8:135–160.",
    "design_effect": "Design effect for clustered units: deff = 1 + (m̄ − 1)ρ. Kish L. Survey Sampling. Wiley, 1965.",
    "bleu_delta": "Corpus BLEU as a smooth function of ten per-segment means (matched and total n-grams of order 1–4, candidate and reference length); "
                  "multivariate delta method with the sample covariance of the per-segment vector. Derived in this package; see metricpower/bleu.py.",
    "koehn2004": "Koehn P. Statistical significance tests for machine translation evaluation. In: Proceedings of the 2004 Conference on Empirical Methods in "
                 "Natural Language Processing (EMNLP). Barcelona, Spain: Association for Computational Linguistics, 2004; 388–395.",
    "cochran1977": "Cochran WG. Sampling Techniques, 3rd ed. Wiley, 1977, ch. 6 (Taylor linearisation of the ratio estimator).",
    "maier_hein2024": "Maier-Hein L, Reinke A, Godau P, et al. Metrics reloaded: recommendations for image analysis validation. Nat Methods 2024;21(2):195–212. doi:10.1038/s41592-023-02151-z",
}
