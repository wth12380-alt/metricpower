"""
metricpower Shiny app.

    shiny run --reload metricpower.app.app:app      (from the repo root, after `pip install -e .`)
    python -m metricpower.app

Left: choose the metric, give the null and alternative, alpha and power, get n.
Right: how n was computed (formula with the numbers substituted) and the reference.
"""
from __future__ import annotations

import html
import math

from shiny import App, reactive, render, req, ui

from .. import __version__
from ..core import design_effect
from ..references import REFERENCES
from .specs import SPECS, CATEGORY_NOTES, cm_value, dropdown_choices, spec_key
from shiny.types import SilentException

MATHJAX = ui.tags.head(
    # the configuration object must exist BEFORE the library is loaded
    ui.tags.script("""
      window.MathJax = {
        tex: { inlineMath: [['\\\\(', '\\\\)']], displayMath: [['\\\\[', '\\\\]']] },
        startup: { typeset: true }
      };
    """),
    ui.tags.script(src="mathjax/tex-mml-chtml.js", id="mathjax-script"),   # vendored copy, works offline
    ui.tags.script("""
      function mpTypeset() {
        if (window.MathJax && typeof MathJax.typesetPromise === 'function') {
          MathJax.typesetPromise().catch(function(){});
        }
      }
      $(document).on('shiny:value', function(e) { setTimeout(mpTypeset, 50); });
      $(document).on('shiny:idle', function(e) { setTimeout(mpTypeset, 50); });
      setTimeout(function(){
        if (!(window.MathJax && typeof MathJax.typesetPromise === 'function')) {
          var el = document.getElementById('mathjax-warning'); if (el) el.style.display = 'block';
        }
      }, 4000);
    """),
    ui.tags.style("""
      .big-n { font-size: 3.2rem; font-weight: 700; line-height: 1.1; }
      .muted { color: #666; }
      .help { color: #666; font-size: .85rem; margin-top: -.5rem; margin-bottom: .8rem; }
      .formula { overflow-x: auto; padding: .4rem .6rem; background: #f7f7f9; border-radius: 6px; margin: .4rem 0; }
      .steps td { padding: .15rem .6rem .15rem 0; }
      .note { border-left: 3px solid #d9822b; padding-left: .6rem; margin: .3rem 0; }
      .ref { font-size: .9rem; }
      code { font-size: .85rem; }
      /* let drop-downs overflow the card instead of being clipped */
      .card, .card-body, .bslib-card { overflow: visible !important; }
    """),
)

app_ui = ui.page_fluid(
    MATHJAX,
    ui.h3("metricpower — sample size for AI performance metrics",
          ui.span(f"  v{__version__}", class_="muted", style="font-size:1rem")),
    ui.layout_columns(
        # ------------------------------------------------ left: inputs
        ui.card(
            ui.h6("1. Choose the metric"),
            ui.input_select("metric", "metric (grey headers = problem category)", dropdown_choices(), selected="dsc_f1|1", width="100%"),
            ui.div("A metric that belongs to two categories is listed under both; the result is the same from either entry.", class_="help"),
            ui.h6("2. Null, alternative and design", style="margin-top:1rem"),
            ui.output_ui("metric_inputs"),
            ui.layout_columns(
                ui.div(ui.input_numeric("alpha", "alpha (type I error)", 0.05, min=0.0001, max=0.5, step=0.01),
                       ui.div("chance of declaring a difference that is not there; 0.05 is standard", class_="help")),
                ui.div(ui.input_numeric("power", "power (1 − beta)", 0.80, min=0.5, max=0.999, step=0.05),
                       ui.div("chance of detecting the difference if it is real; 0.80 or 0.90 is standard", class_="help")),
                col_widths=(6, 6),
            ),
            ui.layout_columns(
                ui.div(ui.input_checkbox("two_sided", "two-sided test", True),
                       ui.div("untick only if a change in one direction is impossible or irrelevant", class_="help")),
                ui.output_ui("test_selector"),
                col_widths=(5, 7),
            ),
            ui.accordion(
                ui.accordion_panel("Clustering (several lesions / slices per patient)",
                    ui.div("Use when the unit you count is not the patient. Units from the same patient are not independent, so n is inflated by the design effect 1 + (m̄ − 1)ρ.", class_="help"),
                    ui.layout_columns(
                        ui.div(ui.input_numeric("m_bar", "units per cluster m̄", 1, min=1, step=1),
                               ui.div("average number of analysis units (lesions, slices, nodules) each patient contributes; "
                                      "leave at 1 if one patient = one unit", class_="help")),
                        ui.div(ui.input_numeric("icc", "intra-cluster correlation ρ", 0.0, min=0, max=1, step=0.05),
                               ui.div("how similar two units from the same patient are: 0 = independent, 1 = identical; "
                                      "unknown in most studies, 0.1–0.3 is a common planning value", class_="help")),
                        col_widths=(6, 6)),
                    ui.output_text("deff_text")),
                open=False,
            ),
            ui.hr(),
            ui.output_ui("answer"),
        ),
        # ------------------------------------------------ right: explanation
        ui.card(
            ui.card_header("How this number is calculated"),
            ui.div("MathJax did not load, so formulas are shown as raw LaTeX.",
                   id="mathjax-warning", class_="note", style="display:none"),
            ui.output_ui("explanation"),
        ),
        col_widths=(5, 7),
    ),
    ui.p(ui.tags.small("Δ has no default: choose the null on clinical grounds. For metrics sized by the delta method the "
                       "variance is optimistic below roughly n = 250; for per-case metrics Wald coverage degrades with skew.",
                       class_="muted")),
)


def _field(f):
    k = f["kind"]
    w = _field_widget(f)
    return ui.div(w, ui.div(f["help"], class_="help")) if f.get("help") else w


def _field_widget(f):
    k = f["kind"]
    if k == "numeric":
        return ui.input_numeric(f["id"], f["label"], f["default"], min=f["min"], max=f["max"], step=f["step"])
    if k == "textarea":
        return ui.input_text_area(f["id"], f["label"], f["default"], rows=3)
    if k == "text":
        return ui.input_text(f["id"], f["label"], f["default"])
    if k == "select":
        return ui.input_select(f["id"], f["label"], f["choices"], selected=f["default"])
    if k == "file":
        return ui.input_file(f["id"], f["label"], accept=[".csv"], multiple=False)
    raise ValueError(k)


def server(input, output, session):

    @reactive.calc
    def key():
        req(input.metric())
        return spec_key(input.metric())

    @reactive.calc
    def spec():
        return SPECS[key()]

    @output
    @render.ui
    def metric_inputs():
        s = spec()
        items = []
        for f in s["fields"]:
            if f["id"] == "null" and f["default"] is None:
                f = dict(f, default=s["null_from_h1"]({x["id"]: x["default"] for x in s["fields"] if x["default"] is not None}))
            items.append(_field(f))
        if s["family"] == "confusion_matrix":
            items.insert(3, ui.output_ui("h1_value"))
        return ui.div(*items)

    @output
    @render.ui
    def h1_value():
        s = spec()
        try:
            v = {f["id"]: input[f["id"]]() for f in s["fields"] if f["id"] != "null"}
            val = cm_value(key(), v)
            return ui.p(ui.strong(f"Metric value under H1 at this operating point: {val:.4f}"),
                        ui.br(), ui.span("Δ = H1 value − H0 value", class_="muted"))
        except Exception:
            return ui.p("")

    PROPORTION_A = {"A_accuracy", "A_ppv", "A_npv", "A_top_n_accuracy", "A_percent_agreement", "A_diagnostic_yield",
                    "A_sensitivity", "A_specificity"}

    @reactive.calc
    def test_recommendation():
        s = spec(); m = key()
        if s["group"] == "B" and s["family"] == "confusion_matrix":
            return "score", ("in simulation at the calculated n (10,000 replicates, every confusion-matrix metric), the score n "
                             "gave power 0.79–0.81 throughout, whereas the Wald n fell short for metrics whose variance is larger "
                             "under H0 than under H1 (specificity 0.78, Dice 0.77, PPV 0.79). The score n is never smaller than "
                             "the Wald n when V0 > V1, and it matches the one-sample proportion tests in presize / MKpower / prop.test. "
                             "Choose Wald if the analysis will report a plug-in-SE 95% CI and nothing else.")
        if s["group"] == "B" and s["family"] == "multiclass":
            return "wald", ("k-class metrics have no single null operating point, so only the Wald (plug-in SE) n is defined; "
                            "it is the n for a 95% CI computed from the data.")
        if m in PROPORTION_A:
            return "score", ("diagnostic-accuracy proportions (accuracy, sensitivity, specificity, PPV, NPV) are classically "
                             "tested with a one-sample proportion test whose SE is taken under H0; presize, MKpower and "
                             "prop.test use this convention.")
        return None, None

    @output
    @render.ui
    def test_selector():
        rec, why = test_recommendation()
        if rec is None:
            return ui.div(ui.strong("Test statistic: "), "not a choice for this metric — its published formula has a single convention.",
                          class_="help")
        lab = lambda k, txt: ui.HTML(f"<b>{txt} — recommended</b>") if k == rec else txt
        return ui.div(
            ui.input_radio_buttons("test", "test statistic",
                                   {"wald": lab("wald", "Wald (plug-in SE)"), "score": lab("score", "score (null SE)")},
                                   selected=rec, inline=True),
            ui.div(ui.strong("Why: "), why, class_="help"),
            ui.div(ui.strong("Disclaimer: "),
                   "the recommendation reflects common practice, not a rule. The sample size must match the statistical test you "
                   "will actually run; planning with one convention and analysing with the other changes the achieved power by "
                   "several percentage points. When unsure, the larger of the two n is the safe choice. This tool does not replace "
                   "review by a statistician.", class_="help"),
        )

    @reactive.calc
    def common():
        m = input.m_bar() or 1; r = input.icc() or 0.0
        rec, _ = test_recommendation()
        try:
            test = input.test() if rec is not None else "wald"
        except SilentException:
            test = rec or "wald"
        return dict(alpha=input.alpha(), power=input.power(), two_sided=input.two_sided(),
                    deff=design_effect(m, r), test=test)

    @output
    @render.text
    def deff_text():
        c = common()
        return f"design effect = 1 + (m̄ − 1)ρ = {c['deff']:.3f}; n is multiplied by this"

    @reactive.calc
    def result():
        s = spec()
        vals = {}
        for f in s["fields"]:
            vals[f["id"]] = input[f["id"]]()
        return s["build"](vals, common())

    @output
    @render.ui
    def answer():
        try:
            r = result()
        except Exception as e:  # show the reason, do not crash
            return ui.div(ui.p(ui.strong("Cannot compute: "), str(e), class_="note"))
        if r.n is None:
            return ui.div(ui.p(ui.strong("No a priori sample size"), " — see the explanation."))
        extra = ""
        if r.group == "B" and r.extra.get("n_score") is not None and r.extra.get("n_wald") is not None:
            extra = f"Wald {r.extra['n_wald']} · score {r.extra['n_score']}"
        if "n1" in r.extra:
            extra = f"{r.extra['n1']} positives + {r.extra['n0']} negatives"
        if "n_subgroup" in r.extra:
            extra = f"of which {r.extra['n_subgroup']} in the subgroup that defines the metric"
        return ui.div(
            ui.p("required sample size", class_="muted", style="margin-bottom:0"),
            ui.div(f"n = {r.n}", class_="big-n"),
            ui.p(extra, class_="muted"),
        )

    @output
    @render.ui
    def explanation():
        try:
            r = result()
        except Exception as e:
            return ui.p(f"Fix the inputs first: {e}", class_="muted")
        s = spec()
        basis = ("the sampling variance comes from the delta method or an elementary result and is inverted to a sample size here"
                 if r.group == "B" else f"an established formula, implemented from the primary reference (R equivalent: {r.extra.get('r_package', '-')})")
        parts = [ui.h5(s["label"]), ui.p(ui.strong("Basis: "), basis)]
        note = CATEGORY_NOTES.get(key())
        if note:
            parts.append(ui.p(ui.strong("Category note: "), note, class_="muted"))
        if r.n is None:
            parts.append(ui.h6("Why there is no a priori sample size"))
            for nt in r.notes:
                parts.append(ui.p(nt, class_="note"))
        else:
            parts.append(ui.h6("General formula"))
            for fx in r.formula_latex:
                parts.append(ui.div(ui.HTML("\\[" + fx + "\\]"), class_="formula"))
            parts.append(ui.h6("With your numbers"))
            for fx in r.formula_numeric:
                parts.append(ui.div(ui.HTML("\\[" + fx + "\\]"), class_="formula"))
            if r.steps:
                rows = [ui.tags.tr(ui.tags.td(ui.tags.code(str(k))), ui.tags.td(str(v))) for k, v in r.steps]
                parts.append(ui.h6("Steps"))
                parts.append(ui.tags.table(*rows, class_="steps"))
            if r.notes:
                parts.append(ui.h6("Read before trusting the number"))
                for nt in r.notes:
                    parts.append(ui.p(nt, class_="note"))
        parts.append(ui.h6("Reference"))
        for k in r.reference:
            parts.append(ui.p(REFERENCES.get(k, k), class_="ref"))
        parts.append(ui.tags.script("if (window.MathJax && MathJax.typesetPromise) { MathJax.typesetPromise(); }"))
        return ui.div(*parts)


from pathlib import Path
app = App(app_ui, server, static_assets=Path(__file__).parent / "www")
