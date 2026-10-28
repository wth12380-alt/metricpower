"""The app must import, and every metric spec must compute with its default inputs."""
import warnings
import pytest

shiny = pytest.importorskip("shiny")


def test_app_object():
    from metricpower.app.app import app
    assert isinstance(app, shiny.App)


def test_every_spec_builds_with_defaults():
    from metricpower.app.specs import SPECS, CATEGORIES, dropdown_choices, spec_key
    common = dict(alpha=0.05, power=0.8, two_sided=True, deff=1.0, test="wald")
    ch = dropdown_choices()
    assert len(ch) == len(CATEGORIES)
    assert all(spec_key(v) in SPECS for grp in ch.values() for v in grp)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for k, s in SPECS.items():
            vals = {f["id"]: f["default"] for f in s["fields"]}
            if vals.get("null", 0) is None:
                vals["null"] = s["null_from_h1"](vals)
            r = s["build"](vals, common)
            assert r.n is not None and r.n > 0, k
