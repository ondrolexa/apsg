"""Automatic identification of components: window fits, search, grading, stacking."""

import itertools
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pytest

from apsg import data
from apsg.math._vector import Vector3
from apsg.paleomag import (
    Component,
    ComponentCriteria,
    ComponentSearchPlot,
    DemagStep,
    Sample,
    Specimen,
    ZijderveldPlot,
)
from apsg.paleomag._components import SEARCH_CRITERIA

LOW, HIGH = Vector3(30, 60), Vector3(320, -15)  # overprint and ChRM directions


def _dir(v):
    return np.asarray(v, dtype=float)


def _angle(a, b):
    a, b = _dir(a), _dir(b)
    return float(
        np.degrees(
            np.arccos(np.clip(a @ b / np.linalg.norm(a) / np.linalg.norm(b), -1, 1))
        )
    )


def _two_segments(scale=1.0, noise=0.0, seed=0, garbage=0, name="X"):
    """The Zijderveld path of a soft overprint removed up to 300 degrees and a ChRM that
    decays to the origin at 600 degrees: two straight segments sharing the step 300."""
    rng = np.random.default_rng(seed)
    levels = np.arange(0, 650, 50)
    steps = []
    for lv in levels:
        over = max(0.0, 1 - lv / 300)
        chrm = 1.0 if lv <= 300 else max(0.0, 1 - (lv - 300) / 300)
        M = 0.3 * chrm * _dir(HIGH) + 0.2 * over * _dir(LOW)
        M = M + rng.normal(0, noise * 0.3, 3)
        steps.append(DemagStep(f"T{lv}", float(lv), Vector3(*(M * scale))))
    if garbage:  # noisy last steps
        for i in range(1, garbage + 1):
            g = rng.normal(0, 0.05 * scale, 3)
            old = steps[-i]
            steps[-i] = DemagStep(old.label, old.level, Vector3(*g))
    return Specimen(name=name, steps=steps, demag="thermal", site="S")


def _straight(n=12):
    steps = [
        DemagStep(f"M{10 * i}", 10.0 * i, Vector3(*(0.2 * (n - i) * _dir(HIGH))))
        for i in range(n)
    ]
    return Specimen(name="L", steps=steps, demag="af")


# -- window fits -----------------------------------------------------------------


@pytest.mark.parametrize("scale", [1.0, 1e-6])
def test_window_fits_match_pca_window_by_window(scale):
    spec = _two_segments(scale, noise=0.02, seed=3)
    table = spec.window_fits(min_steps=4)
    assert len(table) == sum(range(1, len(spec) - 2))  # all windows of >= 4 steps
    assert set(table.columns) >= {
        "start",
        "end",
        "n",
        "mad",
        "dang",
        "alpha",
        "nmad",
        "dec",
        "inc",
    }
    for row in table.iloc[::7].itertuples():
        pos = list(range(row.start, row.end + 1))
        free = spec.pca(pos, coords="specimen")
        anchored = spec.pca(pos, anchor=True, coords="specimen")
        assert row.n == len(pos) == free.n
        assert row.mad == pytest.approx(free.mad, abs=1e-5)
        assert row.dang == pytest.approx(free.dang, abs=1e-4)
        assert row.mad_anchored == pytest.approx(anchored.mad, abs=1e-5)
        assert row.alpha == pytest.approx(
            _angle(free.vector, anchored.vector) % 180, abs=1e-4
        )
        dec, inc = free.vector.geo
        assert (row.dec, row.inc) == pytest.approx((dec % 360, inc), abs=1e-4)
        assert row.nmad == pytest.approx(row.mad / np.sqrt(row.n - 2), rel=1e-9)


def test_window_fits_limits_and_selection():
    spec = _two_segments()
    t = spec.window_fits(min_steps=5, max_steps=6)
    assert set(t["n"]) == {5, 6}
    sub = spec.window_fits(min_steps=4, steps=slice(2, 9))  # 7 steps
    assert sub["start"].min() == 2 and sub["end"].max() == 8
    assert len(spec.window_fits(min_steps=40)) == 0
    assert len(Specimen(name="e").window_fits()) == 0


# -- one component --------------------------------------------------------------


def test_find_component_chrm_at_the_last_steps():
    spec = _two_segments()
    comp = spec.find_component(end="last", coords="specimen")
    assert comp.levels == (300.0, 600.0) or comp.levels[1] == 600.0
    assert comp.mad < 0.01 and _angle(comp.vector, HIGH) < 0.1
    assert comp.alpha < 0.1  # decays to the origin
    assert Component.from_json(comp.to_json()).levels == comp.levels
    # in geographic coordinates the component is rotated with the specimen
    spec.azimuth, spec.hade = 40.0, 20.0
    geo = spec.find_component(end="last")
    assert geo.coords == "geo" and _angle(geo.vector, comp.vector) > 1


def test_find_component_scores_and_origin_test():
    spec = _two_segments(noise=0.01, seed=5)
    best = {s: spec.find_component(score=s) for s in ("mad", "longest", "nmad")}
    assert best["longest"].n >= best["mad"].n
    assert all(
        _angle(c.vector, HIGH) < 3 or _angle(c.vector, LOW) < 15 for c in best.values()
    )
    # the overprint segment points away from the origin: the origin test removes it
    line = _two_segments()
    only_origin = line.find_component(score="longest", alpha_max=1.0)
    assert only_origin is not None and _angle(only_origin.vector, HIGH) < 5
    assert only_origin.alpha < 1.0
    assert line.find_component(criteria=ComponentCriteria(n_min=14)) is None


def test_find_component_tail_and_errors():
    spec = _two_segments(garbage=2, seed=2)
    # the two last steps are noise: a window ending at the last step is not accepted
    assert spec.find_component(end="last", coords="specimen") is None
    comp = spec.find_component(end="last", tail=2, coords="specimen")
    assert comp is not None and comp.levels[1] <= 550
    with pytest.raises(ValueError, match="score"):
        spec.find_component(score="x")
    with pytest.raises(ValueError, match="end"):
        spec.find_component(end="x")


# -- several components ----------------------------------------------------------


def test_find_components_two_segments():
    spec = _two_segments()
    comps = spec.find_components(coords="specimen")
    assert [c.levels for c in comps] == [
        (0.0, 300.0),
        (300.0, 600.0),
    ]  # boundary step shared
    assert _angle(comps[0].vector, LOW) < 0.1 and _angle(comps[1].vector, HIGH) < 0.1
    assert all(c.mad < 0.01 and c.alpha is not None for c in comps)
    assert (
        comps[0].alpha > 10 > comps[1].alpha
    )  # the overprint does not point to the origin
    noisy = _two_segments(noise=0.01, seed=11)
    comps = noisy.find_components(coords="specimen")
    assert len(comps) == 2
    assert _angle(comps[0].vector, LOW) < 5 and _angle(comps[1].vector, HIGH) < 3
    assert comps[0].levels[1] in (250.0, 300.0, 350.0)
    # no shared step: the segments are disjoint
    apart = spec.find_components(share_boundary=False, coords="specimen")
    assert all(a.levels[1] < b.levels[0] for a, b in itertools.pairwise(apart))


def test_find_components_leaves_noisy_steps_out_and_respects_limits():
    spec = _two_segments(garbage=2, seed=2)
    comps = spec.find_components(coords="specimen")
    assert comps and max(c.levels[1] for c in comps) <= 550
    assert len(spec.find_components(max_components=1, coords="specimen")) == 1
    assert spec.find_components(max_components=0) == []
    strict = spec.find_components(
        ComponentCriteria(n_min=4, mad_max=1e-9, dang_max=None)
    )
    assert strict == [] or all(c.mad < 1e-9 for c in strict)


def test_straight_path_is_one_component_and_scale_does_not_matter():
    comps = _straight().find_components(coords="specimen")
    assert len(comps) == 1 and comps[0].n == 12 and comps[0].mad < 1e-6
    a = _two_segments(1.0).find_components(coords="specimen")
    b = _two_segments(1e-7).find_components(coords="specimen")
    assert [c.levels for c in a] == [c.levels for c in b]
    assert _angle(a[1].vector, b[1].vector) < 1e-6


def test_interpret_auto_stores_and_replaces_components():
    spec = _two_segments()
    spec.interpret("ChRM", slice(7, None))
    comps = spec.interpret_auto()
    assert sorted(spec.components) == ["C1", "C2", "ChRM"]
    assert [c.coords for c in comps] == ["geo", "geo"]
    spec.components["C9"] = spec.components["C1"]
    spec.interpret_auto(max_components=1)  # replaces C1, C2 and C9, keeps ChRM
    assert sorted(spec.components) == ["C1", "ChRM"]
    spec.interpret_auto(prefix="A", max_components=1)
    assert sorted(spec.components) == ["A1", "C1", "ChRM"]


# -- criteria and grading ---------------------------------------------------------


def test_criteria_defaults_and_checks():
    legacy = ComponentCriteria()
    assert (legacy.n_min, legacy.mad_max, legacy.dang_max, legacy.alpha_max) == (
        4,
        5.0,
        10.0,
        None,
    )
    assert SEARCH_CRITERIA.mad_max == 10.0 and SEARCH_CRITERIA.dang_max is None
    spec = _two_segments(noise=0.01, seed=7)
    c = spec.pca(slice(0, 5), coords="specimen")
    failed = ComponentCriteria(
        n_min=8, mad_max=0.001, dang_max=0.001, alpha_max=0.001
    ).check(Component.from_json({**c.to_json(), "alpha": 5.0}))
    assert len(failed) == 4 and failed[0].startswith("n = 5 < 8")
    assert any(f.startswith("MAD") for f in failed) and any(
        f.startswith("DANG") for f in failed
    )
    assert any(f.startswith("alpha") for f in failed)
    assert ComponentCriteria(None, None, None, None).passes(c)
    anchored = spec.pca(slice(0, 5), anchor=True, coords="specimen")
    assert not any("DANG" in f for f in ComponentCriteria(dang_max=0.0).check(anchored))


def _sample(n=6):
    specs = []
    for i in range(n):
        s = _two_segments(noise=0.002, seed=i, name=f"S{i}")
        s.interpret("ChRM", slice(7, None), anchor=True)
        specs.append(s)
    return Sample(specs, name="site")


def test_sample_grade_and_filtered_means():
    smp = _sample()
    smp.specimens[0].interpret("ChRM", slice(3, 10))  # straddles the corner: a poor fit
    grade = smp.grade("ChRM", ComponentCriteria(n_min=4, mad_max=1.0, dang_max=None))
    assert list(grade.columns) == [
        "specimen",
        "kind",
        "n",
        "mad",
        "dang",
        "alpha",
        "passed",
        "failed",
    ]
    assert (
        len(grade) == 6
        and not grade["passed"].iloc[0]
        and grade["passed"].iloc[1:].all()
    )
    assert grade["failed"].iloc[0].startswith("MAD")
    crit = ComponentCriteria(n_min=4, mad_max=1.0, dang_max=None)
    with pytest.warns(UserWarning, match="1 of 6 components 'ChRM' fail"):
        mean = smp.mean("ChRM", criteria=crit)
    assert mean["n_lines"] == 5
    with pytest.warns(UserWarning, match="fail"):
        assert len(smp.directions("ChRM", criteria=crit)) == 5
    assert len(smp.directions("ChRM")) == 6  # no criteria, no filtering
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        smp.mean("ChRM", criteria=ComponentCriteria(None, None, None))
    assert len(smp.grade("missing")) == 0


def test_sample_interpret_auto_and_json():
    smp = Sample([_two_segments(name="a"), _two_segments(1e-3, name="b")], name="s")
    out = smp.interpret_auto()
    assert set(out) == {"a", "b"} and all(len(v) == 2 for v in out.values())
    comp = smp["a"].components["C2"]
    assert comp.alpha < 1 and comp.levels == (300.0, 600.0)
    back = Specimen.from_json(smp["a"].to_json())
    assert back.components["C2"].levels == (300.0, 600.0)
    assert back.components["C2"].alpha == pytest.approx(comp.alpha)
    old = {k: v for k, v in comp.to_json().items() if k not in ("alpha", "levels")}
    c = Component.from_json(old)
    assert c.alpha is None and c.levels is None


# -- stacking and fitting by levels ------------------------------------------------


def test_stack_of_identical_specimens_and_options():
    specs = [_two_segments(scale=1.0 + 0.5 * i, name=f"s{i}") for i in range(4)]
    smp = Sample(specs, name="site")
    st = smp.stack(coords="specimen")
    assert (
        st.name == "site-stack" and st.demag == "thermal" and len(st) == len(specs[0])
    )
    base = np.asarray(specs[0].V)
    assert np.allclose(
        np.asarray(st.V), base / np.linalg.norm(base[0])
    )  # NRM-normalized
    unit = smp.stack(coords="specimen", normalize="unit")
    norms = np.linalg.norm(np.asarray(unit.V), axis=1)
    assert (
        np.allclose(norms[:-1], 1) and norms[-1] == 0
    )  # the last step is the zero vector
    raw = smp.stack(coords="specimen", normalize=None)
    assert np.allclose(
        np.asarray(raw.V)[0], np.mean([np.asarray(s.V)[0] for s in specs], axis=0)
    )
    # the stack is a specimen: the search runs on it
    comps = st.find_components(coords="specimen")
    assert [c.levels for c in comps] == [(0.0, 300.0), (300.0, 600.0)]
    # levels that fewer than min_specimens have are dropped
    specs[0].steps.pop()
    specs[1].steps.pop()
    assert len(Sample(specs).stack(coords="specimen", min_specimens=3)) == len(base) - 1
    assert len(Sample(specs).stack(coords="specimen", min_specimens=4)) == len(base) - 1
    with pytest.raises(ValueError, match="shared by 9"):
        smp.stack(min_specimens=9)
    with pytest.raises(ValueError, match="normalize"):
        smp.stack(normalize="x")


def test_stack_in_tilt_coordinates_uses_the_orientation():
    a, b, c = (_two_segments(name=n) for n in "abc")
    a.azimuth = 90.0
    smp = Sample([a, b, c])
    geo, spec = smp.stack("geo"), smp.stack("specimen")
    assert not np.allclose(np.asarray(geo.V), np.asarray(spec.V))
    expected = np.mean([np.asarray(s.vectors("geo"))[0] for s in smp], axis=0)
    assert np.allclose(
        np.asarray(geo.V)[0], expected / np.linalg.norm(np.asarray(a.V)[0])
    )


def test_sample_interpret_by_levels_equals_pca_between():
    smp = _sample(3)
    smp.specimens[2].steps = smp.specimens[2].steps[:8]  # reaches 350 only
    with pytest.warns(UserWarning, match="1 specimens have too few steps"):
        fitted = smp.interpret("R", 400, 600)
    assert fitted == 2 and "R" not in smp.specimens[2].components
    for s in smp.specimens[:2]:
        mask = (s.levels >= 400) & (s.levels <= 600)
        ref = s.pca(mask, coords="specimen")
        assert np.allclose(np.asarray(s.components["R"].vector), np.asarray(ref.vector))
        assert s.components["R"].levels == (400.0, 600.0)
    assert smp.interpret("O", 0, 300, origin=True) == 3  # keywords of pca
    assert smp.specimens[0].components["O"].origin


def test_workflow_stack_then_fit_every_specimen():
    smp = Sample(
        [_two_segments(1.0 + i, noise=0.01, seed=i, name=f"s{i}") for i in range(5)]
    )
    st = smp.stack("specimen")
    chrm = st.find_component(end="last", coords="specimen")
    assert smp.interpret("ChRM", *chrm.levels) == 5
    mean = smp.mean("ChRM", coords="specimen")
    assert _angle(mean["mu"], HIGH) < 3


# -- real specimens: the picks of Remasoft are the reference, not the truth --------


def _pick(site, name):
    return next(s for s in data.load("rs3_components", site=site) if s.name == name)


def test_bundled_rs3_picks_and_the_automatic_search():
    smp = data.load("rs3_components")
    assert sorted(smp) == ["GC083", "K549", "K583", "K599_2", "KL54"]
    k583 = _pick("K583", "A-1")
    assert sorted(k583.components) == ["CHT", "CLT", "CMT"]
    assert k583.components["CHT"].origin and k583.components["CHT"].kind == "line"
    assert any(c.kind == "plane" for c in _pick("KL54", "A-3").components.values())
    # the ChRM of the automatic search is close to the highest Remasoft pick
    for site, name, top in (
        ("K583", "A-1", "CHT"),
        ("GC083", "6-11", "CHT"),
        ("K549", "A-8", "CLT"),
    ):
        spec = _pick(site, name)
        auto = spec.find_component(end="last", coords="specimen")
        assert auto is not None
        assert _angle(auto.vector, spec.components[top].vector) < 6
    # specimens whose last steps are noisy get no result at the last step
    assert _pick("K549", "C-1").find_component(end="last") is None
    assert _pick("KL54", "A-3").find_component(end="last") is None
    levels = [c.levels for c in _pick("K583", "A-1").find_components()]
    assert levels == [(0.0, 500.0), (500.0, 600.0)]


# -- plots ------------------------------------------------------------------------


def test_zijderveld_components_and_component_search_plot():
    spec = _two_segments(noise=0.01, seed=1)
    spec.interpret_auto()
    fig = plt.figure()
    plain = ZijderveldPlot(spec)
    plain.render2fig(fig)
    n_plain = len(fig.axes[0].lines)
    fig2 = plt.figure()
    ZijderveldPlot(spec, components=True).render2fig(fig2)
    # rings and fitted lines in the two projections
    assert len(fig2.axes[0].lines) == n_plain + 4 * len(spec.components)
    labels = [t.get_text() for t in fig2.axes[0].get_legend().get_texts()]
    assert "C1" in labels and "C2" in labels
    fig3 = plt.figure()
    ComponentSearchPlot(spec, min_steps=4).render2fig(fig3)
    assert len(fig3.axes) >= 2  # windows and colour bar
    plt.close("all")
    with pytest.raises(TypeError):
        ComponentSearchPlot(spec, bogus=1)
