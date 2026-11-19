"""Relative paleointensity, susceptibility jump and core sections."""

import itertools

import matplotlib.pyplot as plt
import numpy as np
import pytest

from apsg.math._vector import Vector3
from apsg.paleomag import (
    DemagStep,
    KappaTPlot,
    RPIEstimate,
    Sample,
    Specimen,
    align_core_sections,
    relative_paleointensity,
)

LEVELS = [0, 5, 10, 20, 30, 40, 60]


def _specimen(name, intensities, levels=LEVELS, k=None, treatment=None, demag="af"):
    steps = [
        DemagStep(
            f"M{lv:03g}",
            float(lv),
            Vector3(0.0, 0.0, I),
            k=None if k is None else k[i],
            treatment=treatment,
        )
        for i, (lv, I) in enumerate(zip(levels, intensities, strict=False))
    ]
    return Specimen(name=name, steps=steps, demag=demag)


# -- relative paleointensity ---------------------------------------------------


def test_rpi_susceptibility():
    spec = _specimen("S1", [4e-3, 3e-3, 2e-3], levels=[0, 10, 20], k=[2e-4, 2e-4, 2e-4])
    est = relative_paleointensity(spec, method="susceptibility")
    assert isinstance(est, RPIEstimate)
    assert est.ratio == pytest.approx(20.0) and est.slope is None and est.n == 1
    # the first step with a susceptibility is used
    spec.steps[0] = DemagStep("NRM", 0.0, Vector3(0, 0, 4e-3))
    assert relative_paleointensity(spec).ratio == pytest.approx(20.0)
    with pytest.raises(ValueError, match="susceptibility"):
        relative_paleointensity(_specimen("S2", [1e-3, 1e-3, 1e-3], levels=[0, 10, 20]))


def test_rpi_stepwise_af_normalization():
    arm = np.array([5e-2, 4.5e-2, 4e-2, 3e-2, 2e-2, 1e-2, 4e-3])
    nrm = 0.8 * arm + 1e-4 * np.array([0, 1, -1, 1, -1, 1, -1])  # slope 0.8 with noise
    spec, norm = _specimen("S1", nrm), _specimen("S1", arm)
    est = relative_paleointensity(spec, norm, method="af")
    assert est.method == "af" and est.n == len(LEVELS)
    assert est.ratio == pytest.approx(np.mean(nrm / arm))
    assert est.slope == pytest.approx(0.8, abs=0.02) and est.r2 > 0.999
    assert est.r == pytest.approx(np.sqrt(est.r2))
    # a window of levels
    part = relative_paleointensity(spec, norm, method="af", min_level=10, max_level=40)
    assert list(part.levels) == [10, 20, 30, 40]
    assert part.nrm == pytest.approx(nrm[2:6])
    with pytest.raises(ValueError, match="normalizer"):
        relative_paleointensity(spec, method="af")


def test_rpi_only_shared_levels_are_used():
    spec = _specimen("S1", [4, 3, 2, 1.0], levels=[0, 10, 20, 30])
    norm = _specimen("S1", [8, 4, 2.0], levels=[0, 20, 40])
    est = relative_paleointensity(spec, norm, method="af")
    assert list(est.levels) == [0, 20] and est.ratio == pytest.approx(0.5)
    with pytest.raises(ValueError, match="shared"):
        relative_paleointensity(spec, _specimen("S1", [1.0], levels=[5]), method="af")


def test_rpi_arm_acquisition_has_inverted_slope():
    nrm = np.array([1.0, 0.9, 0.7, 0.5, 0.3, 0.2, 0.1])
    acquired = 1.0 - nrm + 0.05  # ARM grows as the NRM is demagnetized
    spec = _specimen("S1", nrm)
    spec.arm = [
        DemagStep(f"ARM{lv}", float(lv), Vector3(0, 0, a), treatment="arm")
        for lv, a in zip(LEVELS, acquired, strict=True)
    ]
    est = relative_paleointensity(spec, method="arm_acquisition")
    assert est.slope == pytest.approx(1.0) and est.r2 == pytest.approx(1.0)
    assert est.ratio == pytest.approx(np.mean(nrm / acquired))
    other = Specimen(name="S1")
    other.arm = spec.arm
    assert relative_paleointensity(spec, other, method="arm_acquisition").slope == (
        pytest.approx(1.0)
    )
    with pytest.raises(ValueError, match="no ARM"):
        relative_paleointensity(_specimen("S2", nrm), method="arm_acquisition")


def test_rpi_options_are_checked():
    with pytest.raises(ValueError, match="method"):
        relative_paleointensity(_specimen("S1", [1.0, 2.0]), method="other")
    with pytest.raises(ValueError, match="no steps"):
        relative_paleointensity(Specimen(name="E"))


def test_sample_rpi_matches_normalizers_by_name():
    a = _specimen("A", [1.0, 0.5, 0.2], levels=[0, 10, 20])
    b = _specimen("B", [2.0, 1.0, 0.4], levels=[0, 10, 20])
    norm = Sample([_specimen("A", [2.0, 1.0, 0.4], levels=[0, 10, 20])])
    with pytest.warns(UserWarning, match="no normalizer for B"):
        result = Sample([a, b]).rpi(norm, method="af")
    assert list(result) == ["A"] and result["A"].ratio == pytest.approx(0.5)
    ks = [
        _specimen("S", [1.0, 1.0], levels=[0, 10], k=[0.5, 0.5]),
        _specimen("T", [2.0, 1.0], levels=[0, 10], k=[0.5, 0.5]),
    ]
    out = Sample(ks).rpi()
    assert out["S"].ratio == pytest.approx(2.0) and out["T"].ratio == pytest.approx(4.0)


# -- susceptibility jump -------------------------------------------------------


def test_susceptibility_jump():
    levels = [20, 100, 200, 300, 400, 500, 600]
    k = [1.0, 1.1, 1.2, 1.5, 4.0, 9.0, 30.0]
    spec = _specimen("S1", [1.0] * 7, levels=levels, k=k, demag="thermal")
    assert spec.susceptibility_jump() == 400  # 4.0 > 2.5 * 1.5
    assert spec.susceptibility_jump(factor=1.2) == 300
    assert spec.susceptibility_jump(factor=100) is None
    # steps without susceptibility are skipped
    gap = _specimen(
        "S1",
        [1.0] * 4,
        levels=[20, 100, 200, 300],
        k=[1.0, None, np.nan, 3.0],
        demag="thermal",
    )
    assert gap.susceptibility_jump() == 300
    assert _specimen("S2", [1.0] * 3, demag="thermal").susceptibility_jump() is None
    with pytest.raises(ValueError, match="thermal"):
        _specimen("S3", [1.0] * 7, k=k, demag="af").susceptibility_jump()


def test_kappa_t_plot():
    levels = [20, 100, 200, 300, 400, 500]
    k = [1.0, 1.1, 1.2, 1.5, 4.0, 9.0]
    spec = _specimen("S1", [1.0, 0.9, 0.8, 0.6, 0.3, 0.1], levels, k, demag="thermal")
    plot = KappaTPlot(spec)
    assert "jump at 400°C" in plot._default_title()
    fig = plt.figure()
    plot.render2fig(fig)
    assert len(fig.axes) >= 2  # susceptibility and intensity axes
    plt.close("all")
    with pytest.raises(ValueError, match="no susceptibility"):
        KappaTPlot(_specimen("S2", [1.0, 0.5], levels=[20, 100], demag="thermal"))


# -- core sections -------------------------------------------------------------


def _section(name, decs, depth0, turn):
    """Specimens with a ChRM of the declinations `decs` seen turned by `turn`."""
    specs = []
    for i, dec in enumerate(decs):
        M = Vector3((dec - turn) % 360, 40.0) * 1e-3
        s = Specimen(
            name=f"{name}_{i}",
            site=name,
            depth=depth0 + 0.1 * i,
            steps=[DemagStep("NRM", 0.0, M)],
        )
        s.interpret("ChRM", slice(None), kind="line", anchor=True)
        specs.append(s)
    return Sample(specs, name=name)


def _decs(section):
    return [s.component("ChRM", "geo").vector.geo[0] % 360 for s in section]


def _joins(sections):
    """Declination jumps between the bottom of a section and the top of the next."""
    return [
        ((_decs(b)[0] - _decs(a)[-1] + 180) % 360) - 180
        for a, b in itertools.pairwise(sections)
    ]


def test_align_core_sections_top_down():
    true = [350.0, 355.0, 0.0, 5.0, 10.0, 15.0, 20.0, 25.0, 30.0]
    turns = [100.0, 250.0, 40.0]
    sections = [
        _section(f"C{i + 1}", true[3 * i : 3 * i + 3], 0.3 * i, t)
        for i, t in enumerate(turns)
    ]
    offsets = align_core_sections(sections, "ChRM", margin=1, declination=350.0)
    assert set(offsets) == {"C1", "C2", "C3"}
    assert _decs(sections[0])[0] == pytest.approx(350.0, abs=1e-6)  # the first top
    assert _joins(sections) == pytest.approx([0, 0], abs=1e-6)  # no jumps at joins
    # the sections are turned rigidly: the drift within each is unchanged
    for i, s in enumerate(sections):
        d = np.diff(_decs(s)) % 360
        assert d == pytest.approx([5.0, 5.0], abs=1e-6)
    # a section turned by t must be turned back by 360 - t, plus the offset
    # that places its top at the bottom of the section above
    assert offsets["C1"] == pytest.approx(turns[0])


def test_align_core_sections_mean_target_and_dict():
    sections = {
        "A": _section("A", [10.0, 20.0, 30.0], 0.0, 70.0),
        "B": _section("B", [40.0, 50.0, 60.0], 0.3, 200.0),
    }
    offsets = align_core_sections(
        sections, "ChRM", margin=1, target="mean", declination=0.0
    )
    assert set(offsets) == {"A", "B"}
    secs = list(sections.values())
    assert _joins(secs) == pytest.approx([0], abs=1e-6)
    R = np.sum(
        [
            np.asarray(s.component("ChRM", "geo").vector)
            for k in "AB"
            for s in sections[k]
        ],
        axis=0,
    )
    assert np.degrees(np.arctan2(R[1], R[0])) == pytest.approx(0.0, abs=1e-6)


def test_align_core_sections_uses_depth_order_and_validates():
    sec = _section("A", [10.0, 20.0, 30.0], 0.0, 0.0)
    sec.specimens.reverse()  # listed bottom first, ordered by depth
    align_core_sections([sec], "ChRM", margin=1, declination=100.0)
    by_depth = sorted(sec, key=lambda s: s.depth)
    assert _decs(Sample(by_depth))[0] == pytest.approx(100.0, abs=1e-6)
    with pytest.raises(ValueError, match="target"):
        align_core_sections([sec], "ChRM", target="x")
    with pytest.raises(ValueError, match="margin"):
        align_core_sections([sec], "ChRM", margin=0)
    with pytest.raises(ValueError, match="no specimen"):
        align_core_sections([sec], "other")
    # unnamed duplicates are keyed by position
    twin = _section("A", [10.0, 20.0], 1.0, 0.0)
    assert set(align_core_sections([sec, twin], "ChRM")) == {0, 1}
