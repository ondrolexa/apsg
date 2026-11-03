"""Tests of the paleomagnetic Specimen/Sample classes and their readers."""

import re
import warnings
from dataclasses import replace

import matplotlib.pyplot as plt
import numpy as np
import pytest

import apsg
from apsg.feature import Vector3Set
from apsg.math._vector import Vector3
from apsg.paleomag import (
    Component,
    DemagStep,
    Sample,
    Specimen,
    bedding_vector,
    core_rotation,
    demag_plot,
    lines_planes_mean,
    stereo_plot,
    untilt_rotation,
    vgp,
    zijderveld_plot,
)


def _dir(dec, inc):
    return np.array(
        [
            np.cos(np.radians(dec)) * np.cos(np.radians(inc)),
            np.sin(np.radians(dec)) * np.cos(np.radians(inc)),
            np.sin(np.radians(inc)),
        ]
    )


def _angle(a, b):
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    c = a @ b / (np.linalg.norm(a) * np.linalg.norm(b))
    return np.degrees(np.arccos(np.clip(c, -1, 1)))


LEVELS = [0, 100, 200, 300, 350, 400, 450, 500, 525, 550, 575, 600]


def _synthetic(name="S1", **kwargs):
    """Thermal demagnetization of a soft overprint and a hard ChRM."""
    chrm, over = _dir(320, -15), _dir(30, 60)
    steps = []
    for i, lv in enumerate(LEVELS):
        f_over = max(0.0, 1 - lv / 300)
        f_chrm = max(0.02, 1 - (lv / 610) ** 4)
        M = 0.3 * f_chrm * chrm + 0.1 * f_over * over
        label = f"T{lv}"
        steps.append(DemagStep(label, float(lv), Vector3(M), a95=1.0 + i / 10))
    meta = {
        "name": name,
        "demag": "thermal",
        "azimuth": 207.0,
        "hade": 81.0,
        "bedding": bedding_vector(200, 85),
        "volume": 8e-6,
        "site": "SITE",
        "latitude": 47.05,
        "longitude": 91.53,
        "rock": "Basalt",
    }
    return Specimen(**{"steps": steps, **meta, **kwargs})


def _write(spec, path):
    getattr(spec, "to_" + path.suffix[1:])(path)
    return path


def _pmd_columns(path):
    """Dg Ig Ds Is columns of a PMD file."""
    text = path.read_text(encoding="latin-1").split("\x1a")[0]
    rows = re.findall(
        r"E[-+]\d+\s+(-?[\d.]+)\s+(-?[\d.]+)\s+(-?[\d.]+)\s+(-?[\d.]+)\s+[\d.]+", text
    )
    return np.array(rows, dtype=float)


def _rs3_columns(path):
    """Dge Ige Dtc Itc columns of the measurement rows of an RS3 file."""
    rows = []
    for r in path.read_text(encoding="latin-1").splitlines()[3:]:
        if r.strip() and r[:2].strip() not in ("C", "G"):
            rows.append(
                [float(r[a:b]) for a, b in ((39, 45), (45, 51), (51, 57), (57, 63))]
            )
    return np.array(rows)


def _check_columns(spec, cols, tol=0.3):
    for g, t, (dg, ig, ds, is_) in zip(spec.geo, spec.tilt, cols, strict=True):
        assert _angle(g, _dir(dg, ig)) < tol
        assert _angle(t, _dir(ds, is_)) < tol


def _line_specimen(name="S", n=8, offset=(0.0, 0.0, 0.0), **kwargs):
    """Specimen whose magnetization decays along x towards `offset`."""
    steps = [
        DemagStep(f"T{10 * i}", 10.0 * i, Vector3(np.array(offset) + [n - i, 0, 0]))
        for i in range(n)
    ]
    return Specimen(name=name, steps=steps, demag="thermal", **kwargs)


# ---------------------------------------------------------------------------
# geometry
# ---------------------------------------------------------------------------


def test_core_rotation_identity_and_pmtools_matrix():
    assert np.allclose(core_rotation(0, 0), np.eye(3))
    a, b = 123.0, -29.0  # negative hade as in some PMD files
    la, ph = np.radians(a), np.radians(90 - b)
    pmtools = np.array(
        [
            [np.cos(la) * np.sin(ph), -np.sin(la), np.cos(ph) * np.cos(la)],
            [np.sin(ph) * np.sin(la), np.cos(la), np.sin(la) * np.cos(ph)],
            [-np.cos(ph), 0, np.sin(ph)],
        ]
    )
    assert np.allclose(core_rotation(a, b), pmtools)


def test_tilt_correction_tauxe_example():
    # Tauxe (2003) Ex. 3.2: (5.3, 71.6) with strike 135, dip 21 -> (285.7, 76.6)
    R = untilt_rotation(bedding_vector(135 + 90, 21))
    assert _angle(R @ _dir(5.3, 71.6), _dir(285.7, 76.6)) < 0.1


@pytest.mark.parametrize("dip", [0, 30, 90, 120, 180])
def test_untilt_restores_bedding(dip):
    bed = bedding_vector(40, dip)
    up = untilt_rotation(bed) @ np.asarray(bed)
    assert np.allclose(up, [0, 0, -1], atol=1e-12)
    half = untilt_rotation(bed, 0.5) @ np.asarray(bed)
    assert _angle(half, [0, 0, -1]) == pytest.approx(dip / 2, abs=1e-9)


def test_overturned_bed_differs_from_upright():
    upright = untilt_rotation(bedding_vector(220, 60))  # same plane as below
    overturned = untilt_rotation(bedding_vector(40, 120))
    assert not np.allclose(upright, overturned)


# ---------------------------------------------------------------------------
# readers and writers (files generated in the tests)
# ---------------------------------------------------------------------------


def test_rs3_roundtrip_and_metadata(tmp_path):
    spec = _synthetic()
    path = _write(spec, tmp_path / "s1.rs3")
    back = Specimen.from_file(path)
    assert (back.name, back.site, back.rock) == ("S1", "SITE", "Basalt")
    assert (back.latitude, back.longitude) == (47.05, 91.53)
    assert back.labels == spec.labels and back.demag == "thermal"
    assert np.allclose(back.levels, LEVELS)
    assert np.allclose(back.intensity, spec.intensity, rtol=1e-6)
    assert np.allclose(back.rotation("tilt"), spec.rotation("tilt"), atol=1e-3)
    _check_columns(back, _rs3_columns(path))


def test_rs3_af_roundtrip(tmp_path):
    spec = _line_specimen(volume=1e-5)
    spec.demag = "af"
    back = Specimen.from_rs3(_write(spec, tmp_path / "af.rs3"))
    assert back.demag == "af" and back.labels[1] == "M10"


def test_rs3_excludes_fit_rows(tmp_path):
    path = _write(_synthetic(), tmp_path / "s1.rs3")
    fit = (
        f"{'C':<2} {'CHT':<10}{0.13:>14.7g}{78.7:6.1f}{-24.8:6.1f}{320.5:6.1f}"
        f"{-14.0:6.1f}{276.7:6.1f}{-30.7:6.1f}{'':12}{5.2:6.1f}"
    )
    path.write_text(path.read_text(encoding="latin-1") + fit + "\r\n", "latin-1")
    assert len(Specimen.from_rs3(path)) == len(LEVELS)
    assert len(Specimen.from_rs3(path, exclude=())) == len(LEVELS) + 1


def _edit_rs3_header(path, sdec, sinc, p1):
    lines = path.read_text(encoding="latin-1").splitlines()
    h = lines[1]
    lines[1] = h[:74] + f"{sdec:<6g}{sinc:<6g}" + h[86:110] + f"{p1:<3d}" + h[113:]
    path.write_text("\r\n".join(lines), encoding="latin-1")
    return path


def test_rs3_orientation_parameter_6(tmp_path):
    spec = _synthetic()  # written with P1 = 12: SDec 27, SInc 81
    path = _write(spec, tmp_path / "s1.rs3")
    # the mark at 6 o'clock: SDec turned by 180 degrees and SInc negated
    back = Specimen.from_rs3(_edit_rs3_header(path, 207, -81, 6))
    assert np.allclose(back.rotation("geo"), spec.rotation("geo"), atol=1e-6)


def test_rs3_unsupported_orientation_parameters(tmp_path):
    path = _edit_rs3_header(_write(_synthetic(), tmp_path / "s1.rs3"), 27, 81, 3)
    with pytest.raises(NotImplementedError, match="P1=3"):
        Specimen.from_rs3(path)


def test_from_file_rejects_unknown_extension(tmp_path):
    with pytest.raises(ValueError, match="Unsupported"):
        Specimen.from_file(tmp_path / "x.dat")


def test_standard_pmd_roundtrip_and_glued_labels(tmp_path):
    base = _synthetic()
    labels = ["NRM", "90°C", "150°C"] + [f"{lv:g}°C" for lv in base.levels[3:]]
    spec = Specimen(
        name="143",
        steps=[
            DemagStep(lab, s.level, Vector3(-abs(s.M.x), s.M.y, s.M.z), a95=s.a95)
            for lab, s in zip(labels, base.steps, strict=True)
        ],
        azimuth=170.0,
        hade=-29.0,
        bedding=bedding_vector(65, 73),
        volume=11e-6,
    )
    path = _write(spec, tmp_path / "std.pmd")
    # glue a 5-character label to a negative Xc, as some PMD writers do
    text = path.read_text(encoding="latin-1").replace("150°C -", "150°C-")
    assert re.search(r"^150°C-\d", text, re.MULTILINE)
    path.write_text(text, encoding="latin-1")
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        back = Specimen.from_pmd(path)
    assert back.labels == labels
    assert back.levels[1] == 90 and back.demag == "thermal"
    assert back.hade == pytest.approx(-29.0)
    assert back.volume == pytest.approx(11e-6)
    assert np.allclose(back.rotation("tilt"), spec.rotation("tilt"), atol=1e-3)
    assert np.allclose(back.intensity, spec.intensity, rtol=1e-2)
    _check_columns(back, _pmd_columns(path))


def _as_remasoft(path):
    """Rewrite a standard PMD header the way Remasoft exports it: the raw
    sampling azimuth (canonical - 180) and the bedding dip direction as `s`."""
    text = path.read_text(encoding="latin-1")
    a = float(re.search(r"a=\s*([-\d.]+)", text)[1])
    s = float(re.search(r"s=\s*([-\d.]+)", text)[1])
    text = re.sub(r"a=\s*[-\d.]+", f"a={(a - 180) % 360:5.1f}", text, count=1)
    text = re.sub(r"s=\s*[-\d.]+", f"s={(s + 90) % 360:5.1f}", text, count=1)
    path.write_text(text, encoding="latin-1")
    return path


def test_remasoft_pmd_dialect_detected(tmp_path):
    spec = _synthetic()
    path = _as_remasoft(_write(spec, tmp_path / "rem.pmd"))
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        back = Specimen.from_pmd(path)
    assert np.allclose(back.rotation("tilt"), spec.rotation("tilt"), atol=1e-3)
    _check_columns(back, _pmd_columns(path))


def test_pmd_numeric_labels_need_demag(tmp_path):
    spec = _synthetic(steps=[replace(s, label=f"{s.level:g}") for s in _synthetic()])
    path = _write(spec, tmp_path / "num.pmd")
    assert Specimen.from_pmd(path).demag is None
    assert Specimen.from_pmd(path, demag="thermal").demag == "thermal"


def test_pmd_inconsistent_orientation_warns(tmp_path):
    path = _write(_synthetic(), tmp_path / "bad.pmd")
    text = path.read_text(encoding="latin-1")
    path.write_text(re.sub(r"a=\s*[-\d.]+", "a=100.0", text, count=1), "latin-1")
    with pytest.warns(UserWarning, match="does not reproduce"):
        Specimen.from_pmd(path)


def _squid_text(a, b, s, d, rows):
    out = ["site S1  ", f"        {a:5.1f} {b:5.1f} {s:5.1f} {d:5.1f} 8.0"]
    for code, value, m, dc, ic in rows:
        out.append(
            f"{code:<2} {value:>3} 000.0  00.0 000.0  00.0 {m:.2E} 000.5 "
            f"{dc:5.1f} {ic:5.1f} 0.1 0.1 0.1 user  2026-01-18 03:20:55 "
        )
    return "\r\n".join(out) + "\r\n"


def test_squid_reader(tmp_path):
    rows = [
        ("AF", "", 3.2e-5, 177.0, 14.0),
        ("AF", "100", 2.9e-5, 177.2, 12.8),
        ("AF", "150", 2.7e-5, 177.3, 12.3),
        ("ARM", " 1", 2.4e-5, 178.5, 86.5),
        ("AF", "100", 2.3e-5, 176.1, 86.4),
    ]
    path = tmp_path / "x.squid"
    path.write_text(_squid_text(270, 90, 270, 0, rows), encoding="latin-1")
    spec = Specimen.from_file(path)
    assert (spec.site, spec.name) == ("site", "S1")
    assert spec.labels == ["M000", "M010", "M015"]  # ARM block dropped
    assert np.allclose(spec.levels, [0, 10, 15])
    assert spec.demag == "af"
    assert spec.volume == pytest.approx(8e-6)
    assert spec.intensity[0] == pytest.approx(3.2e-2)
    assert (spec.azimuth, spec.hade) == (180.0, 90.0)
    geo0 = core_rotation(180, 90) @ _dir(177.0, 14.0)
    assert _angle(spec.geo[0], geo0) < 1e-6


# ---------------------------------------------------------------------------
# Specimen API
# ---------------------------------------------------------------------------


def test_not_in_default_namespace():
    # `from apsg import *` imports exactly apsg.__all__
    names = {"Specimen", "Sample", "vgp", "zijderveld_plot", "paleomag"}
    assert not names & set(apsg.__all__)
    assert not hasattr(apsg.feature, "Specimen")
    assert not hasattr(apsg.plotting, "zijderveld_plot")


def test_selection():
    spec = _synthetic()
    assert isinstance(spec[0], DemagStep)
    assert spec["T100"].level == 100
    with pytest.raises(KeyError):
        spec["T999"]
    assert spec[2:5].labels == ["T200", "T300", "T350"]
    assert spec[[0, 2]].labels == ["T0", "T200"]
    assert spec[["T0", "T600"]].levels.tolist() == [0, 600]
    assert len(spec[spec.levels > 500]) == 4
    assert spec.between(300, 400).labels == ["T300", "T350", "T400"]
    assert spec[2:5].name == spec.name


def test_validation():
    with pytest.raises(ValueError, match="demag"):
        Specimen(demag="xray")
    with pytest.raises(TypeError):
        Specimen(steps=[1, 2])
    with pytest.raises(ValueError, match="coords"):
        Specimen().rotation("paleo")
    with pytest.raises(ValueError, match="Volume"):
        _ = _line_specimen().moment


def test_json_roundtrip():
    spec = _synthetic()
    spec.interpret("ChRM", slice(4, None))
    back = Specimen.from_json(spec.to_json())
    assert isinstance(back, Specimen)
    assert back.labels == spec.labels and back.site == spec.site
    assert np.allclose(np.asarray(back.tilt), np.asarray(spec.tilt))
    assert back.component("ChRM").vector == spec.component("ChRM").vector
    smp = Sample.from_json(Sample([spec, back], name="SITE").to_json())
    assert isinstance(smp, Sample) and smp.names == ["S1", "S1"]


def test_dataframe(tmp_path):
    spec = _synthetic()
    df = spec.to_dataframe()
    assert len(df) == len(spec)
    cols = _rs3_columns(_write(spec, tmp_path / "s1.rs3"))
    assert np.allclose(df["Ige"], cols[:, 1], atol=0.06)
    assert np.allclose(df["Itc"], cols[:, 3], atol=0.06)


# ---------------------------------------------------------------------------
# processing
# ---------------------------------------------------------------------------


def test_pca_free_line_on_offset_line():
    spec = _line_specimen(offset=(0, 2, 1))
    free = spec.pca(coords="specimen")
    assert free.kind == "line" and free.n == len(spec)
    assert np.allclose(free.vector, [1, 0, 0])  # along the removed component
    assert free.mad == pytest.approx(0, abs=1e-6)
    assert free.dang > 10
    # unnormalized points: an offset line stays a line (MAD 0) only if the
    # vectors are not normalized before the fit
    anchored = spec.pca(coords="specimen", anchor=True)
    assert anchored.mad > 5
    assert anchored.dang is None


def test_pca_anchored_line_through_origin():
    spec = _line_specimen()
    comp = spec.pca(anchor=True, coords="specimen")
    assert np.allclose(comp.vector, [1, 0, 0]) and comp.mad < 1e-6


def test_pca_plane():
    rng = np.random.default_rng(1)
    pts = rng.normal(size=(6, 2))
    steps = [DemagStep(f"M{i}", i, Vector3(x, y, 0.0)) for i, (x, y) in enumerate(pts)]
    comp = Specimen(steps=steps).pca(kind="plane", coords="specimen")
    assert abs(abs(comp.vector[2]) - 1) < 1e-9
    assert comp.mad == pytest.approx(0, abs=1e-6)
    assert np.dot(comp.vector, np.cross(pts_to3(pts[0]), pts_to3(pts[-1]))) > 0
    with pytest.raises(ValueError, match="at least"):
        Specimen(steps=steps[:2]).pca(kind="plane")


def pts_to3(p):
    return np.array([p[0], p[1], 0.0])


def test_pca_frame_invariance_and_interpret():
    spec = _synthetic()
    sp = spec.pca(slice(3, 11), coords="specimen")
    tilt = spec.pca(slice(3, 11), coords="tilt")
    assert tilt.coords == "tilt"
    assert _angle(tilt.vector, spec.rotation("tilt") @ np.asarray(sp.vector)) < 1e-9
    assert tilt.mad == pytest.approx(sp.mad)
    geo = spec.interpret("ChRM", slice(3, 11))
    assert spec.components["ChRM"].coords == "specimen"
    assert _angle(geo.vector, spec.component("ChRM", "geo").vector) < 1e-9
    with pytest.raises(TypeError):
        spec.pca(3)


def test_mdf_and_vds():
    steps = [
        DemagStep(f"M{lv}", lv, Vector3(m, 0, 0))
        for lv, m in ((0, 1.0), (10, 0.8), (20, 0.4))
    ]
    spec = Specimen(steps=steps, demag="af")
    assert spec.mdf() == pytest.approx(17.5)
    assert spec.mdf(vds=True) == pytest.approx(17.5)
    assert spec.vds == pytest.approx(1.0)
    assert spec[:2].mdf() is None


# ---------------------------------------------------------------------------
# collection statistics
# ---------------------------------------------------------------------------


def _comp(vector, kind="line"):
    v = Vector3(vector).normalized()
    return Component(kind, v, 1.0, None, 3, ("a", "b", "c"))


def _sample(vectors, beddings=None, kinds=None):
    beddings = beddings or [Vector3(0, 0, -1)] * len(vectors)
    kinds = kinds or ["line"] * len(vectors)
    return Sample(
        [
            Specimen(name=f"S{i}", bedding=b, components={"ChRM": _comp(v, k)})
            for i, (v, b, k) in enumerate(zip(vectors, beddings, kinds))
        ],
        name="site",
    )


def test_lines_planes_mean_tauxe_example():
    # Tauxe (2003) Ex. 4.3: two directions and one great circle
    res = lines_planes_mean(
        [_dir(152.6, -60.7), _dir(178.1, -54.6)], [_dir(341.1, -26.2)]
    )
    dec, inc = res["mu"].geo
    assert (dec, inc) == pytest.approx((166.7, -60.1), abs=0.05)
    assert res["k"] == pytest.approx(76.3, abs=0.05)
    assert res["alpha"] == pytest.approx(16.7, abs=0.05)
    assert (res["n_lines"], res["n_planes"]) == (2, 1)


def test_sample_mean_lines_and_planes():
    smp = _sample(
        [_dir(152.6, -60.7), _dir(178.1, -54.6), _dir(341.1, -26.2)],
        kinds=["line", "line", "plane"],
    )
    res = smp.mean("ChRM")
    assert res["k"] == pytest.approx(76.3, abs=0.05)
    assert len(smp.directions("ChRM")) == 2 and len(smp.planes("ChRM")) == 1
    lines = _sample([_dir(10, 40), _dir(20, 45), _dir(15, 50)])
    expected = Vector3Set(list(lines.directions("ChRM"))).fisher_statistics()
    got = lines.mean("ChRM")
    assert got["mu"] == expected["mu"] and got["k"] == pytest.approx(expected["k"])
    with pytest.raises(ValueError, match="component"):
        lines.mean("missing")


def test_sample_access(tmp_path):
    paths = [_write(_synthetic(f"S{i}"), tmp_path / f"s{i}.rs3") for i in range(3)]
    smp = Sample.from_files(paths, name="mix")
    assert len(smp) == 3 and smp["S1"].site == "SITE"
    assert smp[0] is smp.specimens[0]
    with pytest.raises(KeyError):
        smp["nope"]
    with pytest.raises(TypeError):
        smp.append("not a specimen")


def _fold_data(pre_folding, n=40, seed=3):
    rng = np.random.default_rng(seed)
    beddings = [bedding_vector(90 if i % 2 else 270, 50) for i in range(n)]
    tilt = Vector3Set.random_fisher(n, mean=Vector3(0, 30), kappa=60, rng=rng)
    if pre_folding:
        geo = [
            np.asarray(untilt_rotation(b)).T @ np.asarray(v)
            for b, v in zip(beddings, tilt)
        ]
    else:
        geo = list(tilt)
    return _sample(geo, beddings)


def test_fold_test():
    pre = _fold_data(True).fold_test("ChRM", n=200, rng=1)
    assert pre["ci"][0] <= 100 <= pre["ci"][1] and pre["ci"][0] > 0
    assert pre["best"] == pytest.approx(100, abs=10)
    post = _fold_data(False).fold_test("ChRM", n=200, rng=1)
    assert post["ci"][0] <= 0 <= post["ci"][1] and post["ci"][1] < 100
    assert len(pre["tau1"]) == len(pre["fracs"]) == 161


def test_reversal_test():
    rng = np.random.default_rng(5)
    normal = Vector3Set.random_fisher(20, mean=Vector3(0, 40), kappa=50, rng=rng)
    reversed_ = Vector3Set.random_fisher(20, mean=Vector3(180, -40), kappa=50, rng=rng)
    ok = _sample(list(normal) + list(reversed_)).reversal_test("ChRM")
    assert not ok.reject()
    shifted = Vector3Set.random_fisher(20, mean=Vector3(215, -40), kappa=50, rng=rng)
    bad = _sample(list(normal) + list(shifted)).reversal_test("ChRM")
    assert bad.reject()
    with pytest.raises(ValueError, match="polarity"):
        _sample(list(normal)).reversal_test("ChRM")


@pytest.mark.parametrize("lat", [-60, 0, 45])
def test_vgp_axial_dipole(lat):
    inc = np.degrees(np.arctan(2 * np.tan(np.radians(lat))))
    assert vgp(0, inc, lat, 30)["lat"] == pytest.approx(90)
    assert vgp(180, -inc, lat, 30)["lat"] == pytest.approx(-90)
    assert vgp(0, inc, lat, 30)["paleolatitude"] == pytest.approx(lat)


@pytest.mark.parametrize(
    "dec, inc, lat, lon", [(25, 60, 40, 250), (200, -30, -20, 10), (300, 10, 70, 0)]
)
def test_vgp_geometry(dec, inc, lat, lon):
    """The pole lies at the magnetic colatitude from the site along `dec`."""
    res = vgp(dec, inc, lat, lon, a95=5)
    p = np.degrees(np.arctan2(2, np.tan(np.radians(inc))))
    site, pole = _dir(lon, lat), _dir(res["lon"], res["lat"])
    assert _angle(site, pole) == pytest.approx(p, abs=1e-6)
    # initial bearing from the site to the pole equals the declination
    la1, la2 = np.radians(lat), np.radians(res["lat"])
    dlon = np.radians(res["lon"] - lon)
    bearing = np.degrees(
        np.arctan2(
            np.sin(dlon) * np.cos(la2),
            np.cos(la1) * np.sin(la2) - np.sin(la1) * np.cos(la2) * np.cos(dlon),
        )
    )
    assert (bearing - dec + 180) % 360 - 180 == pytest.approx(0, abs=1e-6)
    assert res["dp"] == pytest.approx(5 * (1 + 3 * np.cos(np.radians(p)) ** 2) / 2)
    assert res["dm"] == pytest.approx(
        5 * np.sin(np.radians(p)) / np.cos(np.radians(inc))
    )


def test_sample_vgp_uses_site_location():
    lat = 45.0
    inc = np.degrees(np.arctan(2 * np.tan(np.radians(lat))))
    smp = _sample([_dir(355, inc), _dir(5, inc), _dir(0, inc - 3)])
    for s in smp:
        s.latitude, s.longitude = lat, 10.0
    res = smp.vgp("ChRM", coords="geo")
    assert res["lat"] > 85 and "dp" in res
    smp[0].latitude = 0
    with pytest.raises(ValueError, match="locations"):
        smp.vgp("ChRM")


# ---------------------------------------------------------------------------
# plots
# ---------------------------------------------------------------------------


def test_plots():
    spec = _synthetic()
    spec.interpret("ChRM", slice(3, 11))
    spec.interpret("GC", slice(0, 6), kind="plane")
    _fig, (a, b) = plt.subplots(1, 2)
    assert zijderveld_plot(spec, coords="tilt", ax=a) is a
    assert demag_plot(spec, ax=b) is b
    assert b.get_xlabel() == "Level [°C]"
    net = stereo_plot(spec, show=False)
    net.render2fig(plt.figure())
