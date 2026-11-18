"""Tests of the paleomagnetic Specimen/Sample classes and their readers."""

import re
import warnings
from dataclasses import replace
from pathlib import Path

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
    bailey_halls_mean,
    bedding_vector,
    core_rotation,
    demag_plot,
    inclination_only_mean,
    kirschvink_mean,
    lines_planes_mean,
    mcfadden_fold_test,
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


def _edit_rs3_header(path, sdec, sinc, pcodes=None, bdec=None):
    """Rewrite SDec, SInc (and BDec, P1-P4) on the specimen line."""
    lines = path.read_text(encoding="latin-1").splitlines()
    h = lines[1]
    bed = h[86:92] if bdec is None else f"{bdec:<6g}"
    pc = h[110:122] if pcodes is None else "".join(f"{c:<3d}" for c in pcodes)
    lines[1] = h[:74] + f"{sdec:<6g}{sinc:<6g}" + bed + h[92:110] + pc + h[122:]
    path.write_text("\r\n".join(lines), encoding="latin-1")
    return path


# the synthetic specimen: azimuth 207, hade 81, bedding dip direction 200, dip 85
@pytest.mark.parametrize(
    "sdec, sinc, pcodes, bdec",
    [
        (27, 81, (12, 0, 6, 0), 200),  # as written by to_rs3
        (207, -81, (6, 0, 6, 0), 200),  # arrow at 6 o'clock: 180 about the core axis
        (207, 81, (12, 0, 12, 90), 110),  # azimuth of the arrow, bedding strike
        (27, -81, (6, 0, 12, 90), 110),
    ],
)
def test_rs3_orientation_parameters(tmp_path, sdec, sinc, pcodes, bdec):
    spec = _synthetic()
    path = _edit_rs3_header(_write(spec, tmp_path / "s1.rs3"), sdec, sinc, pcodes, bdec)
    back = Specimen.from_rs3(path)
    assert np.allclose(back.rotation("geo"), spec.rotation("geo"), atol=1e-6)
    assert np.allclose(back.rotation("tilt"), spec.rotation("tilt"), atol=1e-6)


@pytest.mark.parametrize(
    "pcodes, match",
    [((3, 0, 6, 0), "P1=3"), ((12, 90, 6, 0), "P2=90"), ((12, 0, 6, 45), "P4=45")],
)
def test_rs3_unsupported_orientation_parameters(tmp_path, pcodes, match):
    path = _edit_rs3_header(_write(_synthetic(), tmp_path / "s1.rs3"), 27, 81, pcodes)
    with pytest.raises(NotImplementedError, match=match):
        Specimen.from_rs3(path)


def test_pmd_greek_header(tmp_path):
    # PaleoMac writes the core azimuth and hade as CP437 alpha and beta
    spec = _synthetic()
    path = _write(spec, tmp_path / "s1.pmd")
    text = (
        path.read_text(encoding="latin-1").replace(" a=", " α=").replace(" b=", " ß=")
    )
    path.write_bytes(text.encode("cp437"))
    back = Specimen.from_pmd(path)
    assert back.name == "S1"
    assert np.allclose(back.rotation("tilt"), spec.rotation("tilt"), atol=1e-3)


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


_KB31_POLES = [  # McFadden & McElhinny (1988) Table 1, poles of the six circles
    (0.263, 0.664, -0.700),
    (0.794, 0.335, -0.507),
    (0.692, 0.358, -0.627),
    (0.593, 0.663, -0.457),
    (0.637, 0.476, -0.606),
    (0.628, 0.524, -0.576),
]


def test_lines_planes_mean_mcfadden_mcelhinny_table1():
    # analyses A (circles only) and C (plus two set points), no arc constraints
    poles = [Vector3(p) for p in _KB31_POLES]
    with pytest.warns(UserWarning, match="Schmidt"):
        a = lines_planes_mean([], poles)
    assert a["R"] == pytest.approx(5.9735, abs=2e-4)
    assert a["k"] == pytest.approx(75.592, abs=0.2)
    assert a["alpha"] == pytest.approx(10.1, abs=0.05)
    c = lines_planes_mean([_dir(146.6, -52.6), _dir(195.5, -57.9)], poles)
    assert c["mu"].geo == pytest.approx((183.1, -51.2), abs=0.15)
    assert c["R"] == pytest.approx(7.8336, abs=2e-4)
    assert c["k"] == pytest.approx(24.032, abs=0.01)
    assert c["alpha"] == pytest.approx(12.5, abs=0.05)


def test_lines_planes_mean_circles_only_warns_about_bias():
    with pytest.warns(UserWarning, match="Schmidt"):
        lines_planes_mean([], [Vector3(p) for p in _KB31_POLES])


_KB31_SETS = [_dir(146.6, -52.6), _dir(195.5, -57.9)]


def test_kirschvink_methods():
    poles = [Vector3(p) for p in _KB31_POLES]
    m1 = kirschvink_mean(_KB31_SETS, poles)
    m2 = kirschvink_mean(_KB31_SETS, poles, method=2)
    # eigenvalues of H_P - H_L sum to N - M
    assert m1["eigenvalues"].sum() == pytest.approx(len(poles) - 2)
    # method 2 = method 1 with line weight 1/2: eigenvalues sum to N - M/2
    assert m2["eigenvalues"].sum() == pytest.approx(len(poles) - 1)
    ref = lines_planes_mean(_KB31_SETS, poles)["mu"]
    for m in (m1, m2):
        assert m["mu"].angle(ref) < 10
        assert m["mu"].dot(Vector3(_KB31_SETS[0]) + Vector3(_KB31_SETS[1])) > 0
    with pytest.raises(ValueError, match="method"):
        kirschvink_mean(_KB31_SETS, poles, method=3)


def test_kirschvink_recovers_common_direction():
    mu = _dir(160.0, -30.0)
    a = np.cross(mu, [0, 0, 1.0])
    a /= np.linalg.norm(a)
    b = np.cross(mu, a)
    poles = [Vector3(np.cos(f) * a + np.sin(f) * b) for f in (0.3, 1.2, 2.4)]
    res = kirschvink_mean([], poles)
    assert abs(res["mu"].dot(mu)) == pytest.approx(1.0)


def test_bailey_halls_mean():
    poles = [Vector3(p) for p in _KB31_POLES]
    res = bailey_halls_mean(_KB31_SETS, poles)
    mm = lines_planes_mean(_KB31_SETS, poles)
    # MM88 p. 165: the two estimates agree to a fraction of a degree
    assert res["mu"].angle(mm["mu"]) < 0.5
    assert res["k"] == pytest.approx(mm["k"], rel=0.1)
    major, minor = res["semi_axes"]
    assert major >= minor > 0
    # ellipse of the order of the circular cone (vault: 24.5 x 12.7 vs 12.5)
    assert 0.5 * mm["alpha"] < minor < major < 3 * mm["alpha"]
    assert abs(res["axes"][0].dot(res["axes"][1])) < 1e-9
    assert abs(res["axes"][0].dot(res["mu"])) < 1e-9
    # circles only: smallest eigenvector of H, omega its eigenvalue
    only = bailey_halls_mean([], poles)
    H = np.array(_KB31_POLES)
    H = H / np.linalg.norm(H, axis=1, keepdims=True)
    lam = np.linalg.eigvalsh(H.T @ H)
    assert only["omega"] == pytest.approx(lam[0])
    with pytest.raises(ValueError, match="circle"):
        bailey_halls_mean(_KB31_SETS, [])


def _bh_S(t, lines, poles):
    """Sum of squares minimised by the Bailey & Halls estimate (eq. 10)."""
    L, P = np.asarray(lines), np.asarray(poles)
    return np.sum((L - t) ** 2) + np.sum((P @ t) ** 2)


def test_bailey_halls_identities():
    poles = np.array(_KB31_POLES)
    poles = poles / np.linalg.norm(poles, axis=1, keepdims=True)
    lines = np.array(_KB31_SETS)
    res = bailey_halls_mean(lines, [Vector3(p) for p in poles])
    t, omega = np.asarray(res["mu"]), res["omega"]
    H, r = poles.T @ poles, lines.sum(axis=0)
    M, N = len(lines), len(poles)
    # (15): (H - omega I) t = r, and omega below the smallest eigenvalue of H
    assert np.allclose((H - omega * np.eye(3)) @ t, r, atol=1e-9)
    assert omega < np.linalg.eigvalsh(H)[0]
    # S = 2M - r.t + omega (the printed (29) uses |r| and is wrong for N > 0)
    S = _bh_S(t, lines, poles)
    assert S == pytest.approx(2 * M - r @ t + omega, abs=1e-9)
    assert res["k"] == pytest.approx((2 * M + N - 2) / S)
    assert abs(np.linalg.norm(r) - r @ t) > 0.01
    # the estimate minimises S on the sphere
    rng = np.random.default_rng(0)
    for _ in range(20):
        u = t + 0.05 * rng.normal(size=3)
        assert _bh_S(u / np.linalg.norm(u), lines, poles) > S
    # (37): S1 - S0 = e'(H - omega I)e for a small tangent step e
    a = np.cross(t, [0.3, 0.5, 0.8])
    a /= np.linalg.norm(a)
    for e in (1e-3 * a, 1e-3 * np.cross(t, a)):
        t1 = (t + e) / np.linalg.norm(t + e)
        assert _bh_S(t1, lines, poles) - S == pytest.approx(
            e @ (H - omega * np.eye(3)) @ e, rel=1e-3
        )
    # (39): semi-axes from the quadratic form restricted to the tangent plane
    from scipy import stats

    B = np.column_stack([a, np.cross(t, a)])
    q = np.linalg.eigvalsh(B.T @ (H - omega * np.eye(3)) @ B)
    C2 = S * stats.f.ppf(0.95, 2, 2 * M + N - 2) / (M + N / 2 - 1)
    semi = sorted(np.degrees(np.sqrt(C2 / q)), reverse=True)
    assert res["semi_axes"] == pytest.approx(semi)
    # the minor semi-axis agrees with the vault check (12.7 deg, k 24.36)
    assert res["k"] == pytest.approx(24.36, abs=0.01)
    assert res["semi_axes"][1] == pytest.approx(12.7, abs=0.1)


def _circle_poles(mu, n, kappa, rng):
    """Poles of `n` great circles through two Fisher populations (Schmidt)."""
    u = np.asarray(
        Vector3Set.random_fisher(n, mean=Vector3(mu[0]), kappa=kappa, rng=rng)
    )
    v = np.asarray(
        Vector3Set.random_fisher(n, mean=Vector3(mu[1]), kappa=kappa, rng=rng)
    )
    P = np.cross(u, v)
    return P / np.linalg.norm(P, axis=1, keepdims=True)


def test_kirschvink_synthetic_common_direction():
    # 21 lines and 7 great circles about a known direction (the vault check
    # recovers it to 1.8 deg, the two methods differ by 1.0 deg)
    rng = np.random.default_rng(1980)
    mu = Vector3(160.0, -30.0)
    lines = np.asarray(Vector3Set.random_fisher(21, mean=mu, kappa=60, rng=rng))
    # circles: great circles through the direction, perturbed by a Fisher error
    poles = []
    for az in np.linspace(0, np.pi, 7, endpoint=False):
        a = np.cross(mu, [0, 0, 1.0])
        a /= np.linalg.norm(a)
        b = np.cross(mu, a)
        n = np.cos(az) * a + np.sin(az) * b
        poles.append(
            np.asarray(
                Vector3Set.random_fisher(1, mean=Vector3(n), kappa=400, rng=rng)
            )[0]
        )
    m1 = kirschvink_mean(lines, [Vector3(p) for p in poles])
    m2 = kirschvink_mean(lines, [Vector3(p) for p in poles], method=2)
    assert np.degrees(np.arccos(abs(np.asarray(m1["mu"]) @ np.asarray(mu)))) < 3
    assert np.degrees(np.arccos(abs(np.asarray(m2["mu"]) @ np.asarray(mu)))) < 3
    assert m1["mu"].angle(m2["mu"]) < 2
    assert m1["eigenvalues"].sum() == pytest.approx(7 - 21)


def test_schmidt_bias_of_circles_only_mean():
    # Schmidt (1985): two populations 30 deg apart, equal kappa, 20 circles; the
    # circles-only mean sits about alpha/2 = 15 deg from either, direct
    # observations of one set reduce the bias (McFadden & McElhinny, 1988)
    rng = np.random.default_rng(1985)
    m1, m2 = Vector3(0, 0), Vector3(180, 30)
    m1v = np.asarray(m1)
    err_free, err_set = [], []
    for _ in range(25):
        P = [Vector3(p) for p in _circle_poles((m1, m2), 20, 100, rng)]
        with pytest.warns(UserWarning, match="Schmidt"):
            free = lines_planes_mean([], P)
        D = np.asarray(Vector3Set.random_fisher(5, mean=m1, kappa=100, rng=rng))
        with_set = lines_planes_mean(D, P)
        err_free.append(
            np.degrees(np.arccos(np.clip(abs(np.asarray(free["mu"]) @ m1v), 0, 1)))
        )
        err_set.append(
            np.degrees(np.arccos(np.clip(np.asarray(with_set["mu"]) @ m1v, -1, 1)))
        )
    assert np.mean(err_free) == pytest.approx(15, abs=3)
    assert np.mean(err_set) < 0.5 * np.mean(err_free)


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


_MF3 = [  # McFadden (1990) Table 3: dec, inc in situ, dec, inc unfolded, dip dir, dip
    [189.0, 56.4, 193.4, -11.8, 199.1, 68.7],
    [173.3, -51.5, 180.3, -11.3, 12.2, 41.5],
    [190.9, 43.4, 191.0, -6.1, 191.2, 49.5],
    [186.9, 10.7, 187.7, -4.3, 229.5, 20.1],
    [169.3, -15.0, 167.0, -13.9, 265.0, 9.0],
    [181.6, -6.9, 180.3, -8.0, 264.9, 10.0],
    [145.9, -32.9, 154.3, -8.1, 12.0, 32.8],
    [168.1, -47.6, 177.5, -16.0, 17.9, 34.6],
]
_MF4 = [  # Table 4: dec, inc, (unfolded dec, inc not needed), dip dir, dip
    [44.4, -56.2, 7.2, -47.5, 131.9, 29.2],
    [5.5, -58.8, 15.2, -76.1, 357.3, 17.7],
    [69.5, -41.3, 287.0, -79.8, 76.4, 57.3],
    [9.7, -81.7, 149.3, -57.5, 337.5, 39.3],
    [70.9, -58.0, 65.6, -69.3, 81.3, 11.6],
    [101.8, -64.4, 226.2, -78.7, 85.2, 33.5],
    [80.4, -79.9, 77.5, -76.3, 249.3, 3.7],
    [50.2, -52.3, 111.7, -74.0, 23.5, 34.1],
    [40.5, -84.0, 110.1, -75.8, 315.2, 13.4],
    [73.8, -78.4, 85.0, -49.9, 270.0, 28.8],
    [55.7, -56.0, 325.8, -63.9, 93.9, 44.7],
    [339.3, -58.6, 22.9, -56.1, 266.2, 26.9],
    [48.6, -61.5, 53.0, -53.5, 250.2, 8.5],
]


def _mcfadden(table, **kw):
    T = np.array(table)
    G = [_dir(r[0], r[1]) for r in T]
    B = [np.asarray(bedding_vector(r[4], r[5])) for r in T]
    return mcfadden_fold_test(G, B, **kw), T


def test_mcfadden_fold_test_table3():
    res, _ = _mcfadden(_MF3)
    # xi_1 = 5.958 in situ and 2.376 unfolded; N = 8
    assert res["geo"] == pytest.approx(5.958, abs=0.01)
    assert res["tilt"] == pytest.approx(2.376, abs=0.01)
    assert res["n"] == 8
    # Table 2 critical values 95 % / 99 % are interpolated by N, here N = 8
    assert res["critical"] == pytest.approx(3.30, abs=0.01)
    assert res["p_values"][0] < 0.01 < res["p_values"][-1]
    # in situ significant, unfolded not: pre-folding remanence
    assert abs(res["geo"]) > res["critical"] > abs(res["tilt"])


def test_mcfadden_fold_test_table4_minimum_at_44_percent():
    res, _ = _mcfadden(_MF4)
    assert abs(res["geo"]) == pytest.approx(5.924, abs=0.01)
    assert abs(res["tilt"]) == pytest.approx(6.676, abs=0.01)
    assert res["best"] == 44
    assert abs(res["xi"][44]) == pytest.approx(0.029, abs=0.005)
    # significant at 0 and 100 %, syn-folding in between
    assert abs(res["geo"]) > res["critical"] and abs(res["tilt"]) > res["critical"]


def test_mcfadden_xi_null_distribution():
    from apsg.paleomag._specimen import _xi_cdf, _xi_critical

    # McFadden (1990) Table 2, from Lord's integral (15)
    table = {
        2: (1.697, 1.937),
        5: (2.609, 3.573),
        10: (3.685, 5.120),
        25: (5.820, 8.174),
    }
    for N, (c95, c99) in table.items():
        assert _xi_critical(N, 0.95) == pytest.approx(c95, abs=0.005)
        assert _xi_critical(N, 0.99) == pytest.approx(c99, abs=0.005)
    # distribution function: symmetric, P(xi <= critical) = level
    assert float(_xi_cdf(0.0, 10)) == pytest.approx(0.5, abs=1e-3)
    assert float(_xi_cdf(_xi_critical(10, 0.95), 10)) == pytest.approx(0.95, abs=1e-3)
    # large N: N(0, N/2) with the exact 99 % constant 2.3263 (the paper's text
    # prints 2.362, which is a transposition)
    assert _xi_critical(400, 0.99) == pytest.approx(2.3263 * np.sqrt(200), abs=1e-3)
    assert _xi_critical(100, 0.99) == pytest.approx(2.3263 * np.sqrt(50), rel=0.01)


def test_mcfadden_fold_test_checks_input():
    with pytest.raises(ValueError, match="same length"):
        mcfadden_fold_test([_dir(0, 10)] * 4, [Vector3(0, 0, -1)] * 3)
    with (
        pytest.warns(UserWarning, match="zero dip"),
        pytest.raises(ValueError, match="three sites"),
    ):
        mcfadden_fold_test([_dir(0, 10)] * 4, [Vector3(0, 0, -1)] * 4)


def test_sample_mcfadden_fold_test():
    smp = _fold_data(True, n=20)
    res = smp.mcfadden_fold_test("ChRM")
    assert res["n"] == 20 and len(res["xi"]) == 101
    # prefolding data: correlated in situ, uncorrelated unfolded
    assert abs(res["geo"]) > res["critical"] > abs(res["tilt"])
    post = _fold_data(False, n=20).mcfadden_fold_test("ChRM")
    assert abs(post["tilt"]) > post["critical"] > abs(post["geo"])


_T42_NORMAL = [
    [330.9, -3.0], [333.7, -22.4], [21.5, 16.8], [39.6, 22.7], [42, 27.9],
    [15, 29.8], [36.5, 32.7], [33.9, 34.2], [27.8, 34.9], [41.8, 36.1],
    [343.3, 36.5], [29.2, 36.9], [72.3, 37.0], [41.1, 37.9], [31.5, 39.5],
    [340.8, 41.1], [66.1, 42.1], [54.4, 43.8], [6.7, 44.9], [18.3, 45.1],
    [73.8, 46.2], [16, 47.8], [51.1, 52.7], [24.5, 56.5], [28.9, 64.5],
]  # fmt: skip
_T42_REVERSED = [
    [216.2, -22.8], [192.2, -23.8], [202.2, -24.5], [220.4, -25.7],
    [216.8, -28.8], [193.7, -29.6], [211.9, -32.5], [196.6, -37.5],
    [202.3, -38.5], [198.1, -39.7], [139.6, -40.2], [160, -40.2],
    [186.3, -43.3], [141.1, -45], [173.3, -45.0], [195.8, -45.4],
    [147.9, -50], [222.5, -51.3], [180.2, -53], [189.4, -53],
    [168.1, -53.2], [190.5, -55.8], [131.2, -56.3], [205, -58.5],
]  # fmt: skip


def test_reversal_test_bootstrap_tauxe_table_4_2():
    # Tauxe (2003) Table 4.2: Watson's F rejects (3.36 > 3.09) while the
    # bootstrap intervals of the Cartesian components overlap (vault check)
    vecs = [_dir(*d) for d in _T42_NORMAL + _T42_REVERSED]
    smp = _sample(vecs)
    watson = smp.reversal_test("ChRM")
    assert watson.statistic == pytest.approx(3.36, abs=0.02)
    assert watson.reject()
    boot = smp.reversal_test("ChRM", method="bootstrap", n=1000, rng=1)
    assert boot.extra["lower"].shape == boot.extra["upper"].shape == (2, 3)
    assert (boot.extra["lower"] <= boot.extra["upper"]).all()
    assert boot.statistic == 0 and not boot.reject()


def test_reversal_test_bootstrap_detects_non_antipodal():
    rng = np.random.default_rng(5)
    normal = Vector3Set.random_fisher(20, mean=Vector3(0, 40), kappa=50, rng=rng)
    rev = Vector3Set.random_fisher(20, mean=Vector3(180, -40), kappa=50, rng=rng)
    ok = _sample(list(normal) + list(rev)).reversal_test(
        "ChRM", method="bootstrap", rng=2
    )
    assert not ok.reject()
    shifted = Vector3Set.random_fisher(20, mean=Vector3(240, -40), kappa=50, rng=rng)
    bad = _sample(list(normal) + list(shifted)).reversal_test(
        "ChRM", method="bootstrap", rng=2
    )
    assert bad.reject() and bad.statistic >= 1
    # the same seed reproduces the intervals
    again = _sample(list(normal) + list(shifted)).reversal_test(
        "ChRM", method="bootstrap", rng=2
    )
    assert np.array_equal(bad.extra["lower"], again.extra["lower"])


_MR82_THETA = [27.6, 28.4, 39.8, 24.8, 36.8, 28.6, 16.0, 29.4, 36.4, 18.2]


def test_inclination_only_mcfadden_reid_appendix():
    # McFadden & Reid (1982) appendix, theta_i = 90 - I_i as printed
    res = inclination_only_mean(90 - np.array(_MR82_THETA))
    assert np.degrees(res["theta0"]) == pytest.approx(27.7, abs=0.05)
    assert res["C"] == pytest.approx(9.917, abs=1e-3)
    assert res["S"] == pytest.approx(-0.1560, abs=1e-2)
    assert res["inc_conditional"] == pytest.approx(61.4, abs=0.05)
    assert res["inc"] == pytest.approx(90 - 27.7, abs=0.05)
    assert res["k"] == pytest.approx(54.2, abs=0.3)
    lo, hi = res["kappa_ci"]
    assert (lo, hi) == pytest.approx((16.3, 114.4), abs=0.8)
    assert res["alpha"] == pytest.approx(6.7, abs=0.05)
    one_sided = inclination_only_mean(90 - np.array(_MR82_THETA), level=0.90)
    assert one_sided["alpha"] == pytest.approx(5.7, abs=0.05)


def test_inclination_only_is_nearly_unbiased_for_fisher_data():
    # I0 = 60, kappa = 40, N = 10: the root of eq. 19 is nearly unbiased (vault
    # simulation -0.2 deg), eq. 40 is biased by about -1.3 deg
    rng = np.random.default_rng(24)
    err, err40 = [], []
    for _ in range(300):
        X = np.asarray(
            Vector3Set.random_fisher(10, mean=Vector3(0, 60), kappa=40, rng=rng)
        )
        res = inclination_only_mean(np.degrees(np.arcsin(X[:, 2])))
        err.append(res["inc"] - 60)
        err40.append(res["inc_conditional"] - 60)
    assert abs(np.mean(err)) < 0.6
    assert np.mean(err40) < np.mean(err) - 0.5


def test_inclination_only_recovers_plain_mean_for_clustered_data():
    inc = np.array([60.0, 61, 59, 60.5, 59.5])
    res = inclination_only_mean(inc)
    assert res["inc"] == pytest.approx(60, abs=0.2) and res["k"] > 1000


def test_inclination_only_input_checks():
    with pytest.raises(ValueError, match="three"):
        inclination_only_mean([50, 60])
    with pytest.raises(ValueError, match="within"):
        inclination_only_mean([50, 60, 100])
    # mixed polarities warn; such data usually have no admissible root either
    with (
        pytest.warns(UserWarning, match="polarities"),
        pytest.raises(ValueError, match="fold back"),
    ):
        inclination_only_mean([50, 55, 52, -48, -52, 45])


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


def test_vgp_dp_dm_literal_values():
    # inc 45: magnetic colatitude p = atan(2 / tan 45) = 63.435 deg, cos^2 p = 1/5
    # exactly, so dp = a95 (1 + 3/5) / 2 = 0.8 a95 and dm = a95 sin p / cos 45
    # = a95 sqrt(8/5) (Butler 1992, eqs. 7.3 and 7.4)
    res = vgp(0, 45, 30, 0, a95=5)
    assert res["dp"] == pytest.approx(4.0)
    assert res["dm"] == pytest.approx(5 * np.sqrt(8 / 5))


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


# ---------------------------------------------------------------------------
# tutorial data (Kilian et al. 2016, docs/notebooks/teel)
# ---------------------------------------------------------------------------

TEEL = Path(__file__).parent.parent / "docs" / "notebooks" / "teel"
teel = pytest.mark.skipif(not TEEL.is_dir(), reason="tutorial data not available")


@teel
def test_teel_files_read_without_orientation_warning():
    files = sorted(TEEL.glob("*.pmd"))
    assert len(files) == 67
    with warnings.catch_warnings():
        warnings.simplefilter("error")  # a misfit of the Dg/Ig columns warns
        specs = [Specimen.from_pmd(f) for f in files]
    assert {s.site for s in specs} == {
        "Z31", "Z34", "Z36", "Z38", "Z40", "Z47", "Z49", "Z51",
    }  # fmt: skip
    assert all(s.demag == "thermal" and s.steps[0].label == "NRM" for s in specs)


@teel
def test_teel_pca_reproduces_the_published_fits():
    import pandas as pd

    comps = pd.read_csv(TEEL / "teel_components.csv")
    mag = comps[comps.component == "MAG"]
    angles = []
    for r in mag.itertuples():
        spec = Specimen.from_pmd(TEEL / f"{r.specimen}.pmd")
        fit = spec.pca(spec.between(r.step_min, r.step_max).labels, coords="geo")
        angles.append(fit.vector.angle(Vector3(r.dec_geo, r.inc_geo)))
    assert np.median(angles) < 0.2 and np.max(angles) < 5


@teel
@pytest.mark.parametrize("site", ["Z31", "Z36", "Z47"])
def test_teel_site_means_agree_with_the_authors(site):
    import pandas as pd

    comps = pd.read_csv(TEEL / "teel_components.csv")
    sites = pd.read_csv(TEEL / "teel_sites.csv", index_col="site")
    smp = Sample.from_files(sorted(TEEL.glob(f"{site}.*.pmd")), name=site)
    for s in smp:
        r = comps[(comps.specimen == s.name) & (comps.component == "MAG")].iloc[0]
        s.interpret("MAG", s.between(r.step_min, r.step_max).labels)
    a = sites.loc[site]
    for coords in ("geo", "tilt"):
        m = smp.mean("MAG", coords)
        ref = Vector3(a[f"dec_{coords}"], a[f"inc_{coords}"])
        assert m["mu"].angle(ref) < 0.3
        assert m["alpha"] == pytest.approx(a[f"alpha95_{coords}"], abs=0.1)
