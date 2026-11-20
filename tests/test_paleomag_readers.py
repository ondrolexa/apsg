"""Readers of the JR6, CIT and 2G formats of apsg.paleomag.

The files are written here from a synthetic ``Specimen``; the geometry is
checked against Remasoft output (values of the JR6 conversion examples of
PuffinPlot) and against the rotated columns the formats carry themselves.
"""

import warnings
from dataclasses import replace

import numpy as np
import pytest

from apsg.math._vector import Vector3
from apsg.paleomag import DemagStep, Sample, Specimen
from tests.test_paleomag import _synthetic


def _dec_inc(v):
    v = np.asarray(v, dtype=float)
    return (
        np.degrees(np.arctan2(v[1], v[0])) % 360,
        np.degrees(np.arcsin(v[2] / np.linalg.norm(v))),
    )


def _assert_same(back, spec, rtol=2e-3, atol=2e-4):
    assert list(back.levels) == pytest.approx(list(spec.levels))
    assert back.labels[0] == "NRM"  # the synthetic T0 is the natural remanence
    M = np.array([np.asarray(s.M) for s in spec.steps])
    Mb = np.array([np.asarray(s.M) for s in back.steps])
    assert np.allclose(Mb, M, rtol=rtol, atol=atol * np.abs(M).max())
    for coords in ("geo", "tilt"):
        a = np.array([np.asarray(v) for v in getattr(back, coords)])
        b = np.array([np.asarray(v) for v in getattr(spec, coords)])
        assert np.allclose(a, b, atol=5e-3 * np.abs(b).max())


# ---------------------------------------------------------------------------
# JR6
# ---------------------------------------------------------------------------


def _jr6_line(name, treatment, M, az, dip, fol=(0, 0), p=(12, 0, 12, 0)):
    exp = int(np.floor(np.log10(np.abs(M).max())))
    x, y, z = (v / 10.0**exp for v in M)
    line = (
        f"{name:<10}{treatment:<8}{x:6.3f}{y:6.3f}{z:6.3f}{exp:4d}"
        f"{az:4.0f}{dip:4.0f}{fol[0]:4.0f}{fol[1]:4.0f}   0   0"
        + "".join(f"{c:3d}" for c in p)
        + "   1"
    )
    assert len(line) == 80, len(line)
    return line


def _jr6_text(spec, p=(12, 0, 12, 0), az=None, dip=None, M=None):
    az = spec.azimuth if az is None else az
    dip = spec.hade if dip is None else dip
    dd, bdip = spec._bedding_dd()
    out = []
    for i, s in enumerate(spec.steps):
        treatment = "NRM" if s.level == 0 else f"T{s.level:g}"
        out.append(
            _jr6_line(
                spec.name,
                treatment,
                np.asarray(s.M if M is None else M[i]),
                az,
                dip,
                (dd, bdip),
                p,
            )
        )
    return "\n".join(out) + "\n"


def test_jr6_roundtrip_two_specimens(tmp_path):
    a, b = _synthetic("S1"), _synthetic("S2", azimuth=100.0, hade=45.0)
    path = tmp_path / "site.jr6"
    path.write_text(_jr6_text(a) + _jr6_text(b))
    smp = Sample.from_file(path)
    assert smp.names == ["S1", "S2"] and smp.name == "site"
    _assert_same(smp["S1"], a)
    _assert_same(smp["S2"], b)
    assert smp["S1"].demag == "thermal"
    with pytest.raises(ValueError, match="2 specimens"):
        Specimen.from_file(path)


@pytest.mark.parametrize("p1, spin", [(3, 90), (9, 270), (6, 180)])
def test_jr6_arrow_clock_position(tmp_path, p1, spin):
    spec = _synthetic()
    c, s = np.cos(np.radians(-spin)), np.sin(np.radians(-spin))
    Rz = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
    turned = [Rz @ np.asarray(st.M) for st in spec.steps]
    path = tmp_path / "s.jr6"
    # the codes keep azimuth and hade, P1 = 6 is absorbed in the orientation
    path.write_text(_jr6_text(spec, p=(p1, 0, 12, 0), M=turned))
    back = Sample.from_file(path)["S1"]
    for got, want in zip(back.geo, spec.geo):
        assert np.allclose(np.asarray(got), np.asarray(want), atol=5e-3 * 0.3)


def test_jr6_plunge_of_core_axis(tmp_path):
    spec = _synthetic()
    path = tmp_path / "s.jr6"
    path.write_text(_jr6_text(spec, p=(12, 90, 12, 0), dip=90 - spec.hade))
    _assert_same(Sample.from_file(path)["S1"], spec)


def test_jr6_matches_remasoft_conversion():
    # intensity, geographic declination and inclination from Remasoft 3.0 for
    # two specimens (codes 12 0 12 0 and 6 0 6 0), converted by PuffinPlot
    import io

    cases = [
        (
            "BC0101A1",
            (0.08e-2, -1.45e-2, -0.46e-2),
            107,
            88,
            (12, 0, 12, 0),
            359.5,
            -3.6,
        ),
        ("Re01", (8.07e-4, -22.02e-4, 22.21e-4), 156, 78, (6, 0, 6, 0), 23.7, 22.8),
    ]
    for name, M, az, dip, p, dec, inc in cases:
        text = _jr6_line(name, "NRM", np.array(M), az, dip, p=p)
        spec = Sample.from_file(io.StringIO(text), fmt="jr6")[name]
        got = _dec_inc(spec.geo[0])
        assert got == pytest.approx((dec, inc), abs=0.1)


def test_jr6_treatments_and_malformed_lines(tmp_path):
    M = np.array([1.0, 2.0, 3.0]) * 1e-3
    lines = [
        _jr6_line("A1", "NRM", M, 10, 20),
        _jr6_line("A1", "A50", M, 10, 20),
        _jr6_line("A1", "M20", M, 10, 20),  # ARM
        "this is not a jr6 line",
    ]
    path = tmp_path / "s.jr6"
    path.write_text("\n".join(lines) + "\n")
    with pytest.warns(UserWarning, match="1 malformed JR6 lines"):
        spec = Sample.from_file(path)["A1"]
    assert spec.labels == ["NRM", "M050"] and spec.demag == "af"
    assert [s.label for s in spec.arm] == ["M020"] and spec.arm[0].treatment == "arm"
    # a bare number needs the demagnetization type
    path.write_text(_jr6_line("A1", "100", M, 10, 20) + "\n")
    with pytest.raises(ValueError, match="pass demag="):
        Sample.from_file(path)
    spec = Sample.from_file(path, demag="thermal")["A1"]
    assert spec.demag == "thermal" and spec.steps[0].level == 100
    # steps listed out of order (as in AGICO example files) are sorted by level
    path.write_text(
        "\n".join(_jr6_line("A1", t, M, 10, 20) for t in ("A40", "NRM", "A10", "A25"))
        + "\n"
    )
    assert Sample.from_file(path)["A1"].labels == ["NRM", "M010", "M025", "M040"]
    # the level first, with a unit (inferred: 120°C is a thermal step, 20M an AF one)
    path.write_text(
        "\n".join(
            _jr6_line("A1", t, M, 10, 20) for t in ("NRM", "20M", "120°C", "400C")
        )
        + "\n",
        encoding="latin-1",
    )
    path.write_text(
        path.read_text(encoding="latin-1") + _jr6_line("A1", "CA", M, 10, 20) + "\n",
        encoding="latin-1",
    )
    with warnings.catch_warnings():
        warnings.simplefilter("error")  # a Remasoft component row is not malformed
        spec = Sample.from_file(path)["A1"]
    assert [(s.level, s.treatment) for s in spec.steps] == [
        (0.0, "nrm"),
        (20.0, "af"),
        (120.0, "thermal"),
        (400.0, "thermal"),
    ]


def test_jr6_unsupported_orientation(tmp_path):
    M = np.array([1.0, 2.0, 3.0]) * 1e-3
    path = tmp_path / "s.jr6"
    path.write_text(_jr6_line("A1", "NRM", M, 10, 20, p=(5, 0, 12, 0)))
    with pytest.raises(NotImplementedError, match="P1=5"):
        Sample.from_file(path)


def test_jr6_sniffed_without_extension(tmp_path):
    spec = _synthetic()
    path = tmp_path / "data.001"
    path.write_text(_jr6_text(spec))
    assert Sample.from_file(path).names == ["S1"]


# ---------------------------------------------------------------------------
# CIT
# ---------------------------------------------------------------------------


def _cit_specimen_text(spec, strike_shift=0.0):
    dd, bdip = spec._bedding_dd()
    lines = [
        f"SITE{spec.name:<9} comment",
        "      0 "
        + " ".join(
            f"{v:5.1f}"
            for v in (
                (spec.azimuth - 270) % 360 + strike_shift,
                spec.hade,
                (dd - 90) % 360,
                bdip,
            )
        )
        + "  1.0",
    ]
    for s, g, t in zip(spec.steps, spec.geo, spec.tilt):
        kind, level = ("NR", "M   ") if s.level == 0 else ("TT", f"{s.level:4.0f}")
        dc, ic = _dec_inc(s.M)
        lines.append(
            f"{kind}{level}{_dec_inc(g)[0]:6.1f}{_dec_inc(g)[1]:6.1f}"
            f"{_dec_inc(t)[0]:6.1f}{_dec_inc(t)[1]:6.1f}"
            f"{abs(s.M) / 1e3:9.3E}{0.5:6.1f}{dc:6.1f}{ic:6.1f}"
            + " " * 27
            + f"{spec.name:<9}{'01-01-2000':>11}{'12:00:00':>8}"
        )
    return "\n".join(lines) + "\n"


def _cit_site(tmp_path, specs, **kw):
    for spec in specs:
        (tmp_path / spec.name).write_text(_cit_specimen_text(spec, **kw))
    index = tmp_path / "site.sam"
    index.write_text("SITE\nINFO\n" + "\n".join(s.name for s in specs) + "\n")
    return index


def test_cit_index_roundtrip(tmp_path):
    specs = [_synthetic("SP01"), _synthetic("SP02", azimuth=100.0, hade=45.0)]
    smp = Sample.from_file(_cit_site(tmp_path, specs))
    assert smp.names == ["SP01", "SP02"] and smp.name == "site"
    for got, want in zip(smp, specs):
        _assert_same(got, want, rtol=5e-3, atol=2e-3)
        assert got.demag == "thermal"
        assert got.azimuth == pytest.approx(want.azimuth, abs=0.1)
        assert got.hade == pytest.approx(want.hade)


def test_cit_single_file_and_af(tmp_path):
    spec = _synthetic("SP01")
    text = _cit_specimen_text(spec).replace("TT", "AF")
    path = tmp_path / "SP01"
    path.write_text(text)
    got = Specimen.from_file(path, fmt="cit")
    assert got.demag == "af" and got.labels[1] == "M100"
    assert Specimen.from_file(path).name == "SP01"  # sniffed


def test_cit_orientation_mismatch_warns(tmp_path):
    index = _cit_site(tmp_path, [_synthetic("SP01")], strike_shift=40.0)
    with pytest.warns(UserWarning, match="does not reproduce"):
        Sample.from_file(index)


def test_cit_missing_file_and_bad_lines(tmp_path):
    index = _cit_site(tmp_path, [_synthetic("SP01")])
    index.write_text(index.read_text() + "GONE\n")
    with pytest.warns(UserWarning, match="'GONE' not found"):
        assert Sample.from_file(index).names == ["SP01"]
    index.write_text(index.read_text().replace("GONE\n", ""))
    (tmp_path / "SP01").write_text(
        (tmp_path / "SP01").read_text() + "CH 123 not a data line\n"
    )
    with pytest.warns(UserWarning, match="1 lines are not CIT data"):
        Sample.from_file(index)


# ---------------------------------------------------------------------------
# 2G
# ---------------------------------------------------------------------------

_2G_COLUMNS = (
    "Sample ID",
    "Meas. type",
    "Treatment Type",
    "AF X",
    "AF Y",
    "AF Z",
    "Temp C",
    "Declination: Unrotated",
    "Inclination: Unrotated",
    "Declination: Sample Rotated",
    "Inclination: Sample Rotated",
    "Declination: Formation Rotated",
    "Inclination: Formation Rotated",
    "Intensity",
    "MS corr",
    "Sample Azimiuth",
    "Sample Dip",
    "Formation Dip Azimuth",
    "Formation Dip",
    "X corr",
    "Y corr",
    "Z corr",
    "IRM Gauss",
    "ARM Gauss",
    "Volume",
    "Depth",
)


def _2g_row(spec, treatment, level, M, k=None, volume=10.8, extra=None, orient=True):
    """One row of a 2G file: the moment in specimen coordinates (A/m)."""
    M = np.asarray(M, dtype=float)
    dd, bdip = spec._bedding_dd()
    geo = tilt = ("NA", "NA")
    if np.any(M):
        dec, inc = _dec_inc(M)
        R_geo = np.asarray(spec.rotation("geo")) @ M
        R_tilt = np.asarray(spec.rotation("tilt")) @ M
        geo, tilt = _dec_inc(R_geo), _dec_inc(R_tilt)
    else:
        dec = inc = "NA"
    cell = dict.fromkeys(_2G_COLUMNS, "NA")
    cell.update(
        {
            "Sample ID": spec.name,
            "Meas. type": "Sample/Discrete",
            "Treatment Type": treatment,
            "Declination: Unrotated": dec,
            "Inclination: Unrotated": inc,
            "Declination: Sample Rotated": geo[0],
            "Inclination: Sample Rotated": geo[1],
            "Declination: Formation Rotated": tilt[0],
            "Inclination: Formation Rotated": tilt[1],
            "Intensity": np.linalg.norm(M) / 1e3 if np.any(M) else "NA",
            "MS corr": "NA" if k is None else k,
            "Sample Azimiuth": spec.azimuth,
            "Sample Dip": 90 - spec.hade,
            "Formation Dip Azimuth": dd,
            "Formation Dip": bdip,
            "X corr": M[0] / 1e3 * volume,
            "Y corr": M[1] / 1e3 * volume,
            "Z corr": M[2] / 1e3 * volume,
            "Volume": volume,
        }
    )
    if treatment.startswith("degauss"):
        cell["AF Z"] = cell["AF X"] = cell["AF Y"] = level * 10  # Oe
    elif treatment == "thermal demag":
        cell["Temp C"] = level
    cell.update(extra or {})
    return "\t".join(
        f"{cell[c]:.6g}" if isinstance(cell[c], float | np.floating) else str(cell[c])
        for c in _2G_COLUMNS
    )


def _2g_text(rows):
    return "\r\n".join(["\t".join(_2G_COLUMNS), *rows]) + "\r\n"


def _2g_thermal_rows(spec, **kw):
    return [
        _2g_row(
            spec,
            "none" if s.level == 0 else "thermal demag",
            s.level,
            s.M,
            **kw,
        )
        for s in spec.steps
    ]


@pytest.mark.parametrize("moment", ["polar", "cartesian"])
def test_2g_roundtrip_two_specimens(tmp_path, moment):
    a, b = _synthetic("S1"), _synthetic("S2", azimuth=100.0, hade=45.0)
    path = tmp_path / "site.dat"
    path.write_text(_2g_text(_2g_thermal_rows(a) + _2g_thermal_rows(b, volume=11.0)))
    with warnings.catch_warnings():
        warnings.simplefilter("error")  # the rotated columns agree
        smp = Sample.from_file(path, moment=moment)
    assert smp.names == ["S1", "S2"]
    _assert_same(smp["S1"], a, rtol=5e-3, atol=1e-3)
    _assert_same(smp["S2"], b, rtol=5e-3, atol=1e-3)
    assert smp["S1"].demag == "thermal"
    assert smp["S1"].volume == pytest.approx(10.8e-6)
    assert smp["S2"].volume == pytest.approx(11.0e-6)
    assert [s.treatment for s in smp["S1"].steps][:2] == ["nrm", "thermal"]


def test_2g_af_and_susceptibility(tmp_path):
    spec = _synthetic()
    rows = [
        _2g_row(spec, "none", 0, spec.steps[0].M, k=1.5e-3),
        _2g_row(spec, "degauss z", 20, spec.steps[1].M, k=1.4e-3),
        # a susceptibility measurement without moment belongs to the step before
        _2g_row(spec, "degauss z", 20, [0, 0, 0], k=2.5e-3),
        _2g_row(spec, "degauss z", 40, spec.steps[2].M),
    ]
    path = tmp_path / "af.dat"
    path.write_text(_2g_text(rows))
    got = Sample.from_file(path)["S1"]
    assert got.demag == "af" and got.labels == ["NRM", "M020", "M040"]
    assert [s.level for s in got.steps] == pytest.approx([0, 20, 40])
    assert [s.k for s in got.steps] == pytest.approx(
        [1.5e-3, 1.4e-3, None], nan_ok=True
    )


def test_2g_arm_and_irm_are_kept_apart(tmp_path):
    spec = _synthetic()
    rows = [
        _2g_row(spec, "none", 0, spec.steps[0].M),
        _2g_row(
            spec, "degauss z - arm axial", 30, spec.steps[1].M, extra={"ARM Gauss": 1.0}
        ),
        _2g_row(spec, "irm", 0, spec.steps[2].M, extra={"IRM Gauss": 500.0}),
    ]
    path = tmp_path / "rm.dat"
    path.write_text(_2g_text(rows))
    got = Sample.from_file(path)["S1"]
    assert got.labels == ["NRM"]
    assert got.arm[0].treatment == "arm" and got.arm[0].level == pytest.approx(30.0)
    assert got.arm[0].field == pytest.approx(0.1)
    assert got.irm[0].treatment == "irm" and got.irm[0].field == pytest.approx(50.0)
    back = Specimen.from_json(got.to_json())
    assert [s.label for s in back.arm] == [s.label for s in got.arm]
    assert back.irm[0].field == pytest.approx(50.0)


def test_2g_orientation_mismatch_warns(tmp_path):
    spec = _synthetic()
    wrong = replace(spec, azimuth=spec.azimuth + 40)
    rows = [_2g_row(spec, "thermal demag", s.level, s.M) for s in spec.steps]
    path = tmp_path / "bad.dat"
    path.write_text(_2g_text(rows).replace("207\t", "247\t"))
    with pytest.warns(UserWarning, match="does not reproduce"):
        Sample.from_file(path)
    assert wrong.azimuth == pytest.approx(247)


def test_2g_bad_files_rejected(tmp_path):
    path = tmp_path / "long.dat"
    path.write_text("Position\tIntensity\r\n1\t2\r\n")
    with pytest.raises(ValueError, match="Sample ID"):
        Sample.from_file(path)
    path.write_text(_2g_text([_2g_row(_synthetic(), "none", 0, [1, 2, 3])]))
    with pytest.raises(ValueError, match="moment"):
        Sample.from_file(path, moment="x")
    with pytest.raises(ValueError, match="protocol"):
        Sample.from_file(path, protocol="x")


def _long_core_text(sections=2, positions=3, levels=(0, 10)):
    rows = []
    for s in range(sections):
        for lv in levels:
            for k in range(positions):
                depth = 0.01 * k
                cell = dict.fromkeys(_2G_COLUMNS, "NA")
                cell.update(
                    {
                        "Sample ID": f"CORE_{s + 1}",
                        "Meas. type": "SAMPLE/CONTINUOUS",
                        "Treatment Type": "none" if lv == 0 else "degauss z",
                        "AF Z": lv * 10,
                        "Declination: Unrotated": 100.0 + 50 * s + 5 * k,
                        "Inclination: Unrotated": 30.0 + lv,
                        "Declination: Sample Rotated": 100.0 + 50 * s + 5 * k,
                        "Inclination: Sample Rotated": 30.0 + lv,
                        "Intensity": 2e-7 / (1 + lv),
                        "Sample Azimiuth": 0.0,
                        "Sample Dip": 90.0,
                        "Depth": depth,
                    }
                )
                rows.append("\t".join(str(cell[c]) for c in _2G_COLUMNS))
    return _2g_text(rows)


def test_2g_continuous_long_core(tmp_path):
    path = tmp_path / "long.dat"
    path.write_text(_long_core_text())
    with warnings.catch_warnings():
        warnings.simplefilter("error")  # the rotated columns agree
        smp = Sample.from_file(path)
    assert len(smp) == 6
    assert smp.names[:3] == ["CORE_1_0", "CORE_1_0.01", "CORE_1_0.02"]
    first = smp[1]
    assert first.site == "CORE_1" and first.depth == pytest.approx(0.01)
    assert first.labels == ["NRM", "M010"] and first.demag == "af"
    # polar: 2e-7 G -> 2e-4 A/m, direction as in the file
    assert abs(first.steps[0].M) == pytest.approx(2e-4)
    assert _dec_inc(first.geo[0]) == pytest.approx((105.0, 30.0), abs=1e-6)
    assert set(smp.by_site()) == {"CORE_1", "CORE_2"}
    assert smp.by_site()["CORE_2"].names[0] == "CORE_2_0"
    back = Specimen.from_json(first.to_json())
    assert back.depth == pytest.approx(0.01)


def test_2g_continuous_options_are_checked(tmp_path):
    path = tmp_path / "long.dat"
    path.write_text(_long_core_text())
    with pytest.raises(ValueError, match="polar"):
        Sample.from_file(path, moment="cartesian")
    with pytest.raises(ValueError, match="discrete"):
        Sample.from_file(path, protocol="tray_normal")


def test_2g_sniffed_without_extension(tmp_path):
    path = tmp_path / "data.001"
    path.write_text(_2g_text(_2g_thermal_rows(_synthetic())))
    assert Sample.from_file(path).names == ["S1"]


# -- tray (holder) corrections -----------------------------------------------

_TRUE = [np.array([3.0, -2.0, 5.0]) * 1e-3 * (1 - 0.1 * i) for i in range(3)]
_TRAY = np.array([0.4, -0.3, 0.2]) * 1e-3


def _tray_rows(spec, protocol):
    row = lambda M: _2g_row(spec, "thermal demag", 100, M)
    rows = []
    for i, v in enumerate(_TRUE):
        tray = _TRAY
        if protocol == "tray_normal":
            rows += [row(tray), row(v + tray)]
        elif protocol == "normal_tray":
            rows += [row(v + tray), row(tray)]
        elif protocol == "tray_normal_ignore":
            # only the first tray line counts, the later ones are junk
            rows += [row(tray * (1 if i == 0 else 7)), row(v + tray)]
        elif protocol == "tray_normal_yflip":
            flipped = v * np.array([-1, 1, -1])
            rows += [row(tray), row(v + tray), row(flipped + tray)]
        else:  # tray_first
            if i == 0:
                rows.append(row(tray))
            rows.append(row(v + tray))
    return rows


@pytest.mark.parametrize(
    "protocol",
    [
        "tray_normal",
        "normal_tray",
        "tray_normal_ignore",
        "tray_normal_yflip",
        "tray_first",
    ],
)
def test_2g_tray_protocols_recover_the_sample(tmp_path, protocol):
    spec = _synthetic()
    path = tmp_path / "tray.dat"
    path.write_text(_2g_text(_tray_rows(spec, protocol)))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # the rotated columns hold raw moments
        got = Sample.from_file(path, moment="cartesian", protocol=protocol)["S1"]
    M = np.array([np.asarray(s.M) for s in got.steps])
    assert M.shape == (3, 3)
    assert np.allclose(M, np.array(_TRUE), rtol=2e-3, atol=1e-8)


def test_2g_tray_incomplete_group_warns(tmp_path):
    spec = _synthetic()
    rows = _tray_rows(spec, "tray_normal")[:-1]
    path = tmp_path / "tray.dat"
    path.write_text(_2g_text(rows))
    with pytest.warns(UserWarning, match="incomplete group"):
        got = Sample.from_file(path, moment="cartesian", protocol="tray_normal")["S1"]
    assert len(got) == 2


def test_dat_and_jr6_in_directory(tmp_path):
    spec = _synthetic()
    (tmp_path / "a.dat").write_text(_2g_text(_2g_thermal_rows(replace(spec, name="A"))))
    (tmp_path / "b.jr6").write_text(_jr6_text(replace(spec, name="B")))
    assert Sample.from_directory(tmp_path).names == ["A", "B"]
    assert Specimen.from_file(tmp_path / "b.jr6").name == "B"


def test_demag_step_json_keeps_treatment():
    step = DemagStep("ARM30", 3.0, Vector3(1, 2, 3), treatment="arm", field=0.1)
    back = DemagStep.from_json(step.to_json())
    assert back.treatment == "arm" and back.field == 0.1
    assert "treatment" not in DemagStep("T1", 1.0, Vector3(1, 2, 3)).to_json()


def test_directory_skips_unreadable_files_with_a_warning(tmp_path):
    base = _synthetic()
    from tests.test_paleomag import _write

    _write(replace(base, name="S1"), tmp_path / "s1.pmd")
    empty = tmp_path / "empty.pmd"  # a header without steps
    empty.write_text(
        "title\r\nEMPTY    a=  0.0   b=  0.0   s= 90.0   d=  0.0   v=11.0E-6m3\r\n"
        "STEP  Xc [Am2]  Yc [Am2]  Zc [Am2]  MAG[A/m]   Dg    Ig    Ds    Is  a95\r\n"
    )
    with pytest.warns(UserWarning, match="empty.pmd skipped"):
        smp = Sample.from_directory(tmp_path)
    assert smp.names == ["S1"]
    with pytest.raises(ValueError, match="no data rows"):
        Sample.from_directory(tmp_path, errors="raise")
    with pytest.raises(ValueError, match="errors"):
        Sample.from_directory(tmp_path, errors="ignore")
