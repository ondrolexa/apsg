"""Format dispatch for paleomagnetic data files.

Every reader returns a list of ``Specimen``, as the instrument formats (2G,
JR6, ...) hold several specimens per file. ``Specimen.from_file`` and
``Sample.from_file`` / ``Sample.from_directory`` go through here.
"""

import warnings
from dataclasses import replace
from pathlib import Path

import numpy as np

from apsg.math._vector import Vector3
from apsg.paleomag._specimen import (
    DemagStep,
    Specimen,
    _agico_orientation,
    _check_geometry,
    _dir,
    _float,
    _read_lines,
    _spin,
    bedding_vector,
)

__all__ = ()

_OE_TO_MT = 0.1  # 1 Oe = 0.1 mT
_G_TO_AM = 1e3  # emu/cm3 (G) to A/m


def _pmd(source, **kwargs):
    return [Specimen.from_pmd(source, **kwargs)]


def _rs3(source, **kwargs):
    return [Specimen.from_rs3(source, **kwargs)]


def _squid(source, **kwargs):
    return [Specimen.from_squid(source, **kwargs)]


def _by_name(items):
    """Group (name, item) pairs by name, in order of first appearance."""
    groups = {}
    for name, item in items:
        groups.setdefault(name, []).append(item)
    return groups


# ---------------------------------------------------------------------------
# AGICO JR6 / JR6A spinner magnetometer
# ---------------------------------------------------------------------------


def _jr6_treatment(text, demag):
    """Return (kind, level) of a JR6 treatment field."""
    s = text.strip().upper()
    if not s or s.startswith("N"):
        return "nrm", 0.0
    kind = {"T": "thermal", "A": "af", "M": "arm"}.get(s[0])
    number = _float(s[1:] if kind else s)
    if number is None:
        raise ValueError(f"unknown treatment {text!r}")
    if kind is None:
        if demag is None:
            raise ValueError(
                f"treatment {text!r} has no type, pass demag='thermal' or 'af'"
            )
        kind = demag
    return kind, number


def _jr6(source, demag=None):
    """Read an AGICO JR6 file: several specimens, 80 columns per measurement."""
    rows = []
    bad = []
    for i, ln in enumerate(_read_lines(source), start=1):
        if not ln.strip():
            continue
        try:
            if len(ln.rstrip()) < 64:
                raise ValueError("line too short")
            xyz = [float(ln[a : a + 6]) for a in (18, 24, 30)]
            exp = int(ln[36:40])
            az, dip, faz, fdip = (float(ln[a : a + 4]) for a in (40, 44, 48, 52))
            p = [int(ln[a : a + 3]) for a in (64, 67, 70, 73)]
            kind, level = _jr6_treatment(ln[10:18], demag)
        except ValueError:
            bad.append(i)
            continue
        rows.append((ln[:10].strip(), kind, level, xyz, exp, az, dip, faz, fdip, p))
    if bad:
        shown = ", ".join(map(str, bad[:3])) + (", ..." if len(bad) > 3 else "")
        warnings.warn(
            f"{source}: {len(bad)} malformed JR6 lines skipped (lines {shown})",
            stacklevel=3,
        )
    if not rows:
        raise ValueError(f"{source}: no JR6 data found")
    specimens = []
    for name, group in _by_name((r[0], r[1:]) for r in rows).items():
        _, _, _, _, az, dip, faz, fdip, p = group[0]
        azimuth, hade, spin, dipdir = _agico_orientation(
            *p, az, dip, faz, source=source
        )
        steps, arm = [], []
        for kind, level, xyz, exp, *_ in group:
            M = _spin(Vector3(np.array(xyz) * 10.0**exp), spin)
            if kind == "nrm":
                label = "NRM"
            elif kind == "thermal":
                label = f"T{level:g}"
            else:
                label = f"M{level:03g}"
            step = DemagStep(label, level, M, treatment=kind)
            (arm if kind == "arm" else steps).append(step)
        specimens.append(
            Specimen(
                name=name,
                steps=steps,
                demag=demag or _demag_of(steps),
                azimuth=azimuth,
                hade=hade,
                bedding=bedding_vector(dipdir, fdip),
                arm=arm,
                filename=None if hasattr(source, "read") else str(source),
            )
        )
    return specimens


def _demag_of(steps):
    """Demagnetization type from the treatment kinds of the steps."""
    kinds = {s.treatment for s in steps} - {"nrm", None}
    return kinds.pop() if len(kinds) == 1 else None


# ---------------------------------------------------------------------------
# Caltech (CIT) format
# ---------------------------------------------------------------------------

_CIT_KINDS = {"NR": "nrm", "AF": "af", "TT": "thermal"}


def _cit_specimen(path):
    lines = _read_lines(path)
    lines = [ln for ln in lines if ln.strip()] if len(lines) < 2 else lines
    if len(lines) < 3:
        raise ValueError(f"{path}: not a CIT specimen file")
    name = lines[0][4:13].strip() or Path(path).stem
    head = lines[1].split()
    if len(head) < 5:
        raise ValueError(f"{path}: orientation line of a CIT file needs 5 numbers")
    _, core_strike, core_dip, bed_strike, bed_dip = (float(h) for h in head[:5])
    steps, geo, tilt, skipped = [], [], [], []
    for i, ln in enumerate(lines[2:], start=3):
        if not ln.strip():
            continue
        kind = _CIT_KINDS.get(ln[:2].upper())
        if kind is None or len(ln) < 57:
            skipped.append(i)
            continue
        level = _float(ln[2:6].replace("M", " ").strip(), 0.0)
        intensity = _float(ln[30:39])
        dec, inc = _float(ln[45:51]), _float(ln[51:57])
        if None in (intensity, dec, inc):
            skipped.append(i)
            continue
        label = {"nrm": "NRM", "af": f"M{level:03g}", "thermal": f"T{level:g}"}[kind]
        steps.append(
            DemagStep(
                label,
                level,
                Vector3(intensity * _G_TO_AM * _dir(dec, inc)),
                treatment=kind,
            )
        )
        pair = []
        for a in (6, 24):
            d, n = _float(ln[a : a + 6]), _float(ln[a + 6 : a + 12])
            pair.append(None if None in (d, n) else _dir(d, n))
        geo.append(pair[0])
        tilt.append(pair[1])
    if skipped:
        shown = ", ".join(map(str, skipped[:3])) + (", ..." if len(skipped) > 3 else "")
        warnings.warn(
            f"{path}: {len(skipped)} lines are not CIT data lines and were "
            f"skipped (lines {shown})",
            stacklevel=4,
        )
    if not steps:
        raise ValueError(f"{path}: no CIT data lines found")
    spec = Specimen(
        name=name,
        steps=steps,
        demag=_demag_of(steps),
        azimuth=(core_strike + 270) % 360,
        hade=core_dip,
        bedding=bedding_vector((bed_strike + 90) % 360, bed_dip),
        filename=str(path),
    )
    _check_geometry(spec, geo, tilt, path)
    return spec


def _cit(source):
    """Read a CIT ``.sam`` index (or a single specimen file)."""
    if hasattr(source, "read"):
        raise ValueError("CIT files are read by name, the index refers to other files")
    path = Path(source)
    if path.suffix.lower() != ".sam":
        return [_cit_specimen(path)]
    specimens = []
    for name in [ln.strip() for ln in _read_lines(path)[2:] if ln.strip()]:
        file = path.parent / name
        if not file.is_file():
            warnings.warn(f"{path}: specimen file {name!r} not found", stacklevel=3)
            continue
        specimens.append(_cit_specimen(file))
    if not specimens:
        raise ValueError(f"{path}: no specimen files found")
    return specimens


# ---------------------------------------------------------------------------
# 2G Enterprises
# ---------------------------------------------------------------------------

_TWOG_PROTOCOLS = (
    "normal",
    "tray_normal",
    "normal_tray",
    "tray_normal_yflip",
    "tray_first",
    "tray_normal_ignore",
)
_TWOG_TREATMENTS = {
    "none": "nrm",
    "degauss x, y, & z": "af",
    "degauss z": "af",
    "degauss z - arm axial": "arm",
    "irm": "irm",
    "thermal demag": "thermal",
}


def _combine_tray(rows, protocol, source):
    """Subtract the holder (tray) measurements.

    `rows` are ``(info, M)`` pairs in file order; returns the corrected ones.
    The formulas follow the protocols of the 2G software: the sample line
    supplies the metadata, tray lines are subtracted. For ``tray_normal_yflip``
    the third line is the sample flipped by 180 degrees about y, so that
    ``N - R = 2 (x, 0, z)``, ``N - T = (x, y, z)`` and ``R - T = (-x, y, -z)``;
    x and z are the mean of the three estimates, y of two.
    """
    size = {"tray_normal": 2, "normal_tray": 2, "tray_normal_ignore": 2}
    size = size.get(protocol, 3 if protocol == "tray_normal_yflip" else 1)
    if protocol == "tray_first":
        if not rows:
            return []
        tray = np.asarray(rows[0][1])
        return [(info, Vector3(np.asarray(M) - tray)) for info, M in rows[1:]]
    if protocol == "tray_normal_ignore":
        tray = np.asarray(rows[0][1]) if rows else np.zeros(3)
    out = []
    if len(rows) % size:
        warnings.warn(
            f"{source}: {len(rows)} lines do not fit the {protocol!r} protocol "
            f"(groups of {size}); the last incomplete group is dropped",
            stacklevel=3,
        )
    for i in range(0, len(rows) - size + 1, size):
        group = rows[i : i + size]
        Ms = [np.asarray(M, dtype=float) for _, M in group]
        if protocol == "tray_normal":
            info, M = group[1][0], Ms[1] - Ms[0]
        elif protocol == "normal_tray":
            info, M = group[0][0], Ms[0] - Ms[1]
        elif protocol == "tray_normal_ignore":
            info, M = group[1][0], Ms[1] - tray
        else:  # tray_normal_yflip
            T, N, R = Ms
            n_t, n_r, r_t = N - T, (N - R) / 2, R - T
            M = np.array(
                [
                    np.mean([n_t[0], n_r[0], -r_t[0]]),
                    np.mean([n_t[1], r_t[1]]),
                    np.mean([n_t[2], n_r[2], -r_t[2]]),
                ]
            )
            info = group[1][0]
        out.append((info, Vector3(M)))
    return out


def _twog_kind(cells, cell, text):
    """Treatment kind of a 2G row: the column, else the first filled field."""
    kind = _TWOG_TREATMENTS.get(text(cells, "Treatment Type").lower())
    if kind:
        return kind
    for column, name in (
        ("ARM Gauss", "arm"),
        ("IRM Gauss", "irm"),
        ("AF X", "af"),
        ("AF Z", "af"),
        ("Temp C", "thermal"),
    ):
        if cell(cells, column) is not None:
            return name
    return "nrm"


def _twog_step(kind, M, k, cells, cell):
    """The ``DemagStep`` of a 2G row of the given treatment kind."""
    af = next((v for v in (cell(cells, f"AF {a}") for a in "ZXY") if v is not None), 0)
    if kind == "thermal":
        temp = cell(cells, "Temp C") or 0.0
        return DemagStep(f"T{temp:g}", temp, M, k=k, treatment=kind)
    if kind == "af":
        level = af * _OE_TO_MT
        return DemagStep(f"M{level:03g}", level, M, k=k, treatment=kind)
    if kind == "irm":
        field = (cell(cells, "IRM Gauss") or 0.0) * _OE_TO_MT
        return DemagStep(f"IRM{field:g}", field, M, k=k, treatment=kind, field=field)
    if kind == "arm":
        level = af * _OE_TO_MT
        bias = cell(cells, "ARM Gauss")
        field = None if bias is None else bias * _OE_TO_MT
        return DemagStep(f"ARM{level:g}", level, M, k=k, treatment=kind, field=field)
    return DemagStep("NRM", 0.0, M, k=k, treatment="nrm")


def _twog(source, moment="polar", protocol="normal", demag=None):
    """Read a 2G Enterprises ``.DAT`` file of discrete samples.

    Tab-separated, the columns are found by their names. NA values are
    missing. Intensities are in G (emu/cm3), AF fields in Oe.

    Continuous (long core) files (``Meas. type`` ``Sample/Continuous``) give one
    specimen per section and depth (``Depth``, or ``Position``): `site` is the
    section (the ``Sample ID``), `depth` the position and the name
    ``<section>_<depth>``. Only ``moment="polar"`` is read for them.
    """
    if moment not in ("polar", "cartesian"):
        raise ValueError(f"moment must be 'polar' or 'cartesian', not {moment!r}")
    if protocol not in _TWOG_PROTOCOLS:
        raise ValueError(f"protocol must be one of {_TWOG_PROTOCOLS}, not {protocol!r}")
    lines = [ln for ln in _read_lines(source) if ln.strip()]
    if len(lines) < 2:
        raise ValueError(f"{source}: not a 2G file")
    cols = {c.strip().strip('"').lower(): i for i, c in enumerate(lines[0].split("\t"))}
    if "sample id" not in cols:
        raise ValueError(f"{source}: no 'Sample ID' column, is this a 2G file?")

    def text(cells, name):
        i = cols.get(name.lower())
        return cells[i].strip().strip('"') if i is not None and i < len(cells) else ""

    def cell(cells, name):
        v = text(cells, name)
        return None if v.upper() in ("", "NA") else _float(v)

    rows = []  # (line number, cells, key, moment or None)
    continuous = False
    for n, ln in enumerate(lines[1:], start=2):
        cells = ln.split("\t")
        if "continuous" in text(cells, "Meas. type").lower():
            continuous = True
            if moment != "polar":
                raise ValueError(
                    f"{source}: continuous (long core) 2G files are read with "
                    "moment='polar' (the sensor response is not modelled)"
                )
            if protocol != "normal":
                raise ValueError(f"{source}: tray protocols are for discrete samples")
        M = None
        if moment == "polar":
            mag = cell(cells, "Intensity")
            dec = cell(cells, "Declination: Unrotated")
            inc = cell(cells, "Inclination: Unrotated")
            if None not in (mag, dec, inc):
                M = Vector3(mag * _G_TO_AM * _dir(dec, inc))
        else:
            vol = cell(cells, "Volume") or 10.8
            for tag in ("corr", "mean"):
                xyz = [cell(cells, f"{a} {tag}") for a in "XYZ"]
                if None not in xyz:
                    M = Vector3(np.array(xyz) * _G_TO_AM / vol)
                    break
        key = text(cells, "Sample ID")
        if continuous:
            # one specimen per position along the core: (section, depth)
            depth = cell(cells, "Depth")
            depth = cell(cells, "Position") if depth is None else depth
            key = (key, depth)
        rows.append((n, cells, key, M))

    specimens = []
    for key, group in _by_name((r[2], r) for r in rows).items():
        sample, depth = key if continuous else (key, None)
        measured = [(r, r[3]) for r in group if r[3] is not None]
        if protocol != "normal":
            measured = _combine_tray(measured, protocol, source)
        steps, arm, irm, geo, tilt = [], [], [], [], []
        numbers = []  # line number of each entry of steps, arm and irm
        for r, M in measured:
            kind = _twog_kind(r[1], cell, text)
            step = _twog_step(kind, M, cell(r[1], "MS corr"), r[1], cell)
            {"arm": arm, "irm": irm}.get(kind, steps).append(step)
            numbers.append((r[0], kind, step))
            if kind not in ("arm", "irm"):
                pair = []
                for tag in ("Sample", "Formation"):
                    d = cell(r[1], f"Declination: {tag} Rotated")
                    i = cell(r[1], f"Inclination: {tag} Rotated")
                    pair.append(None if None in (d, i) else _dir(d, i))
                geo.append(pair[0])
                tilt.append(pair[1])
        # a susceptibility-only line belongs to the step measured before it
        for r in group:
            k = cell(r[1], "MS corr")
            if r[3] is not None or k is None:
                continue
            before = [e for e in numbers if e[0] < r[0] and e[2].k is None]
            if before:
                old = before[-1][2]
                new = replace(old, k=k)
                for lst in (steps, arm, irm):
                    if old in lst:
                        lst[lst.index(old)] = new
                numbers[numbers.index(before[-1])] = (before[-1][0], before[-1][1], new)
        first = group[0][1]
        az, dip = cell(first, "Sample Azimiuth"), cell(first, "Sample Dip")
        faz, fdip = cell(first, "Formation Dip Azimuth"), cell(first, "Formation Dip")
        vol = next((v for v in (cell(r[1], "Volume") for r in group) if v), None)
        spec = Specimen(
            name=sample if depth is None else f"{sample}_{depth:g}",
            site=sample if continuous else "",
            depth=depth,
            steps=steps,
            demag=demag or _demag_of(steps),
            azimuth=(az or 0.0) % 360,
            hade=0.0 if dip is None else 90.0 - dip,
            bedding=(
                bedding_vector(faz, fdip)
                if None not in (faz, fdip)
                else Vector3(0.0, 0.0, -1.0)
            ),
            volume=None if vol is None else vol * 1e-6,
            arm=arm,
            irm=irm,
            filename=None if hasattr(source, "read") else str(source),
        )
        if protocol == "normal" and None not in (az, dip):
            _check_geometry(spec, geo, tilt, source)
        specimens.append(spec)
    return specimens


# format name -> reader returning a list of specimens
_READERS = {
    "pmd": _pmd,
    "rs3": _rs3,
    "squid": _squid,
    "jr6": _jr6,
    "cit": _cit,
    "2g": _twog,
}
# file extension -> format name
_EXTENSIONS = {
    ".pmd": "pmd",
    ".rs3": "rs3",
    ".squid": "squid",
    ".jr6": "jr6",
    ".sam": "cit",
    ".dat": "2g",
}


def _sniff(path):
    """Guess the format of a file from its content, None if unknown."""
    try:
        lines = _read_lines(path)[:3]
    except (OSError, UnicodeDecodeError):
        return None
    head = " ".join(lines)
    nonblank = [ln for ln in lines if ln.strip()]
    if (
        len(nonblank) >= 3
        and "SDec" in nonblank[0]
        and nonblank[2].lstrip().upper().startswith("ID")
    ):
        return "rs3"
    if "a=" in head and "b=" in head and "v=" in head:
        return "pmd"
    if lines and "\t" in lines[0] and "sample id" in lines[0].lower():
        return "2g"
    if lines and all(len(ln.rstrip()) in (79, 80) for ln in lines if ln.strip()):
        return "jr6"
    if len(lines) == 3 and lines[2][:2] in _CIT_KINDS:
        return "cit"
    return None


def detect_format(path):
    """Format name of the file `path` by its extension, else by its content.

    Returns None when the format is not recognized.
    """
    path = Path(path)
    return _EXTENSIONS.get(path.suffix.lower()) or _sniff(path)


def _read_specimens(source, fmt=None, **kwargs):
    """Read all specimens of a file; `fmt` overrides the detection."""
    if fmt is None:
        if hasattr(source, "read"):
            raise ValueError("fmt is required when reading from an open file")
        fmt = detect_format(source)
        if fmt is None:
            ext = Path(source).suffix.lower()
            raise ValueError(
                f"Unsupported file type {ext!r}, expected one of "
                f"{sorted(_EXTENSIONS)} or a recognizable content"
            )
    if fmt not in _READERS:
        raise ValueError(f"Unknown format {fmt!r}, expected one of {sorted(_READERS)}")
    return _READERS[fmt](source, **kwargs)


def _read_directory(
    path, pattern="*", recursive=False, fmt=None, errors="warn", **kwargs
):
    """Read all files of a directory matching `pattern`, sorted by name.

    Without `fmt`, files of an unrecognized type are skipped. A file that cannot
    be read (e.g. a header without steps) is skipped with a warning, or raises
    with ``errors="raise"``.
    """
    if errors not in ("warn", "raise"):
        raise ValueError(f"errors must be 'warn' or 'raise', not {errors!r}")
    path = Path(path)
    if not path.is_dir():
        raise NotADirectoryError(str(path))
    files = sorted(
        f
        for f in (path.rglob(pattern) if recursive else path.glob(pattern))
        if f.is_file()
    )
    specimens = []
    for f in files:
        if fmt is None and detect_format(f) is None:
            continue
        try:
            specimens.extend(_read_specimens(f, fmt=fmt, **kwargs))
        except (ValueError, NotImplementedError) as err:
            if errors == "raise":
                raise
            warnings.warn(f"{f.name} skipped: {err}", stacklevel=3)
    return specimens
