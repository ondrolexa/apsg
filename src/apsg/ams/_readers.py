"""Tables of AMS tensors: ``DataFrame`` and CSV readers and writers."""

import io
import math
import re
import struct
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from apsg.ams._specimen import (
    COMPONENTS,
    AMSSite,
    AMSSpecimen,
    tensor_components,
    tensor_from_components,
)
from apsg.paleomag._specimen import _agico_orientation

_ALIASES = {
    "name": "name",
    "specimen": "name",
    "sample": "name",
    "site": "site",
    "bulk": "bulk",
    "km": "bulk",
    "k_mean": "bulk",
    "std_error": "std_error",
    "sigma": "std_error",
    "azimuth": "azimuth",
    "hade": "hade",
    "dip_direction": "dipdir",
    "dipdir": "dipdir",
    "dip": "dip",
    "frame": "frame",
    **{c: c for c in COMPONENTS},
}


def _value(row, key):
    v = row.get(key)
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return None
    return v


def from_dataframe(df, name="Default"):
    """Return the sites of a table of AMS tensors.

    Required columns are the six tensor components ``k11 k22 k33 k12 k23 k13``
    (the order of AGICO files). Optional columns: ``name`` (``specimen``,
    ``sample``), ``site``, ``bulk`` (``km``, ``k_mean``), ``std_error``
    (``sigma``), ``azimuth`` and ``hade`` of the core, ``dip_direction`` and
    ``dip`` of the bedding, and ``frame`` (``specimen`` or ``geographic``).
    Column names are case-insensitive; other columns are kept in ``meta``.

    Args:
        df: ``DataFrame``.
        name: site name when the table has no ``site`` column.

    Returns:
        dict: site name -> ``AMSSite``, in the order of first appearance.
    """
    cols = {c: _ALIASES.get(str(c).strip().lower(), None) for c in df.columns}
    missing = [c for c in COMPONENTS if c not in cols.values()]
    if missing:
        raise ValueError(f"Missing tensor columns: {', '.join(missing)}")
    sites = {}
    for i, (_, rec) in enumerate(df.iterrows()):
        row = {cols[c]: v for c, v in rec.items() if cols[c] is not None}
        meta = {str(c): v for c, v in rec.items() if cols[c] is None}
        dipdir, dip = _value(row, "dipdir"), _value(row, "dip")
        if (dipdir is None) != (dip is None):
            raise ValueError("dip_direction and dip must be given together")
        site = _value(row, "site")
        site = name if site is None else str(site)
        spec = AMSSpecimen(
            name=str(_value(row, "name") or f"{site}-{i + 1}"),
            tensor=tensor_from_components(*(float(row[c]) for c in COMPONENTS)),
            bulk=_value(row, "bulk"),
            std_error=_value(row, "std_error"),
            azimuth=_value(row, "azimuth"),
            hade=_value(row, "hade"),
            bedding=None if dip is None else (float(dipdir), float(dip)),
            site=site,
            frame=_value(row, "frame") or "specimen",
            meta=meta,
        )
        sites.setdefault(site, AMSSite(name=site)).append(spec)
    return sites


def read_csv(source, name="Default", **kwargs):
    """Read a CSV table of AMS tensors (see `from_dataframe`).

    Args:
        source: file name or open file.
        name: site name when the table has no ``site`` column.
        **kwargs: passed to ``pandas.read_csv``.

    Returns:
        dict: site name -> ``AMSSite``.
    """
    return from_dataframe(pd.read_csv(source, **kwargs), name=name)


def to_dataframe(sites):
    """Return a table of the specimens of sites (inverse of `from_dataframe`).

    Args:
        sites: an ``AMSSite``, or an iterable or dict of them.
    """
    if isinstance(sites, AMSSite):
        sites = [sites]
    elif isinstance(sites, dict):
        sites = list(sites.values())
    rows = []
    for site in sites:
        for s in site:
            row = {"site": s.site or site.name, "name": s.name}
            row.update(zip(COMPONENTS, tensor_components(s.tensor), strict=True))
            row["bulk"] = s.bulk
            row["std_error"] = s.std_error
            row["azimuth"] = s.azimuth
            row["hade"] = s.hade
            if s.bedding is None:
                row["dip_direction"] = row["dip"] = np.nan
            else:
                dd, dip = s.bedding.geo
                row["dip_direction"] = round(float(dd) % 360, 8)
                row["dip"] = round(float(dip) + 90, 8)
            row["frame"] = s.frame
            row.update(s.meta)
            rows.append(row)
    return pd.DataFrame(rows)


# -- instrument files ---------------------------------------------------------


def _read_raw(source):
    """Return ``(raw, name)``: bytes (or str) of a path, bytes or open file."""
    if isinstance(source, bytes | bytearray):
        return bytes(source), None
    if hasattr(source, "read"):
        return source.read(), Path(getattr(source, "name", "") or "").stem or None
    return Path(source).read_bytes(), Path(source).stem


def _decode(raw, name):
    """Return ``(text, name)`` of the content of a text file."""
    if isinstance(raw, str):
        return raw.replace("\r", ""), name
    if b"\x00" in raw[:2048]:
        raise ValueError(
            "binary AMS file (Anisoft .ams); export it to text in Anisoft (report "
            ".asc or table .txt) and read that. Kappabridge .ran files are read by "
            "read_ran"
        )
    for enc in ("utf-8-sig", "latin-1"):
        try:
            return raw.decode(enc).replace("\r", ""), name
        except UnicodeDecodeError:
            continue
    raise ValueError("cannot decode the file")  # pragma: no cover


def _text(source):
    """Return ``(text, name)`` of a path or an open text or binary file."""
    return _decode(*_read_raw(source))


def _numbers(line):
    return [float(x) for x in re.findall(r"[-+]?\d*\.?\d+(?:[Ee][-+]?\d+)?", line)]


def _susceptibility_row(lines):
    """Parse the susceptibility row of a SUFAR, SAFYR or SUSAR page.

    The row below the header with ``susc.`` is ``field, [frequency tag], mean
    susceptibility, error [%], F, F12, F23`` (SUFAR, SAFYR) or ``mean
    susceptibility, norming factor, error [%], F, F12, F23`` (SUSAR, whose header
    reads ``Norming``).

    Returns:
        dict: ``bulk``, ``error_percent``, ``F``, ``F12``, ``F23``, and ``field``
        or ``norming_factor`` depending on the layout.

    Raises:
        ValueError: when the row is missing or has too few numbers.
    """
    for k, ln in enumerate(lines):
        if "susc." not in ln:
            continue
        susar = "Norming" in "\n".join(lines[max(0, k - 2) : k + 1])
        for row in lines[k + 1 :]:
            if re.search(r"\d[Ee][-+]\d", row):
                nums = _numbers(re.sub(r"\bF\d\b", "", row))
                if len(nums) < 6:
                    break
                out = {
                    "bulk": nums[0] if susar else nums[1],
                    "error_percent": nums[-4],
                    "F": nums[-3],
                    "F12": nums[-2],
                    "F23": nums[-1],
                }
                out["norming_factor" if susar else "field"] = nums[1 if susar else 0]
                return out
        break
    raise ValueError("no susceptibility row")


def _page_meta(page, row):
    """Optional metadata of a page (volumes, holder, demag. factor, tests)."""
    meta = {}
    for key, pattern in (
        ("nominal_volume", r"Nom\. vol\.\s*([-+\d.Ee]+)"),
        ("actual_volume", r"Act\. vol\.\s*([-+\d.Ee]+)"),
        ("holder", r"Holder\s+([-+\d.Ee]+)"),
    ):
        m = re.search(pattern, page)
        try:
            meta[key] = float(m.group(1)) if m else None
        except ValueError:
            meta[key] = None
    m = re.search(r"Demag\. fac\.\s*:\s*(YES|NO)", page, re.IGNORECASE)
    meta["demag_factor"] = None if m is None else m.group(1).upper() == "YES"
    meta.update({k: v for k, v in row.items() if k != "bulk"})
    meta.setdefault("field", None)
    return meta


def _page_foliation(lines):
    """Foliation ``(dip direction or strike, dip, tag)`` of a page.

    None when the foliation is not set (dip 0). The tag (``S0``, ``K0``, ...)
    is the first item of the row below the ``T1 F1 L1`` header.
    """
    for k, ln in enumerate(lines):
        if re.match(r"\s*T1\s+F1\s+L1", ln):
            for row in lines[k + 1 :]:
                m = re.search(r"(\d+)/(\d+)", row)
                if m:
                    az, dip = float(m.group(1)), float(m.group(2))
                    return None if dip == 0 else (az, dip, row.split()[0])
            break
    return None


def read_sufar(source, name=None):
    """Read an AGICO SUFAR/SAFYR/SUSAR report (``.asc``) of one site (alias `read_asc`).

    Each page of the report holds one specimen: name, core ``Azi`` and ``Dip``
    with the orientation code ``O.P.`` (P1-P4 of AGICO, see
    ``apsg.paleomag._specimen._agico_orientation``), mean susceptibility, the
    standard errors of the normed principal susceptibilities and the normed
    tensor in specimen coordinates. The tensor, the core orientation (checked
    against the geographic tensor printed on each page for ``O.P. 6 0 6 0``),
    the mean susceptibility (``bulk``, in the unit of the file) and the foliation
    (the bedding, when given) are kept; the ``std_error`` is the mean of the
    printed standard errors. ``meta`` holds the ``O.P.`` code, volumes, holder
    correction, demagnetization factor flag, field, the anisotropy tests
    ``F``, ``F12``, ``F23`` and the error in percent, where present. Pages cut
    off before the tensor are skipped with a warning.

    Args:
        source: file name or open file.
        name: site name. Default the file name.

    Returns:
        AMSSite: the specimens of the file.

    Raises:
        NotImplementedError: for orientation codes other than P1 and P3 of 3,
            6, 9 or 12, P2 of 0 or 90 and P4 of 0 or 90.
    """
    text, stem = _text(source)
    site = AMSSite(name=name or stem or "Default")
    skipped = []
    no_code = []
    for page in text.split("\f"):
        if "ANISOTROPY OF SUSCEPTIBILITY" not in page:
            continue
        lines = page.splitlines()
        sp = lines[0].split("ANISOTROPY")[0].strip()
        try:
            azi = float(re.search(r"Azi\s+(\S+)", page).group(1))
            dip = float(re.search(r"Dip\s+(\S+)", page).group(1))
            i = next(k for k, ln in enumerate(lines) if ln.startswith("Specimen"))
            diag = _numbers(lines[i][8:])
            off = _numbers(lines[i + 1].split("I", 1)[1])
            if len(diag) != 6 or len(off) != 6:
                raise ValueError
            diag, off = diag[-3:], off[-3:]
            susc = _susceptibility_row(lines)
            errs = [
                float(x)
                for ln in lines
                if ln.startswith("+-")
                for x in _numbers(ln[2:])[:3]
            ]
        except (AttributeError, StopIteration, ValueError, IndexError):
            skipped.append(sp)
            continue
        op = re.search(r"O\.P\.\s*:\s*(\d+)\s+(\d+)\s+(\d+)\s+(\d+)", page)
        if op:
            code = tuple(int(g) for g in op.groups())
        else:
            code = (6, 0, 6, 0)
            no_code.append(sp)
        foliation = _page_foliation(lines)
        azimuth, hade, spin, dipdir = _agico_orientation(
            *code,
            azi,
            dip,
            None if foliation is None else foliation[0],
            source=f"{site.name}: specimen {sp}",
        )
        tensor = tensor_from_components(*diag, off[0], off[1], off[2])
        if spin:
            c, s = np.cos(np.radians(spin)), np.sin(np.radians(spin))
            R = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
            tensor = R @ tensor @ R.T
        site.append(
            AMSSpecimen(
                name=sp,
                tensor=tensor,
                bulk=susc["bulk"],
                std_error=float(np.mean(errs)) if errs else None,
                azimuth=azimuth,
                hade=hade,
                bedding=None if foliation is None else (dipdir, foliation[1]),
                site=site.name,
                meta={
                    "op": code,
                    **_page_meta(page, susc),
                    "foliation_tag": None if foliation is None else foliation[2],
                },
            )
        )
    if skipped:
        warnings.warn(
            f"incomplete specimen pages skipped: {', '.join(skipped)}", stacklevel=2
        )
    if no_code:
        warnings.warn(
            f"no O.P. orientation code for {', '.join(no_code)}, assuming 6 0 6 0",
            stacklevel=2,
        )
    if not len(site):
        raise ValueError("no specimens found, is this a SUFAR report?")
    return site


def _tensor_from_directions(L, F, dirs):
    """Normed tensor from ``L``, ``F`` and the (dec, inc) of k1, k2, k3."""
    k3 = 3 / (1 + F + L * F)
    k = np.array([L * F * k3, F * k3, k3])
    v = []
    for dec, inc in dirs:
        d, i = np.radians(dec), np.radians(inc)
        v.append(np.array([np.cos(d) * np.cos(i), np.sin(d) * np.cos(i), np.sin(i)]))
    # the printed directions are rounded, so make the frame exactly orthonormal
    # (k1 and k3 kept as printed, k2 their cross product); otherwise the eigenvalues
    # of weak fabrics are mixed
    v1 = v[0] / np.linalg.norm(v[0])
    v3 = v[2] - (v[2] @ v1) * v1
    v3 /= np.linalg.norm(v3)
    v2 = np.cross(v3, v1)
    T = sum(ki * np.outer(vi, vi) for ki, vi in zip(k, (v1, v2, v3), strict=True))
    return T


def read_anisoft_table(source, name=None, frame="geographic", site=None):
    """Read a text table exported by Anisoft (one row per specimen).

    The first line names the columns. The specimen is ``Name`` or ``Specimen``,
    the mean susceptibility ``Km`` or ``Kmean`` (``bulk``, in the unit of the
    file). Tables with the tensor columns ``K11 K22 K33 K12 K23 K13`` use
    them; tables without (the "tensor-less" export) give a tensor made from
    the principal directions ``K1dec K1inc K2dec K2inc K3dec K3inc`` (also
    ``Kmax_Dec`` ...) and the ratios ``L`` and ``F``, which are exact up to
    their printed precision.

    Args:
        source: file name or open file.
        name: site name. Default the file name.
        frame: coordinates of the tensors of the table, ``"geographic"``
            (default, what Anisoft exports after the orientation was applied) or
            ``"specimen"``.
        site: optional function of a specimen name giving its site, e.g.
            ``lambda n: n.split("_")[0]``; the result is then a dict of sites.

    Returns:
        AMSSite or dict: the specimens (a dict by site if `site` is given).
    """
    text, stem = _text(source)
    rows = [ln for ln in text.splitlines() if ln.strip()]
    header = rows[0].split()
    cols = {h.lower().replace(".", ""): i for i, h in enumerate(header)}

    def col(*names):
        for n in names:
            if n in cols:
                return cols[n]
        return None

    iname = col("name", "specimen")
    ikm = col("km", "kmean")
    if iname is None:
        raise ValueError("no Name or Specimen column, is this an Anisoft table?")
    tensor_cols = [col(c.lower()) for c in COMPONENTS]
    dir_cols = [
        col(*n)
        for n in (
            ("k1dec", "kmax_dec"),
            ("k1inc", "kmax_inc"),
            ("k2dec", "kint_dec"),
            ("k2inc", "kint_inc"),
            ("k3dec", "kmin_dec"),
            ("k3inc", "kmin_inc"),
        )
    ]
    il, jf = col("l"), col("f")
    has_tensor = None not in tensor_cols
    if not has_tensor and (None in dir_cols or il is None or jf is None):
        raise ValueError(
            "the table has neither tensor columns nor directions with L, F"
        )
    site_name = name or stem or "Default"
    specimens = []
    dropped = []
    for n, ln in enumerate(rows[1:], start=2):
        parts = ln.split()
        if len(parts) < len(header):
            dropped.append(n)
            continue
        try:
            if has_tensor:
                T = tensor_from_components(*(float(parts[i]) for i in tensor_cols))
            else:
                d = [float(parts[i]) for i in dir_cols]
                T = _tensor_from_directions(
                    float(parts[il]),
                    float(parts[jf]),
                    list(zip(d[::2], d[1::2], strict=True)),
                )
            bulk = float(parts[ikm]) if ikm is not None else None
        except ValueError:
            dropped.append(n)
            continue
        specimens.append(
            AMSSpecimen(parts[iname], T, bulk=bulk, site=site_name, frame=frame)
        )
    if dropped and specimens:
        shown = ", ".join(map(str, dropped[:3])) + (", ..." if len(dropped) > 3 else "")
        warnings.warn(
            f"{len(dropped)} table rows are short or not numeric and were skipped "
            f"(rows {shown})",
            stacklevel=2,
        )
    if not specimens:
        raise ValueError("no specimens found")
    result = AMSSite(specimens, name=site_name)
    return result.split(site) if site is not None else result


# -- AGICO RAN files and AEX tables ----------------------------------------------

_RAN_HEADER = 64
_RAN_RECORD = struct.Struct("<12s8f2s4h2s4h")  # 64 bytes


def _looks_like_ran(raw):
    """Whether `raw` has the structure of a RAN file: 64-byte header and records
    of 64 bytes whose first 12 bytes are printable text."""
    if len(raw) < 2 * _RAN_HEADER or (len(raw) - _RAN_HEADER) % _RAN_HEADER:
        return False
    names = (
        raw[_RAN_HEADER + i : _RAN_HEADER + i + 12]
        for i in range(0, len(raw) - _RAN_HEADER, _RAN_HEADER)
    )
    return all(n.decode("latin-1").rstrip("\x00 ").isprintable() for n in names)


def read_ran(source, name=None):
    """Read an AGICO ``.RAN`` file, the binary record of a Kappabridge session.

    The file has a 64-byte header and one 64-byte record per measured specimen
    (little-endian). The layout is not published; it was decoded from example
    files and checked against the ``.ASC`` reports and the ``.AEX`` tables of the
    same measurements (all values agree to the digits of the reports).

    Header: an ``int16`` (the number of records plus 2), the locality (30
    characters), a second text field (16 characters, rock or comment) and the
    orientation parameters as text (``12 90 6  0``, P1-P4 of AGICO).

    Record: the specimen name (12 characters, e.g. ``MEL 1/1/1``), two
    ``float32`` of which the second is the mean susceptibility (the norming
    factor, units of 1e-6 SI) and the first repeats it in files of the SUSAR program
    but holds the field in A/m in those of SUFAR 1.x (compared with 34 reports of
    both programs), the normed tensor ``k11 k22 k33 k12 k23 k13`` (6 x ``float32``) in **geographic**
    coordinates, the tag of the first foliation (``S0``, ``K0``, ``00`` for none),
    its dip direction (or strike, see P4), dip, and the trend and plunge of
    the lineation (4 x ``int16``), and the second tag with 4 ``int16`` (unused).

    The core azimuth and dip are not stored in the file (the ``.GED`` file of
    Anisoft holds them, it is not read here), so the specimens are in the
    ``geographic`` frame: ``tensor_in("specimen")`` is not available, the
    tectonic coordinates are (the foliation is the bedding, as the ``Paleo 1``
    system of the reports). Meta: ``op`` (the orientation codes), ``locality``,
    ``rock``, ``norming_factor`` (SI), ``field`` (A/m, None for SUSAR files),
    ``foliation_tag``, ``foliation`` and ``lineation`` (degrees).

    Args:
        source: file name, open binary file or bytes.
        name: site name. Default the file name.

    Returns:
        AMSSite: the specimens of the file, in the order of the file.

    Raises:
        ValueError: if the content does not have the structure of a RAN file.
    """
    raw, stem = _read_raw(source)
    if isinstance(raw, str):
        raise TypeError("RAN files are binary, open the file in 'rb' mode")
    if len(raw) < 2 * _RAN_HEADER:
        raise ValueError("not a RAN file: shorter than a header and one record")
    head = raw[:_RAN_HEADER]

    def field(a, b):
        return head[a:b].decode("latin-1").replace("\x00", " ").strip()

    locality, rock = field(2, 32), field(32, 48)
    digits = re.findall(r"\d+", field(48, 62))
    codes = tuple(int(d) for d in digits) if len(digits) == 4 else None
    if codes is None:
        warnings.warn(
            "no orientation parameters P1-P4 in the header of the RAN file, "
            "the foliation is read as a dip direction (P4 = 0)",
            stacklevel=2,
        )
    p4 = codes[3] if codes and codes[3] in (0, 90) else 0
    n, extra = divmod(len(raw) - _RAN_HEADER, _RAN_HEADER)
    if extra:
        warnings.warn(
            f"{extra} trailing bytes of the RAN file are not a record and were ignored",
            stacklevel=2,
        )
    site = AMSSite(name=name or stem or "Default")
    odd = []
    for i in range(n):
        vals = _RAN_RECORD.unpack_from(raw, _RAN_HEADER + i * _RAN_HEADER)
        nm, mean, norm, k, tag = vals[0], vals[1], vals[2], vals[3:9], vals[9]
        f_az, f_dip, l_az, l_pl = vals[10:14]
        sp = nm.decode("latin-1").rstrip("\x00 ")
        if not sp or not sp.isprintable() or not np.all(np.isfinite([mean, *k])):
            raise ValueError(
                f"record {i + 1}: not a RAN file (no name or finite values)"
            )
        if abs(k[0] + k[1] + k[2] - 3) > 0.1:
            odd.append(sp)
        tag = tag.decode("latin-1").rstrip("\x00 ")
        bedding = None
        if tag != "00" and f_dip > 0:
            bedding = ((f_az + p4) % 360, float(f_dip))
        site.append(
            AMSSpecimen(
                name=sp,
                tensor=tensor_from_components(*k),
                bulk=float(norm) * 1e-6,
                azimuth=None,
                hade=None,
                bedding=bedding,
                site=site.name,
                frame="geographic",
                meta={
                    "op": codes,
                    "locality": locality,
                    "rock": rock,
                    "norming_factor": float(norm) * 1e-6,
                    "field": None
                    if abs(mean - norm) <= 1e-6 * abs(norm)
                    else float(mean),
                    "foliation_tag": tag,
                    "foliation": (float(f_az), float(f_dip)),
                    "lineation": (float(l_az), float(l_pl)),
                },
            )
        )
    if odd:
        warnings.warn(
            f"the normed tensor of {', '.join(odd[:3])}"
            f"{' and others' if len(odd) > 3 else ''} does not have trace 3, "
            "is this the layout of a RAN file?",
            stacklevel=2,
        )
    if not len(site):
        raise ValueError("no records found")
    return site


def read_aex(source, names=None):
    """Read an Anisoft ``.AEX`` export: a table of parameters per measurement.

    The first line names the columns, e.g. ``Km P T`` (mean susceptibility in
    SI, the anisotropy degree and the shape parameter of Jelinek) or ``Spec Name
    Km K1 K2 K3 L F P P' T U dK1geo iK1geo ...`` (the principal susceptibilities and
    the directions of the axes in geographic coordinates as well). Each following
    line has the values of one measurement in the order of the report. Exports
    with the column ``Spec Name`` (or ``Name``, ``Specimen``) start each line with
    the specimen name, which may contain blanks; they become the index. The file
    holds no tensors, so the result is a table, not an ``AMSSite``; compare it
    with ``AMSSite.table()`` or ``AMSSpecimen.params()``.

    Args:
        source: file name or open file.
        names: optional specimen names (one per row) for the index, replacing
            the names of the file.

    Returns:
        pandas.DataFrame: one row per measurement, the numeric columns as named
        in the file. Rows that are short or not numeric are skipped with a warning.
    """
    text, _ = _text(source)
    rows = [ln for ln in text.splitlines() if ln.strip()]
    if not rows:
        raise ValueError("empty AEX file")
    header = rows[0].split()
    if any(_float(h) is not None for h in header):
        raise ValueError("the first line of an AEX file must name the columns")
    lead = [h.lower() for h in header[:2]]
    n_name = 2 if lead == ["spec", "name"] else int(lead[0] in ("name", "specimen"))
    columns = header[n_name:]
    data, index, dropped = [], [], []
    for n, ln in enumerate(rows[1:], start=2):
        parts = ln.split()
        values = [_float(x) for x in parts[len(parts) - len(columns) :]]
        # the name (if any) takes the leading items, so there are more than `columns`
        enough = len(parts) > len(columns) if n_name else len(parts) == len(columns)
        if not enough or None in values:
            dropped.append(n)
            continue
        data.append(values)
        index.append(" ".join(parts[: len(parts) - len(columns)]))
    if dropped:
        shown = ", ".join(map(str, dropped[:3])) + (", ..." if len(dropped) > 3 else "")
        warnings.warn(
            f"{len(dropped)} AEX rows are short or not numeric and were skipped "
            f"(rows {shown})",
            stacklevel=2,
        )
    df = pd.DataFrame(data, columns=columns)
    if names is not None:
        if len(names) != len(df):
            raise ValueError(f"{len(names)} names for {len(df)} rows")
        df.index = list(names)
    elif n_name:
        df.index = pd.Index(index, name="Name")
    return df


def _float(text):
    try:
        return float(text)
    except ValueError:
        return None


def read_ams(source, name=None, **kwargs):
    """Read an Anisoft/AGICO file, whichever of the layouts it has.

    A report (``.asc``, see `read_sufar`) is recognised by its title, a binary
    ``.ran`` file (any case of the extension, or the structure of one) is read
    by `read_ran`, anything else is read as a table (`read_anisoft_table`, to
    which `kwargs` go). ``.aex`` tables hold no tensors, use `read_aex`; binary
    ``.ams`` files must be exported to text in Anisoft.
    """
    raw, stem = _read_raw(source)
    ext = "" if hasattr(source, "read") else Path(str(source)).suffix.lower()
    if isinstance(raw, bytes) and (
        ext == ".ran" or (b"\x00" in raw[:2048] and _looks_like_ran(raw))
    ):
        if kwargs:
            raise TypeError("a RAN file takes no table options")
        return read_ran(raw, name=name or stem)
    if ext == ".aex":
        raise ValueError("an AEX file holds parameters, not tensors; use read_aex")
    text, stem = _decode(raw, stem)
    buf = io.StringIO(text)
    buf.name = stem or ""
    if "ANISOTROPY OF SUSCEPTIBILITY" in text[:400]:
        if kwargs:
            raise TypeError("a SUFAR report takes no table options")
        return read_sufar(buf, name=name)
    return read_anisoft_table(buf, name=name, **kwargs)


read_asc = read_sufar
