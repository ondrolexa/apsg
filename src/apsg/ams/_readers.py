"""Tables of AMS tensors: ``DataFrame`` and CSV readers and writers."""

import io
import math
import re
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


def _text(source):
    """Return ``(text, name)`` of a path or an open text or binary file."""
    if hasattr(source, "read"):
        raw = source.read()
        name = Path(getattr(source, "name", "") or "").stem or None
    else:
        raw = Path(source).read_bytes()
        name = Path(source).stem
    if isinstance(raw, str):
        return raw.replace("\r", ""), name
    if b"\x00" in raw[:2048]:
        raise ValueError(
            "binary AMS file (Anisoft .ams/.ran); export it to text in Anisoft "
            "(report .asc or table .txt) and read that"
        )
    for enc in ("utf-8-sig", "latin-1"):
        try:
            return raw.decode(enc).replace("\r", ""), name
        except UnicodeDecodeError:
            continue
    raise ValueError("cannot decode the file")  # pragma: no cover


def _numbers(line):
    return [float(x) for x in re.findall(r"[-+]?\d*\.?\d+(?:[Ee][-+]?\d+)?", line)]


def read_sufar(source, name=None):
    """Read an AGICO SUFAR/Anisoft report (``.asc``) of one site.

    Each page of the report holds one specimen: name, core ``Azi`` and ``Dip``,
    mean susceptibility, the standard errors of the normed principal
    susceptibilities and the normed tensor in specimen coordinates. The
    tensor, core orientation (``azimuth = Azi``, ``hade = -Dip``, checked
    against the geographic tensor printed on each page) and the mean
    susceptibility (``bulk``, in the unit of the file) are kept; the
    ``std_error`` is the mean of the printed standard errors. Pages cut off
    before the tensor are skipped with a warning.

    Args:
        source: file name or open file.
        name: site name. Default the file name.

    Returns:
        AMSSite: the specimens of the file.
    """
    text, stem = _text(source)
    site = AMSSite(name=name or stem or "Default")
    skipped = []
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
            k = next(k for k, ln in enumerate(lines) if "susc." in ln)
            bulk = next(
                _numbers(ln)[1]
                for ln in lines[k:]
                if re.match(r"\s+\d+\s+[\d.]+E[-+]\d+", ln)
            )
            errs = [
                float(x)
                for ln in lines
                if ln.startswith("+-")
                for x in _numbers(ln[2:])[:3]
            ]
        except (AttributeError, StopIteration, ValueError, IndexError):
            skipped.append(sp)
            continue
        site.append(
            AMSSpecimen(
                name=sp,
                tensor=tensor_from_components(*diag, off[0], off[1], off[2]),
                bulk=bulk,
                std_error=float(np.mean(errs)) if errs else None,
                azimuth=azi,
                hade=-dip,
                site=site.name,
            )
        )
    if skipped:
        warnings.warn(
            f"incomplete specimen pages skipped: {', '.join(skipped)}", stacklevel=2
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
    for ln in rows[1:]:
        parts = ln.split()
        if len(parts) < len(header):
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
            continue
        specimens.append(
            AMSSpecimen(parts[iname], T, bulk=bulk, site=site_name, frame=frame)
        )
    if not specimens:
        raise ValueError("no specimens found")
    result = AMSSite(specimens, name=site_name)
    return result.split(site) if site is not None else result


def read_ams(source, name=None, **kwargs):
    """Read an Anisoft/SUFAR text file, whichever of the two layouts it has.

    A report (``.asc``, see `read_sufar`) is recognised by its title, anything
    else is read as a table (`read_anisoft_table`, to which `kwargs` go).
    Binary ``.ams`` and ``.ran`` files must be exported to text in Anisoft.
    """
    text, stem = _text(source)
    buf = io.StringIO(text)
    buf.name = stem or ""
    if "ANISOTROPY OF SUSCEPTIBILITY" in text[:400]:
        if kwargs:
            raise TypeError("a SUFAR report takes no table options")
        return read_sufar(buf, name=name)
    return read_anisoft_table(buf, name=name, **kwargs)
