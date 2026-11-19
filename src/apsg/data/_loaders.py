"""Readers that turn the stored data into apsg objects."""

import io
import json
import re
import tempfile
import zipfile
from functools import cache
from importlib import resources
from pathlib import Path

import numpy as np
import pandas as pd

from apsg.feature import (
    Ellipsoid,
    EllipsoidSet,
    FaultSet,
    FoliationSet,
    LineationSet,
    Stress3,
    Stress3Set,
    Vector2Set,
    Vector3Set,
)

FILES = resources.files("apsg.data") / "files"


def file(name):
    """Path of a bundled file."""
    return Path(str(FILES / name))


@cache
def read_json(name):
    """Content of a bundled JSON file (do not modify the result)."""
    return json.loads((FILES / name).read_text(encoding="utf-8"))


def _vectors(d, name):
    return Vector3Set.from_array(d["azi"], d["inc"], name=name)


def _lines(d, name):
    return LineationSet.from_array(d["azi"], d["inc"], name=name)


def _planes(d, name):
    return FoliationSet.from_array(d["azi"], d["inc"], name=name)


def _faults(d, name):
    return FaultSet.from_array(
        d["fazi"], d["finc"], d["lazi"], d["linc"], d["sense"], name=name
    )


def _circular(d, name):
    return Vector2Set.from_directions(d["angle"], name=name)


def _grouped(d, name):
    start = np.asarray(d["start"], float)
    return pd.DataFrame(
        {
            "start": start,
            "center": start + d["width"] / 2,
            "count": np.asarray(d["count"], int),
        }
    )


def _table(d, name):
    return pd.DataFrame(d["columns"])


def _tensors(d, name):
    return EllipsoidSet([Ellipsoid(m) for m in d["matrices"]], name=name)


def _stress(d, name):
    return Stress3Set([Stress3(m) for m in d["matrices"]], name=name)


def _ams(d, name):
    from apsg.ams import AMSSite, AMSSpecimen

    sites = {}
    for sp in d["specimens"]:
        s = AMSSpecimen.from_json(sp)
        sites.setdefault(s.site, AMSSite(name=s.site)).append(s)
    return sites


KINDS = {
    "ams": _ams,
    "vectors": _vectors,
    "lines": _lines,
    "planes": _planes,
    "faults": _faults,
    "circular": _circular,
    "grouped": _grouped,
    "table": _table,
    "tensors": _tensors,
    "stress": _stress,
}


def build(kind, data, name):
    """Create the object of a stored entry; ``groups`` give a dict of objects."""
    if "groups" in data:
        return {
            g: build(sub["kind"], {k: v for k, v in sub.items() if k != "kind"}, g)
            for g, sub in data["groups"].items()
        }
    return KINDS[kind](data, name)


def count(kind, data):
    """Number of items of a stored entry (per group for ``groups``)."""
    if "groups" in data:
        return sum(count(s["kind"], s) for s in data["groups"].values())
    if kind == "ams":
        return len(data["specimens"])
    if kind in ("tensors", "stress"):
        return len(data["matrices"])
    if kind == "table":
        return len(next(iter(data["columns"].values())))
    if kind == "grouped":
        return int(sum(data["count"]))
    first = data.get("angle", data.get("azi", data.get("fazi")))
    return len(first)


# -- files -------------------------------------------------------------------
def load_faults(**kw):
    return FaultSet.from_csv(file("faults.csv"), name=kw.pop("name", "faults"), **kw)


def load_structures(**kw):
    return pd.read_csv(file("structures.csv"), **kw)


def load_structures_wide(**kw):
    kw.setdefault("dtype_backend", "numpy_nullable")
    return pd.read_csv(file("structures_wide.csv"), **kw)


def _pmdata_names():
    with zipfile.ZipFile(file("pmdata.zip")) as z:
        return z.namelist()


def load_pmdata_sites(**kw):
    with zipfile.ZipFile(file("pmdata.zip")) as z, z.open("sites.csv") as f:
        return pd.read_csv(f, index_col="site", **kw)


def load_pmdata(site=None, **kw):
    """Specimens of the artificial paleomagnetic study, grouped by site."""
    from apsg.paleomag import Sample, Specimen  # not imported by `import apsg`

    names = sorted(n for n in _pmdata_names() if n.endswith(".pmd"))
    sites = sorted({n.split(".")[0] for n in names})
    if site is not None and site not in sites:
        raise KeyError(f"unknown site {site!r}, available: {sites}")
    out = {}
    with zipfile.ZipFile(file("pmdata.zip")) as z:
        for s in [site] if site else sites:
            specs = []
            for n in (n for n in names if n.split(".")[0] == s):
                with z.open(n) as f:
                    specs.append(Specimen.from_pmd(io.TextIOWrapper(f, "latin-1")))
            out[s] = Sample(specs, name=s)
    return out[site] if site else out


def pmdata_folder():
    """Folder with the PMD files of ``pmdata``, extracted once into a cache."""
    from apsg import __version__

    folder = Path(tempfile.gettempdir()) / f"apsg-data-{__version__}" / "pmdata"
    names = _pmdata_names()
    if not all((folder / n).exists() for n in names):
        folder.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(file("pmdata.zip")) as z:
            z.extractall(folder)
    return folder


def _natural(text):
    """Sort key putting ``MEL2`` before ``MEL10``."""
    return [int(s) if s.isdigit() else s for s in re.split(r"(\d+)", text)]


def _ams_zip_names(zipname):
    with zipfile.ZipFile(file(zipname)) as z:
        return z.namelist()


def _ams_zip_sites(zipname):
    names = [n for n in _ams_zip_names(zipname) if n.endswith(".ASC")]
    return sorted((n[:-4] for n in names), key=_natural)


def _load_ams_zip(zipname, format="asc", site=None):
    """AMS specimens of the sites of an archive of ASC reports and RAN files."""
    from apsg.ams import read_ran, read_sufar  # not imported by `import apsg`

    readers = {"asc": ("ASC", read_sufar), "ran": ("RAN", read_ran)}
    if format not in readers:
        raise ValueError(f"format must be 'asc' or 'ran', not {format!r}")
    sites = _ams_zip_sites(zipname)
    if site is not None and site not in sites:
        raise KeyError(f"unknown site {site!r}, available: {sites}")
    ext, reader = readers[format]
    out = {}
    with zipfile.ZipFile(file(zipname)) as z:
        for s in [site] if site else sites:
            out[s] = reader(z.read(f"{s}.{ext}"), name=s)
    return out[site] if site else out


def _load_ams_aex(zipname, report_names):
    """The AEX tables of an archive, one after the other in the order of the sites.

    Tables without specimen names are indexed by the names of the ASC reports."""
    import pandas as pd

    from apsg.ams import read_aex

    with zipfile.ZipFile(file(zipname)) as z:
        files = sorted((n for n in z.namelist() if n.endswith(".AEX")), key=_natural)
        df = pd.concat([read_aex(z.read(n)) for n in files], ignore_index=report_names)
    if report_names:
        df.index = [s.name for site in _load_ams_zip(zipname).values() for s in site]
    return df


def load_melechov(format="asc", site=None, **kw):
    """AMS specimens of five sites, from the ASC reports or the RAN files."""
    return _load_ams_zip("melechov.zip", format, site)


def load_melechov_aex(**kw):
    """The ``Km P T`` export of the Melechov specimens, indexed by specimen name."""
    return _load_ams_aex("melechov.zip", True)


def load_trebic(format="asc", site=None, **kw):
    """AMS specimens of two sites, from the ASC reports (SUFAR 1.1, SUSAR) or RAN files."""
    return _load_ams_zip("trebic.zip", format, site)


def load_trebic_aex(**kw):
    """The AEX exports of the two sites (with specimen names and principal values)."""
    return _load_ams_aex("trebic.zip", False)


def _folder(name, zipname):
    from apsg import __version__

    folder = Path(tempfile.gettempdir()) / f"apsg-data-{__version__}" / name
    names = _ams_zip_names(zipname)
    if not all((folder / n).exists() for n in names):
        folder.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(file(zipname)) as z:
            z.extractall(folder)
    return folder


def melechov_folder():
    """Folder with the RAN, ASC and AEX files of ``melechov``, extracted once."""
    return _folder("melechov", "melechov.zip")


def trebic_folder():
    """Folder with the RAN, ASC and AEX files of ``trebic``, extracted once."""
    return _folder("trebic", "trebic.zip")


def load_ams_table(**kw):
    """Anisoft table with the tensor components (geographic coordinates)."""
    from apsg.ams import read_anisoft_table  # not imported by `import apsg`

    name = kw.pop("name", "ams_table")
    return {name: read_anisoft_table(file("ams_table.txt"), name=name, **kw)}


def load_ams_table_no_tensor(**kw):
    """Tensor-less Anisoft export: tensors rebuilt from directions with L and F."""
    from apsg.ams import read_anisoft_table

    kw.setdefault("site", lambda n: n.split("_")[0])
    return read_anisoft_table(file("ams_table_no_tensor.txt"), **kw)
