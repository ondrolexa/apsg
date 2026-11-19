"""Readers that turn the stored data into apsg objects."""

import io
import json
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
def load_mele(**kw):
    return FaultSet.from_csv(file("mele.csv"), name=kw.pop("name", "mele"), **kw)


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
