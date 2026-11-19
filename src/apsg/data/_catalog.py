"""Registry of the bundled datasets."""

from dataclasses import dataclass

from apsg.data import _loaders as L


@dataclass(frozen=True)
class Dataset:
    """Description of a bundled dataset.

    Attributes:
        name: name used by ``apsg.data.load``.
        kind: ``vectors``, ``lines``, ``planes``, ``faults``, ``circular``,
            ``grouped``, ``table``, ``tensors``, ``stress``, ``groups`` (a dict of
            several of them) or ``paleomag``.
        description: what the data are.
        source: citation, or how synthetic data were made.
        n: number of items (total over groups).
        note: remarks on the data, e.g. corrections of a printed table.
        file: name of the file in the package; ``None`` for entries stored in
            the JSON catalogs.
    """

    name: str
    kind: str
    description: str
    source: str
    n: int
    note: str = ""
    file: str | None = None

    def __str__(self):
        out = f"{self.name} [{self.kind}, n = {self.n}]: {self.description}\n"
        out += f"  Source: {self.source}"
        return out + (f"\n  Note: {self.note}" if self.note else "")


_FILES = {
    "mele": Dataset(
        "mele",
        "faults",
        "Fault-slip data: fault plane, slickenline and sense of movement.",
        "tutorial data of APSG",
        94,
        file="mele.csv",
    ),
    "structures": Dataset(
        "structures",
        "table",
        "Lineation and foliation measurements of several sites, one row per "
        "measurement (columns site, structure, azi, inc).",
        "tutorial data of APSG",
        253,
        file="structures.csv",
    ),
    "structures_wide": Dataset(
        "structures_wide",
        "table",
        "The same kind of data in wide format: one row per measurement with the "
        "columns of S1, L1, S2 and L2 (empty where not measured).",
        "tutorial data of APSG",
        132,
        file="structures_wide.csv",
    ),
    "pmdata": Dataset(
        "pmdata",
        "paleomag",
        "71 thermally demagnetized drill-core specimens of 10 sites (dict of "
        "``Sample`` by site, or one ``Sample`` with ``site=``).",
        "artificial, seeded random generator: normal and reversed ChRM acquired "
        "before tilting, soft overprint, some great-circle and poorly behaved specimens",
        71,
        "Names, locations and numbers are invented.",
        file="pmdata.zip",
    ),
    "pmdata_sites": Dataset(
        "pmdata_sites",
        "table",
        "Site table of ``pmdata``: location of the hypothetical study, bedding and "
        "number of specimens.",
        "artificial, seeded random generator",
        10,
        file="pmdata.zip",
    ),
}

_FILE_LOADERS = {
    "mele": L.load_mele,
    "structures": L.load_structures,
    "structures_wide": L.load_structures_wide,
    "pmdata": L.load_pmdata,
    "pmdata_sites": L.load_pmdata_sites,
}


def _stored():
    out = {}
    for fname in ("published.json", "synthetic.json"):
        for name, e in L.read_json(fname).items():
            n = L.count(e["kind"], e["data"])
            out[name] = Dataset(
                name, e["kind"], e["description"], e["source"], n, e["note"]
            )
    return out


def registry():
    """All datasets by name: files first, then published, then synthetic."""
    return {**_FILES, **_stored()}


def loader(name):
    """Function building the object of dataset ``name``."""
    if name in _FILE_LOADERS:
        return _FILE_LOADERS[name]
    for fname in ("published.json", "synthetic.json"):
        entries = L.read_json(fname)
        if name in entries:
            e = entries[name]

            def load(e=e, **kw):
                if kw:
                    raise TypeError(f"{name!r} takes no arguments, got {sorted(kw)}")
                return L.build(e["kind"], e["data"], name)

            return load
    raise KeyError(name)
