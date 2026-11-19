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
    "faults": Dataset(
        "faults",
        "faults",
        "Fault-slip data: fault plane, slickenline and sense of movement.",
        "tutorial data of APSG",
        94,
        file="faults.csv",
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
    "melechov": Dataset(
        "melechov",
        "ams",
        "AMS of 51 measured specimens of 5 sites (MEL1-MEL5) of the Melechov massif "
        '(dict of ``AMSSite`` by site, read from the SUSAR reports; ``format="ran"`` '
        "reads the binary RAN files of the same measurements, ``site=`` one site).",
        "author's unpublished data: measured on an AGICO Kappabridge, reports of "
        "SUSAR, RAN files and AEX table exported from Anisoft",
        51,
        "Core orientation P1-P4 = 12 90 6 0 (checked against the geographic and "
        "`Paleo 1` tensors of the reports). The mean of the 51 geographic tensors is "
        "reproduced by the AS1 statistics file of Anisoft: eigenvalues 1.027, 1.004, "
        "0.969 and directions 253.3/30.2, 9.4/37.1, 136.1/38.2 (D/I of k1, k2, k3). "
        "The RAN files hold the geographic tensors only (no core orientation).",
        file="melechov.zip",
    ),
    "melechov_aex": Dataset(
        "melechov_aex",
        "table",
        "Anisoft AEX export of ``melechov``: mean susceptibility Km (SI), anisotropy "
        "degree P and shape parameter T of every measurement, indexed by specimen name.",
        "author's unpublished data: measured on an AGICO Kappabridge, exported from Anisoft",
        51,
        file="melechov.zip",
    ),
    "trebic": Dataset(
        "trebic",
        "ams",
        "AMS of 17 measured specimens of two sites (E236-2, TR1), as dict of "
        "``AMSSite`` by site, read from the ASC reports (E236-2 in the layout of SUFAR "
        '1.1 with a field column, TR1 of SUSAR with the code 12 90 12 0); ``format="ran"`` '
        "reads the RAN files of the same measurements, ``site=`` one site.",
        "author's unpublished data: measured on an AGICO Kappabridge, reports of "
        "SUFAR 1.1 and SUSAR, RAN files and AEX tables exported from Anisoft",
        17,
        "The RAN files of SUFAR hold the field (A/m) in their first value, those of "
        "SUSAR repeat the mean susceptibility; the tensors of the reports and RAN files "
        "agree to the digits of the reports, and the AEX tables give the principal "
        "susceptibilities and directions of every specimen.",
        file="trebic.zip",
    ),
    "trebic_aex": Dataset(
        "trebic_aex",
        "table",
        "Anisoft AEX tables of ``trebic``: specimen name, mean susceptibility Km, the "
        "principal susceptibilities K1-K3, L, F, P, P', T, U and the directions of the "
        "axes in geographic coordinates (the table names differ from those of the reports).",
        "author's unpublished data: measured on an AGICO Kappabridge, exported from Anisoft",
        17,
        file="trebic.zip",
    ),
    "ams_table": Dataset(
        "ams_table",
        "ams",
        "Table exported by Anisoft with the tensor components k11 ... k13 of three "
        'specimens (geographic coordinates; ``frame="specimen"`` reads them as '
        "specimen coordinates).",
        "author's unpublished data: AMS measured on an AGICO Kappabridge, exported "
        "from Anisoft",
        3,
        file="ams_table.txt",
    ),
    "ams_table_no_tensor": Dataset(
        "ams_table_no_tensor",
        "ams",
        "Anisoft export without tensor components: six specimens of two sites "
        "(GC2, GC4) with the principal directions and L, F, from which the "
        "normed tensors are rebuilt.",
        "author's unpublished data: AMS measured on an AGICO Kappabridge, exported "
        "from Anisoft",
        6,
        file="ams_table_no_tensor.txt",
    ),
}

_FILE_LOADERS = {
    "faults": L.load_faults,
    "structures": L.load_structures,
    "structures_wide": L.load_structures_wide,
    "pmdata": L.load_pmdata,
    "pmdata_sites": L.load_pmdata_sites,
    "melechov": L.load_melechov,
    "melechov_aex": L.load_melechov_aex,
    "trebic": L.load_trebic,
    "trebic_aex": L.load_trebic_aex,
    "ams_table": L.load_ams_table,
    "ams_table_no_tensor": L.load_ams_table_no_tensor,
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
