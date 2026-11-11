"""Paleomagnetic demagnetization data: ``Specimen`` and ``Sample``.

A ``Specimen`` holds the ordered demagnetization steps of one specimen (one
demagnetization type, thermal or alternating field) together with its
orientation, and reads the PMD, Remasoft RS3 and SQUID formats. A ``Sample``
collects specimens and provides the statistics of their interpreted
components.

Orientation convention: the canonical core orientation is the PMD one. The
specimen z axis has trend ``azimuth`` and inclination ``90 - hade``, the x axis
lies in the vertical plane through it, and the rotation from specimen to
geographic coordinates is ``getRotationMatrix(azimuth, 90 - hade)`` of PMTools.
Bedding is stored as a ``Vector3``, the normal of the bed pointing to its
stratigraphic top (upward for upright, downward for overturned beds).
"""

import re
import warnings
from dataclasses import dataclass, field, replace
from datetime import datetime
from pathlib import Path

import numpy as np

from apsg.feature._container import FoliationSet, Vector3Set
from apsg.feature._distributions._base import as_rng
from apsg.feature._geodata import Foliation, Lineation
from apsg.feature._hypothesis import common_mean_test
from apsg.feature._tensor3 import OrientationTensor3, Rotation3
from apsg.helpers._helper import eformat
from apsg.helpers._math import acosd, asind, atan2d, atand, cosd, sind, tand
from apsg.math._vector import Vector3

__all__ = (
    "Component",
    "DemagStep",
    "Sample",
    "Specimen",
    "lines_planes_mean",
    "vgp",
)

_COORDS = ("specimen", "geo", "tilt")
_DEMAG = (None, "thermal", "af")
_UP = (0.0, 0.0, -1.0)
_NUM = r"[-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?"
_ENUM = r"[-+]?\d*\.?\d+[eE][-+]?\d+"
_PMD_HEAD = re.compile(rf"\b([abdsv])=\s*({_NUM})")
_PMD_DATE = re.compile(r"(\d{1,2}-\d{1,2}-\d{4}\s+\d{1,2}:\d{2})")
_PMD_ROW = re.compile(
    rf"^(?P<label>.*?)\s*(?P<x>{_ENUM})\s*(?P<y>{_ENUM})\s*(?P<z>{_ENUM})"
    rf"\s*(?P<mag>{_ENUM})(?P<rest>.*)$"
)
# PMD headers written by PaleoMac label azimuth and hade with CP437 alpha and
# beta (bytes 0xE0, 0xE1), which arrive as "à", "á" when decoded as latin-1
_PMD_GREEK = str.maketrans({"à": "a", "á": "b", "α": "a", "β": "b", "ß": "b"})


# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------


def core_rotation(azimuth, hade):
    """Rotation from specimen to geographic coordinates.

    The columns are the specimen axes in geographic coordinates (x north,
    y east, z down): z has trend `azimuth` and inclination ``90 - hade``, x lies
    in the vertical plane through z and y is horizontal. This is
    ``getRotationMatrix(azimuth, 90 - hade)`` of PMTools, the PMD convention.
    The matrix is built directly, as `hade` may be negative or above 90.
    """
    sa, ca, sb, cb = sind(azimuth), cosd(azimuth), sind(hade), cosd(hade)
    return Rotation3([[ca * cb, -sa, ca * sb], [sa * cb, ca, sa * sb], [-sb, 0.0, cb]])


def bedding_vector(dipdir, dip):
    """Return the bedding normal pointing to the stratigraphic top.

    Args:
        dipdir: dip direction of the bed in degrees.
        dip: dip in degrees, 0-180; a dip above 90 is an overturned bed.
    """
    return Vector3(dipdir, dip - 90)


def untilt_rotation(bedding, frac=1.0):
    """Rotation restoring `bedding` (way-up normal) to horizontal.

    The rotation turns the bedding normal to the upward vertical about the
    strike (``bedding x up``); ``frac`` scales the angle (partial untilting,
    used by the fold test).
    """
    bedding = Vector3(bedding)
    up = Vector3(_UP)
    angle = bedding.angle(up)
    if angle < 1e-9:
        return Rotation3(np.eye(3))
    axis = bedding.cross(up)
    if abs(axis) < 1e-12:  # upside-down horizontal bed, strike undefined
        axis = Vector3(1, 0, 0)
    return Rotation3.from_axisangle(axis, frac * angle)


def _angles(A, B):
    """Angles in degrees between corresponding rows of two arrays of vectors."""
    A = np.asarray(A, dtype=float)
    B = np.asarray(B, dtype=float)
    na = np.linalg.norm(A, axis=1)
    nb = np.linalg.norm(B, axis=1)
    ok = (na > 0) & (nb > 0)
    c = np.sum(A[ok] * B[ok], axis=1) / (na[ok] * nb[ok])
    return np.degrees(np.arccos(np.clip(c, -1, 1)))


def _dir(azi, inc):
    """Unit vector array of declination/inclination (any inclination)."""
    return np.array([cosd(azi) * cosd(inc), sind(azi) * cosd(inc), sind(inc)])


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass(frozen=True, eq=False)
class DemagStep:
    """One demagnetization step.

    Attributes:
        label: step label as in the source file (``"NRM"``, ``"T350"``,
            ``"M010"``, ...).
        level: treatment level, degrees C (thermal) or mT (AF), 0 for NRM.
        M: magnetization in specimen coordinates, A/m (``Vector3``).
        a95: measurement precision angle in degrees (0 when not given).
        k: bulk susceptibility or None.
        comment: free text.
    """

    label: str
    level: float
    M: Vector3
    a95: float = 0.0
    k: float | None = None
    comment: str = ""

    def to_json(self):
        """Return as JSON dict."""
        return {
            "label": self.label,
            "level": self.level,
            "M": list(self.M),
            "a95": self.a95,
            "k": self.k,
            "comment": self.comment,
        }

    @classmethod
    def from_json(cls, d):
        """Return ``DemagStep`` from a dict of `to_json`."""
        return cls(**{**d, "M": Vector3(d["M"])})


@dataclass(frozen=True, eq=False)
class Component:
    """A magnetization component fitted by principal component analysis.

    Attributes:
        kind: ``"line"`` or ``"plane"``.
        vector: line direction (pointing along the removed magnetization) or
            plane pole (``Vector3``, unit length).
        mad: maximum angular deviation in degrees (Kirschvink, 1980). A
            geometric precision index of the fit, not a confidence limit. A
            free plane need not pass through the origin; an anchored plane of
            unit vectors is the great circle used by `lines_planes_mean`.
        dang: deviation angle between the line and the centre of mass of the
            data (Tauxe & Staudigel, 2004), None for planes and anchored fits.
        n: number of steps used.
        labels: labels of the steps used.
        anchor: True when the fit was forced through the origin.
        coords: coordinate system of `vector`.
    """

    kind: str
    vector: Vector3
    mad: float
    dang: float | None
    n: int
    labels: tuple
    anchor: bool = False
    coords: str = "specimen"

    def __repr__(self):
        azi, inc = self.vector.geo
        return (
            f"Component({self.kind} {azi:.1f}/{inc:.1f} MAD={self.mad:.1f} "
            f"n={self.n} {self.labels[0]}-{self.labels[-1]} {self.coords})"
        )

    def transform(self, R, coords):
        """Return the component rotated by `R` into `coords`."""
        return replace(self, vector=self.vector.transform(R), coords=coords)

    @property
    def lin(self):
        """The line direction as ``Lineation``."""
        return Lineation(self.vector)

    @property
    def fol(self):
        """The plane (pole `vector`) as ``Foliation``."""
        return Foliation(self.vector)

    def to_json(self):
        """Return as JSON dict."""
        d = {k: getattr(self, k) for k in self.__dataclass_fields__}
        return {**d, "vector": list(self.vector), "labels": list(self.labels)}

    @classmethod
    def from_json(cls, d):
        """Return ``Component`` from a dict of `to_json`."""
        return cls(
            **{**d, "vector": Vector3(d["vector"]), "labels": tuple(d["labels"])}
        )


# ---------------------------------------------------------------------------
# Parsing helpers
# ---------------------------------------------------------------------------


def _float(text, default=None):
    try:
        return float(text)
    except (TypeError, ValueError):
        return default


def _level(label):
    """Treatment level parsed from a step label (NRM -> 0)."""
    if label.strip().upper().startswith("NRM"):
        return 0.0
    match = re.search(r"\d+(?:\.\d+)?", label)
    return float(match.group()) if match else 0.0


def _demag_from_labels(labels):
    """Guess the demagnetization type from step labels."""
    for label in labels:
        s = label.strip().upper()
        if s.startswith("NRM") or not s:
            continue
        if "°C" in label or s.startswith("T"):
            return "thermal"
        if s.startswith(("M", "AF", "A")) or "MT" in s:
            return "af"
    return None


def _read_lines(filename):
    with open(filename, encoding="latin-1") as f:
        text = f.read()
    return text.split("\x1a")[0].splitlines()


# ---------------------------------------------------------------------------
# Specimen
# ---------------------------------------------------------------------------


@dataclass(eq=False)
class Specimen:
    """Demagnetization data of one specimen.

    Args:
        name: specimen name.
        steps: list of ``DemagStep`` in measurement order.
        demag: ``"thermal"``, ``"af"`` or None (unknown).
        azimuth: core azimuth, PMD convention (see `core_rotation`).
        hade: core hade, PMD convention.
        bedding: bedding normal pointing to the stratigraphic top
            (``Vector3``; see `bedding_vector`). Default horizontal.
        foldaxis: fold axis (``Lineation``) or None; stored only.
        volume: specimen volume in m3 or None.
        site: site name.
        latitude: site latitude in degrees or None.
        longitude: site longitude in degrees or None.
        height: site elevation or None.
        rock: rock type.
        age: age.
        formation: formation.
        date: date of measurement or None.
        filename: source file or None.
        components: fitted components by name, in specimen coordinates (see
            `interpret`).

    Examples:
        >>> s = Specimen.from_file("K583B-15.rs3")  # doctest: +SKIP
        >>> s.interpret("ChRM", slice(3, 12))  # doctest: +SKIP
    """

    name: str = "Default"
    steps: list = field(default_factory=list)
    demag: str | None = None
    azimuth: float = 0.0
    hade: float = 0.0
    bedding: Vector3 = field(default_factory=lambda: Vector3(_UP))
    foldaxis: Lineation | None = None
    volume: float | None = None
    site: str = ""
    latitude: float | None = None
    longitude: float | None = None
    height: float | None = None
    rock: str = ""
    age: str = ""
    formation: str = ""
    date: datetime | None = None
    filename: str | None = None
    components: dict = field(default_factory=dict)

    def __post_init__(self):
        self.steps = list(self.steps)
        if not all(isinstance(s, DemagStep) for s in self.steps):
            raise TypeError("steps must be DemagStep instances")
        if self.demag not in _DEMAG:
            raise ValueError(f"demag must be one of {_DEMAG}, got {self.demag!r}")
        self.bedding = Vector3(self.bedding)
        if abs(self.bedding) == 0:
            raise ValueError("bedding must be a non-zero vector")

    def __repr__(self):
        demag = f" {self.demag}" if self.demag else ""
        site = f"{self.site}/" if self.site else ""
        return f"Specimen {site}{self.name}: {len(self)} steps{demag}"

    def __len__(self):
        return len(self.steps)

    def __iter__(self):
        return iter(self.steps)

    def __getitem__(self, key):
        """Step by position or label, or a new ``Specimen`` for a selection
        (slice, sequence of positions or labels, boolean mask)."""
        if isinstance(key, (int, np.integer)):
            return self.steps[key]
        if isinstance(key, str):
            for step in self.steps:
                if step.label == key:
                    return step
            raise KeyError(f"Step {key!r} not found in {self.name}")
        if isinstance(key, slice):
            return self._subset(self.steps[key])
        key = list(key)
        if key and all(isinstance(k, str) for k in key):
            return self._subset([self[k] for k in key])
        arr = np.asarray(key)
        if arr.dtype == bool:
            if len(arr) != len(self):
                raise IndexError("Boolean mask must have one value per step")
            return self._subset([s for s, ok in zip(self.steps, arr) if ok])
        return self._subset([self.steps[int(i)] for i in arr])

    def _subset(self, steps):
        return replace(self, steps=list(steps), components=dict(self.components))

    def between(self, lo=None, hi=None):
        """Return the steps with ``lo <= level <= hi`` as a new ``Specimen``."""
        lv = self.levels
        ok = np.ones(len(lv), dtype=bool)
        if lo is not None:
            ok &= lv >= lo
        if hi is not None:
            ok &= lv <= hi
        return self[ok]

    # -- per-step arrays ---------------------------------------------------

    @property
    def labels(self):
        """List of step labels."""
        return [s.label for s in self.steps]

    @property
    def levels(self):
        """Treatment levels as numpy array."""
        return np.array([s.level for s in self.steps], dtype=float)

    @property
    def intensity(self):
        """Magnetization intensities in A/m as numpy array."""
        return np.array([abs(s.M) for s in self.steps], dtype=float)

    @property
    def moment(self):
        """Magnetic moments in Am2 as numpy array (needs `volume`)."""
        if self.volume is None:
            raise ValueError(f"Volume of specimen {self.name} is unknown")
        return self.intensity * self.volume

    # -- coordinates -------------------------------------------------------

    def rotation(self, coords="geo"):
        """Return ``Rotation3`` from specimen coordinates to `coords`.

        Args:
            coords: ``"specimen"``, ``"geo"`` (geographic, in situ) or
                ``"tilt"`` (tilt corrected, stratigraphic).
        """
        if coords not in _COORDS:
            raise ValueError(f"coords must be one of {_COORDS}, got {coords!r}")
        if coords == "specimen":
            return Rotation3(np.eye(3))
        R = core_rotation(self.azimuth, self.hade)
        if coords == "tilt":
            R = untilt_rotation(self.bedding) @ R
        return R

    def vectors(self, coords="geo"):
        """Return magnetization vectors (A/m) in `coords` as ``Vector3Set``."""
        g = Vector3Set([s.M for s in self.steps], name=self.name)
        if coords == "specimen":
            return g
        return g.transform(self.rotation(coords))

    @property
    def V(self):
        """Magnetization vectors in specimen coordinates (``Vector3Set``)."""
        return self.vectors("specimen")

    @property
    def geo(self):
        """Magnetization vectors in geographic coordinates (``Vector3Set``)."""
        return self.vectors("geo")

    @property
    def tilt(self):
        """Magnetization vectors in tilt-corrected coordinates (``Vector3Set``)."""
        return self.vectors("tilt")

    # -- processing --------------------------------------------------------

    def pca(self, steps=None, *, kind="line", anchor=False, coords="geo"):
        """Principal component analysis of demagnetization data.

        The best-fit line (``kind="line"``) or plane (``kind="plane"``) of the
        selected steps (Kirschvink, 1980; Tauxe, 2003, §3.8, eq. 3.4-3.10). For a
        free fit the points are centred on their centre of mass; an anchored
        fit (``anchor=True``) passes through the origin. The magnetization
        vectors are used unnormalized. The line points along the removed
        magnetization (first minus last step; along the data for anchored
        fits), the plane pole along ``first x last``, so the sense of the
        demagnetization path is kept.

        Args:
            steps: selection of steps (slice, positions, labels or boolean
                mask); None for all.
            kind: ``"line"`` or ``"plane"``.
            anchor: force the fit through the origin. Default False.
            coords: coordinate system of the result.

        Returns:
            Component: the fitted component.

        References:
            Kirschvink, J.L. (1980). The least-squares line and plane and the
                analysis of palaeomagnetic data. Geophysical Journal of the
                Royal Astronomical Society 62, 699-718.
        """
        if kind not in ("line", "plane"):
            raise ValueError(f"kind must be 'line' or 'plane', got {kind!r}")
        sub = self if steps is None else self[steps]
        if isinstance(sub, DemagStep):
            raise TypeError("pca needs a selection of steps, not a single step")
        n_min = (2 if kind == "line" else 3) - int(anchor)
        if len(sub) < n_min:
            raise ValueError(f"{kind} fit needs at least {n_min} steps")
        X = np.asarray(sub.V, dtype=float)
        center = X.mean(axis=0)
        Y = X if anchor else X - center
        T = OrientationTensor3(Y.T @ Y / len(Y))
        if kind == "line":
            v = np.asarray(T.V1, dtype=float)
            ref = center if anchor or len(X) < 2 else X[0] - X[-1]
            if np.dot(v, ref) < 0:
                v = -v
            mad = T.MAD_l
            dang = None
            if not anchor and np.linalg.norm(center) > 0:
                dang = float(_angles([v], [center])[0])
        else:
            v = np.asarray(T.V3, dtype=float)
            ref = np.cross(X[0], X[-1])
            if np.dot(v, ref) < 0:
                v = -v
            mad = T.MAD_p
            dang = None
        comp = Component(
            kind, Vector3(v), float(mad), dang, len(sub), tuple(sub.labels), anchor
        )
        if coords == "specimen":
            return comp
        return comp.transform(self.rotation(coords), coords)

    def interpret(self, name, steps=None, *, kind="line", anchor=False):
        """Fit a component by `pca` and store it under `name`.

        Returns:
            Component: the stored component in geographic coordinates.
        """
        comp = self.pca(steps, kind=kind, anchor=anchor, coords="specimen")
        self.components[name] = comp
        return comp.transform(self.rotation("geo"), "geo")

    def component(self, name, coords="geo"):
        """Return the stored component `name` in `coords`."""
        return self.components[name].transform(self.rotation(coords), coords)

    @property
    def vds(self):
        """Vector difference sum, the total magnetization removed along the
        demagnetization path plus the last remaining vector (Gee et al.,
        1993)."""
        X = np.asarray(self.V, dtype=float)
        return float(
            np.linalg.norm(np.diff(X, axis=0), axis=1).sum() + np.linalg.norm(X[-1])
        )

    def mdf(self, vds=False):
        """Median destructive field (or temperature).

        The level at which the remaining magnetization falls to half of the
        initial value, interpolated linearly between steps; the intensity
        relative to the first step or, with ``vds=True``, the remaining part of
        the vector difference sum (Gee et al., 1993). None when never reached.
        """
        X = np.asarray(self.V, dtype=float)
        if len(X) < 2:
            return None
        if vds:
            d = np.linalg.norm(np.diff(X, axis=0), axis=1)
            rest = np.append(np.cumsum(d[::-1])[::-1], 0.0) + np.linalg.norm(X[-1])
            f = rest / rest[0]
        else:
            f = np.linalg.norm(X, axis=1) / np.linalg.norm(X[0])
        lv = self.levels
        below = np.flatnonzero(f <= 0.5)
        if len(below) == 0:
            return None
        i = below[0]
        if i == 0:
            return float(lv[0])
        return float(np.interp(0.5, [f[i], f[i - 1]], [lv[i], lv[i - 1]]))

    # -- conversion --------------------------------------------------------

    def to_dataframe(self):
        """Steps as ``pandas.DataFrame`` with directions in all coordinates."""
        import pandas as pd

        data = {"label": self.labels, "level": self.levels, "M": self.intensity}
        for coords, suffix in zip(_COORDS, ("sp", "ge", "tc")):
            dirs = np.array([v.geo for v in self.vectors(coords)]).reshape(-1, 2)
            data[f"D{suffix}"] = dirs[:, 0]
            data[f"I{suffix}"] = dirs[:, 1]
        data["a95"] = [s.a95 for s in self.steps]
        data["comment"] = [s.comment for s in self.steps]
        return pd.DataFrame(data)

    _META = (
        "name", "demag", "azimuth", "hade", "volume", "site", "latitude",
        "longitude", "height", "rock", "age", "formation", "filename",
    )  # fmt: skip

    def to_json(self):
        """Return as JSON dict."""
        kwargs = {k: getattr(self, k) for k in self._META}
        kwargs["steps"] = [s.to_json() for s in self.steps]
        kwargs["bedding"] = list(self.bedding)
        kwargs["foldaxis"] = None if self.foldaxis is None else list(self.foldaxis)
        kwargs["date"] = None if self.date is None else self.date.isoformat()
        kwargs["components"] = {k: c.to_json() for k, c in self.components.items()}
        return {"datatype": type(self).__name__, "kwargs": kwargs}

    @classmethod
    def from_json(cls, obj_json):
        """Return ``Specimen`` from a dict of `to_json`."""
        kw = dict(obj_json["kwargs"])
        kw["steps"] = [DemagStep.from_json(s) for s in kw["steps"]]
        kw["bedding"] = Vector3(kw["bedding"])
        if kw.get("foldaxis") is not None:
            kw["foldaxis"] = Lineation(kw["foldaxis"])
        if kw.get("date") is not None:
            kw["date"] = datetime.fromisoformat(kw["date"])
        kw["components"] = {
            k: Component.from_json(c) for k, c in kw.get("components", {}).items()
        }
        return cls(**kw)

    # -- readers -----------------------------------------------------------

    @classmethod
    def from_file(cls, filename, **kwargs):
        """Read a ``.pmd``, ``.rs3`` or ``.squid`` file by its extension."""
        ext = Path(filename).suffix.lower()
        readers = {".pmd": cls.from_pmd, ".rs3": cls.from_rs3, ".squid": cls.from_squid}
        if ext not in readers:
            raise ValueError(f"Unsupported file type {ext!r}, expected {list(readers)}")
        return readers[ext](filename, **kwargs)

    @classmethod
    def from_pmd(cls, filename, demag=None):
        """Read a PMD file (Enkin's PaleoMac format).

        Line 2 holds the name and the orientation ``a``, ``b`` (core azimuth and
        hade, written as ``α``, ``β`` by PaleoMac), ``s``, ``d`` (bedding) and ``v`` (volume, m3), the data rows the
        step label, the moment components ``Xc Yc Zc`` (Am2), the intensity
        and the geographic and tilt-corrected directions. Rows are parsed by
        their number format, not by fixed columns, so labels such as
        ``150°C`` running into the next column are read too.

        Two dialects exist: standard PMD (``s`` is the strike, dip to the
        right) and files exported by Remasoft, which store the raw
        ``SDec``/``BDec`` of RS3 (core azimuth turned by 180 degrees, ``s`` the
        dip direction). The dialect reproducing the ``Dg Ig Ds Is`` columns
        of the file is used; a warning is issued when neither does.

        Args:
            filename: path of the file.
            demag: ``"thermal"`` or ``"af"``; guessed from the labels if None.
        """
        lines = [ln.translate(_PMD_GREEK) for ln in _read_lines(filename)]
        ih = next((i for i, ln in enumerate(lines) if "a=" in ln and "b=" in ln), None)
        if ih is None:
            raise ValueError(f"{filename}: no orientation line (a=, b=, ...) found")
        head = lines[ih]
        o = {k: float(v) for k, v in _PMD_HEAD.findall(head)}
        missing = {"a", "b", "s", "d", "v"} - set(o)
        if missing:
            raise ValueError(
                f"{filename}: missing {sorted(missing)} in orientation line"
            )
        name = head[: head.index("a=")].strip()
        title = lines[ih - 1].strip() if ih > 0 else ""
        site = title.split("/")[0].strip() if "/" in title else ""
        dm = _PMD_DATE.search(head)
        date = None
        if dm:
            try:
                date = datetime.strptime(
                    " ".join(dm.group(1).split()), "%m-%d-%Y %H:%M"
                )
            except ValueError:
                date = None
        steps, dg, ds = [], [], []
        for ln in lines[ih + 2 :]:
            m = _PMD_ROW.match(ln)
            if not m:
                continue
            label = m["label"].strip()
            xyz = np.array([float(m["x"]), float(m["y"]), float(m["z"])])
            rest = m["rest"].split(None, 5)
            vals = [_float(t) for t in rest[:5]]
            nvals = next((i for i, v in enumerate(vals) if v is None), len(vals))
            vals = vals[:nvals] + [None] * (5 - nvals)
            comment = " ".join(rest[nvals:]).strip()
            steps.append(
                DemagStep(
                    label,
                    _level(label),
                    Vector3(xyz / o["v"]),
                    a95=vals[4] if vals[4] is not None else 0.0,
                    comment=comment,
                )
            )
            dg.append(None if None in vals[0:2] else _dir(vals[0], vals[1]))
            ds.append(None if None in vals[2:4] else _dir(vals[2], vals[3]))
        if not steps:
            raise ValueError(f"{filename}: no data rows found")
        dialects = {
            "pmd": (o["a"], o["b"], bedding_vector(o["s"] + 90, o["d"])),
            "remasoft": (o["a"] + 180, o["b"], bedding_vector(o["s"], o["d"])),
        }
        best, misfit = None, np.inf
        for key, (a, b, bed) in dialects.items():
            spec = cls(steps=steps, azimuth=a, hade=b, bedding=bed)
            err = []
            for computed, stored in ((spec.geo, dg), (spec.tilt, ds)):
                pairs = [(c, s) for c, s in zip(computed, stored) if s is not None]
                if pairs:
                    err.append(np.median(_angles(*zip(*pairs))))
            score = float(np.sum(err)) if err else 0.0
            if score < misfit:
                best, misfit = key, score
        if misfit > 2:
            warnings.warn(
                f"{filename}: orientation does not reproduce the Dg/Ig and Ds/Is "
                f"columns (misfit {misfit:.1f} deg); read as {best!r} PMD",
                stacklevel=2,
            )
        a, b, bed = dialects[best]
        return cls(
            name=name,
            steps=steps,
            demag=demag or _demag_from_labels([s.label for s in steps]),
            azimuth=a % 360,
            hade=b,
            bedding=bed,
            volume=o["v"],
            site=site,
            date=date,
            filename=str(filename),
        )

    @classmethod
    def from_rs3(cls, filename, exclude=("C", "G"), demag=None):
        """Read a Remasoft RS3 file (AGICO).

        Fixed-width columns. The specimen line holds name, site, location,
        rock, age, formation, the sampling orientation ``SDec SInc``, bedding
        ``BDec BInc`` (dip direction, dip), fold axis ``FDec FInc`` and the
        orientation parameters ``P1-P4``. The data rows hold the step, the
        intensity (A/m) and the direction in specimen coordinates.

        The orientation parameters follow AGICO (2009), as described by
        Sagnotti (2013): P1 is the clock position of the reference arrow
        (specimen x axis) on the frontal base of the core, P2 is 0 when
        ``SInc`` is the dip of the frontal base (the hade of the core axis)
        and 90 when it is the plunge of the core axis, P3 is the clock
        position of the azimuth ``SDec`` measured in the field and P4 is 0
        when ``BDec`` is the dip direction of the bedding and 90 when it is
        the strike (right-hand rule). Supported are P1 and P3 of 12 or 6,
        P2 = 0 and P4 of 0 or 90: P3 = 6 turns ``SDec`` by 180 degrees and
        P1 = 6 turns the specimen by 180 degrees about the core axis. The
        codes 12 0 6 0, 6 0 6 0 and 12 0 12 90 were checked against the
        ``Dge Ige Dtc Itc`` columns of Remasoft files.

        Args:
            filename: path of the file.
            exclude: row IDs to skip, by default the component (``C``) and
                great circle (``G``) fits of Remasoft.
            demag: ``"thermal"`` or ``"af"``; from the step units or row IDs
                if None.
        """
        lines = [ln for ln in _read_lines(filename) if ln.strip()]
        if len(lines) < 3:
            raise ValueError(f"{filename}: not an RS3 file")
        h = lines[1]

        def num(a, b):
            return _float(h[a:b].strip())

        def txt(a, b):
            return h[a:b].strip()

        sdec, sinc = num(74, 80) or 0.0, num(80, 86) or 0.0
        bdec, binc = num(86, 92), num(92, 98)
        fdec, finc = num(98, 104), num(104, 110)
        p1, p2, p3, p4 = (int(num(a, a + 3) or 0) for a in (110, 113, 116, 119))
        if p1 not in (12, 6) or p2 != 0 or p3 not in (12, 6) or p4 not in (0, 90):
            raise NotImplementedError(
                f"{filename}: RS3 orientation parameters P1={p1}, P2={p2}, "
                f"P3={p3}, P4={p4} are not supported (supported: P1 and P3 "
                "12 or 6, P2 0, P4 0 or 90)"
            )
        azimuth = sdec + (180.0 if p3 == 6 else 0.0) + (180.0 if p1 == 6 else 0.0)
        hade = -sinc if p1 == 6 else sinc
        units = lines[2][3:14]
        if demag is None:
            if "mT" in units:
                demag = "af"
            elif "C" in units:
                demag = "thermal"
        steps = []
        for r in lines[3:]:
            rid = r[0:2].strip()
            if rid in exclude:
                continue
            step = r[3:13].strip()
            M = _float(r[13:27].strip())
            dsp, isp = _float(r[27:33].strip()), _float(r[33:39].strip())
            if M is None or dsp is None or isp is None:
                continue
            if rid == "N":
                label = "NRM"
            elif rid in ("M", "A"):
                label = f"M{step}"
            else:
                label = f"{rid}{step}"
            steps.append(
                DemagStep(
                    label,
                    _float(step, _level(label)),
                    Vector3(M * _dir(dsp, isp)),
                    a95=_float(r[75:81].strip(), 0.0),
                    k=_float(r[81:95].strip()),
                    comment=r[115:].strip(),
                )
            )
        if demag is None:
            demag = _demag_from_labels([s.label for s in steps])
        bedding = Vector3(_UP)
        if bdec is not None and binc is not None:
            bedding = bedding_vector(bdec + p4, binc)
        return cls(
            name=txt(0, 10),
            steps=steps,
            demag=demag,
            azimuth=azimuth % 360,
            hade=hade,
            bedding=bedding,
            foldaxis=Lineation(fdec, finc) if None not in (fdec, finc) else None,
            site=txt(10, 20),
            latitude=num(20, 30),
            longitude=num(30, 41),
            height=num(41, 51),
            rock=txt(51, 66),
            age=txt(66, 71),
            formation=txt(71, 74),
            filename=str(filename),
        )

    @classmethod
    def from_squid(cls, filename, demag=None):
        """Read a SQUID magnetometer file.

        Line 1 holds site and specimen name, line 2 the core azimuth and hade,
        bedding strike and dip and the volume (cm3). The azimuth is turned by
        -90 degrees to the PMD convention, AF fields are converted from Oe to
        mT (labelled ``M%03d``) and everything from the first ``ARM`` row on
        is dropped, following PMTools. Intensities are converted to A/m
        (x 1000) and the specimen-coordinate directions are used.

        Args:
            filename: path of the file.
            demag: ``"thermal"`` or ``"af"``; from the row codes if None.
        """
        lines = [ln for ln in _read_lines(filename) if ln.strip()]
        if len(lines) < 3:
            raise ValueError(f"{filename}: not a SQUID file")
        title = lines[0].split()
        name = title[-1] if title else ""
        site = " ".join(title[:-1])
        head = [float(t) for t in lines[1].split()[:5]]
        a, b, s, d, v = head
        steps = []
        for ln in lines[2:]:
            if ln.startswith("ARM"):
                break
            code, value = ln[:2].strip(), ln[2:6].strip()
            tokens = ln[6:].split()
            M = float(tokens[4]) * 1e3
            dc, ic = float(tokens[6]), float(tokens[7])
            if code == "AF":
                level = float(value) / 10 if value else 0.0
                label = f"M{round(level):03d}"
            elif code.startswith("N"):
                level, label = 0.0, "NRM"
            else:
                level = _float(value, 0.0)
                label = f"{code}{level:g}"
            steps.append(DemagStep(label, level, Vector3(M * _dir(dc, ic))))
        if demag is None:
            demag = _demag_from_labels([st.label for st in steps])
        return cls(
            name=name,
            steps=steps,
            demag=demag,
            azimuth=(a - 90) % 360,
            hade=b,
            bedding=bedding_vector(s + 90, d),
            volume=v * 1e-6,
            site=site,
            filename=str(filename),
        )

    # -- writers -----------------------------------------------------------

    def _bedding_dd(self):
        """Dip direction and dip (0-180) of the bedding."""
        azi, inc = self.bedding.geo
        return azi % 360, inc + 90

    def to_pmd(self, filename):
        """Write the specimen as a standard PMD file.

        Without a known volume ``v=1`` is written, so the moment columns hold
        the magnetization.
        """
        v = self.volume or 1.0
        dd, dip = self._bedding_dd()
        date = (self.date or datetime.now()).strftime("%m-%d-%Y %H:%M")
        rows = [
            f"{self.site}/{self.name}" if self.site else self.name,
            (
                f"{self.name[:8]:<8}  a={self.azimuth % 360:5.1f}   b={self.hade:5.1f}"
                f"   s={(dd - 90) % 360:5.1f}   d={dip:5.1f}   v={eformat(v, 2)}m3  {date}"
            ),
            "STEP  Xc [Am2]  Yc [Am2]  Zc [Am2]  MAG[A/m]   Dg    Ig    Ds    Is   a95 ",
        ]
        geo, tilt = self.geo, self.tilt
        for step, g, t in zip(self.steps, geo, tilt):
            x, y, z = np.asarray(step.M, dtype=float) * v
            rows.append(
                f"{step.label:<4}{x:10.2E}{y:10.2E}{z:10.2E}{abs(step.M):10.2E}"
                f"{g.geo[0]:6.1f}{g.geo[1]:6.1f}{t.geo[0]:6.1f}{t.geo[1]:6.1f}"
                f"{step.a95:5.1f} {step.comment}"
            )
        with open(filename, "w", encoding="latin-1", errors="replace", newline="") as f:
            f.write("\r\n".join(rows) + "\r\n\x1a")

    def to_rs3(self, filename):
        """Write the specimen as a Remasoft RS3 file (orientation parameters
        12 0 6 0)."""
        dd, dip = self._bedding_dd()
        unit = "mT" if self.demag == "af" else "°C"
        prefix = "M" if self.demag == "af" else "T"

        def f(value, fmt):
            return "" if value is None else format(value, fmt)

        fold = ("", "")
        if self.foldaxis is not None:
            fold = (f"{self.foldaxis.geo[0]:.0f}", f"{self.foldaxis.geo[1]:.0f}")
        rows = [
            (
                "Name      Site      Latitude  Longitude  Height    Rock           Age  "
                "Fm SDec  SInc  BDec  BInc  FDec  FInc  P1 P2 P3 P4 Note"
            ),
            (
                f"{self.name[:9]:<10}{self.site[:9]:<10}{f(self.latitude, 'g'):<10}"
                f"{f(self.longitude, 'g'):<11}{f(self.height, 'g'):<10}"
                f"{self.rock[:14]:<15}{self.age[:4]:<5}{self.formation[:2]:<3}"
                f"{(self.azimuth - 180) % 360:<6.1f}{self.hade:<6.1f}{dd:<6.1f}{dip:<6.1f}"
                f"{fold[0]:<6}{fold[1]:<6}12 0  6  0      "
            ),
            (
                f"ID {'Step[' + unit + ']':<10}       M[A/m]   Dsp   Isp   Dge   Ige   Dtc"
                "   Itc   Dfc   Ifc   Prec    K[e-06 SI] Limit1    Limit2    Note      "
            ),
        ]
        for step, sp, g, t in zip(self.steps, self.V, self.geo, self.tilt):
            rid = "N" if step.label.upper().startswith("NRM") else prefix
            k = "" if step.k is None else f"{step.k:g}"
            rows.append(
                f"{rid:<2} {step.level:<10g}{abs(step.M):>14.7g}"
                f"{sp.geo[0]:6.1f}{sp.geo[1]:6.1f}{g.geo[0]:6.1f}{g.geo[1]:6.1f}"
                f"{t.geo[0]:6.1f}{t.geo[1]:6.1f}{'':12}{step.a95:6.1f}{k:>14}"
                f"{'':20} {step.comment}"
            )
        with open(
            filename, "w", encoding="latin-1", errors="replace", newline=""
        ) as fh:
            fh.write("\r\n".join(rows) + "\r\n")


# ---------------------------------------------------------------------------
# Collection statistics
# ---------------------------------------------------------------------------


def lines_planes_mean(lines, poles, level=0.95, tol=1e-10, max_iter=1000):
    """Mean direction of directions combined with great circles.

    McFadden & McElhinny (1988), after Tauxe (2003, §4.4, Ex. 4.3): each
    great circle contributes the point on it nearest to the current mean,
    iterated to convergence. The precision is ``k = (2M + N - 2) / (2(M + N -
    R))`` and the confidence cone uses ``N' = M + N/2`` in Fisher's formula,
    ``cos(alpha) = 1 - (N' - 1) / (k R) ((1 - level)^(-1/(N' - 1)) - 1)``.
    With planes only, the start is their common intersection (oriented
    downward).

    Note:
        The method assumes the circles are scattered about the sought
        direction. Circles from two populations with similar precisions and
        means not exactly antiparallel converge on a direction biased by
        about half their angular separation (Schmidt, 1985); direct
        observations of the sought component reduce the bias. McFadden &
        McElhinny note that `k` tends to be too large and the cone slightly
        too small (coverage a bit below `level`). No arc constraints are
        applied, so their analyses A and C are reproduced, not B and D.

    Args:
        lines: ``(M, 3)`` unit vectors (directions).
        poles: ``(N, 3)`` unit poles of the great circles.
        level: confidence level. Default 0.95.

    Returns:
        dict: ``mu`` (``Vector3``), ``k``, ``alpha``, ``R``, ``n_lines``,
            ``n_planes`` and ``points`` (the fitted points on the circles).

    References:
        McFadden, P.L. & McElhinny, M.W. (1988). The combined analysis of
            remagnetization circles and direct observations in
            palaeomagnetism. Earth and Planetary Science Letters 87, 161-172.
        Schmidt, P.W. (1985). Bias in converging great circle methods.
            Earth and Planetary Science Letters 72, 427-432.
    """
    L = np.asarray(lines, dtype=float).reshape(-1, 3)
    P = np.asarray(poles, dtype=float).reshape(-1, 3)
    L = L / np.linalg.norm(L, axis=1, keepdims=True) if len(L) else L
    P = P / np.linalg.norm(P, axis=1, keepdims=True)
    M, N = len(L), len(P)
    if M + N < 2:
        raise ValueError("At least two directions or great circles are needed")
    if M:
        m = L.sum(axis=0)
    else:
        if N < 2:
            raise ValueError("At least two great circles are needed without lines")
        warnings.warn(
            "Great circles without direct observations give a biased mean when "
            "the circles come from two populations of similar precision "
            "(Schmidt, 1985); add direct observations of the sought component.",
            UserWarning,
            stacklevel=2,
        )
        m = np.linalg.eigh(P.T @ P)[1][:, 0]
        m = m if m[2] >= 0 else -m
    m = m / np.linalg.norm(m)
    for _ in range(max_iter):
        G = m - (P @ m)[:, None] * P
        G /= np.linalg.norm(G, axis=1, keepdims=True)
        s = L.sum(axis=0) + G.sum(axis=0)
        m_new = s / np.linalg.norm(s)
        done = np.linalg.norm(m_new - m) < tol
        m = m_new
        if done:
            break
    R = float(np.linalg.norm(s))
    k = (2 * M + N - 2) / (2 * (M + N - R)) if M + N - R > 0 else np.inf
    Np = M + N / 2
    if Np > 1 and np.isfinite(k):
        cos_a = 1 - (Np - 1) / (k * R) * ((1 - level) ** (-1 / (Np - 1)) - 1)
        alpha = float(acosd(np.clip(cos_a, -1, 1)))
    else:
        alpha = 0.0 if np.isinf(k) else 180.0
    return {
        "mu": Vector3(m),
        "k": float(k),
        "alpha": alpha,
        "R": R,
        "n_lines": M,
        "n_planes": N,
        "points": [Vector3(g) for g in G],
    }


def _orient(v, lines):
    """Orient an axis along the mean of `lines`, or downward without lines."""
    v = np.asarray(v, dtype=float)
    ref = lines.sum(axis=0) if len(lines) else np.array([0.0, 0.0, 1.0])
    return v if v @ ref >= 0 else -v


def _unit_rows(a):
    a = np.asarray(a, dtype=float).reshape(-1, 3)
    return a / np.linalg.norm(a, axis=1, keepdims=True) if len(a) else a


def kirschvink_mean(lines, poles, method=1):
    """Mean direction of directions and great circles by Kirschvink's methods.

    Kirschvink (1980, §6): with ``H_L`` and ``H_P`` the sums of ``u u^T`` over
    the lines and the plane poles, the common direction is the eigenvector of
    the smallest eigenvalue of ``H_P - H_L`` (method 1). Method 2 replaces each
    line by two orthogonal directions of weight 1/2, which equals method 1 with
    the line matrix weighted by 1/2. Neither method gives an error region; use
    `lines_planes_mean` or `bailey_halls_mean` for that. The directions are
    treated as axial and the result is oriented along the mean of the lines
    (downward when there are none).

    Args:
        lines: ``(M, 3)`` directions.
        poles: ``(N, 3)`` poles of the great circles.
        method: 1 or 2. Default 1.

    Returns:
        dict: ``mu`` (``Vector3``), ``eigenvalues`` (ascending, of the matrix
            whose smallest eigenvector is ``mu``), ``n_lines``, ``n_planes``.

    References:
        Kirschvink, J.L. (1980). The least-squares line and plane and the
            analysis of palaeomagnetic data. Geophysical Journal of the
            Royal Astronomical Society 62, 699-718.
    """
    if method not in (1, 2):
        raise ValueError(f"method must be 1 or 2, got {method!r}")
    L, P = _unit_rows(lines), _unit_rows(poles)
    if len(L) + len(P) < 2:
        raise ValueError("At least two directions or great circles are needed")
    w = 1.0 if method == 1 else 0.5
    H = P.T @ P - w * (L.T @ L)
    val, vec = np.linalg.eigh(H)
    return {
        "mu": Vector3(_orient(vec[:, 0], L)),
        "eigenvalues": val,
        "n_lines": len(L),
        "n_planes": len(P),
    }


def bailey_halls_mean(lines, poles, level=0.95):
    """Maximum likelihood mean of directions and great circles with an ellipse.

    Bailey & Halls (1984): directions are Fisher distributed about the true
    direction ``t`` and the pole of each circle has density
    ``k I0(k sin(phi)) / (4 pi sinh(k))`` about it. The estimate maximises
    ``r.t - t.H.t / 2`` with ``r`` the resultant of the lines and
    ``H = sum p p^T`` of the poles, i.e. ``(H - omega I) t = r`` with ``omega``
    the smallest root of the secular equation (eq. 23). For circles only,
    ``t`` is the smallest eigenvector of ``H`` and ``omega`` its eigenvalue.
    The precision is ``k = S / (2M + N - 2)`` with ``S = 2M - r.t + omega``
    (the printed (29) uses the length of ``r``, which is wrong for ``N > 0``).
    The confidence region is the ellipse ``e'(H - omega I)e <= C^2``,
    ``C^2 = S F / (M + N/2 - 1)`` with ``F`` the quantile of
    ``F(2, 2M + N - 2)`` (eqs. 32, 37-39), computed in the tangent plane at
    ``t``.

    Args:
        lines: ``(M, 3)`` directions.
        poles: ``(N, 3)`` poles of the great circles (``N >= 1``).
        level: confidence level. Default 0.95.

    Returns:
        dict: ``mu`` (``Vector3``), ``k``, ``omega``, ``semi_axes`` (degrees,
            major first), ``axes`` (two ``Vector3`` along them), ``n_lines``
            and ``n_planes``.

    References:
        Bailey, R.C. & Halls, H.C. (1984). Estimate of confidence in
            palaeomagnetic directions derived from mixed remagnetization
            circle and direct observational data. Journal of Geophysics 54,
            174-182.
    """
    from scipy import optimize, stats

    L, P = _unit_rows(lines), _unit_rows(poles)
    M, N = len(L), len(P)
    if N < 1:
        raise ValueError("At least one great circle is needed")
    if M + N < 2:
        raise ValueError("At least two directions or great circles are needed")
    H = P.T @ P
    lam, E = np.linalg.eigh(H)
    r = L.sum(axis=0)
    b = E.T @ r
    if M == 0 or abs(b[0]) < 1e-12 * max(np.linalg.norm(r), 1.0):
        # circles only (or r orthogonal to the minimum eigenvector)
        t, omega = E[:, 0], lam[0]
        if M:
            t = t if t @ r >= 0 else -t
    else:
        f = lambda w: np.sum(b**2 / (lam - w) ** 2) - 1.0
        hi = lam[0] - 1e-12 * max(1.0, abs(lam[0]))
        lo = lam[0] - np.linalg.norm(r) - 1.0
        omega = optimize.brentq(f, lo, hi, xtol=1e-14)
        t = E @ (b / (lam - omega))
        t /= np.linalg.norm(t)
    t = _orient(t, L)
    S = 2 * M - r @ t + omega
    df = 2 * M + N - 2
    k = df / S if S > 0 else np.inf
    # tangent plane at t
    a = np.cross(t, E[:, 2] if abs(t @ E[:, 2]) < 0.9 else E[:, 1])
    a /= np.linalg.norm(a)
    B = np.column_stack([a, np.cross(t, a)])
    A = B.T @ (H - omega * np.eye(3)) @ B
    mu_, W = np.linalg.eigh(A)
    Fq = stats.f.ppf(level, 2, df)
    C2 = S * Fq / (M + N / 2 - 1)
    semi = np.degrees(np.sqrt(C2 / np.maximum(mu_, 1e-300)))
    order = np.argsort(semi)[::-1]
    return {
        "mu": Vector3(t),
        "k": float(k),
        "omega": float(omega),
        "semi_axes": tuple(float(semi[i]) for i in order),
        "axes": tuple(Vector3(B @ W[:, i]) for i in order),
        "n_lines": M,
        "n_planes": N,
    }


def vgp(dec, inc, lat, lon, a95=None):
    """Virtual geomagnetic pole of a mean direction at a site.

    For a geocentric axial dipole ``tan(I) = 2 cot(p)`` gives the magnetic
    colatitude ``p`` and the pole is at angular distance ``p`` from the site
    along the declination (Butler, 1992, ch. 7; Tauxe, 2003, §1.3.4).
    The confidence oval has semi-axes ``dp = a95 (1 + 3 cos^2 p) / 2`` and
    ``dm = a95 sin p / cos I``.

    Args:
        dec: mean declination in degrees.
        inc: mean inclination in degrees.
        lat: site latitude in degrees.
        lon: site longitude in degrees.
        a95: confidence cone of the mean direction, degrees (optional).

    Returns:
        dict: ``lat``, ``lon`` (pole), ``paleolatitude`` (site), and ``dp``,
            ``dm`` when `a95` is given.

    References:
        Butler, R.F. (1992). Paleomagnetism: Magnetic Domains to Geologic
            Terranes. Blackwell, Boston.
    """
    p = atan2d(2 * cosd(inc), sind(inc))
    plat = asind(np.clip(sind(lat) * cosd(p) + cosd(lat) * sind(p) * cosd(dec), -1, 1))
    cos_plat = cosd(plat)
    if cos_plat < 1e-12:
        beta = 0.0
    else:
        beta = asind(np.clip(sind(p) * sind(dec) / cos_plat, -1, 1))
    if cosd(p) >= sind(lat) * sind(plat):
        plon = lon + beta
    else:
        plon = lon + 180 - beta
    res = {
        "lat": float(plat),
        "lon": float(plon % 360),
        "paleolatitude": float(atand(tand(inc) / 2)),
    }
    if a95 is not None:
        res["dp"] = float(a95 * (1 + 3 * cosd(p) ** 2) / 2)
        res["dm"] = float(a95 * sind(p) / cosd(inc))
    return res


class Sample:
    """Collection of ``Specimen`` objects and statistics of their components.

    Components are fitted per specimen by ``Specimen.interpret`` and referred
    to here by name; specimens without the component are skipped.

    Args:
        specimens: iterable of ``Specimen``.
        name: sample (or site) name.

    Examples:
        >>> smp = Sample.from_files(glob("*.rs3"), name="GC083")  # doctest: +SKIP
        >>> for s in smp:  # doctest: +SKIP
        ...     s.interpret("ChRM", slice(5, None))
        >>> smp.mean("ChRM", coords="tilt")  # doctest: +SKIP
    """

    def __init__(self, specimens=(), name="Default"):
        self.specimens = list(specimens)
        if not all(isinstance(s, Specimen) for s in self.specimens):
            raise TypeError("Sample holds Specimen instances only")
        self.name = name

    @classmethod
    def from_files(cls, filenames, name="Default", **kwargs):
        """Read specimens from files (see ``Specimen.from_file``)."""
        return cls([Specimen.from_file(f, **kwargs) for f in filenames], name=name)

    def __repr__(self):
        return f"Sample {self.name}: {len(self)} specimens"

    def __len__(self):
        return len(self.specimens)

    def __iter__(self):
        return iter(self.specimens)

    def __getitem__(self, key):
        if isinstance(key, str):
            for s in self.specimens:
                if s.name == key:
                    return s
            raise KeyError(f"Specimen {key!r} not found in {self.name}")
        return self.specimens[key]

    def append(self, specimen):
        """Add a ``Specimen``."""
        if not isinstance(specimen, Specimen):
            raise TypeError("Sample holds Specimen instances only")
        self.specimens.append(specimen)

    @property
    def names(self):
        """List of specimen names."""
        return [s.name for s in self.specimens]

    def to_json(self):
        """Return as JSON dict."""
        return {
            "datatype": type(self).__name__,
            "kwargs": {"name": self.name, "specimens": [s.to_json() for s in self]},
        }

    @classmethod
    def from_json(cls, obj_json):
        """Return ``Sample`` from a dict of `to_json`."""
        kw = obj_json["kwargs"]
        return cls([Specimen.from_json(s) for s in kw["specimens"]], name=kw["name"])

    def _components(self, name, coords, kind=None):
        pairs = [
            (s, s.component(name, coords))
            for s in self.specimens
            if name in s.components
        ]
        if kind is not None:
            pairs = [(s, c) for s, c in pairs if c.kind == kind]
        return pairs

    def directions(self, name, coords="geo"):
        """Line components `name` as ``Vector3Set`` of unit vectors."""
        comps = self._components(name, coords, "line")
        return Vector3Set([c.vector.normalized() for _, c in comps], name=self.name)

    def planes(self, name, coords="geo"):
        """Plane components `name` as ``FoliationSet``."""
        comps = self._components(name, coords, "plane")
        return FoliationSet([c.fol for _, c in comps], name=self.name)

    def mean(self, name, coords="geo", level=0.95):
        """Mean direction of the components `name`.

        Fisher statistics (``Vector3Set.fisher_statistics``) when all
        components are lines; with plane components the combined mean of
        `lines_planes_mean` (McFadden & McElhinny, 1988), which can be biased
        when the planes are not anchored by direct observations (see there).

        Returns:
            dict: ``mu``, ``k``, ``alpha`` (and ``kappa`` for lines only),
                ``n_lines`` and ``n_planes``.
        """
        comps = self._components(name, coords)
        if not comps:
            raise ValueError(f"No specimen of {self.name} has component {name!r}")
        lines = [c.vector for _, c in comps if c.kind == "line"]
        poles = [c.vector for _, c in comps if c.kind == "plane"]
        if not poles:
            res = Vector3Set(lines).fisher_statistics(level)
            return {**res, "n_lines": len(lines), "n_planes": 0}
        res = lines_planes_mean(lines, poles, level)
        return {k: res[k] for k in ("mu", "k", "alpha", "n_lines", "n_planes")}

    def fold_test(self, name, n=1000, fracs=None, level=0.95, rng=None):
        """Bootstrap fold test of Tauxe & Watson (1994).

        For each bootstrap resample of the line components (geographic
        coordinates) the percentage of untilting maximizing the largest
        eigenvalue ``tau1`` of the orientation tensor is found; each specimen
        is untilted by its own bedding. A confidence interval of these maxima
        including 100 % (and excluding 0 %) supports a pre-folding
        magnetization. Polarity does not matter.

        Args:
            name: component name.
            n: number of bootstrap resamples.
            fracs: percentages of untilting tried, default -10 to 150.
            level: confidence level of the interval.
            rng: None, seed or ``numpy.random.Generator``.

        Returns:
            dict: ``best`` (percentage of the data), ``ci`` (interval of the
                bootstrap maxima), ``fracs``, ``tau1`` (curve of the data) and
                ``maxima``.

        References:
            Tauxe, L. & Watson, G.S. (1994). The fold test: an eigen analysis
                approach. Earth and Planetary Science Letters 122, 331-341.
        """
        comps = self._components(name, "geo", "line")
        if len(comps) < 3:
            raise ValueError("Fold test needs at least three line components")
        fracs = np.arange(-10, 151) if fracs is None else np.asarray(fracs, dtype=float)
        G = np.array([np.asarray(c.vector.normalized()) for _, c in comps])
        U = np.empty((len(fracs), len(G), 3))
        for j, (s, _) in enumerate(comps):
            for i, f in enumerate(fracs):
                U[i, j] = np.asarray(untilt_rotation(s.bedding, f / 100)) @ G[j]

        def tau1(idx):
            A = U[:, idx, :]
            T = np.einsum("fni,fnj->fij", A, A) / len(idx)
            return np.linalg.eigvalsh(T)[:, -1]

        curve = tau1(np.arange(len(G)))
        gen = as_rng(rng)
        maxima = np.array(
            [fracs[np.argmax(tau1(gen.integers(0, len(G), len(G))))] for _ in range(n)]
        )
        q = (1 - level) / 2
        return {
            "best": float(fracs[np.argmax(curve)]),
            "ci": tuple(float(v) for v in np.quantile(maxima, [q, 1 - q])),
            "fracs": fracs,
            "tau1": curve,
            "maxima": maxima,
        }

    def reversal_test(self, name, coords="tilt", method=None, level=0.95):
        """Reversal test: common mean of the normal and the flipped reversed
        line components.

        The directions are split by the sign of their projection on the
        principal axis of the orientation tensor, the reversed mode is turned
        to its antipode and ``common_mean_test`` is applied (Watson's F test
        by default; Tauxe, 2003, §4.13-4.14).

        Returns:
            TestResult: ``reject()`` is True when the modes are not antipodal.
        """
        D = np.asarray(self.directions(name, coords), dtype=float)
        if len(D) < 3:
            raise ValueError("Reversal test needs at least three line components")
        axis = np.linalg.eigh(D.T @ D)[1][:, -1]
        side = D @ axis >= 0
        if side.all() or not side.any():
            raise ValueError("All directions have the same polarity")
        return common_mean_test([D[side], -D[~side]], method=method, level=level)

    def vgp(self, name, coords="tilt", latitude=None, longitude=None, level=0.95):
        """Virtual geomagnetic pole of the mean of the components `name`.

        The site location defaults to the common ``latitude``/``longitude`` of
        the specimens. See `vgp` for the formulas.
        """
        if latitude is None or longitude is None:
            locs = {
                (s.latitude, s.longitude)
                for s in self.specimens
                if s.latitude is not None and s.longitude is not None
            }
            if len(locs) != 1:
                raise ValueError(
                    "Give latitude and longitude, the specimens have "
                    f"{len(locs)} different locations"
                )
            latitude, longitude = locs.pop()
        m = self.mean(name, coords, level)
        dec, inc = m["mu"].geo
        return vgp(dec, inc, latitude, longitude, m["alpha"])
