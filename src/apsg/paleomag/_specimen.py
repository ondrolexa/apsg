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
from functools import cache
from pathlib import Path

import numpy as np

from apsg.feature._container import FoliationSet, Vector3Set
from apsg.feature._distributions._base import ConfidenceRegion, as_rng
from apsg.feature._geodata import Foliation, Lineation
from apsg.feature._hypothesis import TestResult, common_mean_test
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
    "mcfadden_fold_test",
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


_CLOCK_AZIMUTH = {12: 0.0, 3: 270.0, 6: 180.0, 9: 90.0}
_CLOCK_SPIN = {12: 0.0, 3: 90.0, 6: 0.0, 9: 270.0}


def _agico_orientation(p1, p2, p3, p4, azimuth, dip, bedding=None, source=""):
    """Core orientation from the AGICO orientation parameters ``P1-P4``.

    Used by the RS3 and JR6 readers and the SUFAR report reader. The
    parameters follow AGICO (2009) as described by Sagnotti (2013) and the
    conversion of PuffinPlot: P1 is the clock position (12, 3, 6, 9) of the
    reference arrow (specimen x axis) on the frontal base of the core, P2 is 0
    when `dip` is the dip of the frontal base (the hade of the core axis) and
    90 when it is the plunge of the core axis, P3 is the clock position of the
    measured `azimuth` and P4 is 0 when `bedding` is a dip direction and 90
    when it is the strike (right-hand rule).

    A reference arrow at 6 o'clock is a half turn of the specimen about the core
    axis, which is absorbed by ``azimuth + 180`` and ``-hade``. The codes 12 0 6 0,
    6 0 6 0 and 12 0 12 90 were checked against Remasoft files and 12 90 6 0 (with
    P4 = 0 for the foliation) against SUSAR reports of AGICO (the specimen and
    geographic tensors printed on 159 pages); the arrows at 3 and 9 o'clock follow
    PuffinPlot and are not checked against instrument files. They cannot
    be absorbed and are returned as `spin`, the angle about the specimen z axis
    by which the measured vectors (or tensors) must be rotated,
    ``M' = Rz(spin) M``.

    Args:
        p1, p2, p3, p4: orientation parameters.
        azimuth: measured core azimuth in degrees.
        dip: measured core dip (see P2) in degrees.
        bedding: bedding strike or dip direction (see P4) or None.
        source: file name for the error message.

    Returns:
        ``(azimuth, hade, spin, dipdir)`` in the PMD convention of
        `core_rotation`; `dipdir` is None without `bedding`.

    Raises:
        NotImplementedError: for parameter values other than the supported ones.
    """
    if p1 not in _CLOCK_AZIMUTH or p3 not in _CLOCK_AZIMUTH or p2 not in (0, 90):
        raise NotImplementedError(
            f"{source}: orientation parameters P1={p1}, P2={p2}, P3={p3}, "
            f"P4={p4} are not supported (supported: P1 and P3 3, 6, 9 or 12, "
            "P2 0 or 90, P4 0 or 90)"
        )
    if p4 not in (0, 90):
        raise NotImplementedError(
            f"{source}: orientation parameters P1={p1}, P2={p2}, P3={p3}, "
            f"P4={p4} are not supported (supported: P4 0 or 90)"
        )
    hade = dip if p2 == 0 else 90.0 - dip
    azimuth = azimuth + _CLOCK_AZIMUTH[p3]
    if p1 == 6:
        azimuth, hade = azimuth + 180.0, -hade
    dipdir = None if bedding is None else (bedding + p4) % 360
    return azimuth % 360, hade, _CLOCK_SPIN[p1], dipdir


def _spin(M, angle):
    """Rotate the vector `M` about the z axis by `angle` degrees."""
    if not angle:
        return M
    c, s = cosd(angle), sind(angle)
    x, y, z = np.asarray(M, dtype=float)
    return Vector3(c * x - s * y, s * x + c * y, z)


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
        treatment: ``"nrm"``, ``"af"``, ``"thermal"``, ``"arm"``, ``"irm"`` or
            None when not recorded.
        field: applied field in mT of an ARM (bias field) or IRM step, else None.
    """

    label: str
    level: float
    M: Vector3
    a95: float = 0.0
    k: float | None = None
    comment: str = ""
    treatment: str | None = None
    field: float | None = None

    def to_json(self):
        """Return as JSON dict."""
        d = {
            "label": self.label,
            "level": self.level,
            "M": list(self.M),
            "a95": self.a95,
            "k": self.k,
            "comment": self.comment,
        }
        for key in ("treatment", "field"):
            if getattr(self, key) is not None:
                d[key] = getattr(self, key)
        return d

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
        anchor: True when the fit was forced through the origin, ``"last"`` when it was
            forced through the last of the steps (``Anchor to last point`` of
            Remasoft), False for a free fit.
        coords: coordinate system of `vector`.
        origin: True when the origin was added as a data point of the fit
            (the ``O`` variant of Remasoft).
        alpha: angle in degrees between the free and the origin-anchored line fit
            of the same steps (None when not computed, see `Specimen.find_components`).
        levels: first and last treatment level of the steps used (None when
            unknown).
    """

    kind: str
    vector: Vector3
    mad: float
    dang: float | None
    n: int
    labels: tuple
    anchor: bool | str = False
    coords: str = "specimen"
    origin: bool = False
    alpha: float | None = None
    levels: tuple | None = None

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
        return {
            **d,
            "vector": list(self.vector),
            "labels": list(self.labels),
            "levels": None if self.levels is None else list(self.levels),
        }

    @classmethod
    def from_json(cls, d):
        """Return ``Component`` from a dict of `to_json`."""
        levels = d.get("levels")
        return cls(
            **{
                **d,
                "vector": Vector3(d["vector"]),
                "labels": tuple(d["labels"]),
                "levels": None if levels is None else tuple(levels),
            }
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
    """Guess the demagnetization type from step labels.

    Thermal: ``T250``, ``250°C``, ``250C``; AF: ``M010``, ``H010``, ``AF10``,
    ``100M``, ``100MT``.
    """
    for label in labels:
        s = label.strip().upper()
        if s.startswith("NRM") or not s:
            continue
        if "°C" in label or s.startswith("T") or re.fullmatch(r"[\d.]+\s*C", s):
            return "thermal"
        if (
            s.startswith(("M", "AF", "A", "H"))
            or "MT" in s
            or re.fullmatch(r"[\d.]+\s*M", s)
        ):
            return "af"
    return None


def _demag_from_levels(levels):
    """Guess the demagnetization type from the treatment levels alone.

    For files that carry no unit (RS3 rows of a Remasoft file are all ``M``). Thermal
    series reach more than 200 degrees or start at 50 or more; AF series start
    at a few mT and rarely pass 200 mT. Checked on 90 thermal (first step 120-160,
    last 240-600) and 44 AF specimens (first step 1, last 10-100 mT); a series that
    fits neither, or has fewer than two levels, gives None.
    """
    lv = [x for x in levels if x > 0]
    if len(lv) < 2:
        return None
    return "thermal" if max(lv) > 200 or min(lv) >= 50 else "af"


def _check_geometry(spec, geo, tilt, source, tol=2.0, stacklevel=3):
    """Warn when the orientation does not reproduce the file's own directions.

    `geo` and `tilt` are lists of unit vectors or None, one per step.
    """
    errors = []
    for computed, stored in ((spec.geo, geo), (spec.tilt, tilt)):
        pairs = [(c, s) for c, s in zip(computed, stored) if s is not None]
        if pairs:
            errors.append(np.median(_angles(*zip(*pairs))))
    if errors and max(errors) > tol:
        warnings.warn(
            f"{source}: orientation of {spec.name!r} does not reproduce the "
            f"directions stored in the file (misfit {max(errors):.1f} deg)",
            stacklevel=stacklevel,
        )


def _read_lines(filename):
    if hasattr(filename, "read"):  # open file (text or binary)
        text = filename.read()
        if isinstance(text, bytes):
            text = text.decode("latin-1")
    else:
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
        depth: depth of a continuous (long core) measurement or None.
        arm: steps of the anisotropy of remanence (ARM) treatment, which are
            not part of the demagnetization series `steps` (``DemagStep`` with
            ``treatment="arm"``: `level` is the AF field of the demagnetization
            of the ARM and `field` the bias field).
        irm: steps of an IRM treatment (`level` is the applied field in mT).

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
    depth: float | None = None
    arm: list = field(default_factory=list)
    irm: list = field(default_factory=list)

    def __post_init__(self):
        self.steps = list(self.steps)
        self.arm = list(self.arm)
        self.irm = list(self.irm)
        if not all(
            isinstance(s, DemagStep) for s in (*self.steps, *self.arm, *self.irm)
        ):
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

    def pca(self, steps=None, *, kind="line", anchor=False, origin=False, coords="geo"):
        """Principal component analysis of demagnetization data.

        The best-fit line (``kind="line"``) or plane (``kind="plane"``) of the
        selected steps (Kirschvink, 1980; Tauxe, 2003, §3.8, eq. 3.4-3.10). For a
        free fit the points are centred on their centre of mass; an anchored
        fit (``anchor=True``) passes through the origin and ``anchor="last"``
        through the last selected step; ``origin=True`` adds
        the origin as one more point of the centred fit of a line (the ``O``
        variant of Remasoft), which pulls the line towards the origin without
        forcing it through. The four combinations of Remasoft's ``Include ORIGIN``
        and ``Anchor to last point`` are: free fit, ``anchor="last"``,
        ``origin=True`` and ``anchor=True`` (both clicked: anchored to the origin).
        The magnetization
        vectors are used unnormalized. The line points along the removed
        magnetization (first minus last step; along the data for anchored
        fits), the plane pole along ``first x last``, so the sense of the
        demagnetization path is kept.

        Args:
            steps: selection of steps (slice, positions, labels or boolean
                mask); None for all.
            kind: ``"line"`` or ``"plane"``.
            anchor: True forces the fit through the origin, ``"last"`` through the
                last selected step (lines only). Default False.
            origin: include the origin as a data point (lines only, not with
                `anchor`). Default False.
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
        if anchor not in (False, True, "last"):
            raise ValueError(f"anchor must be False, True or 'last', got {anchor!r}")
        if anchor == "last" and kind != "line":
            raise ValueError("anchor='last' applies to lines only")
        if origin and (anchor or kind != "line"):
            raise ValueError("origin applies to free line fits only")
        sub = self if steps is None else self[steps]
        if isinstance(sub, DemagStep):
            raise TypeError("pca needs a selection of steps, not a single step")
        n_min = (2 if kind == "line" else 3) - int(anchor is True)
        if len(sub) < n_min:
            raise ValueError(f"{kind} fit needs at least {n_min} steps")
        X = np.asarray(sub.V, dtype=float)
        P = np.vstack([X, np.zeros(3)]) if origin else X
        center = X[-1] if anchor == "last" else P.mean(axis=0)
        Y = P if anchor is True else P - center
        # the direction and the MAD do not depend on the scale; the eigenvalues
        # of a tensor of weak vectors (< 1e-4 A/m) would be rounded to zero
        scale = float(np.abs(Y).max())
        if scale == 0:
            raise ValueError("the selected vectors are identical, no line or plane")
        Y = Y / scale
        T = OrientationTensor3(Y.T @ Y / len(Y))
        if kind == "line":
            v = np.asarray(T.V1, dtype=float)
            ref = center if anchor is True or len(X) < 2 else X[0] - X[-1]
            if np.dot(v, ref) < 0:
                v = -v
            mad = T.MAD_l
            dang = None
            if not anchor and not origin and np.linalg.norm(center) > 0:
                dang = float(_angles([v], [center])[0])
        else:
            v = np.asarray(T.V3, dtype=float)
            ref = np.cross(X[0], X[-1])
            if np.dot(v, ref) < 0:
                v = -v
            mad = T.MAD_p
            dang = None
        comp = Component(
            kind,
            Vector3(v),
            float(mad),
            dang,
            len(sub),
            tuple(sub.labels),
            anchor,
            origin=origin,
            levels=(float(np.min(sub.levels)), float(np.max(sub.levels))),
        )
        if coords == "specimen":
            return comp
        return comp.transform(self.rotation(coords), coords)

    def interpret(self, name, steps=None, *, kind="line", anchor=False, origin=False):
        """Fit a component by `pca` and store it under `name`.

        Returns:
            Component: the stored component in geographic coordinates.
        """
        comp = self.pca(
            steps, kind=kind, anchor=anchor, origin=origin, coords="specimen"
        )
        self.components[name] = comp
        return comp.transform(self.rotation("geo"), "geo")

    def window_fits(self, min_steps=4, max_steps=None, steps=None):
        """Line fits of every contiguous window of steps, as a ``DataFrame``.

        See `apsg.paleomag._components.window_fits`.
        """
        from apsg.paleomag._components import window_fits

        return window_fits(self, min_steps, max_steps, steps)

    def find_component(self, criteria=None, **kwargs):
        """The best single line component by the acceptance criteria, or None.

        Keyword arguments: `score`, `end`, `tail`, `alpha_max`, `steps`, `coords`; see
        `apsg.paleomag._components.find_component`.
        """
        from apsg.paleomag._components import find_component

        return find_component(self, criteria, **kwargs)

    def find_components(self, criteria=None, **kwargs):
        """Split the demagnetization path into a few linear components.

        Keyword arguments: `max_components`, `segment_penalty`, `skip_penalty`,
        `share_boundary`, `steps`, `coords`; see
        `apsg.paleomag._components.find_components`. The result is a heuristic and
        should be checked on the Zijderveld diagram.
        """
        from apsg.paleomag._components import find_components

        return find_components(self, criteria, **kwargs)

    def interpret_auto(self, prefix="C", criteria=None, **kwargs):
        """Find the components automatically and store them as ``C1``, ``C2``, ...

        The components found by `find_components` replace those stored under
        `prefix` followed by a number.

        Returns:
            list of Component: the components in geographic coordinates.
        """
        kwargs["coords"] = "specimen"
        comps = self.find_components(criteria, **kwargs)
        for key in [
            k for k in self.components if re.fullmatch(f"{re.escape(prefix)}\\d+", k)
        ]:
            del self.components[key]
        for i, comp in enumerate(comps, start=1):
            self.components[f"{prefix}{i}"] = comp
        R = self.rotation("geo")
        return [c.transform(R, "geo") for c in comps]

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

    def susceptibility_jump(self, factor=2.5):
        """Temperature of the first sudden increase of the susceptibility.

        The first step whose susceptibility `k` exceeds `factor` times that of the
        preceding step with a susceptibility, in a thermal demagnetization
        series. It marks the growth of new magnetic minerals, as in the
        susceptibility monitoring of PuffinPlot (limit 2.5).

        Args:
            factor: ratio of the susceptibilities of consecutive steps.

        Returns:
            float or None: the level (degrees C) of the step, None without a jump.

        Raises:
            ValueError: for an AF demagnetization.
        """
        if self.demag == "af":
            raise ValueError("the susceptibility jump is for thermal demagnetization")
        prev = None
        for step in self.steps:
            if step.k is None or not np.isfinite(step.k):
                continue
            if prev is not None and prev > 0 and step.k > prev * factor:
                return step.level
            prev = step.k
        return None

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
        "longitude", "height", "rock", "age", "formation", "filename", "depth",
    )  # fmt: skip

    def to_json(self):
        """Return as JSON dict."""
        kwargs = {k: getattr(self, k) for k in self._META}
        kwargs["steps"] = [s.to_json() for s in self.steps]
        for key in ("arm", "irm"):
            if getattr(self, key):
                kwargs[key] = [s.to_json() for s in getattr(self, key)]
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
        for key in ("arm", "irm"):
            kw[key] = [DemagStep.from_json(s) for s in kw.get(key, [])]
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
    def from_file(cls, filename, fmt=None, **kwargs):
        """Read a file holding one specimen.

        The format is taken from the extension (``.pmd``, ``.rs3``,
        ``.squid``) or the content, or from `fmt`. Use ``Sample.from_file`` for
        files with several specimens.
        """
        from apsg.paleomag._readers import _read_specimens

        specimens = _read_specimens(filename, fmt=fmt, **kwargs)
        if len(specimens) != 1:
            raise ValueError(
                f"{filename}: file holds {len(specimens)} specimens, "
                "use Sample.from_file"
            )
        return specimens[0]

    @classmethod
    def from_pmd(cls, filename, demag=None, dialect=None):
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
        of the file is used; a warning is issued when neither does or when
        the file has no such columns to decide with (then standard PMD is
        assumed). Lines after the column header that are not data rows are
        reported in a warning.

        Args:
            filename: path of the file, or an open file object.
            demag: ``"thermal"`` or ``"af"``; guessed from the labels if None.
            dialect: ``"pmd"`` or ``"remasoft"`` to skip the detection.
        """
        if dialect not in (None, "pmd", "remasoft"):
            raise ValueError(f"dialect must be 'pmd' or 'remasoft', not {dialect!r}")
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
        if o["v"] <= 0:
            raise ValueError(f"{filename}: volume v={o['v']:g} must be positive")
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
        steps, dg, ds, skipped = [], [], [], []
        for i, ln in enumerate(lines[ih + 2 :], start=ih + 3):
            m = _PMD_ROW.match(ln)
            if not m:
                if ln.strip():
                    skipped.append(i)
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
        if skipped:
            shown = ", ".join(map(str, skipped[:3])) + (
                ", ..." if len(skipped) > 3 else ""
            )
            warnings.warn(
                f"{filename}: {len(skipped)} lines are not data rows and were "
                f"skipped (lines {shown})",
                stacklevel=2,
            )
        dialects = {
            "pmd": (o["a"], o["b"], bedding_vector(o["s"] + 90, o["d"])),
            "remasoft": (o["a"] + 180, o["b"], bedding_vector(o["s"], o["d"])),
        }
        best, misfit, comparable = dialect or "pmd", np.inf, False
        for key, (a, b, bed) in dialects.items() if dialect is None else ():
            spec = cls(steps=steps, azimuth=a, hade=b, bedding=bed)
            err = []
            for computed, stored in ((spec.geo, dg), (spec.tilt, ds)):
                pairs = [(c, s) for c, s in zip(computed, stored) if s is not None]
                if pairs:
                    err.append(np.median(_angles(*zip(*pairs))))
            comparable = comparable or bool(err)
            score = float(np.sum(err)) if err else 0.0
            if score < misfit:
                best, misfit = key, score
        if dialect is None and not comparable:
            warnings.warn(
                f"{filename}: no Dg/Ig/Ds/Is columns to tell standard PMD from "
                "Remasoft PMD; read as standard PMD (pass dialect='remasoft' "
                "for a Remasoft export)",
                stacklevel=2,
            )
        elif dialect is None and misfit > 2:
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
    def from_rs3(cls, filename, exclude=("C", "G"), demag=None, components=False):
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
        the strike (right-hand rule). See `_agico_orientation`: P1 and P3 of
        3, 6, 9 or 12, P2 of 0 or 90 and P4 of 0 or 90 are supported. The codes
        12 0 6 0, 6 0 6 0 and 12 0 12 90 were checked against the
        ``Dge Ige Dtc Itc`` columns of Remasoft files, the codes of P2 = 90 against
        SUSAR reports (see `_agico_orientation`), the arrows at 3 and 9 o'clock
        follow the conversion of PuffinPlot.

        Args:
            filename: path of the file.
            exclude: row IDs to skip, by default the component (``C``) and
                great circle (``G``) fits of Remasoft.
            demag: ``"thermal"`` or ``"af"``; from the step units or row IDs
                if None. Remasoft files usually carry no unit (``Step[ ]``, every step
                row ``M``), then the type is guessed from the levels (thermal above 200
                or from 50 up, otherwise AF, see `_demag_from_levels`); pass `demag`
                for a series that is not typical. Thermal steps are labelled ``T120``.
            components: also import the component (``C``) and great circle
                (``G``) rows as ``Specimen.components``. Each row is refitted by
                `pca` from the steps between its two limits (a ``C`` row is a free line;
                the Note flags select the variant: ``O`` adds the origin as a data
                point, ``A`` anchors the line to the last step and ``OA`` to the
                origin; a ``G`` row is a plane through the origin) and a warning is issued when the printed
                direction (specimen coordinates, the pole for a ``G`` row) or MAD
                is not reproduced. Default False.

        Warnings are issued for a missing core orientation, missing orientation
        parameters (12 0 12 0 is assumed), step rows that cannot be read and
        when the orientation does not reproduce the geographic and tilt-corrected
        directions (``Dge Ige Dtc Itc``) printed in the rows.

        Raises:
            ValueError: if the file lacks the column line (``SDec``) or the
                units line (``ID``).
        """
        numbered = [
            (i, ln) for i, ln in enumerate(_read_lines(filename), start=1) if ln.strip()
        ]
        lines = [ln for _, ln in numbered]
        if (
            len(lines) < 3
            or "SDec" not in lines[0]
            or not lines[2].lstrip().upper().startswith("ID")
        ):
            raise ValueError(
                f"{filename}: not an RS3 file (expected a column line with SDec, "
                "the specimen line and a units line starting with ID)"
            )
        h = lines[1]

        def num(a, b):
            return _float(h[a:b].strip())

        def txt(a, b):
            return h[a:b].strip()

        sdec, sinc = num(74, 80), num(80, 86)
        if sdec is None or sinc is None:
            warnings.warn(
                f"{filename}: no core orientation (SDec, SInc), the specimen is "
                "taken as unoriented",
                stacklevel=2,
            )
        oriented = sdec is not None and sinc is not None
        sdec, sinc = sdec or 0.0, sinc or 0.0
        bdec, binc = num(86, 92), num(92, 98)
        fdec, finc = num(98, 104), num(104, 110)
        codes = [num(a, a + 3) for a in (110, 113, 116, 119)]
        if all(c is None for c in codes):
            warnings.warn(
                f"{filename}: no orientation parameters P1-P4, assuming 12 0 12 0",
                stacklevel=2,
            )
            codes = [12, 0, 12, 0]
        p1, p2, p3, p4 = (int(c or 0) for c in codes)
        azimuth, hade, spin, dipdir = _agico_orientation(
            p1, p2, p3, p4, sdec, sinc, bdec, source=filename
        )
        if not oriented:
            azimuth, hade = 0.0, 0.0
        units = lines[2][3:14]
        if demag is None:
            if "mT" in units:
                demag = "af"
            elif "C" in units:
                demag = "thermal"
        steps, geo, tilt, skipped, fits = [], [], [], [], []
        for i, r in numbered[3:]:
            rid = r[0:2].strip()
            if rid in ("C", "G"):
                fits.append((i, r))
            if rid in exclude:
                continue
            step = r[3:13].strip()
            M = _float(r[13:27].strip())
            dsp, isp = _float(r[27:33].strip()), _float(r[33:39].strip())
            if M is None or dsp is None or isp is None:
                skipped.append(i)
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
                    _spin(Vector3(M * _dir(dsp, isp)), spin),
                    a95=_float(r[75:81].strip(), 0.0),
                    k=_float(r[81:95].strip()),
                    comment=r[115:].strip(),
                )
            )
            pair = []
            for a in (39, 51):
                d, n = _float(r[a : a + 6]), _float(r[a + 6 : a + 12])
                pair.append(None if None in (d, n) else _dir(d, n))
            geo.append(pair[0])
            tilt.append(pair[1])
        if skipped:
            shown = ", ".join(map(str, skipped[:3])) + (
                ", ..." if len(skipped) > 3 else ""
            )
            warnings.warn(
                f"{filename}: {len(skipped)} step rows without intensity or direction "
                f"were skipped (lines {shown})",
                stacklevel=2,
            )
        if demag is None:
            # the rows of Remasoft files are all "M", whatever the treatment
            ids = {r[0:2].strip() for _, r in numbered[3:]}
            demag = (
                "thermal"
                if "T" in ids
                else _demag_from_levels([st.level for st in steps])
                or _demag_from_labels([s.label for s in steps])
            )
        if demag == "thermal":
            steps = [
                replace(st, label="T" + st.label[1:])
                if st.label.startswith("M") and st.label[1:2].isdigit()
                else st
                for st in steps
            ]
        bedding = Vector3(_UP)
        if bdec is not None and binc is not None:
            bedding = bedding_vector(dipdir, binc)
        spec = cls(
            name=txt(0, 10),
            steps=steps,
            demag=demag,
            azimuth=azimuth,
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
        _check_geometry(spec, geo, tilt, filename, stacklevel=3)
        if components:
            spec._rs3_components(fits, filename)
        return spec

    def _rs3_components(self, fits, source):
        """Refit the ``C`` and ``G`` rows of an RS3 file as components.

        The rows give a name, the limits (step labels) of the fit, the printed
        direction in specimen coordinates and the MAD (``Prec``); they are fitted
        again by `pca` from the steps between the limits and a warning is issued
        when the printed values are not reproduced.
        """

        def find(token):
            if token.upper().startswith("NRM"):
                level = 0.0
            else:
                m = re.search(r"\d+(?:\.\d+)?", token)
                if m is None:
                    return None
                level = float(m.group())
            hits = [k for k, s in enumerate(self.steps) if abs(s.level - level) < 1e-6]
            return hits[0] if hits else None

        for i, r in fits:
            name = r[3:13].strip() or r[0:2].strip()
            tail = r[95:].split()
            ends = [find(tok) for tok in tail[:2]] if len(tail) >= 2 else [None]
            if len(ends) < 2 or None in ends:
                warnings.warn(
                    f"{source}: line {i}: steps of the limits {tail[:2]} of "
                    f"component {name!r} not found, skipped",
                    stacklevel=4,
                )
                continue
            a, b = sorted(ends)
            line = r[0:2].strip() == "C"
            note = tail[2].upper() if len(tail) > 2 else ""
            flags = note if note in ("O", "A", "OA", "AO") else ""  # Note column
            if not line:
                anchor = True
            elif "A" in flags:  # anchored to the origin when O is set as well
                anchor = True if "O" in flags else "last"
            else:
                anchor = False
            try:
                comp = self.pca(
                    slice(a, b + 1),
                    kind="line" if line else "plane",
                    anchor=anchor,
                    origin=line and "O" in flags and not anchor,
                    coords="specimen",
                )
            except ValueError as err:
                warnings.warn(
                    f"{source}: line {i}: component {name!r} skipped ({err})",
                    stacklevel=4,
                )
                continue
            dsp, isp = _float(r[27:33].strip()), _float(r[33:39].strip())
            mad = _float(r[75:81].strip())
            if dsp is not None and isp is not None and mad is not None:
                ang = float(_angles([np.asarray(comp.vector)], [_dir(dsp, isp)])[0])
                ang = min(ang, 180 - ang)  # the sense of the line is not compared
                if ang > 1 or abs(comp.mad - mad) > 1:
                    warnings.warn(
                        f"{source}: line {i}: component {name!r} differs from its refit "
                        f"({ang:.1f} deg, MAD {mad:.1f} printed, {comp.mad:.1f} fitted)",
                        stacklevel=4,
                    )
            key, n = name, 2
            while key in self.components:
                key, n = f"{name}{n}", n + 1
            self.components[key] = comp

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
            "STEP  Xc [Am2]  Yc [Am2]  Zc [Am2]  MAG[A/m]   Dg    Ig    Ds    Is   a95",
        ]
        geo, tilt = self.geo, self.tilt
        for step, g, t in zip(self.steps, geo, tilt):
            x, y, z = np.asarray(step.M, dtype=float) * v
            rows.append(
                f"{step.label:<4}{x:10.2E}{y:10.2E}{z:10.2E}{abs(step.M):10.2E}"
                f"{g.geo[0]:6.1f}{g.geo[1]:6.1f}{t.geo[0]:6.1f}{t.geo[1]:6.1f}"
                f"{step.a95:5.1f} {step.comment}".rstrip()  # no trailing space
            )
        with open(filename, "w", encoding="latin-1", errors="replace", newline="") as f:
            # DOS end-of-file marker on its own line, newline-terminated
            f.write("\r\n".join(rows) + "\r\n\x1a\r\n")

    def _rs3_head_rows(self, unit):
        """The column line, the specimen line and the units line of an RS3 file."""
        dd, dip = self._bedding_dd()

        def f(value, fmt):
            return "" if value is None else format(value, fmt)

        fold = ("", "")
        if self.foldaxis is not None:
            fold = (f"{self.foldaxis.geo[0]:.0f}", f"{self.foldaxis.geo[1]:.0f}")
        return [
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

    def _rs3_step_rows(self, prefix):
        """The measurement rows of an RS3 file (`prefix` is the row ID of the steps)."""
        rows = []
        for step, sp, g, t in zip(self.steps, self.V, self.geo, self.tilt):
            rid = "N" if step.label.upper().startswith("NRM") else prefix
            k = "" if step.k is None else f"{step.k:g}"
            rows.append(
                f"{rid:<2} {step.level:<10g}{abs(step.M):>14.7g}"
                f"{sp.geo[0]:6.1f}{sp.geo[1]:6.1f}{g.geo[0]:6.1f}{g.geo[1]:6.1f}"
                f"{t.geo[0]:6.1f}{t.geo[1]:6.1f}{'':12}{step.a95:6.1f}{k:>14}"
                f"{'':20} {step.comment}"
            )
        return rows

    def _rs3_component_rows(self, names, unit):
        """Component (``C``) and great circle (``G``) rows of RS3 for the components `names`.

        Returns:
            tuple: the rows and a list of ``(name, reason)`` of the components that
            RS3 cannot hold (a line forced through the origin, steps that are not
            steps of the specimen).
        """
        labels = {lab: i for i, lab in enumerate(self.labels)}
        rows, skipped = [], []

        def limit(level):
            return "NRM" if level == 0 else f"{level:g}{unit}"

        def direction(vec):
            azi, inc = vec.geo
            return f"{azi % 360:6.1f}{inc:6.1f}"

        for name in names:
            comp = self.components[name]
            idx = [labels.get(lab) for lab in comp.labels]
            if not idx or None in idx:
                skipped.append((name, "its steps are not steps of the specimen"))
                continue
            first, last = min(idx), max(idx)
            if comp.kind == "line":
                # Remasoft's Note flags: O include origin, A anchor to the last point,
                # OA (both) anchor to the origin
                rid = "C"
                note = {True: "OA", "last": "A"}.get(comp.anchor, "O" * comp.origin)
                V = np.asarray(self.V, dtype=float)
                v = np.asarray(comp.vector, dtype=float)
                # the length of the removed vector (first step to the last one, or to
                # the origin when anchored there); Remasoft's own value is not known
                end = 0 if comp.anchor is True else V[last]
                intensity = abs((V[first] - end) @ v)
            else:
                rid, note, intensity = "G", "", 1.0  # the pole is a unit vector
            if len(name) > 10:
                warnings.warn(
                    f"component name {name!r} is cut to 10 characters in the RS3 file",
                    stacklevel=4,
                )
            rows.append(
                (
                    f"{rid:<2} {name[:10]:<10}{intensity:>14.7g}"
                    f"{direction(comp.vector)}"
                    f"{direction(self.component(name, 'geo').vector)}"
                    f"{direction(self.component(name, 'tilt').vector)}"
                    f"{'':12}{comp.mad:6.1f}{'':14}"
                    f"{limit(self.steps[first].level):<10}"
                    f"{limit(self.steps[last].level):<10}{note}"
                ).rstrip()
            )
        return rows, skipped

    def to_rs3(self, filename, components=True):
        """Write the specimen as a Remasoft RS3 file (orientation parameters 12 0 6 0).

        The file holds the column, specimen and units lines, one row per step and,
        after them, one row per stored component: a ``C`` row for a line and a
        ``G`` row for a plane (the direction columns hold its pole). The Note column of
        a ``C`` row holds the Remasoft flags of the fit: ``O`` (`Include ORIGIN`,
        `pca(origin=True)`), ``A`` (`Anchor to last point`, `pca(anchor="last")`) and
        ``OA`` (both: anchored to the origin, `pca(anchor=True)`). The rows give the
        directions of the component in specimen, geographic and tilt-corrected
        coordinates, the MAD and the first and last step of the fit as limits (``NRM``,
        ``300°C``, ``20mT``); `from_rs3` with ``components=True`` reads them back.

        The intensity column of a ``C`` row is the length of the removed vector (the
        projection of first minus last step on the fitted direction, first step minus
        origin for ``OA``); it differs from the number Remasoft writes, whose
        definition is not published, and `from_rs3` ignores it. Component names are
        cut to 10 characters.

        Args:
            filename: path of the file.
            components: ``True`` writes all components, ``False`` none, a name or a
                list of names those.

        Raises:
            KeyError: for a name that is not a stored component.
        """
        if components is True:
            names = list(self.components)
        elif components is False or components is None:
            names = []
        else:
            names = [components] if isinstance(components, str) else list(components)
            unknown = [n for n in names if n not in self.components]
            if unknown:
                raise KeyError(f"no component {unknown[0]!r}")
        unit = "mT" if self.demag == "af" else "°C"
        rows = self._rs3_head_rows(unit)
        rows += self._rs3_step_rows("M" if self.demag == "af" else "T")
        fits, skipped = self._rs3_component_rows(names, unit)
        for name, reason in skipped:
            warnings.warn(
                f"component {name!r} is not written to the RS3 file: {reason}",
                stacklevel=2,
            )
        with open(
            filename, "w", encoding="latin-1", errors="replace", newline=""
        ) as fh:
            fh.write("\r\n".join(rows + fits) + "\r\n")


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
            major first), ``axes`` (two ``Vector3`` along them), ``region``
            (the ellipse as ``ConfidenceRegion``, plottable by
            ``StereoNet.confidence``), ``n_lines`` and ``n_planes``.

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
    semi_axes = tuple(float(semi[i]) for i in order)
    axes = tuple(Vector3(B @ W[:, i]) for i in order)
    return {
        "mu": Vector3(t),
        "k": float(k),
        "omega": float(omega),
        "semi_axes": semi_axes,
        "axes": axes,
        "region": ConfidenceRegion(
            mu=Vector3(t), level=level, axes=axes, gamma=semi_axes
        ),
        "n_lines": M,
        "n_planes": N,
    }


@cache
def _xi_null(N, h=0.002):
    """Null distribution of the McFadden (1990) correlation statistic.

    Under the null hypothesis ``xi`` is the sum of ``N`` independent terms
    ``cos(theta_i)`` with ``theta_i`` uniform on the circle, whose
    characteristic function is ``J0(t)^N``; this is the distribution of
    Lord's integral (McFadden 1990, eq. 15). It is evaluated by convolving
    the exact cell masses of the arcsine law of one term on a grid of width
    `h` (FFT). Returns the cell centres and the cumulative probability there.
    """
    edges = np.arange(-1.0, 1.0 + h / 2, h)
    cdf1 = 1 - np.arccos(np.clip(edges, -1, 1)) / np.pi
    p = np.diff(cdf1)
    size = N * (len(p) - 1) + 1
    P = np.fft.irfft(np.fft.rfft(p, size) ** N, size)
    P = np.maximum(P, 0)
    P /= P.sum()
    # the mass of cell k sits at its centre N * (-1 + h/2) + h * k
    return N * (-1 + h / 2) + h * np.arange(size), np.cumsum(P) - P / 2


def _xi_cdf(x, N):
    """Distribution function of the null law of ``xi`` for `N` terms."""
    if N > 100:  # N(0, N/2), McFadden (1990) for large N
        from scipy.special import ndtr

        return ndtr(np.asarray(x, dtype=float) / np.sqrt(N / 2))
    edges, cum = _xi_null(int(N))
    return np.interp(x, edges, cum, left=0.0, right=1.0)


def _xi_critical(N, level):
    """Critical value of ``xi`` (McFadden 1990, Table 2) at `level`."""
    if N > 100:
        from scipy.special import ndtri

        return float(ndtri(level) * np.sqrt(N / 2))
    edges, cum = _xi_null(int(N))
    return float(np.interp(level, cum, edges))


def _rodrigues(axes, angles):
    """Rotation matrices about unit `axes` (n, 3) by `angles` in radians."""
    K = np.zeros((len(axes), 3, 3))
    K[:, 0, 1], K[:, 0, 2] = -axes[:, 2], axes[:, 1]
    K[:, 1, 0], K[:, 1, 2] = axes[:, 2], -axes[:, 0]
    K[:, 2, 0], K[:, 2, 1] = -axes[:, 1], axes[:, 0]
    s = np.sin(angles)[:, None, None]
    c = (1 - np.cos(angles))[:, None, None]
    return np.eye(3) + s * K + c * (K @ K)


def mcfadden_fold_test(directions, beddings, fracs=None, level=0.95, small=5.0):
    """Correlation fold test of McFadden (1990) for complex folding.

    Every direction comes from its own limb with its own bedding. The mean
    ``mu`` of the directions in the tested position is moved by a small tilt
    about the strike of each bed (the folding sense, `small` degrees; the
    amount is immaterial for small tilts) and ``xi = sum(cos(theta_i))``,
    where ``theta_i`` is the angle, in the plane perpendicular to ``mu``,
    between the deviation of the i-th direction from ``mu`` and the
    displacement of ``mu`` (definition 1). Directions that scatter because
    they are tilted differently correlate with the tilt, so a large ``|xi|``
    means the directions are not in their primary position. The
    null distribution is that of Lord's integral (eq. 15), the critical values
    are those of Table 2. The test is run for each percentage of untilting;
    a pre-folding magnetization is significant in situ (0 %) but not when
    unfolded (100 %), a syn-folding one has the minimum of ``|xi|`` between.

    Note:
        ``|xi|`` is compared with the critical value, as over- and
        under-correction both produce a correlation. McFadden's definition 2
        (the mean taken from the other position), recommended when the in
        situ and unfolded means differ much, and his tests for sites on
        common limbs are not implemented. The paper prints 2.362 for the
        large-N 99 % constant, the exact value of its Table 2 is 2.326
        (``sqrt(N/2)`` times the normal quantile), which is used here for
        ``N > 100``; below that the exact null law is used. Beds with zero
        dip have no strike and are dropped.

    Args:
        directions: ``(N, 3)`` directions (geographic coordinates, x north,
            y east, z down), at least 5 are recommended.
        beddings: ``(N, 3)`` bedding normals pointing to the stratigraphic
            top, see `bedding_vector`.
        fracs: percentages of untilting, default 0 to 100 in steps of 1.
        level: confidence level of the critical value.
        small: angle of the small tilt of the mean in degrees.

    Returns:
        dict: ``fracs``, ``xi`` (signed statistic at each percentage),
            ``p_values`` (``1 - F(|xi|)`` of the null law), ``critical``,
            ``geo`` and ``tilt`` (``xi`` at 0 and 100 %, 0 and 100 % need
            to be in `fracs`), ``best`` (percentage with the smallest
            ``|xi|``), ``n`` and ``level``.

    References:
        McFadden, P.L. (1990). A new fold test for palaeomagnetic studies.
            Geophysical Journal International 103, 163-169.
        Lord, R.D. (1948). The use of the Hankel transform in statistics I.
            Biometrika 35, 44-55.
    """
    G = _unit_rows(directions)
    B = _unit_rows(beddings)
    if len(G) != len(B):
        raise ValueError("directions and beddings need the same length")
    up = np.asarray(_UP)
    axes = np.cross(B, up)
    norm = np.linalg.norm(axes, axis=1)
    flat = norm < 1e-9
    if flat.any():
        warnings.warn(
            f"{int(flat.sum())} beds with zero dip have no strike and are dropped",
            UserWarning,
            stacklevel=2,
        )
        G, B, axes, norm = G[~flat], B[~flat], axes[~flat], norm[~flat]
    N = len(G)
    if N < 3:
        raise ValueError("The correlation fold test needs at least three sites")
    if N < 5:
        warnings.warn(
            "McFadden (1990) recommends at least 5 sites", UserWarning, stacklevel=2
        )
    axes = axes / norm[:, None]
    dip = np.arccos(np.clip(B @ up, -1, 1))
    fracs = np.arange(0, 101.0) if fracs is None else np.asarray(fracs, dtype=float)
    fold = _rodrigues(axes, -np.radians(small) * np.ones(N))
    xi = np.empty(len(fracs))
    for j, f in enumerate(fracs):
        P = np.einsum("nij,nj->ni", _rodrigues(axes, f / 100 * dip), G)
        mu = P.sum(axis=0)
        mu /= np.linalg.norm(mu)
        tau = np.einsum("nij,j->ni", fold, mu)
        dev = P - (P @ mu)[:, None] * mu
        shift = tau - (tau @ mu)[:, None] * mu
        den = np.linalg.norm(dev, axis=1) * np.linalg.norm(shift, axis=1)
        ok = den > 0
        xi[j] = np.sum(np.einsum("ni,ni->n", dev, shift)[ok] / den[ok])

    def at(f):
        hit = np.flatnonzero(np.isclose(fracs, f))
        return float(xi[hit[0]]) if len(hit) else None

    return {
        "fracs": fracs,
        "xi": xi,
        "p_values": 1 - _xi_cdf(np.abs(xi), N),
        "critical": _xi_critical(N, level),
        "geo": at(0.0),
        "tilt": at(100.0),
        "best": float(fracs[np.argmin(np.abs(xi))]),
        "n": N,
        "level": level,
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
    def from_file(cls, filename, name=None, fmt=None, **kwargs):
        """Read all specimens of a file.

        The format is taken from the extension or the content, or from `fmt`;
        `name` defaults to the file stem.
        """
        from apsg.paleomag._readers import _read_specimens

        if name is None:
            name = "Default" if hasattr(filename, "read") else Path(filename).stem
        return cls(_read_specimens(filename, fmt=fmt, **kwargs), name=name)

    @classmethod
    def from_files(cls, filenames, name="Default", fmt=None, **kwargs):
        """Read the specimens of several files (see `from_file`)."""
        from apsg.paleomag._readers import _read_specimens

        specimens = []
        for f in filenames:
            specimens.extend(_read_specimens(f, fmt=fmt, **kwargs))
        return cls(specimens, name=name)

    @classmethod
    def from_directory(
        cls,
        path,
        pattern="*",
        recursive=False,
        name=None,
        fmt=None,
        errors="warn",
        **kwargs,
    ):
        """Read all files of a directory, sorted by name.

        Files whose format is not recognized are skipped unless `fmt` is given.
        A file that cannot be read (for instance one with a header but no steps) is
        skipped with a warning, unless ``errors="raise"``.

        Args:
            path: directory.
            pattern: glob pattern of the files.
            recursive: also search subdirectories.
            name: sample name, the directory name by default.
            fmt: format name overriding the detection.
            errors: ``"warn"`` (default) or ``"raise"`` for files that cannot be read.
            **kwargs: passed to the readers.
        """
        from apsg.paleomag._readers import _read_directory

        specimens = _read_directory(path, pattern, recursive, fmt, errors, **kwargs)
        return cls(specimens, name=Path(path).name if name is None else name)

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

    def rpi(self, normalizer=None, method="susceptibility", **kwargs):
        """Relative paleointensity of the specimens, see
        ``relative_paleointensity``.

        Args:
            normalizer: ``Sample`` whose specimens of the same name are the
                normalizers (AF demagnetization of the ARM or IRM); for
                ``"arm_acquisition"`` the `arm` steps of the specimens themselves are
                used if None.
            method: ``"susceptibility"``, ``"af"`` or ``"arm_acquisition"``.
            **kwargs: `min_level` and `max_level`.

        Returns:
            dict: ``RPIEstimate`` by specimen name. Specimens without a
                normalizer are skipped with a warning.
        """
        from apsg.paleomag._procedures import relative_paleointensity

        result, missing = {}, []
        for s in self.specimens:
            norm = None
            if normalizer is not None:
                if s.name not in normalizer.names:
                    missing.append(s.name)
                    continue
                norm = normalizer[s.name]
            result[s.name] = relative_paleointensity(s, norm, method, **kwargs)
        if missing:
            warnings.warn(
                f"no normalizer for {', '.join(missing)}", UserWarning, stacklevel=2
            )
        return result

    def by_site(self):
        """Dict of ``Sample`` by the `site` of the specimens, in order of appearance.

        For long-core data the site of a specimen is its core section.
        """
        groups = {}
        for s in self.specimens:
            groups.setdefault(s.site, []).append(s)
        return {k: Sample(v, name=k) for k, v in groups.items()}

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

    def _components(self, name, coords, kind=None, criteria=None):
        pairs = [
            (s, s.component(name, coords))
            for s in self.specimens
            if name in s.components
        ]
        if kind is not None:
            pairs = [(s, c) for s, c in pairs if c.kind == kind]
        if criteria is not None:
            passed = [(s, c) for s, c in pairs if criteria.passes(c)]
            if len(passed) < len(pairs):
                warnings.warn(
                    f"{len(pairs) - len(passed)} of {len(pairs)} components {name!r} "
                    "fail the acceptance criteria and were left out",
                    stacklevel=3,
                )
            pairs = passed
        return pairs

    def grade(self, name, criteria=None):
        """Grade the components `name` against acceptance criteria.

        Args:
            name: component name.
            criteria: ``ComponentCriteria``, the PmagPy defaults (at least 4 steps,
                MAD <= 5, DANG <= 10 degrees) if None.

        Returns:
            pandas.DataFrame: ``specimen``, ``kind``, ``n``, ``mad``, ``dang``, ``alpha``,
            ``passed`` and the text of the criteria ``failed``.
        """
        from apsg.paleomag._components import grade_sample

        return grade_sample(self, name, criteria)

    def interpret_auto(self, prefix="C", criteria=None, **kwargs):
        """Find the components of every specimen (`Specimen.interpret_auto`).

        Returns:
            dict: the components in geographic coordinates by specimen name.
        """
        return {
            s.name: s.interpret_auto(prefix, criteria, **kwargs) for s in self.specimens
        }

    def interpret(
        self, name, min_level, max_level, kind="line", anchor=False, origin=False
    ):
        """Fit the same range of treatment levels in every specimen.

        Returns:
            int: the number of specimens fitted (those with too few steps are skipped
            with a warning).
        """
        from apsg.paleomag._components import interpret_levels

        return interpret_levels(self, name, min_level, max_level, kind, anchor, origin)

    def stack(self, coords="geo", normalize="nrm", min_specimens=3):
        """Stack the specimens step by step into one pseudo-specimen.

        See `apsg.paleomag._components.stack_sample`: the vectors of the
        specimens at each shared treatment level are averaged, in `coords`.
        """
        from apsg.paleomag._components import stack_sample

        return stack_sample(self, coords, normalize, min_specimens)

    def directions(self, name, coords="geo", criteria=None):
        """Line components `name` as ``Vector3Set`` of unit vectors.

        Components failing `criteria` (``ComponentCriteria``) are left out."""
        comps = self._components(name, coords, "line", criteria)
        return Vector3Set([c.vector.normalized() for _, c in comps], name=self.name)

    def planes(self, name, coords="geo", criteria=None):
        """Plane components `name` as ``FoliationSet``.

        Components failing `criteria` (``ComponentCriteria``) are left out."""
        comps = self._components(name, coords, "plane", criteria)
        return FoliationSet([c.fol for _, c in comps], name=self.name)

    def mean(self, name, coords="geo", level=0.95, criteria=None):
        """Mean direction of the components `name`.

        Components failing `criteria` (``ComponentCriteria``) are left out.

        Fisher statistics (``Vector3Set.fisher_statistics``) when all
        components are lines; with plane components the combined mean of
        `lines_planes_mean` (McFadden & McElhinny, 1988), which can be biased
        when the planes are not anchored by direct observations (see there).

        Returns:
            dict: ``mu``, ``k``, ``alpha`` (and ``kappa`` for lines only),
                ``n_lines`` and ``n_planes``.
        """
        comps = self._components(name, coords, criteria=criteria)
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

    def mcfadden_fold_test(self, name, fracs=None, level=0.95, small=5.0):
        """Correlation fold test of McFadden (1990) on the line components.

        Every specimen is taken as a site on its own limb with its own bedding;
        see `mcfadden_fold_test`, which also describes the result.
        """
        comps = self._components(name, "geo", "line")
        if len(comps) < 3:
            raise ValueError("Fold test needs at least three line components")
        return mcfadden_fold_test(
            [np.asarray(c.vector.normalized()) for _, c in comps],
            [np.asarray(s.bedding) for s, _ in comps],
            fracs=fracs,
            level=level,
            small=small,
        )

    def reversal_test(
        self, name, coords="tilt", method=None, level=0.95, n=1000, rng=None
    ):
        """Reversal test: common mean of the normal and the flipped reversed
        line components.

        The directions are split by the sign of their projection on the
        principal axis of the orientation tensor, the reversed mode is turned
        to its antipode and ``common_mean_test`` is applied (Watson's F test
        by default; Tauxe, 2003, §4.13-4.14).

        With ``method="bootstrap"`` the confidence intervals of the three
        Cartesian components of the Fisher mean are found by bootstrap (`n`
        resamples, `rng`) for each mode, and the modes share a mean when the
        intervals overlap in all components (Tauxe, 2003, §4.13-4.14; Tauxe et
        al., 1991). The statistic is the number of components with disjoint
        intervals and the test is rejected for any, so ``extra`` holds the
        intervals ``lower`` and ``upper`` (2 x 3 arrays, normal mode first).
        The two criteria can disagree: on Tauxe's Table 4.2 Watson's F rejects
        while the intervals overlap.

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
        modes = [D[side], -D[~side]]
        if method != "bootstrap":
            return common_mean_test(modes, method=method, level=level)
        gen = as_rng(rng)
        lo, hi = [], []
        for X in modes:
            idx = gen.integers(0, len(X), (n, len(X)))
            m = X[idx].sum(axis=1)
            m /= np.linalg.norm(m, axis=1, keepdims=True)
            q = np.quantile(m, [(1 - level) / 2, (1 + level) / 2], axis=0)
            lo.append(q[0])
            hi.append(q[1])
        lo, hi = np.array(lo), np.array(hi)
        disjoint = (lo[0] > hi[1]) | (lo[1] > hi[0])
        return TestResult(
            name="Bootstrap reversal test",
            statistic=float(disjoint.sum()),
            critical=0.0,
            level=level,
            source="Tauxe (2003), sect. 4.13-4.14",
            extra={"lower": lo, "upper": hi},
        )

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
