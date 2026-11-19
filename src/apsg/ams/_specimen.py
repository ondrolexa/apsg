"""AMS specimens and sites: tensors with bulk susceptibility, orientation and bedding."""

from dataclasses import dataclass, field

import numpy as np

from apsg.feature import Ellipsoid, EllipsoidSet, Vector3
from apsg.feature._statistics import (
    anisotropy_test,
    compare_tensor_means,
    jelinek_shape,
    tensor_bootstrap,
    tensor_error_bootstrap,
)
from apsg.math._matrix import Matrix3
from apsg.paleomag._specimen import bedding_vector, core_rotation, untilt_rotation

COORDS = ("specimen", "geographic", "tectonic")
FRAMES = ("specimen", "geographic")

# AGICO order of the six independent components of the symmetric tensor
COMPONENTS = ("k11", "k22", "k33", "k12", "k23", "k13")


def tensor_from_components(k11, k22, k33, k12, k23, k13):
    """Return the symmetric 3x3 array of the six components (AGICO order)."""
    return np.array([[k11, k12, k13], [k12, k22, k23], [k13, k23, k33]], dtype=float)


def tensor_components(tensor):
    """Return ``(k11, k22, k33, k12, k23, k13)`` of a symmetric 3x3 tensor."""
    T = np.asarray(tensor, dtype=float)
    return (T[0, 0], T[1, 1], T[2, 2], T[0, 1], T[1, 2], T[0, 2])


def _check_coords(coords):
    if coords not in COORDS:
        raise ValueError(f"coords must be one of {COORDS}, got {coords!r}")


@dataclass(eq=False)
class AMSSpecimen:
    """Anisotropy of magnetic susceptibility of one specimen.

    Args:
        name: specimen name.
        tensor: symmetric 3x3 susceptibility tensor in the coordinates `frame`.
            It may be normalised (AGICO files give a tensor of mean 1 and the
            bulk susceptibility separately).
        bulk: mean susceptibility (``k_mean``) when it is not the mean of the
            tensor; used for reporting and weighting only.
        std_error: standard error of the tensor fit of the instrument
            (``sigma`` of the six-position design), used by parametric
            bootstraps.
        azimuth: core azimuth of the specimen z axis and
        hade: its hade (PMD convention of `apsg.paleomag.core_rotation`),
            required for geographic and tectonic coordinates of a tensor in
            specimen coordinates.
        bedding: ``(dip direction, dip)`` of the bedding, or the normal
            pointing to the stratigraphic top as `Vector3`; required for
            tectonic coordinates (dip up to 180 for overturned beds).
        site: site name.
        frame: ``"specimen"`` (default) or ``"geographic"``, the coordinates of
            `tensor`.
        meta: free dictionary.

    Examples:
        >>> s = AMSSpecimen("S1", tensor_from_components(1.1, 1.0, 0.9, 0.02, 0, 0))
        >>> s.params()["P"]  # doctest: +SKIP
    """

    name: str
    tensor: np.ndarray
    bulk: float | None = None
    std_error: float | None = None
    azimuth: float | None = None
    hade: float | None = None
    bedding: Vector3 | tuple | None = None
    site: str | None = None
    frame: str = "specimen"
    meta: dict = field(default_factory=dict)

    def __post_init__(self):
        T = np.array(self.tensor, dtype=float)
        if T.shape != (3, 3) or not np.all(np.isfinite(T)):
            raise ValueError("tensor must be a finite 3x3 matrix")
        if not np.allclose(T, T.T, atol=1e-9 * max(1.0, np.abs(T).max())):
            raise ValueError("tensor must be symmetric")
        self.tensor = (T + T.T) / 2
        if self.frame not in FRAMES:
            raise ValueError(f"frame must be one of {FRAMES}, got {self.frame!r}")
        if self.bedding is not None and not isinstance(self.bedding, Vector3):
            dipdir, dip = self.bedding
            self.bedding = bedding_vector(dipdir, dip)
        if (self.azimuth is None) != (self.hade is None):
            raise ValueError("azimuth and hade must be given together")

    def __repr__(self):
        return f"AMSSpecimen {self.name}: k_mean = {self.k_mean:.4g}"

    @property
    def k_mean(self):
        """float: mean susceptibility, `bulk` or the mean of the tensor."""
        return self.bulk if self.bulk is not None else float(np.trace(self.tensor) / 3)

    @property
    def normalized(self):
        """ndarray: the tensor divided by its mean ``trace / 3``."""
        return self.tensor / (np.trace(self.tensor) / 3)

    def rotation(self, coords):
        """Return the `Rotation3` from the tensor `frame` to `coords`.

        Raises:
            ValueError: if the orientation or the bedding needed is missing.
        """
        _check_coords(coords)
        R = np.eye(3)
        if coords == "specimen":
            if self.frame != "specimen":
                raise ValueError("the tensor is not given in specimen coordinates")
            return Matrix3(R)
        if self.frame == "specimen":
            if self.azimuth is None:
                raise ValueError(f"{self.name}: core orientation is not known")
            R = np.asarray(core_rotation(self.azimuth, self.hade))
        if coords == "tectonic":
            if self.bedding is None:
                raise ValueError(f"{self.name}: bedding is not known")
            R = np.asarray(untilt_rotation(self.bedding)) @ R
        return Matrix3(R)

    def tensor_in(self, coords="specimen", normalize=False):
        """Return the tensor rotated to `coords` as a 3x3 array.

        Args:
            coords: ``"specimen"``, ``"geographic"`` (x north, y east, z down)
                or ``"tectonic"`` (geographic, bedding restored to horizontal).
            normalize: divide by the mean of the tensor.
        """
        R = np.asarray(self.rotation(coords))
        T = self.normalized if normalize else self.tensor
        return R @ T @ R.T

    def ellipsoid(self, coords="specimen", normalize=False):
        """Return the tensor as `Ellipsoid` (see `tensor_in`)."""
        return Ellipsoid(self.tensor_in(coords, normalize))

    def params(self, coords="specimen"):
        """Return the Jelinek (1981) parameters of the tensor.

        Returns:
            dict: ``k_mean``, ``L``, ``F``, ``P``, ``Pj``, ``T`` and ``U``
            (see `jelinek_shape`).
        """
        k = np.linalg.eigvalsh(self.tensor_in(coords))
        return {"k_mean": self.k_mean, **jelinek_shape(k)}

    def bootstrap(self, coords="specimen", n_resamples=1000, level=0.95, rng=None):
        """Bootstrap the uncertainty of the tensor from its measurement error.

        Normal noise of standard deviation `std_error` is added to the six
        components of the tensor in its own frame and the simulated tensors
        are normalised and rotated to `coords` (see
        ``tensor_error_bootstrap``).

        Returns:
            TensorBootstrap: eigenvalues, eigenvectors and regions of the axes.

        Raises:
            ValueError: if `std_error` is not known.
        """
        if self.std_error is None:
            raise ValueError(f"{self.name}: std_error is not known")
        return tensor_error_bootstrap(
            self.tensor,
            self.std_error,
            n_resamples=n_resamples,
            level=level,
            normalize=True,
            rng=rng,
            rotation=np.asarray(self.rotation(coords)),
        )

    def to_json(self):
        """Return as JSON dict."""
        return {
            "name": self.name,
            "tensor": [list(map(float, row)) for row in self.tensor],
            "bulk": self.bulk,
            "std_error": self.std_error,
            "azimuth": self.azimuth,
            "hade": self.hade,
            "bedding": None if self.bedding is None else list(map(float, self.bedding)),
            "site": self.site,
            "frame": self.frame,
            "meta": dict(self.meta),
        }

    @classmethod
    def from_json(cls, d):
        """Return ``AMSSpecimen`` from a dict of `to_json`."""
        d = dict(d)
        if d.get("bedding") is not None:
            d["bedding"] = Vector3(d["bedding"])
        return cls(**d)


class AMSSite:
    """Collection of ``AMSSpecimen`` of one site with group statistics.

    Group statistics use normalised tensors (each divided by its mean), as
    usual for AMS, and the classic linear perturbation method of Jelinek
    (1978) after Hext (1963).

    Args:
        specimens: iterable of ``AMSSpecimen``.
        name: site name.

    Examples:
        >>> site = AMSSite([AMSSpecimen("a", np.diag([1.1, 1, 0.9]))])
        >>> site.table()  # doctest: +SKIP
    """

    def __init__(self, specimens=(), name="Default"):
        self.specimens = list(specimens)
        if not all(isinstance(s, AMSSpecimen) for s in self.specimens):
            raise TypeError("AMSSite holds AMSSpecimen instances only")
        self.name = name

    def __repr__(self):
        return f"AMSSite {self.name}: {len(self)} specimens"

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
        """Add an ``AMSSpecimen``."""
        if not isinstance(specimen, AMSSpecimen):
            raise TypeError("AMSSite holds AMSSpecimen instances only")
        self.specimens.append(specimen)

    @property
    def names(self):
        """list: specimen names."""
        return [s.name for s in self.specimens]

    def split(self, key):
        """Return a dict of sites, grouping the specimens by ``key(specimen name)``.

        Args:
            key: function of the specimen name giving the site name, e.g.
                ``lambda n: n.split("_")[0]``.
        """
        out = {}
        for s in self.specimens:
            k = key(s.name)
            out.setdefault(k, AMSSite(name=k)).append(s)
            s.site = k
        return out

    def set_bedding(self, dipdir, dip):
        """Set the same bedding (dip direction, dip) for all specimens."""
        n = bedding_vector(dipdir, dip)
        for s in self.specimens:
            s.bedding = n

    def tensors(self, coords="specimen", normalize=True):
        """Return the specimen tensors as `EllipsoidSet` in `coords`.

        Args:
            coords: ``"specimen"``, ``"geographic"`` or ``"tectonic"``.
            normalize: divide each tensor by its mean (default True).
        """
        if not self.specimens:
            raise ValueError(f"site {self.name} has no specimens")
        return EllipsoidSet(
            [s.ellipsoid(coords, normalize) for s in self.specimens], name=self.name
        )

    def mean(self, coords="specimen", level=0.95, anisoft=False):
        """Return the mean tensor and confidence ellipses of its axes.

        See `EllipsoidSet.mean_tensor`; the tensors are normalised.
        """
        return self.tensors(coords).mean_tensor(
            level=level, normalize=True, anisoft=anisoft
        )

    def bootstrap(self, coords="specimen", n_resamples=1000, level=0.95, rng=None):
        """Bootstrap the mean tensor of the site (Constable and Tauxe, 1990).

        See `EllipsoidSet.bootstrap_tensor`; the tensors are normalised.
        """
        return tensor_bootstrap(
            [np.asarray(e) for e in self.tensors(coords)],
            n_resamples=n_resamples,
            level=level,
            rng=rng,
        )

    def compare(
        self,
        other,
        coords="specimen",
        method="hotelling",
        level=0.95,
        n_resamples=2000,
        rng=None,
    ):
        """Test whether this and another site have the same mean tensor.

        See `compare_tensor_means`; the tensors are normalised.

        Returns:
            TestResult: ``reject()`` is True when the mean tensors differ.
        """
        if not isinstance(other, AMSSite):
            raise TypeError("other must be an AMSSite")
        return compare_tensor_means(
            [np.asarray(e) for e in self.tensors(coords)],
            [np.asarray(e) for e in other.tensors(coords)],
            level=level,
            method=method,
            n_resamples=n_resamples,
            rng=rng,
        )

    def anisotropy_test(self, coords="specimen", level=0.95):
        """Return the Jelinek (1978) test that the mean tensor is isotropic."""
        return anisotropy_test(
            [np.asarray(e) for e in self.tensors(coords)], level=level
        )

    def table(self, coords="specimen"):
        """Return a `DataFrame` of the parameters of each specimen.

        Columns: ``name``, ``k_mean``, ``L``, ``F``, ``P``, ``Pj``, ``T``,
        ``U`` and the direction (``k1_azi``, ``k1_inc``, ...) of the principal
        axes in `coords`, as dip direction and plunge.
        """
        import pandas as pd

        rows = []
        for s in self.specimens:
            e = s.ellipsoid(coords, normalize=True)
            row = {"name": s.name, **s.params(coords)}
            for i, lin in enumerate(e.eigenlins(), 1):
                azi, inc = lin.geo
                row[f"k{i}_azi"], row[f"k{i}_inc"] = float(azi), float(inc)
            rows.append(row)
        return pd.DataFrame(rows)

    @classmethod
    def from_dataframe(cls, df, name="Default"):
        """Return the ``AMSSite`` of a table (see `from_dataframe`).

        Raises:
            ValueError: if the table has more than one site.
        """
        from apsg.ams._readers import from_dataframe

        sites = from_dataframe(df, name=name)
        if len(sites) != 1:
            raise ValueError(
                f"the table holds {len(sites)} sites ({', '.join(sites)}); "
                "use apsg.ams.from_dataframe"
            )
        return next(iter(sites.values()))

    @classmethod
    def from_csv(cls, source, name="Default", **kwargs):
        """Read one site from a CSV table (see `read_csv`)."""
        import pandas as pd

        return cls.from_dataframe(pd.read_csv(source, **kwargs), name=name)

    def to_dataframe(self):
        """Return a table of the specimens (see `to_dataframe`)."""
        from apsg.ams._readers import to_dataframe

        return to_dataframe(self)

    def to_csv(self, path=None, **kwargs):
        """Write the specimens as CSV (see `to_dataframe`); return text if `path` is None."""
        return self.to_dataframe().to_csv(path, index=False, **kwargs)

    def to_json(self):
        """Return as JSON dict."""
        return {
            "datatype": type(self).__name__,
            "kwargs": {"name": self.name, "specimens": [s.to_json() for s in self]},
        }

    @classmethod
    def from_json(cls, obj_json):
        """Return ``AMSSite`` from a dict of `to_json`."""
        kw = obj_json["kwargs"]
        return cls([AMSSpecimen.from_json(s) for s in kw["specimens"]], name=kw["name"])
