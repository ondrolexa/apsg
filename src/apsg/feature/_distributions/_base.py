"""
Common interface of the orientation distributions.

Densities are taken with respect to arc length on the circle and area on the
sphere. Axial distributions are symmetric densities on the full circle or
sphere, so the uniform density is ``1/2pi`` or ``1/4pi`` for both symmetries.
"""

import os
import warnings
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import ClassVar

import numpy as np

from apsg.feature._container import FeatureSet, G
from apsg.helpers._helper import check_count
from apsg.math._vector import Axial2, Axial3, Vector2, Vector3

_DISTRIBUTION_REGISTRY = {}

# warnings point to the first frame outside apsg
_APSG_DIR = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))


def finite_param(name, value):
    """Return parameter `value` as float, raising unless it is a finite number."""
    if isinstance(value, (str, bytes, bool, np.bool_)):
        raise TypeError(f"{name} must be a number, got {value!r}")
    try:
        x = float(value)
    except (TypeError, ValueError, OverflowError):
        raise TypeError(f"{name} must be a number, got {value!r}") from None
    if not np.isfinite(x):
        raise ValueError(f"{name} must be finite, got {value!r}")
    return x


def unit_vector(name, value):
    """Return `value` as a unit vector array, raising for non-finite or zero
    vectors."""
    v = np.asarray(value, dtype=float)
    if not np.all(np.isfinite(v)):
        raise ValueError(f"{name} must be finite, got {v}")
    norm = np.linalg.norm(v)
    if norm == 0:
        raise ValueError(f"{name} must not be a zero vector")
    return v / norm


def get_distribution(name, dim=None):
    """Return the distribution class registered under `name`.

    Args:
        name: class name (``"VonMisesFisher"``) or short name (``"fisher"``),
            case-insensitive, or a distribution class, returned unchanged.

    Keyword Args:
        dim: 2 or 3, to choose between distributions sharing a short name
            (e.g. ``"uniform"``). Default None.

    Returns:
        type: subclass of ``OrientationDistribution``.
    """
    if isinstance(name, type) and issubclass(name, OrientationDistribution):
        return name
    key = str(name).lower()
    named = [
        cls
        for cls in _DISTRIBUTION_REGISTRY.values()
        if key in (cls.__name__.lower(), cls.name)
    ]
    if not named:
        raise ValueError(f"Unknown distribution {name!r}")
    found = [cls for cls in named if dim is None or cls.dim == dim]
    if not found:
        kind = "pair" if dim == "pair" else f"{dim}D"
        raise ValueError(f"Distribution {name!r} is not {kind}")
    if len(found) > 1:
        names = ", ".join(cls.__name__ for cls in found)
        raise ValueError(f"Distribution {name!r} is ambiguous ({names}), give dim")
    return found[0]


def as_rng(rng=None):
    """Return a ``numpy.random.Generator``.

    Args:
        rng: None, an integer seed or a ``Generator``. With None the seed is
            drawn from numpy's global random state, so ``np.random.seed()``
            makes the result reproducible.
    """
    if isinstance(rng, np.random.Generator):
        return rng
    if rng is None:
        rng = np.random.randint(0, 2**31 - 1)
    return np.random.default_rng(rng)


def legacy_rng():
    """Return numpy's global ``RandomState``.

    Used by the ``random_*`` methods of the feature sets, so their samples stay
    the same for a given ``np.random.seed()`` as in earlier versions.
    """
    return np.random.mtrand._rand


@dataclass(frozen=True)
class FitInfo:
    """Diagnostics of a fitted distribution.

    Attributes:
        n: number of data used for the fit.
        method: fitting method.
        loglik: log-likelihood of the data under the fitted distribution.
        converged: False when a numerical solver did not converge.
        stats: sufficient statistics used by the fit, e.g. the mean resultant
            length or the eigenvalues of the orientation tensor.
    """

    n: int
    method: str
    loglik: float
    converged: bool = True
    stats: dict = field(default_factory=dict)


@dataclass(frozen=True)
class ConfidenceRegion:
    """Confidence region of the mean direction or axis.

    A cone has ``alpha``, the half-angle in degrees. An ellipse has ``axes``,
    the two directions spanning it, and ``gamma``, the semi-angles in degrees
    along them.
    """

    mu: object
    level: float
    alpha: float | None = None
    axes: tuple | None = None
    gamma: tuple | None = None

    @property
    def kind(self):
        return "cone" if self.axes is None else "ellipse"

    def as_dict(self):
        """Return the region in the layout of the ``*_statistics`` methods."""
        if self.kind == "cone":
            return {"mu": self.mu, "alpha": self.alpha}
        return {"mu": self.mu, "axes": self.axes, "gamma": self.gamma}


class OrientationDistribution(ABC):
    """Base class of probability distributions of orientations.

    Subclasses define the class attributes ``dim`` (2 or 3), ``axial``,
    ``n_params`` (number of free parameters, for AIC/BIC), ``name`` (short
    name) and ``feature`` (default feature class of samples), and implement
    ``logpdf_array``, ``rvs_array``, ``params`` and, if they can be fitted,
    ``_fit``.

    Data and evaluation points can be apsg features, feature sets or arrays
    of shape ``(dim,)`` or ``(N, dim)``. They are normalized to unit length.
    """

    dim: ClassVar[int]
    axial: ClassVar[bool]
    n_params: ClassVar[int]
    name: ClassVar[str]
    feature: ClassVar[type]

    fit_info = None

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        if "name" in cls.__dict__:
            _DISTRIBUTION_REGISTRY[cls.__name__] = cls

    # data handling

    @classmethod
    def as_array(cls, x):
        """Return `x` as an ``(N, dim)`` array of unit vectors."""
        arr = np.atleast_2d(np.asarray(x, dtype=float))
        if arr.ndim != 2 or arr.shape[1] != cls.dim:
            raise ValueError(
                f"{cls.__name__} expects {cls.dim}D data, got shape {arr.shape}"
            )
        norms = np.linalg.norm(arr, axis=1)
        if np.any(norms == 0):
            raise ValueError("Zero-length vectors have no orientation")
        return arr / norms[:, None]

    @staticmethod
    def _is_single(x):
        if isinstance(x, FeatureSet):
            return False
        if isinstance(x, (Vector2, Vector3)):
            return True
        return np.ndim(x) == 1

    # density

    @abstractmethod
    def logpdf_array(self, X):
        """Log-density at the rows of the ``(N, dim)`` unit vector array `X`."""

    def logpdf(self, x):
        """Log-density at `x` (feature, feature set or array).

        Returns:
            float for a single feature or vector, otherwise an array.
        """
        values = self.logpdf_array(self.as_array(x))
        return float(values[0]) if self._is_single(x) else values

    def pdf(self, x):
        """Density at `x` (feature, feature set or array).

        Returns:
            float for a single feature or vector, otherwise an array.
        """
        return np.exp(self.logpdf(x))

    def loglik(self, data):
        """Log-likelihood of `data`."""
        return float(np.sum(self.logpdf_array(self.as_array(data))))

    def aic(self, data):
        """Akaike information criterion of `data` under the distribution."""
        return 2 * self.n_params - 2 * self.loglik(data)

    def bic(self, data):
        """Bayesian information criterion of `data` under the distribution."""
        n = len(self.as_array(data))
        return self.n_params * np.log(n) - 2 * self.loglik(data)

    # sampling

    @abstractmethod
    def rvs_array(self, n, rng):
        """Return ``(n, dim)`` array of unit vectors sampled with Generator `rng`."""

    def rvs(self, n=100, *, rng=None, feature=None, name="Default"):
        """Return random sample of the distribution.

        Args:
            n: sample size. Default 100.

        Keyword Args:
            rng: None, an integer seed or a ``numpy.random.Generator``.
                Default None uses a seed drawn from numpy's global random state.
            feature: feature class of the sample, e.g. ``Foliation``. Default
                is the class attribute ``feature`` of the distribution. Samples
                of vectorial distributions can be axial features (the sign is
                then dropped), not the other way round.
            name: name of the returned set. Default 'Default'.

        Returns:
            FeatureSet: the sample.
        """
        n = check_count(n)
        rng = as_rng(rng)
        X = self.rvs_array(n, rng)
        if self.axial:
            X = X * rng.choice([-1.0, 1.0], size=(len(X), 1))
        feature = self.feature if feature is None else feature
        if not issubclass(feature, Vector2 if self.dim == 2 else Vector3):
            raise TypeError(f"{feature.__name__} is not a {self.dim}D feature")
        if self.axial and not issubclass(feature, (Axial2, Axial3)):
            # the sign of an axial sample is random
            raise TypeError(f"{feature.__name__} does not hold axial data")
        return G([feature(x) for x in X], name=name, feature=feature)

    # fitting

    @classmethod
    def fit(cls, data, *, method="mle", **options):
        """Fit the distribution to `data`.

        Args:
            data: feature set or ``(N, dim)`` array. Axial data (e.g.
                ``LineationSet``) given to a model of vectorial data are
                flipped into one hemisphere first, with a warning.

        Keyword Args:
            method: fitting method. Default "mle".

        Returns:
            OrientationDistribution: the fitted distribution, with fit
            diagnostics in ``fit_info``.
        """
        if not cls.axial and _is_axial_set(data):
            warnings.warn(
                f"{cls.__name__} is a model of vectorial data; the axial data of "
                f"{type(data).__name__} are fitted after flipping them into one "
                "hemisphere (halfspace), which is only adequate for clustered data.",
                UserWarning,
                skip_file_prefixes=(_APSG_DIR,),
            )
            data = data.halfspace()
        X = cls.as_array(data)
        cls._check_dispersion(X)
        params, converged, stats = cls._fit(X, method=method, **options)
        for key, value in params.items():
            if np.ndim(value) == 0 and not np.isfinite(value):
                raise ValueError(
                    f"Cannot fit {cls.__name__}: the data are degenerate (e.g. all "
                    f"on one great circle), the fitted {key} is {value}"
                )
        dist = cls(**params)
        dist.fit_info = FitInfo(
            n=len(X),
            method=method,
            loglik=float(np.sum(dist.logpdf_array(X))),
            converged=converged,
            stats=stats,
        )
        return dist

    @classmethod
    def _check_dispersion(cls, X):
        """Raise unless `X` has the spread needed to estimate the parameters
        (at least two data, not all equal), as the concentration of identical
        data is infinite."""
        if cls.n_params == 0:
            return
        if len(X) < 2:
            raise ValueError(
                f"{cls.__name__} needs at least 2 data to fit, got {len(X)}"
            )
        # cosines to the first datum (per vector of a pair frame); a pair, like
        # an axis, is the same after reversing its vectors
        cos = np.sum(X * X[0], axis=-1)
        if cls.axial:
            cos = np.abs(cos)
        if np.all(cos > 1 - 1e-12):
            raise ValueError(
                f"Cannot fit {cls.__name__}: all {len(X)} data are identical "
                "(zero dispersion, infinite concentration)"
            )

    @classmethod
    def _fit(cls, X, method="mle", **options):
        """Return ``(params, converged, stats)`` fitted to unit vector array `X`."""
        raise NotImplementedError(f"{cls.__name__} can't be fitted")

    # description

    @property
    @abstractmethod
    def params(self):
        """dict: parameters, which recreate the distribution as ``cls(**params)``."""

    @property
    def mean(self):
        """Mean direction or axis, None when the distribution has none."""
        return None

    def mean_confidence(self, level=0.95):
        """Confidence region of the mean of the data the distribution was fitted to.

        Args:
            level: confidence level. Default 0.95.

        Returns:
            ConfidenceRegion: the region.
        """
        if self.fit_info is None:
            raise ValueError("Confidence region needs a distribution fitted to data")
        return self._mean_confidence(level)

    def _mean_confidence(self, level):
        raise NotImplementedError(
            f"{type(self).__name__} has no confidence region of the mean"
        )

    # serialization

    def to_json(self):
        return {
            "datatype": type(self).__name__,
            "args": (),
            "kwargs": {k: _param_to_json(v) for k, v in self.params.items()},
        }

    @classmethod
    def from_json(cls, obj_json):
        dist_cls = _DISTRIBUTION_REGISTRY[obj_json["datatype"]]
        kwargs = {k: _param_from_json(v) for k, v in obj_json["kwargs"].items()}
        return dist_cls(**kwargs)

    def __eq__(self, other):
        if type(self) is not type(other):
            return NotImplemented
        a, b = self.params, other.params
        return a.keys() == b.keys() and all(np.allclose(a[k], b[k]) for k in a)

    def __hash__(self):
        return hash(type(self))

    def __repr__(self):
        args = ", ".join(f"{k}={_param_repr(v)}" for k, v in self.params.items())
        return f"{type(self).__name__}({args})"


def _is_axial_set(data):
    return isinstance(data, FeatureSet) and issubclass(
        data.__feature_class__, (Axial2, Axial3)
    )


def _param_to_json(value):
    if hasattr(value, "to_json"):
        return value.to_json()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


def _param_from_json(value):
    if isinstance(value, dict) and "datatype" in value:
        from apsg.feature import feature_from_json

        return feature_from_json(value)
    return value


def _param_repr(value):
    if isinstance(value, float | np.floating):
        return f"{value:.6g}"
    return repr(value)
