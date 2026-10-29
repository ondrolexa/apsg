"""Sets of 2D and 3D vectors and of linear and planar features."""

import csv
import warnings
from os.path import basename

import numpy as np
from scipy import stats
from scipy.spatial.distance import cdist
from scipy.spatial.transform import Rotation

from apsg.config import apsg_conf
from apsg.feature._container._base import (
    FeatureSet,
    G,
    _deprecated_alias,
    _per_item,
    _read_csv_columns,
)
from apsg.feature._distributions import (
    ACG,
    ESAG,
    AxialUniform2,
    AxialUniform3,
    AxialVonMises,
    Bingham,
    Cardioid,
    JonesPewsey,
    Kent,
    Uniform2,
    Uniform3,
    VonMises,
    VonMisesFisher,
    Watson,
    WrappedCauchy,
    WrappedNormal,
    axial_cross_moments,
    bingham_ellipse,
)
from apsg.feature._distributions._base import _APSG_DIR
from apsg.feature._distributions._circular import (
    _vonmises_kappa_approx,
    double_angles,
    upton_interval,
)
from apsg.feature._distributions._spherical import _tangent_basis, fisher_precision
from apsg.feature._geodata import (
    Direction,
    Foliation,
    Lineation,
    Pair,
)
from apsg.feature._hypothesis import (
    axial_common_mean_test,
    bingham_test,
    common_mean_test,
    equal_kappa_test,
    hodges_ajne_test,
    kuiper_test,
    rao_spacing_test,
    rayleigh_test,
    watson_u2_test,
)
from apsg.feature._tensor2 import OrientationTensor2
from apsg.feature._tensor3 import (
    OrientationTensor3,
    Rotation3,
)
from apsg.helpers._helper import check_count
from apsg.helpers._math import acosd
from apsg.math._vector import Axial2, Axial3, Vector2, Vector3


def _pairwise_angles(fset, dim, axial):
    """Angles in degrees between all pairs of items of `fset`, in the order of
    ``itertools.combinations``; axes (`axial`) are compared by absolute
    cosines."""
    X = np.array(fset, dtype=float).reshape(-1, dim)
    norms = np.linalg.norm(X, axis=1)
    if np.any(norms == 0):
        raise ValueError("The angle to a zero vector is undefined")
    U = X / norms[:, None]
    i, j = np.triu_indices(len(U), 1)
    cos = np.einsum("ij,ij->i", U[i], U[j])
    if axial:
        cos = np.abs(cos)
    return np.degrees(np.arccos(np.clip(cos, -1, 1)))


def _warn_axial(fset, name):
    """Warn that resultant statistic `name` of axial 3D data is deprecated."""
    if issubclass(fset.__feature_class__, Axial3):
        warnings.warn(
            f"{name}() of axial data ({type(fset).__name__}) is deprecated and will "
            "raise TypeError in apsg 2.2. Use the watson or bingham distributions or "
            "the orientation tensor (ortensor().strength, ...), or convert the axes "
            f"to vectors, e.g. .halfspace().to_vec().{name}().",
            FutureWarning,
            skip_file_prefixes=(_APSG_DIR,),
        )


def _samples(fset, others):
    """Unit vector arrays of `fset` and `others` (sets of the same kind) for
    the sample comparison tests; doubled angles for axes on the circle."""
    sets = (fset, *others)
    kind = type(fset)
    for other in others:
        if type(other) is not kind:
            raise TypeError(
                f"Cannot compare {kind.__name__} with {type(other).__name__}"
            )
    samples = [np.asarray(s.normalized(), dtype=float) for s in sets]
    if issubclass(fset.__feature_class__, Axial2):
        samples = [double_angles(x) for x in samples]
    return samples


def _hotelling_test(X, Y, mean_diff):
    """Two-sample Hotelling's T² test of equal means; return (T², p-value).

    `mean_diff` is passed in rather than computed from X and Y, because the
    mean of axial data is the set's resultant (``R()``), not a coordinate mean.
    """
    n, m = len(X), len(Y)
    if n + m <= 4:
        raise ValueError(
            "Hotelling requires sum of lengths of dataset to be greater than 4"
        )
    S_pooled = (
        (n - 1) * np.cov(X, rowvar=False) + (m - 1) * np.cov(Y, rowvar=False)
    ) / (n + m - 2)
    T2 = (n * m / (n + m)) * (mean_diff @ np.linalg.pinv(S_pooled) @ mean_diff)
    # T² of p=3 dimensional data scaled to an F(3, n + m - 4) statistic
    F_stat = T2 * (n + m - 4) / ((n + m - 2) * 3)
    return T2, float(1.0 - stats.f.cdf(F_stat, 3, n + m - 4))


def _energy_statistic(X, Y):
    """Energy distance statistic (Székely & Rizzo 2004) of two samples."""
    n, m = len(X), len(Y)
    XY = np.mean(cdist(X, Y))  # cross-distances
    XX = np.mean(cdist(X, X))  # within X
    YY = np.mean(cdist(Y, Y))  # within Y
    return float((n * m / (n + m)) * (2 * XY - XX - YY))


def _rbf_mmd2(X, Y, bandwidth):
    """Unbiased squared Maximum Mean Discrepancy with a Gaussian kernel."""
    n, m = len(X), len(Y)
    gamma = 1.0 / (2.0 * bandwidth**2)
    K_XX = np.exp(-gamma * cdist(X, X, metric="sqeuclidean"))
    K_YY = np.exp(-gamma * cdist(Y, Y, metric="sqeuclidean"))
    K_XY = np.exp(-gamma * cdist(X, Y, metric="sqeuclidean"))
    # unbiased: zero out diagonal contributions
    np.fill_diagonal(K_XX, 0.0)
    np.fill_diagonal(K_YY, 0.0)
    return K_XX.sum() / (n * (n - 1)) + K_YY.sum() / (m * (m - 1)) - 2.0 * K_XY.mean()


def _permutation_test(statistic, X, Y, n_permutations, rng):
    """Return the observed `statistic` of samples X, Y and its permutation
    p-value: the share of random relabellings of the pooled sample giving a
    statistic at least as large (counting the observed one)."""
    observed = statistic(X, Y)
    pooled = np.vstack([X, Y])
    n = len(X)
    count = 0
    for _ in range(n_permutations):
        perm = rng.permutation(len(pooled))
        if statistic(pooled[perm[:n]], pooled[perm[n:]]) >= observed:
            count += 1
    return observed, (count + 1) / (n_permutations + 1)


class Vector2Set(FeatureSet):
    """
    Class to store set of ``Vector2`` features.
    """

    __feature_class__ = Vector2

    def __init__(self, data, name="Default"):
        super().__init__(data, name=name)
        if not all(isinstance(obj, Vector2) for obj in data):
            raise TypeError("Data must be instances of Vector2")

    def __repr__(self):
        return f"V2({len(self)}) {self.name}"

    def __array__(self, dtype=None, copy=None):
        out = np.column_stack([self.x, self.y])
        return out if dtype is None else out.astype(dtype)

    def __abs__(self):
        """Returns array of euclidean norms."""

        return np.asarray([abs(e) for e in self])

    x = _per_item("x", "Return numpy array of x-components.")
    y = _per_item("y", "Return numpy array of y-components.")

    @property
    def direction(self):
        """Return array of direction angles."""

        return np.asarray([e.direction for e in self]).T

    def to_vec2(self):
        """Return ``Vector2Set`` object with all data converted to ``Vector2``."""

        return Vector2Set([Vector2(e) for e in self], name=self.name)

    def to_dir2(self):
        """Return ``Direction2Set`` object with all data converted to ``Direction``."""

        return Direction2Set([Direction(e) for e in self], name=self.name)

    def proj(self, vec):
        """Return projections of all features in ``Vector2Set`` onto vector."""

        return type(self)([e.project(vec) for e in self], name=self.name)

    def dot(self, vec):
        """Return array of dot products of all features in ``Vector2Set`` with vector."""

        return np.array([e.dot(vec) for e in self])

    def cross(self, other=None):
        """Return cross products of all features in ``Vector2Set``."""
        res = []
        if other is None:
            X = np.array(self, dtype=float).reshape(-1, 2)
            i, j = np.triu_indices(len(X), 1)
            return X[i, 0] * X[j, 1] - X[i, 1] * X[j, 0]
        elif isinstance(other, FeatureSet):
            self._check_same_length(other, "cross")
            res = [e.cross(f) for e, f in zip(self, other)]
        elif isinstance(other, Vector2):
            res = [e.cross(other) for e in self]
        else:
            raise TypeError("Wrong argument type!")
        return np.asarray(res)

    __pow__ = cross

    def angle(self, other=None):
        """Return angles of all data in ``Vector2Set`` object."""
        res = []
        if other is None:
            return _pairwise_angles(self, 2, self._axial)
        elif isinstance(other, FeatureSet):
            self._check_same_length(other, "angle")
            res = [e.angle(f) for e, f in zip(self, other)]
        elif isinstance(other, Vector2):
            res = [e.angle(other) for e in self]
        else:
            raise TypeError("Wrong argument type!")
        return np.asarray(res)

    def normalized(self):
        """Return ``Vector2Set`` object with normalized (unit length) elements."""

        return type(self)([e.normalized() for e in self], name=self.name)

    uv = normalized

    def transform(self, F, **kwargs):
        """Return affine transformation of all features ``Vector2Set`` by matrix 'F'.

        Args:
            F: Transformation matrix. Array-like value e.g. ``DeformationGradient3``

        Keyword Args:
            norm: normalize transformed features. True or False. Default False

        Returns:
            Vector2Set: The transformed feature set.
        """
        return type(self)([e.transform(F, **kwargs) for e in self], name=self.name)

    @property
    def _axial(self):
        return issubclass(self.__feature_class__, Axial2)

    def _doubled(self):
        """Return ``Vector2Set`` with doubled direction angles of axial data."""
        return Vector2Set([abs(e) * Vector2(2 * e.direction) for e in self])

    def R(self, mean=False):
        """Return resultant of data in ``Vector2Set`` object.

        Resultant is of same type as features in ``Vector2Set``. For axial
        data (``Direction2Set``) the angles are doubled before summation, so
        the direction of the resultant is half the direction of the doubled
        resultant and its length is the length of the doubled resultant.

        Args:
            mean: if True returns mean resultant. Default False
        Returns:
            Vector2: The resultant vector.
        """
        self._require(1, "R")
        if self._axial:
            R2 = sum(self._doubled())
            R = abs(R2) * self.__feature_class__(R2.direction / 2)
        else:
            R = sum(self)
        if mean:
            R = R / len(self)
        return R

    def fisher_statistics(self, level=0.95, *, weighted=False):
        """Fit von Mises distribution and return statistics.

        The concentration is the piecewise approximation of the
        maximum-likelihood estimate (Fisher, 1993, Statistical Analysis of
        Circular Data). The confidence interval of the mean direction is the
        large-sample approximation of Upton (1986), as given by Zar
        (Biostatistical Analysis, eq. 26.24-26.25). For axial data
        (``Direction2Set``) the angles are doubled, the distribution
        is fitted and the mean direction and the interval are halved.

        Args:
            level: confidence level. Default 0.95 for 95 %.
            weighted: weight the data by their magnitudes. Default False (unit
                vectors).
        Returns:
            dict: with keys ``mu`` (mean direction), ``k`` (concentration
                parameter; of the doubled angles for axial data), and ``alpha``
                (half-width of the confidence interval of the mean direction
                in degrees, 180 (90 for axial data) when the data are too
                dispersed to give one).

        Note:
            Calculated alpha for level 0.95 means, you can be 95% confident that
            the true population mean direction lies within alpha degrees of mu.
            ``VonMises.fit(data, method="approx")`` (``AxialVonMises`` for axial
            data) gives the fitted distribution itself.

            With ``weighted=True`` the mean is the direction of ``R()``, the
            concentration is estimated from the weighted mean resultant length
            ``|sum(w u)| / sum(w)`` and the interval uses Kish's effective sample
            size ``sum(w)^2 / sum(w^2)``. This is an approximation without a
            published source; for unit vectors it equals the unweighted result.
        """
        if weighted:
            self._require(2, "fisher_statistics")
            w = abs(self)
            Rbar = float(abs(self.R()) / w.sum())
            n_eff = float(w.sum() ** 2 / (w**2).sum())
            alpha = upton_interval(n_eff, Rbar, level)
            return {
                "mu": self.R().normalized(),
                "k": _vonmises_kappa_approx(Rbar),
                "alpha": alpha / 2 if self._axial else alpha,
            }
        model = AxialVonMises if self._axial else VonMises
        dist = model.fit(self, method="approx")
        alpha = dist.mean_confidence(level).alpha
        return {"mu": dist.mean, "k": dist.kappa, "alpha": alpha}

    def bootstrap_statistics(self, level=0.95, n=1000, *, weighted=False):
        """Bootstrap confidence interval of the mean direction.

        The data are resampled `n` times with replacement and the interval is
        the equal-tailed `level` percentile interval of the resampled mean
        directions, measured from the mean of the data. For axial data
        (``Direction2Set``) the mean axes are compared by their doubled angles.

        Args:
            level (float): confidence level. Default 0.95.
            n (int): number of bootstrap resamples. Default 1000.
            weighted (bool): weight the data by their magnitudes, i.e. the mean
                is the direction of ``R()``. Default False (unit vectors).

        Returns:
            dict: ``mu`` (mean direction), ``low`` and ``high`` (bounds of the
            interval in degrees, ``low <= mu.direction <= high`` up to a
            multiple of 360, or of 180 for axial data).

        Note:
            This is the basic percentile interval, not the pivotal (studentized)
            interval of Fisher (1993, sec. 4.4.4 and 8.3.2), which has a smaller
            coverage error for small samples. ``fisher_statistics`` gives the
            von Mises interval.
        """
        self._require(1, "bootstrap_statistics")
        scale = 2 if self._axial else 1
        data = self if weighted else self.normalized()
        mu = data.R()
        m = np.radians(scale * mu.direction)
        dev = np.array([scale * s.R().direction for s in data.bootstrap(n=n)])
        dev = (np.radians(dev) - m + np.pi) % (2 * np.pi) - np.pi
        tail = 100 * (1 - level) / 2
        low, high = np.degrees(np.percentile(dev, [tail, 100 - tail])) / scale
        return {
            "mu": mu.normalized(),
            "low": float(mu.direction + low),
            "high": float(mu.direction + high),
        }

    def csd(self):
        """Circular standard deviation ``sqrt(-2 ln(Rbar))`` (Mardia & Jupp,
        2000, eq. 2.3.11; Fisher, 1993, eq. 2.12).

        For axial data (``Direction2Set``) it is computed from the doubled
        angles and halved.

        Returns:
            float: circular standard deviation in degrees.
        """
        self._require(1, "csd")
        Rbar = abs(self.normalized().R(mean=True))
        if Rbar <= 0:
            return np.inf
        csd = float(np.degrees(np.sqrt(-2 * np.log(min(Rbar, 1.0)))))
        return csd / 2 if self._axial else csd

    def uniformity_test(self, level=0.95, method="rayleigh"):
        """Test of uniformity on the circle.

        Axes (``Direction2Set``) are tested by their doubled angles (Fisher,
        1993, eq. 2.36); the Rayleigh test of the doubled angles is the Bingham
        test on the circle (Mardia & Jupp, 2000, p. 233).

        Args:
            level: confidence level. Default 0.95 for 95 %.
            method: "rayleigh" (default; against a unimodal alternative,
                ``rayleigh_test``), "kuiper" (``kuiper_test``), "watson"
                (Watson's U², ``watson_u2_test``), "hodges_ajne"
                (``hodges_ajne_test``) or "rao" (Rao's spacing test,
                ``rao_spacing_test``). Kuiper, Watson and Rao tests detect any
                departure from uniformity, multimodal included.

        Returns:
            TestResult: ``reject()`` is True when the data are not uniform at
            the given confidence level.
        """
        self._require(1, "uniformity_test")
        theta = self._angles()
        if method == "rayleigh":
            return rayleigh_test(np.column_stack([np.cos(theta), np.sin(theta)]), level)
        tests = {
            "kuiper": kuiper_test,
            "watson": watson_u2_test,
            "hodges_ajne": hodges_ajne_test,
            "rao": rao_spacing_test,
        }
        if method not in tests:
            raise ValueError(
                f"Unknown method {method!r}, expected rayleigh, " + ", ".join(tests)
            )
        return tests[method](theta, level)

    def _angles(self):
        """Direction angles in radians, doubled for axial data."""
        X = np.asarray(self.normalized(), dtype=float)
        if self._axial:
            X = double_angles(X)
        return np.arctan2(X[:, 1], X[:, 0])

    def moment(self, p=1, centered=False):
        """Sample trigonometric moment of order `p`.

        ``m_p = mean(exp(i p theta))`` (Mardia & Jupp, 2000, eq. 2.4.2-2.4.4),
        about the zero direction or, if `centered`, about the mean direction.
        For axial data the doubled angles are used.

        Returns:
            tuple: length ``Rbar_p`` and direction ``theta_p`` in degrees.
        """
        self._require(1, "moment")
        theta = self._angles()
        if centered:
            theta = theta - np.angle(np.mean(np.exp(1j * theta)))
        m = np.mean(np.exp(1j * p * theta))
        return float(abs(m)), float(np.degrees(np.angle(m)) % 360)

    def circular_dispersion(self):
        """Circular dispersion ``delta = (1 - rho2) / (2 Rbar^2)``.

        ``rho2`` is the mean of ``cos 2(theta - mean)`` (Mardia & Jupp, 2000,
        sec. 2.3.4; Fisher, 1993, sec. 2.3). ``delta / n`` is the squared
        circular standard error of the mean direction, see
        ``dispersion_interval``. For axial data the doubled angles are used.
        """
        Rbar, _ = self.moment(1)
        if Rbar <= 0:
            return np.inf
        R2, t2 = self.moment(2, centered=True)
        return float((1 - R2 * np.cos(np.radians(t2))) / (2 * Rbar**2))

    def angular_deviation(self):
        """Angular deviation ``sqrt(2 (1 - Rbar))`` in degrees.

        An approximation of the circular standard deviation ``csd`` (Mardia &
        Jupp, 2000, eq. 2.3.12). For axial data it is halved.
        """
        Rbar, _ = self.moment(1)
        s = float(np.degrees(np.sqrt(2 * (1 - Rbar))))
        return s / 2 if self._axial else s

    def skewness(self):
        """Circular skewness ``Rbar_2 sin(theta_2 - 2 theta_1) / (1 - Rbar)^(3/2)``
        (Mardia & Jupp, 2000, eq. 2.4.12), from the moments about the zero
        direction; for axial data of the doubled angles."""
        Rbar, t1 = self.moment(1)
        R2, t2 = self.moment(2)
        if Rbar >= 1:
            return 0.0
        return float(R2 * np.sin(np.radians(t2 - 2 * t1)) / (1 - Rbar) ** 1.5)

    def kurtosis(self):
        """Circular kurtosis ``(Rbar_2 cos(theta_2 - 2 theta_1) - Rbar^4) / (1 -
        Rbar)^2`` (Mardia & Jupp, 2000, eq. 2.4.13); for axial data of the
        doubled angles."""
        Rbar, t1 = self.moment(1)
        R2, t2 = self.moment(2)
        if Rbar >= 1:
            return 0.0
        return float((R2 * np.cos(np.radians(t2 - 2 * t1)) - Rbar**4) / (1 - Rbar) ** 2)

    def median(self):
        """Circular median direction.

        The direction minimizing the mean arc distance to the data (Fisher,
        1993, sec. 2.3.2); it lies at a datum or at the antipode of one. Ties
        are resolved by taking the first candidate in the order of the data.
        For axial data the doubled angles are used and the result is halved.

        Returns:
            Vector2: the median, of the feature class of the set.
        """
        self._require(1, "median")
        theta = self._angles()
        cand = np.concatenate([theta, theta + np.pi])
        diff = np.abs(cand[:, None] - theta[None, :]) % (2 * np.pi)
        dist = np.minimum(diff, 2 * np.pi - diff).mean(axis=1)
        best = np.degrees(cand[np.argmin(np.round(dist, 12))])
        if self._axial:
            best = best / 2
        return self.__feature_class__(best % 360)

    def common_mean_test(self, *others, method=None, level=0.95):
        """Test whether this and other sets share the mean direction.

        Axes (``Direction2Set``) are compared by their doubled angles.

        Args:
            *others: one or more ``Vector2Set`` of the same kind.
            method: "watson_williams" (default), "uniform_scores" or
                "watson_u2" (two samples), see ``common_mean_test``.
            level: confidence level. Default 0.95.

        Returns:
            TestResult: ``reject()`` is True when the mean directions differ.
        """
        return common_mean_test(_samples(self, others), method=method, level=level)

    def equal_kappa_test(self, *others, method=None, level=0.95):
        """Test whether this and other sets share the concentration.

        Axes (``Direction2Set``) are compared by their doubled angles.

        Args:
            *others: one or more ``Vector2Set`` of the same kind.
            method: "f" (default for two sets) or "bartlett", see
                ``equal_kappa_test``.
            level: confidence level. Default 0.95.

        Returns:
            TestResult: ``reject()`` is True when the concentrations differ.
        """
        return equal_kappa_test(_samples(self, others), method=method, level=level)

    def var(self):
        """Circular variance ``V = 1 - Rbar`` (Mardia & Jupp, 2000, eq. 2.3.3).

        For axial data (``Direction2Set``) ``Rbar`` is that of the doubled
        angles. Some authors use ``2 (1 - Rbar)``.
        """
        self._require(1, "var")
        return 1 - abs(self.normalized().R(mean=True))

    def delta(self):
        """Angle whose cosine is the mean resultant length, in degrees.

        For axial data (``Direction2Set``) it is computed from the doubled
        angles and halved.
        """
        self._require(1, "delta")
        delta = acosd(abs(self.R(mean=True)))
        return delta / 2 if self._axial else delta

    def rdegree(self):
        """Degree of preferred orientation ``100 (2 R - N) / N`` in percent."""
        self._require(1, "rdegree")
        N = len(self)
        return 100 * (2 * abs(self.normalized().R()) - N) / N

    def ortensor(self):
        """Return orientation tensor ``OrientationTensor2`` of ``Vector2Set``."""
        self._require(1, "ortensor")

        return self._ortensor

    @property
    def _ortensor(self):
        if "ortensor" not in self._cache:
            self._cache["ortensor"] = OrientationTensor2.from_features(self)
        return self._cache["ortensor"]

    @property
    def _svd(self):
        if "svd" not in self._cache:
            self._cache["svd"] = np.linalg.svd(self._ortensor)
        return self._cache["svd"]

    def halfspace(self):
        """Change orientation of vectors in ``Vector2Set``, so all have angle<=90 with."""
        arr = np.array(self)
        while True:
            resultant = arr.sum(axis=0)
            needs_flip = (arr @ resultant) < 0
            if not needs_flip.any():
                break
            arr[needs_flip] *= -1
        return type(self)([self.__feature_class__(row) for row in arr], name=self.name)

    @classmethod
    def from_directions(cls, angles, name="Default"):
        """Create ``Vector2Set`` object from arrays of direction angles.

        Args:
            angles: list or angles.

        Keyword Args:
            name: name of ``Vector2Set`` object. Default is 'Default'.

        Examples:
            >>> f = vec2set.from_angles([120,130,140,125, 132. 131])
        Returns:
            Vector2Set: The created feature set.
        """
        return cls([cls.__feature_class__(a) for a in angles], name=name)

    @classmethod
    def from_xy(cls, x, y, name="Default"):
        """Create ``Vector2Set`` object from arrays of x and y components.

        Args:
            x: list or array of x components.
            y: list or array of y components.

        Keyword Args:
            name: name of ``Vector2Set`` object. Default is 'Default'.

        Examples:
            >>> v = vec2set.from_xy([-0.4330127, -0.4330127, -0.66793414],
                                    [0.75, 0.25, 0.60141061])
        Returns:
            Vector2Set: The created feature set.
        """
        return cls([cls.__feature_class__(xx, yy) for xx, yy in zip(x, y)], name=name)

    @classmethod
    def _uniform_model(cls):
        return Uniform2()

    @classmethod
    def random_vonmises(
        cls, n=100, mean=None, kappa=5, name="Default", *, rng=None, position=None
    ):
        """Return a set of random vectors sampled from the von Mises
        distribution ([`VonMises`][apsg.feature.VonMises]).

        Args:
            n: number of features. Default 100.
            mean: mean direction given as azimuth. Default 0.
            kappa: concentration. Default 5.
            name: name of the set. Default 'Default'.

        Keyword Args:
            rng: None, an integer seed or a ``numpy.random.Generator``.
                Default None uses numpy's global random state.
            position: deprecated alias of `mean`.

        Examples:
            >>> v = vec2set.random_vonmises(mean=120, kappa=10)

        Returns:
            Vector2Set: the sample.
        """
        mean = _deprecated_alias(mean, position, "position", "mean", "random_vonmises")
        mean = 0 if mean is None else mean
        return cls._sample(VonMises(mean, kappa), n, rng, name)

    @classmethod
    def random_wrapped_normal(cls, n=100, mean=0, rho=0.5, name="Default", *, rng=None):
        """Return a set of random vectors sampled from the wrapped normal
        distribution ([`WrappedNormal`][apsg.feature.WrappedNormal]).

        Args:
            n: number of features. Default 100.
            mean: mean direction given as azimuth. Default 0.
            rho: mean resultant length, ``0 < rho < 1``. Default 0.5.
            name: name of the set. Default 'Default'.

        Keyword Args:
            rng: None, an integer seed or a ``numpy.random.Generator``.
                Default None uses numpy's global random state.

        Returns:
            Vector2Set: the sample.
        """
        return cls._sample(WrappedNormal(mean, rho), n, rng, name)

    @classmethod
    def random_wrapped_cauchy(cls, n=100, mean=0, rho=0.5, name="Default", *, rng=None):
        """Return a set of random vectors sampled from the wrapped Cauchy
        distribution ([`WrappedCauchy`][apsg.feature.WrappedCauchy]).

        Args:
            n: number of features. Default 100.
            mean: mean direction given as azimuth. Default 0.
            rho: mean resultant length, ``0 <= rho < 1``. Default 0.5.
            name: name of the set. Default 'Default'.

        Keyword Args:
            rng: None, an integer seed or a ``numpy.random.Generator``.
                Default None uses numpy's global random state.

        Returns:
            Vector2Set: the sample.
        """
        return cls._sample(WrappedCauchy(mean, rho), n, rng, name)

    @classmethod
    def random_cardioid(cls, n=100, mean=0, rho=0.25, name="Default", *, rng=None):
        """Return a set of random vectors sampled from the cardioid
        distribution ([`Cardioid`][apsg.feature.Cardioid]).

        Args:
            n: number of features. Default 100.
            mean: mean direction given as azimuth. Default 0.
            rho: mean resultant length, ``0 <= rho <= 0.5``. Default 0.25.
            name: name of the set. Default 'Default'.

        Keyword Args:
            rng: None, an integer seed or a ``numpy.random.Generator``.
                Default None uses numpy's global random state.

        Returns:
            Vector2Set: the sample.
        """
        return cls._sample(Cardioid(mean, rho), n, rng, name)

    @classmethod
    def random_jones_pewsey(
        cls, n=100, mean=0, kappa=1, psi=0, name="Default", *, rng=None
    ):
        """Return a set of random vectors sampled from the Jones-Pewsey
        distribution ([`JonesPewsey`][apsg.feature.JonesPewsey]).

        Args:
            n: number of features. Default 100.
            mean: mean direction given as azimuth. Default 0.
            kappa: concentration. Default 1.
            psi: shape (0 von Mises, 1 cardioid, -1 wrapped Cauchy).
                Default 0.
            name: name of the set. Default 'Default'.

        Keyword Args:
            rng: None, an integer seed or a ``numpy.random.Generator``.
                Default None uses numpy's global random state.

        Returns:
            Vector2Set: the sample.
        """
        return cls._sample(JonesPewsey(mean, kappa, psi), n, rng, name)


class Direction2Set(Vector2Set):
    """
    Class to store set of ``Direction`` features.
    """

    __feature_class__ = Direction

    def __init__(self, data, name="Default"):
        super().__init__(data, name=name)
        if not all(isinstance(obj, Direction) for obj in data):
            raise TypeError("Data must be instances of Direction")

    def __repr__(self):
        return f"D2({len(self)}) {self.name}"

    @classmethod
    def _uniform_model(cls):
        return AxialUniform2()

    @classmethod
    def random_axial_vonmises(cls, n=100, mean=0, kappa=1, name="Default", *, rng=None):
        """Return a set of random directions sampled from the axial von Mises
        distribution ([`AxialVonMises`][apsg.feature.AxialVonMises]).

        Args:
            n: number of features. Default 100.
            mean: mean axis given as azimuth. Default 0.
            kappa: concentration. Default 1.
            name: name of the set. Default 'Default'.

        Keyword Args:
            rng: None, an integer seed or a ``numpy.random.Generator``.
                Default None uses numpy's global random state.

        Returns:
            Direction2Set: the sample.
        """
        return cls._sample(AxialVonMises(mean, kappa), n, rng, name)


class Vector3Set(FeatureSet):
    """
    Class to store set of ``Vector3`` features.
    """

    __feature_class__ = Vector3

    def __init__(self, data, name="Default"):
        super().__init__(data, name=name)
        if not all(isinstance(obj, Vector3) for obj in data):
            raise TypeError("Data must be instances of Vector3")

    def __repr__(self):
        return f"V3({len(self)}) {self.name}"

    def __array__(self, dtype=None, copy=None):
        out = np.column_stack([self.x, self.y, self.z])
        return out if dtype is None else out.astype(dtype)

    def __abs__(self):
        """Returns array of euclidean norms."""

        return np.asarray([abs(e) for e in self])

    x = _per_item("x", "Return numpy array of x-components.")
    y = _per_item("y", "Return numpy array of y-components.")
    z = _per_item("z", "Return numpy array of z-components.")

    @property
    def geo(self):
        """Return arrays of azi and inc according to apsg_conf.notation."""

        azi, inc = np.asarray([e.geo for e in self]).T
        return azi, inc

    def to_lin(self):
        """Return ``LineationSet`` object with all data converted to ``Lineation``."""

        return LineationSet([Lineation(e) for e in self], name=self.name)

    def to_fol(self):
        """Return ``FoliationSet`` object with all data converted to ``Foliation``."""

        return FoliationSet([Foliation(e) for e in self], name=self.name)

    def to_vec(self):
        """Return ``Vector3Set`` object with all data converted to ``Vector3``."""

        return Vector3Set([Vector3(e) for e in self], name=self.name)

    def project(self, vec):
        """Return projections of all features in ``FeatureSet`` onto vector."""

        return type(self)([e.project(vec) for e in self], name=self.name)

    proj = project

    def reject(self, vec):
        """Return rejections of all features in ``FeatureSet`` onto vector."""

        return type(self)([e.reject(vec) for e in self], name=self.name)

    def dot(self, vec):
        """Return array of dot products of all features in ``FeatureSet`` with vector."""

        return np.array([e.dot(vec) for e in self])

    def cross(self, other=None):
        """Return cross products of all features in ``FeatureSet``."""
        res = []
        # the feature class of the products, also for no products at all
        feature = type(
            self.__feature_class__(1, 0, 0).cross(self.__feature_class__(0, 1, 0))
        )
        if other is None:
            X = np.array(self, dtype=float).reshape(-1, 3)
            i, j = np.triu_indices(len(X), 1)
            res = [feature(c) for c in np.cross(X[i], X[j])]
        elif isinstance(other, FeatureSet):
            self._check_same_length(other, "cross")
            res = [e.cross(f) for e, f in zip(self, other)]
        elif isinstance(other, Vector3):
            res = [e.cross(other) for e in self]
        else:
            raise TypeError("Wrong argument type!")
        return G(res, name=self.name, feature=feature)

    __pow__ = cross

    def angle(self, other=None):
        """Return angles of all data in ``FeatureSet`` object."""
        res = []
        if other is None:
            return _pairwise_angles(self, 3, issubclass(self.__feature_class__, Axial3))
        elif isinstance(other, FeatureSet):
            self._check_same_length(other, "angle")
            res = [e.angle(f) for e, f in zip(self, other)]
        elif isinstance(other, Vector3):
            res = [e.angle(other) for e in self]
        else:
            raise TypeError("Wrong argument type!")
        return np.asarray(res)

    def normalized(self):
        """Return ``FeatureSet`` object with normalized (unit length) elements."""

        return type(self)([e.normalized() for e in self], name=self.name)

    uv = normalized

    def transform(self, F, **kwargs):
        """Return affine transformation of all features ``FeatureSet`` by matrix 'F'.

        Args:
            F: Transformation matrix. Array-like value e.g. ``DeformationGradient3``

        Keyword Args:
            norm: normalize transformed features. True or False. Default False

        Returns:
            FeatureSet: The transformed feature set.
        """
        return type(self)([e.transform(F, **kwargs) for e in self], name=self.name)

    def is_upper(self):
        """Return boolean array of z-coordinate negative test."""

        return np.asarray([e.is_upper() for e in self])

    def R(self, mean=False):
        """Return resultant of data in ``FeatureSet`` object.

        Resultant is of same type as features in ``FeatureSet``. For axial
        data (``LineationSet``, ``FoliationSet``) it is the principal axis of
        the orientation tensor, with the length of the resultant of the axes
        oriented to the side of the principal axis (``halfspace``), so it does
        not depend on the signs the axes are stored with. It describes
        clustered axial data only; the orientation tensor and the ``Watson``
        and ``Bingham`` distributions describe axial data in general.

        Args:
            mean: if True returns mean resultant. Default False
        Returns:
            Vector3: The resultant vector.
        """
        self._require(1, "R")
        if issubclass(self.__feature_class__, Axial3):
            axis = self._principal_axis()
            folded = np.array(self.halfspace(), dtype=float).sum(axis=0)
            R = self.__feature_class__(np.linalg.norm(folded) * axis)
        else:
            R = sum(self)
        if mean:
            R = R / len(self)
        return R

    def fisher_statistics(self, level=0.95):
        """Fisher statistics: mean direction, precision and confidence cone.

        The precision is Fisher's ``k = (N - 1) / (N - R)`` (Fisher, 1953, p.
        303; Fisher, Lewis & Embleton, 1987, eq. 5.25), the confidence cone
        ``alpha`` (alpha95 for ``level=0.95``) is the cone of Fisher (1953),
        ``cos(alpha) = 1 - (N - R) / R ((1 - level)^(-1 / (N - 1)) - 1)``.

        Args:
            level: confidence level. Default 0.95 for 95 %.
        Returns:
            dict: with keys ``mu`` (mean direction), ``k`` (Fisher's
                precision), ``kappa`` (maximum-likelihood estimate of the
                concentration, ``coth(kappa) - 1/kappa = Rbar``) and ``alpha``
                (confidence cone half-angle in degrees).

        Note:
            Calculated alpha for level 0.95 means, you can be 95% confident that
            the true population mean direction lies within alpha degrees of mu.
            ``VonMisesFisher.fit(data)`` gives the fitted distribution
            (``method="precision"`` for ``k``). Before apsg 2.1 ``k`` was the
            maximum-likelihood estimate, now ``kappa``.

            Deprecated for axial data (``LineationSet``, ``FoliationSet``),
            which are flipped to the side of their principal axis
            (``halfspace``), adequate for clustered data only. It will raise
            ``TypeError`` in apsg 2.2; use ``watson_statistics``,
            ``bingham_statistics`` or convert the axes to vectors,
            ``.halfspace().to_vec()``.

        References:
            Fisher, R.A. (1953). Dispersion on a sphere. Proceedings of the
                Royal Society of London A 217, 295-305.
        """
        _warn_axial(self, "fisher_statistics")
        return self._fisher_statistics(level)

    def _fisher_statistics(self, level=0.95):
        data = self.normalized()
        if issubclass(self.__feature_class__, Axial3):
            data = data.halfspace()
        X = np.asarray(data, dtype=float)
        dist = VonMisesFisher.fit(X)
        R = float(np.linalg.norm(X.sum(axis=0)))
        alpha = dist.mean_confidence(level).alpha
        return {
            "mu": dist.mean,
            "k": fisher_precision(len(X), R),
            "kappa": dist.kappa,
            "alpha": alpha,
        }

    def csd(self):
        """Angular standard deviation ``81 / sqrt(k)`` in degrees.

        The approximate radius of the cone about the mean containing 63 % of
        the directions of a Fisher distribution with precision ``k``
        (``fisher_statistics``), theta63 (Borradaile, 2003, Statistics of
        Earth Science Data, p. 271).

        Deprecated for axial data, see ``fisher_statistics``.

        Returns:
            float: angular standard deviation in degrees.
        """
        self._require(1, "csd")
        _warn_axial(self, "csd")
        k = self._fisher_statistics()["k"]
        return 81 / np.sqrt(k) if k > 0 else np.inf

    def uniformity_test(self, level=0.95):
        """Test of uniformity.

        Vectors are tested by the Rayleigh test against a preferred direction
        (``rayleigh_test``, Mardia & Jupp, 2000, eq. 10.4.4-10.4.6), axes
        (``LineationSet``, ``FoliationSet``) by the Bingham test against a
        preferred orientation, a cluster or a girdle (``bingham_test``, Mardia &
        Jupp, 2000, eq. 10.7.1-10.7.3). The Rayleigh test can't detect axial
        alternatives, as the resultant of antipodal data vanishes.

        Args:
            level: confidence level. Default 0.95 for 95 %.

        Returns:
            TestResult: ``reject()`` is True when the data are not uniform at
            the given confidence level.
        """
        self._require(1, "uniformity_test")
        if issubclass(self.__feature_class__, Axial3):
            return bingham_test(self.ortensor(), len(self), level)
        return rayleigh_test(np.asarray(self.normalized(), dtype=float), level)

    def watson_statistics(self, level=0.95):
        """Fit Watson distribution and return statistics.

        Fits both the bipolar (``k > 0``, mean axis is the major eigenvector of
        the orientation tensor) and the girdle (``k < 0``, mean axis is the
        minor eigenvector, i.e. the pole of the girdle) form by maximum
        likelihood and returns the one with the higher likelihood. The
        normalizing constant ``M(1/2, 3/2, k)`` (Kummer's function) is evaluated
        in closed form through Dawson's integral (``k > 0``) and the error
        function (``k < 0``).

        The confidence cone is the large-sample cone of Fisher, Lewis &
        Embleton (1987, eq. 6.11-6.13), ``sin^2(alpha) = -ln(1 - level)
        sigma^2``, with the variance ``sigma^2`` of the sample axis from the
        moments of the data (see ``Watson``); it does not assume the Watson
        model.

        Args:
            level: confidence level. Default 0.95 for 95 %.
        Returns:
            dict: with keys ``mu`` (mean axis; the girdle pole when ``k < 0``),
                ``k`` (concentration parameter, negative for girdle
                distributions), and ``alpha`` (confidence cone half-angle).

        Note:
            Calculated alpha for level 0.95 means, you can be 95% confident that
            the true population mean axis lies within alpha degrees of the mu.
            ``Watson.fit(data)`` gives the fitted distribution.
        """
        dist = Watson.fit(self)
        mu = self.ortensor().eigenvectors(dist.fit_info.stats["which"])
        alpha = dist.mean_confidence(level).alpha
        return {"mu": mu, "k": dist.kappa, "alpha": alpha}

    def bingham_statistics(self, level=0.95, which=0):
        """Confidence ellipse of an eigenvector of the orientation tensor and
        concentrations of the fitted Bingham distribution.

        The ellipse is the large-sample (central limit theorem) approximation
        of the sampling distribution of the eigenvector, see
        ``apsg.feature._distributions.bingham_ellipse``; its long axis lies
        along the direction of the larger spread of the data. Under the
        Bingham model it tends to the ellipse of Bingham (1974, eq. 7.24).

        Args:
            level: confidence level. Default 0.95 for 95 %.
            which: index (0, 1 or 2) of the eigenvector the confidence ellipse
                is centered on. 0 is the major eigenvector, 2 is the minor
                (pole) eigenvector. Default 0.

        Returns:
            dict: with keys ``mu`` (center eigenvector), ``axes`` (tuple of
                the other two eigenvectors, spanning the ellipse), ``gamma``
                (tuple of the corresponding semi-angles in degrees), ``n``
                (sample size), ``level``, ``which`` and ``k`` (concentrations
                ``kappa_major`` and ``kappa_minor`` of the fitted ``Bingham``
                distribution, along the second and third eigenvectors).
        """

        ot = self.ortensor()
        vecs = ot.eigenvectors()
        moments = axial_cross_moments(np.asarray(self.normalized()), vecs)
        region = bingham_ellipse(
            ot.eigenvalues(), vecs, moments, len(self), level, which
        )
        dist = Bingham.fit(self)
        return region.as_dict() | {
            "n": len(self),
            "level": level,
            "which": which,
            "k": (dist.kappa_major, dist.kappa_minor),
        }

    def bootstrap_statistics(self, level=0.95, n=1000, method="cone"):
        """Bootstrap confidence region of the principal orientation.

        The mean is the principal eigenvector of the orientation tensor, which
        applies to axial as well as directional data. The data are resampled
        `n` times with replacement and the principal eigenvectors of the
        resamples, flipped to the side of the mean, give the region.

        Args:
            level (float): confidence level. Default 0.95.
            n (int): number of bootstrap resamples. Default 1000.
            method (str): "cone" (default), the `level` percentile of the
                angles between the mean and the resampled eigenvectors, or
                "ellipse", the region of Constable & Tauxe (1990, eq. 13-15):
                semi-angles ``arcsin(sigma_i sqrt(g))`` along the principal
                axes of the tangent-plane covariance of the resampled
                eigenvectors, ``g = -2 ln(1 - level) / mu^2`` with ``mu`` the
                length of their mean.

        Returns:
            dict: ``mu`` (mean orientation) and ``alpha`` (radius of the
            confidence cone in degrees) for "cone"; ``mu``, ``axes`` and
            ``gamma`` (as ``bingham_statistics``) for "ellipse".

        Note:
            With the moments of the bootstrap means the constant has no factor
            ``1 / n``; Tauxe (2003, eq. 4.16) prints ``g = -2 ln(0.05) / (n
            mu^2)``, which applies to moments of the data and covers about 10 %
            when used with bootstrap means.

        References:
            Constable, C. & Tauxe, L. (1990). The bootstrap for magnetic
                susceptibility tensors. Journal of Geophysical Research 95,
                8383-8395.
        """
        mu = self.ortensor().eigenvectors(0)
        m = np.asarray(mu, dtype=float)
        V = np.array(
            [np.asarray(s.ortensor().eigenvectors(0)) for s in self.bootstrap(n=n)]
        )
        # eigenvectors have an arbitrary sign
        V[V @ m < 0] *= -1
        match method:
            case "cone":
                angles = np.degrees(np.arccos(np.clip(V @ m, -1, 1)))
                return {"mu": mu, "alpha": float(np.percentile(angles, 100 * level))}
            case "ellipse":
                mean = V.mean(axis=0)
                length = np.linalg.norm(mean)
                center = mean / length
                e1, e2 = _tangent_basis(center)
                P = V @ np.column_stack([e1, e2])
                lam, vec = np.linalg.eigh(np.cov(P, rowvar=False))
                g = -2 * np.log(1 - level) / length**2
                axes, gamma = [], []
                for i in (1, 0):
                    axes.append(Vector3(vec[0, i] * e1 + vec[1, i] * e2))
                    arg = np.sqrt(max(lam[i], 0) * g)
                    gamma.append(
                        90.0 if arg >= 1 else float(np.degrees(np.arcsin(arg)))
                    )
                return {
                    "mu": Vector3(center),
                    "axes": tuple(axes),
                    "gamma": tuple(gamma),
                }
        raise ValueError(f"Unknown method {method!r}, expected cone or ellipse")

    def common_mean_test(self, *others, method=None, level=0.95):
        """Test whether this and other sets share the mean direction or axis.

        Vectors are compared by ``common_mean_test``, axes (``LineationSet``,
        ``FoliationSet``) by their principal axes, ``axial_common_mean_test``.
        ``similarity`` tests whether the whole distributions differ.

        Args:
            *others: one or more sets of the same kind.
            method: for vectors "watson" (default), "mcfadden_lowes" or
                "watson_debiche" (unequal concentrations); not used for axes.
            level: confidence level. Default 0.95.

        Returns:
            TestResult: ``reject()`` is True when the mean directions differ.
        """
        samples = _samples(self, others)
        if issubclass(self.__feature_class__, Axial3):
            if method is not None:
                raise ValueError("Axes are compared by the axial test only")
            return axial_common_mean_test(samples, level=level)
        return common_mean_test(samples, method=method, level=level)

    def equal_kappa_test(self, *others, method=None, level=0.95):
        """Test whether this and other sets of vectors share the concentration.

        Args:
            *others: one or more sets of the same kind.
            method: "f" (default for two sets) or "bartlett", see
                ``equal_kappa_test``.
            level: confidence level. Default 0.95.

        Returns:
            TestResult: ``reject()`` is True when the concentrations differ.
        """
        if issubclass(self.__feature_class__, Axial3):
            raise TypeError("equal_kappa_test compares sets of vectors, not axes")
        return equal_kappa_test(_samples(self, others), method=method, level=level)

    def var(self):
        """Spherical variance ``1 - Rbar``.

        Deprecated for axial data, see ``fisher_statistics``; the orientation
        tensor (e.g. ``1 - ortensor().eigenvalues(0)``) describes their spread.
        """
        self._require(1, "var")
        _warn_axial(self, "var")
        return 1 - abs(self.normalized().R(mean=True))

    def delta(self):
        """Angle ``arccos(Rbar)`` in degrees, from the mean resultant length.

        It decreases from 90 to 0 degrees as the directions cluster; for the
        radius of the cone containing 63 % of the data see ``csd``.

        Deprecated for axial data, see ``fisher_statistics``.
        """
        self._require(1, "delta")
        _warn_axial(self, "delta")
        return acosd(abs(self.R(mean=True)))

    def rdegree(self):
        """Degree of preferred orientation ``100 (2 R - N) / N`` in percent.

        Deprecated for axial data, see ``fisher_statistics``; the orientation
        tensor measures (``ortensor().strength``, ``.Intensity``, ...) describe
        the fabric of axial data.
        """
        self._require(1, "rdegree")
        _warn_axial(self, "rdegree")
        N = len(self)
        return 100 * (2 * abs(self.normalized().R()) - N) / N

    def ortensor(self):
        """Return orientation tensor ``Ortensor`` of ``Group``."""
        self._require(1, "ortensor")

        return self._ortensor

    @property
    def _ortensor(self):
        if "ortensor" not in self._cache:
            self._cache["ortensor"] = OrientationTensor3.from_features(self)
        return self._cache["ortensor"]

    @property
    def _svd(self):
        if "svd" not in self._cache:
            self._cache["svd"] = np.linalg.svd(self._ortensor)
        return self._cache["svd"]

    def centered(self, max_vertical=False):
        """Rotate ``FeatureSet`` object to position that eigenvectors are parallel
        to axes of coordinate system: E1||X (north-south), E2||X(east-west),
        E3||X(vertical).

        Args:
            max_vertical: If True E1 is rotated to vertical. Default False.

        Returns:
            FeatureSet: The rotated feature set.
        """
        if max_vertical:
            return self.transform(self._svd[2]).rotate(Vector3(0, -1, 0), 90)
        else:
            return self.transform(self._svd[2])

    def halfspace(self):
        """Change orientation of vectors in ``FeatureSet``, so all have angle<=90
        with the principal axis of the orientation tensor.

        The vectors are treated as axes: the result depends only on the axes, not
        on the signs they are stored with.

        Returns:
            FeatureSet: The flipped feature set.
        """
        if len(self) == 0:
            return self.copy()
        arr = np.array(self, dtype=float)
        arr[arr @ self._principal_axis() < 0] *= -1
        return type(self)([self.__feature_class__(row) for row in arr], name=self.name)

    def _principal_axis(self):
        """Principal eigenvector of the orientation tensor as an array, pointing
        to the lower hemisphere (if horizontal, to positive y, then x)."""
        v = np.asarray(self.ortensor().eigenvectors(0), dtype=float)
        ref = v[np.flatnonzero(np.abs(v) > 1e-12)[::-1][0]]
        return -v if ref < 0 else v

    def similarity(
        self,
        other,
        *,
        method="energy_distance",
        alpha=0.05,
        n_permutations=999,
        random_state=42,
        bandwidth=None,
    ):
        """Tests whether two sets of 3D vectors are sampled from the same distribution.
        H0 hypothesis is that two samples are from same distribution. For the
        parametric tests of a common mean direction or concentration see
        ``common_mean_test`` and ``equal_kappa_test``.

        Note on methods:

        Hotelling's T² Test: Parametric. Multivariate generalisation of the
            two-sample t-test. Sensitive to differences in mean but assumes
            multivariate normality.

        Energy Distance Test: Non-parametric permutation test based on the
            energy statistic (Székely & Rizzo 2004). Detects any distributional
            difference (mean, variance, shape, etc.).

        MMD Test (RBF kernel): Non-parametric permutation test using Maximum
            Mean Discrepancy with a Gaussian kernel. Also detects arbitrary
            distributional differences.

        Note:
            The vectors are compared as stored, as points in 3D. For axial data
            (``LineationSet``, ``FoliationSet``) a vector and its antipode are
            the same feature, but here they count as different, so the result
            depends on the signs the vectors happen to have. Give both sets
            consistent signs first, e.g. flip every vector of both sets to the
            same side of one common reference axis. ``halfspace()`` is not
            enough, as it flips each set towards its own principal axis.

        Args:
            other (Vector3Set): the set to compare with.

        Keyword Args:
            method (str): One of "hotelling", "energy_distance" or "mmd".
                Default "energy_distance".
            alpha (float): significance level. Default 0.05.
            n_permutations (int): Number of permutations. Default 999.
            random_state (int): Random state. Default 42.
            bandwidth (float): Bandwidth for RBF kernel. Default None for
                median heuristic.

        Returns:
            tuple: (statistic, p-value, same_distribution) where
            same_distribution is True when H0 cannot be rejected at the given
            ``alpha`` (p-value >= alpha), i.e. False for different distributions.
        """

        X, Y = np.array(self), np.array(other)
        rng = np.random.default_rng(random_state)
        match method:
            case "hotelling":
                mean_diff = X.mean(axis=0) - Y.mean(axis=0)
                statistic, p_value = _hotelling_test(X, Y, mean_diff)
            case "mmd":
                if bandwidth is None:
                    # median heuristic
                    pooled = np.vstack([X, Y])
                    dists = cdist(pooled, pooled, metric="euclidean")
                    bandwidth = float(np.median(dists[dists > 0]))
                statistic, p_value = _permutation_test(
                    lambda A, B: _rbf_mmd2(A, B, bandwidth), X, Y, n_permutations, rng
                )
            case "energy_distance":
                statistic, p_value = _permutation_test(
                    _energy_statistic, X, Y, n_permutations, rng
                )
            case _:
                raise ValueError(
                    f"Unknown method {method!r}, expected one of "
                    "'hotelling', 'energy_distance' or 'mmd'"
                )
        return float(statistic), float(p_value), p_value >= alpha

    def align(self, other):
        """Return best estimate rotation as `Rotation3` to align with others."""

        R = Rotation.align_vectors(np.array(other), np.array(self))[0]
        return Rotation3(R.as_matrix())

    @classmethod
    def from_csv(cls, filename, acol=0, icol=1, name=None, delimiter=None):
        """Create ``FeatureSet`` object from csv file of azimuths and inclinations.

        Args:
            filename (str): name of CSV file to load.

        Keyword Args:
            acol (int or str): azimuth column (starts from 0). Default 0.
            icol (int or str): inclination column (starts from 0). Default 1.
                When acol and icol are strings they are used as column headers.
            name (str): name of the created feature set. Default is the
                filename without its directory part.
            delimiter (str): values delimiter. Default None, i.e. detected
                from the file.

        Examples:
            >>> gf = folset.from_csv('file1.csv')                 #doctest: +SKIP
            >>> gl = linset.from_csv('file2.csv', acol=1, icol=2) #doctest: +SKIP
            >>> gl = linset.from_csv('file2.csv', name='My data') #doctest: +SKIP

        Returns:
            FeatureSet: The created feature set.
        """
        azi, inc = _read_csv_columns(filename, (acol, icol), delimiter)
        return cls.from_array(
            azi, inc, name=name if name is not None else basename(filename)
        )

    def to_csv(self, filename, delimiter=","):
        """Save ``FeatureSet`` object to csv file of azimuths and inclinations.

        Args:
            filename (str): name of CSV file to save.

        Keyword Args:
            delimiter (str): values delimiter. Default ','.

        Note:
            Written values are rounded according to `ndigits` settings in apsg_conf.

        Returns:
            None
        """

        n = apsg_conf.ndigits

        with open(filename, "w", newline="") as csvfile:
            fieldnames = ["azi", "inc"]
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames, delimiter=delimiter)
            writer.writeheader()
            for dt in self:
                azi, inc = dt.geo
                writer.writerow({"azi": round(azi, n), "inc": round(inc, n)})

    @classmethod
    def from_array(cls, azis, incs, name="Default"):
        """Create ``FeatureSet`` object from arrays of azimuths and inclinations.

        Args:
            azis: list or array of azimuths.
            incs: list or array of inclinations.

        Keyword Args:
            name: name of ``FeatureSet`` object. Default is 'Default'.

        Examples:
            >>> f = folset.from_array([120,130,140], [10,20,30])
            >>> l = linset.from_array([120,130,140], [10,20,30])
        Returns:
            FeatureSet: The created feature set.
        """
        if len(azis) != len(incs):
            raise ValueError(
                f"azis and incs must have the same length, got {len(azis)} and "
                f"{len(incs)}"
            )
        return cls(
            [cls.__feature_class__(azi, inc) for azi, inc in zip(azis, incs)], name=name
        )

    @classmethod
    def from_xyz(cls, x, y, z, name="Default"):
        """Create ``FeatureSet`` object from arrays of x, y and z components.

        Args:
            x: list or array of x components.
            y: list or array of y components.
            z: list or array of z components.

        Keyword Args:
            name: name of ``FeatureSet`` object. Default is 'Default'.

        Examples:
            >>> v = vecset.from_xyz([-0.4330127, -0.4330127, -0.66793414],
                                    [0.75, 0.25, 0.60141061],
                                    [0.5, 0.8660254, 0.43837115])
        Returns:
            FeatureSet: The created feature set.
        """
        return cls(
            [cls.__feature_class__(xx, yy, zz) for xx, yy, zz in zip(x, y, z)],
            name=name,
        )

    @classmethod
    def _uniform_model(cls):
        return Uniform3()

    @classmethod
    def random_normal(cls, n=100, position=None, sigma=20, name="Default"):
        """Method to create ``FeatureSet`` of normaly distributed features.

        Deprecated: the sample does not follow a defined distribution on the
        sphere; use ``random_fisher`` or ``random_kent``.

        Keyword Args:
            n: number of objects to be generated.
            position: mean orientation given as ``Vector3``. Default Vector3(0, 0, 1).
            sigma: sigma of normal distribution. Default 20.
            name: name of dataset. Default is 'Default'.

        Examples:
            >>> np.random.seed(58463123)
            >>> l = linset.random_normal(100, lin(120, 40))
            >>> l.R
            L:120/39

        Returns:
            FeatureSet: The created feature set.
        """
        warnings.warn(
            "random_normal is deprecated and will be removed, use random_fisher "
            "or random_kent instead.",
            FutureWarning,
            stacklevel=2,
        )
        n = check_count(n)
        if position is None:
            position = Vector3(0, 0, 1)
        data = []
        orig = Vector3(0, 0, 1)
        ax = orig.cross(position)
        ang = orig.angle(position)
        if not any(ax):  # position parallel to orig: any horizontal axis will do
            ax = Vector3(1, 0, 0)
        for s, r in zip(
            180 * np.random.uniform(low=0, high=180, size=n),
            np.random.normal(loc=0, scale=sigma, size=n),
        ):
            v = orig.rotate(Vector3(s, 0), r).rotate(ax, ang)
            data.append(cls.__feature_class__(v))
        return cls(data, name=name)

    @classmethod
    def random_fisher(
        cls, n=100, mean=None, kappa=20, name="Default", *, rng=None, position=None
    ):
        """Return a set of random vectors sampled from the von Mises-Fisher
        distribution ([`VonMisesFisher`][apsg.feature.VonMisesFisher]).

        Args:
            n: number of features. Default 100.
            mean: mean direction given as ``Vector3``. Default Vector3(0, 0, 1).
            kappa: concentration. Default 20.
            name: name of the set. Default 'Default'.

        Keyword Args:
            rng: None, an integer seed or a ``numpy.random.Generator``.
                Default None uses numpy's global random state.
            position: deprecated alias of `mean`.

        Examples:
            >>> l = linset.random_fisher(mean=lin(120, 50))

        Returns:
            FeatureSet: the sample.
        """
        mean = _deprecated_alias(mean, position, "position", "mean", "random_fisher")
        mean = Vector3(0, 0, 1) if mean is None else mean
        return cls._sample(VonMisesFisher(mean, kappa), n, rng, name)

    @classmethod
    def random_kent(
        cls,
        mean=None,
        n=100,
        kappa=20,
        beta=None,
        name="Default",
        *,
        major=None,
        rng=None,
        p=None,
    ):
        """Return a set of random vectors sampled from the Kent distribution
        ([`Kent`][apsg.feature.Kent]), the 5-parameter Fisher-Bingham
        distribution (Kent, 1982).

        Args:
            mean: mean direction given as ``Vector3`` (then `major` is needed),
                or a ``Pair`` whose lineation is the mean and whose in-plane
                direction perpendicular to it is the major axis.
            n: number of features. Default 100.
            kappa: concentration. Default 20.
            beta: ellipticity, ``0 <= beta < kappa / 2`` (the distribution is
                bimodal for larger ``beta``). Default ``kappa / 4``.
            name: name of the set. Default 'Default'.

        Keyword Args:
            major: major axis of the elliptical contours, perpendicular to
                `mean`. Only with a vector `mean`.
            rng: None, an integer seed or a ``numpy.random.Generator``.
                Default None uses numpy's global random state.
            p: deprecated alias of `mean`.

        Examples:
            >>> l = linset.random_kent(pair(150, 40, 150, 40), n=300, kappa=30)
            >>> v = vecset.random_kent(vec(120, 40), major=vec(30, 0), kappa=30)

        Returns:
            FeatureSet: the sample.
        """
        mean = _deprecated_alias(mean, p, "p", "mean", "random_kent")
        if mean is None:
            raise TypeError("random_kent() needs the mean direction or a Pair")
        if isinstance(mean, Pair):
            if major is not None:
                raise TypeError("major can't be given with a Pair")
            mean, major = mean.lvec, mean.fvec.cross(mean.lvec)
        elif major is None:
            raise TypeError("random_kent() needs major with a vector mean")
        if beta is None:
            beta = kappa / 4
        if beta < 0:
            raise ValueError("beta must be non-negative.")
        if 2 * beta >= kappa:
            warnings.warn(
                "Kent distribution with beta >= kappa / 2 is not unimodal.",
                UserWarning,
                stacklevel=2,
            )
        return cls._sample(Kent(mean, major, kappa, beta), n, rng, name)

    @classmethod
    def random_esag(
        cls, n=100, mu=(0, 0, 3), major=(1, 0, 0), rho=1, name="Default", *, rng=None
    ):
        """Return a set of random vectors sampled from the elliptically
        symmetric angular Gaussian distribution ([`ESAG`][apsg.feature.ESAG]).

        Args:
            n: number of features. Default 100.
            mu: mean vector of the underlying Gaussian; its direction is the
                mean direction and its length the concentration.
                Default (0, 0, 3).
            major: major axis of the elliptical contours. Default (1, 0, 0).
            rho: ellipticity, ``0 < rho <= 1`` (1 is rotationally symmetric).
                Default 1.
            name: name of the set. Default 'Default'.

        Keyword Args:
            rng: None, an integer seed or a ``numpy.random.Generator``.
                Default None uses numpy's global random state.

        Returns:
            FeatureSet: the sample.
        """
        return cls._sample(ESAG(mu, major, rho), n, rng, name)

    @classmethod
    def sfs(cls, n=100, name="Default"):
        """Deprecated alias of `uniform_sfs`."""
        warnings.warn(
            "sfs() is deprecated and will be removed in apsg 2.2, use uniform_sfs() "
            "instead.",
            FutureWarning,
            stacklevel=2,
        )
        return cls.uniform_sfs(n, name)

    @classmethod
    def gss(cls, n=100, name="Default"):
        """Deprecated alias of `uniform_gss`."""
        warnings.warn(
            "gss() is deprecated and will be removed in apsg 2.2, use uniform_gss() "
            "instead.",
            FutureWarning,
            stacklevel=2,
        )
        return cls.uniform_gss(n, name)

    @classmethod
    def uniform_sfs(cls, n=100, name="Default"):
        """Return a set of `n` evenly spread (not random) vectors, the
        Spherical Fibonacci Spiral points on a sphere, after John Burkardt.

        http://people.sc.fsu.edu/~jburkardt/

        Args:
            n: number of features. Default 100.
            name: name of the set. Default 'Default'.

        Examples:
            >>> v = vecset.uniform_sfs(300)
            >>> v.ortensor().eigenvalues()
            (0.3334645347163635, 0.33333474915201167, 0.33320071613162483)
        Returns:
            FeatureSet: The created feature set.
        """
        phi = (1 + np.sqrt(5)) / 2
        n = check_count(n, minimum=1)
        i2 = 2 * np.arange(n) - n + 1
        theta = 2 * np.pi * i2 / phi
        sp = i2 / n
        cp = np.sqrt((n + i2) * (n - i2)) / n
        dc = np.array([cp * np.sin(theta), cp * np.cos(theta), sp]).T
        return cls([cls.__feature_class__(d) for d in dc], name=name)

    @classmethod
    def uniform_gss(cls, n=100, name="Default"):
        """Return a set of `n` evenly spread (not random) vectors, the Golden
        Section Spiral points on a sphere.

        http://www.softimageblog.com/archives/115

        Args:
            n: number of features. Default 100.
            name: name of the set. Default 'Default'.

        Examples:
            >>> v = vecset.uniform_gss(300)
            >>> v.ortensor().eigenvalues()
            (0.33335688569571587, 0.33332315115436933, 0.33331996314991513)
        Returns:
            FeatureSet: The created feature set.
        """
        inc = np.pi * (3 - np.sqrt(5))
        n = check_count(n, minimum=1)
        off = 2 / n
        k = np.arange(n)
        y = k * off - 1 + (off / 2)
        r = np.sqrt(1 - y * y)
        phi = k * inc
        dc = np.array([np.cos(phi) * r, y, np.sin(phi) * r]).T
        return cls([cls.__feature_class__(d) for d in dc], name=name)


class _Axial3Samplers:
    """Generators of the axial models, shared by ``LineationSet`` and
    ``FoliationSet``."""

    __slots__ = ()

    @classmethod
    def _uniform_model(cls):
        return AxialUniform3()

    @classmethod
    def random_watson(cls, n=100, mean=(0, 0, 1), kappa=1, name="Default", *, rng=None):
        """Return a set of random axes sampled from the Watson distribution
        ([`Watson`][apsg.feature.Watson]).

        Args:
            n: number of features. Default 100.
            mean: mean axis (pole of the girdle for negative `kappa`).
                Default (0, 0, 1).
            kappa: concentration, positive for bipolar and negative for
                girdle distributions. Default 1.
            name: name of the set. Default 'Default'.

        Keyword Args:
            rng: None, an integer seed or a ``numpy.random.Generator``.
                Default None uses numpy's global random state.

        Examples:
            >>> l = linset.random_watson(mean=lin(120, 30), kappa=10)
            >>> f = folset.random_watson(mean=lin(120, 30), kappa=-20)  # girdle

        Returns:
            FeatureSet: the sample.
        """
        return cls._sample(Watson(mean, kappa), n, rng, name)

    @classmethod
    def random_bingham(
        cls,
        n=100,
        mean=(0, 0, 1),
        major=(1, 0, 0),
        kappa_major=-1,
        kappa_minor=-2,
        name="Default",
        *,
        rng=None,
    ):
        """Return a set of random axes sampled from the Bingham distribution
        ([`Bingham`][apsg.feature.Bingham]).

        Args:
            n: number of features. Default 100.
            mean: mean axis. Default (0, 0, 1).
            major: major axis, made perpendicular to `mean`. Default (1, 0, 0).
            kappa_major: concentration along the major axis. Default -1.
            kappa_minor: concentration along the minor axis. Default -2.
            name: name of the set. Default 'Default'.

        Keyword Args:
            rng: None, an integer seed or a ``numpy.random.Generator``.
                Default None uses numpy's global random state.

        Returns:
            FeatureSet: the sample.
        """
        dist = Bingham(mean, major, kappa_major, kappa_minor)
        return cls._sample(dist, n, rng, name)

    @classmethod
    def random_acg(
        cls,
        n=100,
        mean=(0, 0, 1),
        major=(1, 0, 0),
        shape=(2, 0.6, 0.4),
        name="Default",
        *,
        rng=None,
    ):
        """Return a set of random axes sampled from the angular central
        Gaussian distribution ([`ACG`][apsg.feature.ACG]).

        Args:
            n: number of features. Default 100.
            mean: mean axis. Default (0, 0, 1).
            major: major axis, made perpendicular to `mean`. Default (1, 0, 0).
            shape: eigenvalues ``(a1, a2, a3)``. Default (2, 0.6, 0.4).
            name: name of the set. Default 'Default'.

        Keyword Args:
            rng: None, an integer seed or a ``numpy.random.Generator``.
                Default None uses numpy's global random state.

        Returns:
            FeatureSet: the sample.
        """
        return cls._sample(ACG(mean, major, shape), n, rng, name)


class LineationSet(_Axial3Samplers, Vector3Set):
    """
    Class to store set of ``Lineation`` features.
    """

    __feature_class__ = Lineation

    def __init__(self, data, name="Default"):
        super().__init__(data, name=name)
        if not all(isinstance(obj, Lineation) for obj in data):
            raise TypeError("Data must be instances of Lineation")

    def __repr__(self):
        return f"L({len(self)}) {self.name}"


class FoliationSet(_Axial3Samplers, Vector3Set):
    """
    Class to store set of ``Foliation`` features.
    """

    __feature_class__ = Foliation

    def __init__(self, data, name="Default"):
        super().__init__(data, name=name)
        if not all(isinstance(obj, Foliation) for obj in data):
            raise TypeError("Data must be instances of Foliation")

    def __repr__(self):
        return f"S({len(self)}) {self.name}"

    def dipvec(self):
        """Return ``FeatureSet`` object with plane dip vector."""

        return Vector3Set([e.dipvec() for e in self], name=self.name)

    def strike(self):
        """Return ``Direction2Set`` object with strikes of planar features."""

        return Direction2Set([e.strike() for e in self], name=self.name)
