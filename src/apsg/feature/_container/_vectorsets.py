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
    _per_item,
    _read_csv_columns,
)
from apsg.feature._distributions import (
    AxialVonMises,
    Bingham,
    Kent,
    VonMises,
    VonMisesFisher,
    Watson,
    axial_cross_moments,
    bingham_ellipse,
    legacy_rng,
)
from apsg.feature._distributions._base import _APSG_DIR
from apsg.feature._geodata import (
    Direction,
    Foliation,
    Lineation,
    Pair,
)
from apsg.feature._tensor2 import OrientationTensor2
from apsg.feature._tensor3 import (
    OrientationTensor3,
    Rotation3,
)
from apsg.helpers._helper import check_count
from apsg.helpers._math import acosd
from apsg.math._vector import Axial2, Axial3, Vector2, Vector3

_WATSON_U2_ALPHA = np.array([0.005, 0.01, 0.025, 0.05, 0.10])


_WATSON_U2_CRITICAL = np.array([0.302, 0.267, 0.221, 0.187, 0.152])


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


def _watson_u2_test(w, level=0.95):
    """Watson's U² one-sample EDF test of uniformity on [0, 1].

    Uses the small-sample-corrected statistic and asymptotic critical values of
    Stephens (1970), "Use of the Kolmogorov-Smirnov, Cramér-von Mises and Related
    Statistics Without Extensive Tables," J. R. Statist. Soc. B.
    """
    w = np.sort(np.asarray(w, dtype=float))
    n = len(w)
    i = np.arange(1, n + 1)
    u2 = (
        np.sum((w - (2 * i - 1) / (2 * n)) ** 2)
        - n * (w.mean() - 0.5) ** 2
        + 1 / (12 * n)
    )
    u2_star = (u2 - 0.1 / n + 0.1 / n**2) * (1.0 + 0.8 / n)
    alpha = 1.0 - level
    critical_value = float(np.interp(alpha, _WATSON_U2_ALPHA, _WATSON_U2_CRITICAL))
    return {
        "statistic": float(u2_star),
        "critical_value": critical_value,
        "uniform": bool(u2_star <= critical_value),
    }


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

    def fisher_statistics(self, level=0.95):
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
        """
        model = AxialVonMises if self._axial else VonMises
        dist = model.fit(self, method="approx")
        alpha = dist.mean_confidence(level).alpha
        return {"mu": dist.mean, "k": dist.kappa, "alpha": alpha}

    def csd(self):
        """Circular standard deviation ``sqrt(-2 ln(Rbar))`` (Mardia, 1972).

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

    def uniformity_test(self, level=0.95):
        """Rayleigh test of uniformity against a preferred direction.

        Uses the large-sample chi-square approximation ``2 * N * Rbar**2 ~ chi2(df=2)``
        (Mardia & Jupp, 2000, Directional Statistics, sec. 10.4.1) for planar data.

        Args:
            level: confidence level for the uniformity decision. Default 0.95 for 95 %.

        Returns:
            dict: with keys ``R`` (resultant length), ``statistic`` (chi-square
                statistic), ``p_value``, and ``uniform`` (True if data is consistent
                with a uniform/random distribution at the given confidence level).
        """
        self._require(1, "uniformity_test")
        N = len(self)
        R = abs(self.normalized().R())
        statistic = 2 * R**2 / N
        p_value = float(stats.chi2.sf(statistic, df=2))
        return {
            "R": R,
            "statistic": statistic,
            "p_value": p_value,
            "uniform": bool(p_value >= 1.0 - level),
        }

    def var(self):
        """Circular variance based on resultant length (Mardia 1972)."""
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
        """Degree of preffered orientation of vectors in ``Vector2Set``."""
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
    def random(cls, n=100, name="Default"):
        """Method to create ``Vector2Set`` of features with uniformly distributed
        random orientation.

        Keyword Args:
            n: number of objects to be generated.
            name: name of dataset. Default is 'Default'.

        Examples:
            >>> np.random.seed(58463123)
            >>> l = vec2set.random(100)

        Returns:
            Vector2Set: The created feature set.
        """
        n = check_count(n)
        return cls([cls.__feature_class__.random() for i in range(n)], name=name)

    @classmethod
    def random_vonmises(cls, n=100, position=0, kappa=5, name="Default"):
        """Return ``Vector2Set`` of random vectors sampled from von Mises distribution
        around center position with concentration kappa.

        Args:
            n: number of objects to be generated.
            position: mean orientation given as angle. Default 0.
            kappa: concentration parameter of the distribution. Default 5.
            name: name of dataset. Default is 'Default'.

        Examples:
            >>> v = vec2set.random_vonmises(position=120, kappa=10)
        Returns:
            Vector2Set: The created feature set.
        """
        X = VonMises(position, kappa).rvs_array(check_count(n), legacy_rng())
        return cls([cls.__feature_class__(x) for x in X], name=name)


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

    def uniformity_test(self, level=0.95):
        """Rayleigh's uniformity test for axial data, via angle doubling.

        Doubles each direction angle (mod 360°) to map the 180°-periodic axial data
        onto a full circle before applying the standard circular Rayleigh test - see
        ``Vector2Set.uniformity_test``.

        Args:
            level: confidence level for the uniformity decision. Default 0.95.

        Returns:
            dict: with keys ``R``, ``statistic``, ``p_value`` and ``uniform``.
        """
        self._require(1, "uniformity_test")
        doubled = Vector2Set([Vector2(2 * e.direction) for e in self])
        return doubled.uniformity_test(level=level)


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
        """Fit Fisher distribution and return statistics.

        Args:
            level: confidence level. Default 0.95 for 95 %.
        Returns:
            dict: with keys ``mu`` (mean axis), ``k`` (precision parameter), and
                ``alpha`` (confidence cone half-angle).

        Note:
            Calculated alpha for level 0.95 means, you can be 95% confident that
            the true population mean axis lies within alpha degrees of the mu.
            ``VonMisesFisher.fit(data)`` gives the fitted distribution.

            Deprecated for axial data (``LineationSet``, ``FoliationSet``),
            which are flipped to the side of their principal axis
            (``halfspace``), adequate for clustered data only. It will raise
            ``TypeError`` in apsg 2.2; use ``watson_statistics``,
            ``bingham_statistics`` or convert the axes to vectors,
            ``.halfspace().to_vec()``.
        """
        _warn_axial(self, "fisher_statistics")
        return self._fisher_statistics(level)

    def _fisher_statistics(self, level=0.95):
        data = self.normalized()
        if issubclass(self.__feature_class__, Axial3):
            data = data.halfspace()
        dist = VonMisesFisher.fit(np.asarray(data))
        alpha = dist.mean_confidence(level).alpha
        return {"mu": dist.mean, "k": dist.kappa, "alpha": alpha}

    def csd(self):
        """Angular standard deviation based on Fisher's statistics (Fisher, 1953).

        Deprecated for axial data, see ``fisher_statistics``.

        Returns:
            float: angular standard deviation in degrees.
        """
        self._require(1, "csd")
        _warn_axial(self, "csd")
        k = self._fisher_statistics()["k"]
        return 81 / np.sqrt(k) if k > 0 else np.inf

    def uniformity_test(self, level=0.95):
        """Rayleigh test of uniformity against a preferred direction.

        Uses the large-sample chi-square approximation ``3 * N * Rbar**2 ~ chi2(df=3)``
        (Mardia & Jupp, 2000, Directional Statistics, sec. 10.4.1) for spherical data.

        Args:
            level: confidence level for the uniformity decision. Default 0.95 for 95 %.

        Returns:
            dict: with keys ``R`` (resultant length), ``statistic`` (chi-square
                statistic), ``p_value``, and ``uniform`` (True if data is consistent
                with a uniform/random distribution at the given confidence level).
        """
        self._require(1, "uniformity_test")
        N = len(self)
        R = abs(self.normalized().R())
        statistic = 3 * R**2 / N
        p_value = float(stats.chi2.sf(statistic, df=3))
        return {
            "R": R,
            "statistic": statistic,
            "p_value": p_value,
            "uniform": bool(p_value >= 1.0 - level),
        }

    def watson_statistics(self, level=0.95):
        """Fit Watson distribution and return statistics.

        Fits both the bipolar (``k > 0``, mean axis is the major eigenvector of
        the orientation tensor) and the girdle (``k < 0``, mean axis is the
        minor eigenvector, i.e. the pole of the girdle) form by maximum
        likelihood and returns the one with the higher likelihood. The
        normalizing constant ``M(1/2, 3/2, k)`` (Kummer's function) is evaluated
        in closed form through Dawson's integral (``k > 0``) and the error
        function (``k < 0``).

        The confidence cone is the large-sample approximation
        ``sin^2(alpha) = -ln(1 - level) / (N * |k| * |tau - tau_m|)``, where
        ``tau`` is the eigenvalue of the mean axis and ``tau_m`` the mean of the
        other two eigenvalues.

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
        along the direction of the larger spread of the data.

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

    def bootstrap_statistics(self, level=0.95, n=1000):
        """Bootstrap confidence cone of the principal orientation.

        The mean is the principal eigenvector of the orientation tensor and
        `alpha` is the `level` percentile of the angles between it and the
        principal eigenvectors of `n` bootstrap resamples. Like the
        orientation tensor itself, it applies to axial as well as directional
        data.

        Args:
            level (float): confidence level. Default 0.95.
            n (int): number of bootstrap resamples. Default 1000.

        Returns:
            dict: ``mu`` (mean orientation) and ``alpha`` (radius of the
            confidence cone in degrees).
        """
        mu = self.ortensor().eigenvectors(0)
        deviations = []
        for sample in self.bootstrap(n=n):
            v = sample.ortensor().eigenvectors(0)
            # eigenvectors have an arbitrary sign
            if v.dot(mu) < 0:
                v = -v
            deviations.append(mu.angle(v))
        return {"mu": mu, "alpha": float(np.percentile(deviations, 100 * level))}

    def var(self):
        """Spherical variance based on resultant length (Mardia 1972).

        Deprecated for axial data, see ``fisher_statistics``; the orientation
        tensor (e.g. ``1 - ortensor().eigenvalues(0)``) describes their spread.
        """
        self._require(1, "var")
        _warn_axial(self, "var")
        return 1 - abs(self.normalized().R(mean=True))

    def delta(self):
        """Cone angle containing ~63% of the data in degrees.

        Deprecated for axial data, see ``fisher_statistics``.
        """
        self._require(1, "delta")
        _warn_axial(self, "delta")
        return acosd(abs(self.R(mean=True)))

    def rdegree(self):
        """Degree of preffered orientation of vectors in ``FeatureSet``.

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
        H0 hypothesis is that two samples are from same distribution.

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
    def random(cls, n=100, name="Default"):
        """Method to create ``Vector3Set`` of features with uniformly distributed
        random orientation.

        Keyword Args:
            n: number of objects to be generated.
            name: name of dataset. Default is 'Default'.

        Examples:
            >>> np.random.seed(58463123)
            >>> v = vecset.random(100)

        Returns:
            Vector3Set: The created feature set.
        """
        n = check_count(n)
        u1 = np.random.random(n)
        u2 = np.random.random(n)
        z = 2 * u2 - 1
        theta = 2 * np.pi * u1
        r = np.sqrt(1 - z**2)
        x = r * np.cos(theta)
        y = r * np.sin(theta)
        data = [cls.__feature_class__(v) for v in np.array([x, y, z]).T]
        return cls(data, name=name)

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
    def random_fisher(cls, n=100, position=None, kappa=20, name="Default"):
        """Return ``FeatureSet`` of random vectors sampled from von Mises Fisher
        distribution around center position with concentration kappa.

        Args:
            n: number of objects to be generated.
            position: mean orientation given as ``Vector3``. Default Vector3(0, 0, 1).
            kappa: precision parameter of the distribution. Default 20.
            name: name of dataset. Default is 'Default'.

        Examples:
            >>> l = linset.random_fisher(position=lin(120,50))
        Returns:
            FeatureSet: The created feature set.
        """
        n = check_count(n)
        if position is None:
            position = Vector3(0, 0, 1)
        X = VonMisesFisher(position, kappa).rvs_array(n, legacy_rng())
        return cls([cls.__feature_class__(x) for x in X], name=name)

    @classmethod
    def random_kent(cls, p, n=100, kappa=20, beta=None, name="Default"):
        """Return ``FeatureSet`` of random vectors sampled from Kent distribution
        (Kent, 1982) - The 5-parameter Fisher–Bingham distribution.

        Args:
            p: Pair object defining orientation of data.
            n: number of objects to be generated.
            kappa: concentration parameter. Default 20.
            beta: ellipticity, ``0 <= beta < kappa / 2`` (the distribution is
                bimodal for larger ``beta``). Default ``kappa / 4``.
            name: name of dataset. Default is 'Default'.

        Examples:
            >>> p = pair(150, 40, 150, 40)
            >>> l = linset.random_kent(p, n=300, kappa=30)
        Returns:
            FeatureSet: The created feature set.
        """
        if not isinstance(p, Pair):
            raise TypeError("Argument must be Pair object.")
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
        kd = Kent(p.lvec, p.fvec.cross(p.lvec), kappa, beta)
        X = kd.rvs_array(check_count(n), legacy_rng())
        return cls([cls.__feature_class__(x) for x in X], name=name)

    @classmethod
    def sfs(cls, n=100, name="Default"):
        """Method to create ``FeatureSet`` of uniformly distributed vectors.
        Spherical Fibonacci Spiral points on a sphere algorithm adopted from
        John Burkardt.

        http://people.sc.fsu.edu/~jburkardt/

        Keyword Args:
            n: number of objects to be generated. Default 1000.
            name: name of dataset. Default is 'Default'.

        Examples:
            >>> v = vecset.sfs(300)
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
    def gss(cls, n=100, name="Default"):
        """Method to create ``FeatureSet`` of uniformly distributed vectors.
        Golden Section Spiral points on a sphere algorithm.

        http://www.softimageblog.com/archives/115

        Args:
            n: number of objects to be generated. Default 1000.
            name: name of dataset. Default is 'Default'.

        Examples:
            >>> v = vecset.gss(300)
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


class LineationSet(Vector3Set):
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

    def uniformity_test(self, level=0.95):
        """Watson's U² test of uniformity for axial (spherical) data.

        Rayleigh's test (used by plain ``Vector3Set``) is insensitive to axial/bipolar
        concentration since the resultant vector cancels for antipodal data. Instead,
        tests whether ``|xᵢ · v|`` (v = major eigenvector of the orientation tensor) is
        uniformly distributed on [0, 1] - true for the (folded) z-coordinate of points
        uniformly distributed on the sphere.

        Args:
            level: confidence level for the uniformity decision. Default 0.95.

        Returns:
            dict: with keys ``statistic`` (Watson's U²), ``critical_value`` and
                ``uniform``.
        """
        self._require(1, "uniformity_test")
        v = self.ortensor().eigenvectors(0)
        w = [e.dot(v) for e in self]
        return _watson_u2_test(w, level=level)


class FoliationSet(Vector3Set):
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

    def uniformity_test(self, level=0.95):
        """Watson's U² test of uniformity for axial (spherical) data.

        Rayleigh's test (used by plain ``Vector3Set``) is insensitive to axial/bipolar
        concentration since the resultant vector cancels for antipodal data. Instead,
        tests whether ``|xᵢ · v|`` (v = major eigenvector of the orientation tensor) is
        uniformly distributed on [0, 1] - true for the (folded) z-coordinate of points
        uniformly distributed on the sphere.

        Args:
            level: confidence level for the uniformity decision. Default 0.95.

        Returns:
            dict: with keys ``statistic`` (Watson's U²), ``critical_value`` and
                ``uniform``.
        """
        self._require(1, "uniformity_test")
        v = self.ortensor().eigenvectors(0)
        w = [e.dot(v) for e in self]
        return _watson_u2_test(w, level=level)

    def dipvec(self):
        """Return ``FeatureSet`` object with plane dip vector."""

        return Vector3Set([e.dipvec() for e in self], name=self.name)

    def strike(self):
        """Return ``Direction2Set`` object with strikes of planar features."""

        return Direction2Set([e.strike() for e in self], name=self.name)
