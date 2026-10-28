"""
Distributions on the circle.

References are abbreviated MJ (Mardia & Jupp, 2000, Directional Statistics),
FLE (Fisher, Lewis & Embleton, 1987, Statistical Analysis of Spherical Data)
and F93 (Fisher, 1993, Statistical Analysis of Circular Data).
"""

import warnings

import numpy as np
from scipy import special, stats
from scipy.optimize import brentq, minimize

from apsg.feature._distributions._base import (
    ConfidenceRegion,
    OrientationDistribution,
    finite_param,
    unit_vector,
)
from apsg.feature._geodata import Direction
from apsg.helpers._math import acosd
from apsg.math._vector import Vector2


class Uniform2(OrientationDistribution):
    """Uniform distribution of directions on the circle."""

    dim = 2
    axial = False
    n_params = 0
    name = "uniform"
    feature = Vector2

    def logpdf_array(self, X):
        return np.full(len(X), -np.log(2 * np.pi))

    def rvs_array(self, n, rng):
        theta = rng.uniform(0, 2 * np.pi, n)
        return np.column_stack([np.cos(theta), np.sin(theta)])

    @classmethod
    def _fit(cls, X, method="mle", **options):
        return {}, True, {}

    @property
    def params(self):
        return {}


class AxialUniform2(Uniform2):
    """Uniform distribution of axes (undirected lines) on the circle."""

    axial = True
    name = "axial_uniform"
    feature = Direction


class VonMises(OrientationDistribution):
    """von Mises distribution of directions on the circle.

    The density is ``exp(kappa * cos(theta - mu)) / (2 pi I0(kappa))``.

    Args:
        mean: mean direction as ``Vector2`` or angle in degrees. Default 0.
        kappa: concentration parameter. Default 1.

    Fitting methods are "mle" (maximum likelihood, ``I1(kappa)/I0(kappa) =
    Rbar``, MJ eq. 5.3.2-5.3.6), "approx" (the piecewise approximation of the
    maximum-likelihood estimate of Best & Fisher, 1981, F93 eq. 4.40, MJ eq.
    5.3.7-5.3.10, maximum relative error 0.9 %) and "corrected" (the
    small-sample correction of F93 eq. 4.41 of the maximum-likelihood
    estimate, recommended for ``N <= 15``). Sampling uses scipy's
    ``vonmises`` (Best & Fisher, 1979).

    References:
        Fisher, N.I. (1993). Statistical Analysis of Circular Data. Cambridge
            University Press.
        Mardia, K.V. & Jupp, P.E. (2000). Directional Statistics. Wiley.
    """

    dim = 2
    axial = False
    n_params = 2
    name = "vonmises"
    feature = Vector2

    def __init__(self, mean=0, kappa=1):
        self._mean = Vector2(unit_vector("mean", Vector2(mean)))
        kappa = finite_param("kappa", kappa)
        if not kappa >= 0:
            raise ValueError("kappa must be non-negative")
        self.kappa = kappa

    @property
    def params(self):
        return {"mean": self._mean, "kappa": self.kappa}

    @property
    def mean(self):
        return self._mean

    def logpdf_array(self, X):
        c = X @ np.asarray(self._mean, dtype=float)
        if np.isinf(self.kappa):
            return np.where(np.isclose(c, 1), np.inf, -np.inf)
        return self.kappa * (c - 1) - np.log(2 * np.pi * special.i0e(self.kappa))

    def rvs_array(self, n, rng):
        loc = np.radians(self._mean.direction)
        theta = stats.vonmises.rvs(self.kappa, loc=loc, size=n, random_state=rng)
        return np.column_stack([np.cos(theta), np.sin(theta)])

    @classmethod
    def _fit(cls, X, method="mle", **options):
        resultant = X.sum(axis=0)
        R = np.linalg.norm(resultant)
        Rbar = R / len(X)
        mean = Vector2(resultant) if R > 0 else Vector2(1, 0)
        n = len(X)
        match method:
            case "mle":
                kappa = _vonmises_kappa_mle(Rbar)
            case "approx":
                kappa = _vonmises_kappa_approx(Rbar)
            case "corrected":
                kappa = vonmises_kappa_corrected(_vonmises_kappa_mle(Rbar), n)
            case _:
                raise ValueError(f"Unknown fitting method {method!r}")
        # mean of cos 2(theta - mean), for the circular dispersion
        rho2 = float(np.mean(2 * (X @ np.asarray(mean.normalized())) ** 2 - 1))
        return {"mean": mean, "kappa": kappa}, True, {"Rbar": Rbar, "rho2": rho2}

    def mean_confidence(self, level=0.95, method="upton"):
        """Confidence interval of the mean direction of the fitted distribution.

        Args:
            level: confidence level. Default 0.95.
            method: "upton" (default, see ``upton_interval``), "vonmises"
                (``arcsin(z / sqrt(N Rbar kappa))``, F93 eq. 4.42-4.43, for the
                fitted kappa, adequate for small samples) or "dispersion" (no
                model assumed, F93 eq. 4.21-4.22, for ``N >= 25``, see
                ``dispersion_interval``).

        Returns:
            ConfidenceRegion: cone with the half-width ``alpha`` in degrees.
        """
        if self.fit_info is None:
            raise ValueError("Confidence region needs a distribution fitted to data")
        alpha = _circular_mean_interval(self, level, method)
        return ConfidenceRegion(mu=self._mean, level=level, alpha=alpha)

    def _kappa_confidence(self, level):
        return vonmises_kappa_interval(
            self.fit_info.n, self.fit_info.stats["Rbar"], self.kappa, level
        )


class WrappedNormal(OrientationDistribution):
    """Wrapped normal distribution of directions on the circle.

    The normal distribution ``N(mu, sigma^2)`` wrapped around the circle, with
    ``rho = exp(-sigma^2 / 2)`` (MJ eq. 3.5.63-3.5.66). The density is
    evaluated as a Fourier series for wide distributions and as a sum of
    wrapped normal densities for narrow ones.

    Args:
        mean: mean direction as ``Vector2`` or angle in degrees. Default 0.
        rho: concentration parameter, the mean resultant length,
            ``0 <= rho < 1``. Default 0.5.

    Fitting methods are "mle" (maximum likelihood, numerical) and "moments"
    (mean direction and ``rho = Rbar``, MJ eq. 3.5.66). Sampling wraps normal
    deviates (F93 sec. 3.3.5). The confidence interval of the mean direction is
    ``dispersion_interval``.

    References:
        Mardia, K.V. & Jupp, P.E. (2000). Directional Statistics. Wiley.
    """

    dim = 2
    axial = False
    n_params = 2
    name = "wrapped_normal"
    feature = Vector2

    def __init__(self, mean=0, rho=0.5):
        self._mean = Vector2(unit_vector("mean", Vector2(mean)))
        rho = finite_param("rho", rho)
        if not 0 <= rho < 1:
            raise ValueError("rho must be in [0, 1)")
        self.rho = rho

    @property
    def sigma(self):
        """Standard deviation (radians) of the wrapped normal distribution."""
        return np.sqrt(-2 * np.log(self.rho)) if self.rho > 0 else np.inf

    @property
    def params(self):
        return {"mean": self._mean, "rho": self.rho}

    @property
    def mean(self):
        return self._mean

    def logpdf_array(self, X):
        mu = np.radians(self._mean.direction)
        theta = np.arctan2(X[:, 1], X[:, 0]) - mu
        return np.log(_wrapped_normal_pdf(theta, self.sigma))

    def rvs_array(self, n, rng):
        theta = rng.normal(np.radians(self._mean.direction), self.sigma, n)
        return np.column_stack([np.cos(theta), np.sin(theta)])

    @classmethod
    def _fit(cls, X, method="mle", **options):
        resultant = X.sum(axis=0)
        Rbar = np.linalg.norm(resultant) / len(X)
        mean = Vector2(resultant) if Rbar > 0 else Vector2(1, 0)
        rho = min(Rbar, 1 - 1e-12)
        # mean of cos 2(theta - mean), for the circular dispersion
        rho2 = float(np.mean(2 * (X @ np.asarray(mean.normalized())) ** 2 - 1))
        stats_ = {"Rbar": Rbar, "rho2": rho2}
        match method:
            case "moments":
                return {"mean": mean, "rho": rho}, True, stats_
            case "mle":
                theta = np.arctan2(X[:, 1], X[:, 0])

                def nll(p):
                    sigma = np.exp(p[1])
                    return -np.sum(np.log(_wrapped_normal_pdf(theta - p[0], sigma)))

                start = [np.radians(mean.direction), np.log(np.sqrt(-2 * np.log(rho)))]
                res = minimize(nll, start, method="Nelder-Mead", options=_NM_OPTIONS)
                fitted = {
                    "mean": np.degrees(res.x[0]),
                    "rho": float(np.exp(-np.exp(2 * res.x[1]) / 2)),
                }
                return fitted, bool(res.success), stats_
        raise ValueError(f"Unknown fitting method {method!r}")

    def _mean_confidence(self, level):
        stats_ = self.fit_info.stats
        alpha = dispersion_interval(
            self.fit_info.n, stats_["Rbar"], stats_["rho2"], level
        )
        return ConfidenceRegion(mu=self._mean, level=level, alpha=alpha)


_NM_OPTIONS = {"xatol": 1e-10, "fatol": 1e-12, "maxiter": 2000}


def _wrapped_normal_pdf(theta, sigma):
    """Density of the wrapped normal distribution at angles `theta` (radians)."""
    if np.isinf(sigma):
        return np.full(np.shape(theta), 1 / (2 * np.pi))
    theta = (np.asarray(theta) + np.pi) % (2 * np.pi) - np.pi
    if sigma**2 < 4:
        # sum of the normal densities over the windings
        k = np.arange(-int(np.ceil(6 * sigma / (2 * np.pi))) - 1, 0)
        k = np.concatenate([k, [0], -k[::-1]])
        z = (theta[..., None] + 2 * np.pi * k) / sigma
        return np.exp(-(z**2) / 2).sum(axis=-1) / (sigma * np.sqrt(2 * np.pi))
    # Fourier series (MJ eq. 3.5.64)
    rho = np.exp(-(sigma**2) / 2)
    k = np.arange(1, int(np.ceil(np.sqrt(-np.log(1e-17) / -np.log(rho)))) + 1)
    series = (rho ** (k**2) * np.cos(theta[..., None] * k)).sum(axis=-1)
    return (1 + 2 * series) / (2 * np.pi)


class AxialVonMises(OrientationDistribution):
    """Axial von Mises distribution, the von Mises distribution of doubled angles.

    The density is ``exp(kappa * cos(2 (theta - mu))) / (2 pi I0(kappa))``,
    the doubly wrapped von Mises distribution (MJ eq. 3.6.4-3.6.6). It is the
    Bingham distribution on the circle. Following the doubling convention (F93
    eq. 2.36), kappa and the mean resultant length refer to the doubled angles,
    while the mean axis and its confidence interval are halved.

    Args:
        mean: mean axis as ``Direction``, ``Vector2`` or angle in degrees.
            Default 0.
        kappa: concentration parameter of the doubled angles. Default 1.

    Fitting methods are those of ``VonMises``, applied to the doubled angles.
    """

    dim = 2
    axial = True
    n_params = 2
    name = "axial_vonmises"
    feature = Direction

    def __init__(self, mean=0, kappa=1):
        self._mean = Direction(unit_vector("mean", Vector2(mean)))
        self._doubled = VonMises(2 * self._mean.direction, kappa)

    @property
    def kappa(self):
        return self._doubled.kappa

    @property
    def params(self):
        return {"mean": self._mean, "kappa": self.kappa}

    @property
    def mean(self):
        return self._mean

    def logpdf_array(self, X):
        return self._doubled.logpdf_array(double_angles(X))

    def rvs_array(self, n, rng):
        return halve_angles(self._doubled.rvs_array(n, rng))

    @classmethod
    def _fit(cls, X, method="mle", **options):
        params, converged, stats = VonMises._fit(double_angles(X), method=method)
        params["mean"] = params["mean"].direction / 2
        return params, converged, stats

    def mean_confidence(self, level=0.95, method="upton"):
        """Confidence interval of the mean axis of the fitted distribution.

        The interval of the doubled angles (see ``VonMises.mean_confidence``)
        halved.

        Args:
            level: confidence level. Default 0.95.
            method: "upton" (default), "vonmises" or "dispersion".

        Returns:
            ConfidenceRegion: cone with the half-width ``alpha`` in degrees.
        """
        if self.fit_info is None:
            raise ValueError("Confidence region needs a distribution fitted to data")
        alpha = _circular_mean_interval(self, level, method)
        return ConfidenceRegion(mu=self._mean, level=level, alpha=alpha / 2)

    def _kappa_confidence(self, level):
        return vonmises_kappa_interval(
            self.fit_info.n, self.fit_info.stats["Rbar"], self.kappa, level
        )


def _circular_mean_interval(dist, level, method):
    """Half-width (degrees) of the confidence interval of the mean of a fitted
    von Mises distribution (of the doubled angles for axial data)."""
    n, s = dist.fit_info.n, dist.fit_info.stats
    match method:
        case "upton":
            return upton_interval(n, s["Rbar"], level)
        case "vonmises":
            if s["Rbar"] >= 1:
                return 0.0
            if dist.kappa <= 0 or s["Rbar"] <= 0:
                return 180.0
            arg = stats.norm.ppf((1 + level) / 2) / np.sqrt(n * s["Rbar"] * dist.kappa)
            return 180.0 if arg >= 1 else float(np.degrees(np.arcsin(arg)))
        case "dispersion":
            return dispersion_interval(n, s["Rbar"], s["rho2"], level)
    raise ValueError(
        f"Unknown method {method!r}, expected upton, vonmises or dispersion"
    )


def vonmises_kappa_corrected(kappa, n):
    """Small-sample correction of the maximum-likelihood von Mises kappa.

    ``max(kappa - 2 / (n kappa), 0)`` for ``kappa < 2``, otherwise ``(n - 1)^3
    kappa / (n^3 + n)`` (Best & Fisher, 1981; F93 eq. 4.41), recommended for ``n
    <= 15``.
    """
    if not np.isfinite(kappa):
        return kappa
    if kappa < 2:
        return max(kappa - 2 / (n * kappa), 0.0) if kappa > 0 else 0.0
    return (n - 1) ** 3 * kappa / (n**3 + n)


def vonmises_kappa_interval(n, Rbar, kappa, level=0.95):
    """Confidence interval of the von Mises concentration.

    ``kappa = (1 + sqrt(1 + 3a)) / (4a)`` with ``a = (n - R) / chi2_n-1`` at the
    two quantiles (F93 eq. 4.48-4.55). It is valid for ``kappa >= 2``; for less
    concentrated data a warning is issued (use a bootstrap instead).

    Returns:
        tuple: lower and upper limit.
    """
    if kappa < 2:
        warnings.warn(
            f"The interval of kappa is valid for kappa >= 2, got {kappa:.3g}",
            UserWarning,
            stacklevel=3,
        )
    R = n * Rbar
    if n - R <= 0:
        return (np.inf, np.inf)
    limits = []
    for q in ((1 - level) / 2, (1 + level) / 2):
        a = (n - R) / stats.chi2.ppf(q, n - 1)
        limits.append(float((1 + np.sqrt(1 + 3 * a)) / (4 * a)))
    return tuple(limits)


def double_angles(X):
    """Return unit vectors of ``(N, 2)`` array `X` with doubled angles."""
    x, y = X[:, 0], X[:, 1]
    return np.column_stack([x**2 - y**2, 2 * x * y])


def halve_angles(X):
    """Return unit vectors of ``(N, 2)`` array `X` with halved angles."""
    theta = np.arctan2(X[:, 1], X[:, 0]) / 2
    return np.column_stack([np.cos(theta), np.sin(theta)])


def _vonmises_kappa_mle(Rbar):
    """Solve ``I1(kappa) / I0(kappa) = Rbar`` for kappa."""
    if Rbar < 1e-12:
        return 0.0
    if Rbar >= 1 - 1e-12:
        return np.inf
    return brentq(
        lambda k: special.i1e(k) / special.i0e(k) - Rbar, 0, 1 / (1 - Rbar) + 10
    )


def _vonmises_kappa_approx(Rbar):
    """Piecewise approximation of the maximum-likelihood kappa (Best & Fisher,
    1981; F93 eq. 4.40)."""
    if Rbar >= 1:
        return np.inf
    if Rbar < 0.53:
        return 2 * Rbar + Rbar**3 + (5 / 6) * Rbar**5
    if Rbar < 0.85:
        return -0.4 + 1.39 * Rbar + 0.43 / (1.0 - Rbar)
    return 1.0 / (Rbar**3 - 4 * Rbar**2 + 3 * Rbar)


def upton_interval(N, Rbar, level=0.95):
    """Half-width (degrees) of the confidence interval of the circular mean.

    Large-sample approximation of Upton (1986), as given by Zar
    (Biostatistical Analysis, eq. 26.24-26.25). Returns 180 when the data are
    too dispersed to give an interval.

    Args:
        N: sample size.
        Rbar: mean resultant length.
        level: confidence level. Default 0.95.
    """
    if Rbar >= 1.0 or N <= 1:
        return 0.0
    R = N * Rbar
    chi2 = stats.chi2.ppf(level, df=1)
    if Rbar <= 0.9:
        if Rbar <= np.sqrt(chi2 / (2 * N)):
            return 180.0
        cos_d = np.sqrt(2 * N * (2 * R**2 - N * chi2) / (4 * N - chi2)) / R
    else:
        cos_d = np.sqrt(N**2 - (N**2 - R**2) * np.exp(chi2 / N)) / R
    return acosd(np.clip(cos_d, -1.0, 1.0))


def dispersion_interval(N, Rbar, rho2, level=0.95):
    """Half-width (degrees) of the confidence interval of the circular mean.

    Large-sample interval for any unimodal distribution (F93 eq. 4.21-4.22,
    for ``N >= 25``): ``arcsin(z * sqrt(delta / N))`` with the circular
    dispersion ``delta = (1 - rho2) / (2 Rbar^2)`` (MJ sec. 2.3.4). Returns
    180 when the data are too dispersed to give an interval.

    Args:
        N: sample size.
        Rbar: mean resultant length.
        rho2: mean of ``cos 2(theta - mean)``.
        level: confidence level. Default 0.95.
    """
    if Rbar >= 1.0 or N <= 1:
        return 0.0
    if Rbar <= 0:
        return 180.0
    delta = (1 - rho2) / (2 * Rbar**2)
    arg = stats.norm.ppf((1 + level) / 2) * np.sqrt(delta / N)
    return 180.0 if arg >= 1 else float(np.degrees(np.arcsin(arg)))


def vonmises_kde(theta, centers, kappa, *, axial=False, weights=None):
    """Kernel density estimate with von Mises kernels.

    Args:
        theta: angles (radians) at which the density is evaluated.
        centers: angles (radians) of the data.
        kappa: concentration of the kernel.

    Keyword Args:
        axial: if True, each datum contributes two kernels of half weight at
            ``center`` and ``center + pi``. Default False.
        weights: weights of the data, normalized to mean 1. Default None
            (equal weights).

    Returns:
        ndarray: the density at `theta`.
    """
    theta = np.atleast_1d(theta)[:, None]
    centers = np.asarray(centers, dtype=float)[None, :]
    density = stats.vonmises.pdf(theta, kappa, loc=centers)
    if axial:
        density = (density + stats.vonmises.pdf(theta, kappa, loc=centers + np.pi)) / 2
    if weights is not None:
        density = density * np.asarray(weights, dtype=float)[None, :]
    return density.sum(axis=1) / centers.shape[1]


def _angles_from(X, mean):
    """Angles (radians) of the rows of `X` from the direction `mean`, in
    (-pi, pi]."""
    mu = np.radians(Vector2(mean).direction)
    theta = np.arctan2(X[:, 1], X[:, 0]) - mu
    return (theta + np.pi) % (2 * np.pi) - np.pi


def _moment_stats(X, mean):
    """Mean resultant length and mean of cos 2(theta - mean) of `X`, for the
    confidence interval of the mean (``dispersion_interval``)."""
    Rbar = np.linalg.norm(X.sum(axis=0)) / len(X)
    rho2 = float(np.mean(np.cos(2 * _angles_from(X, mean))))
    return {"Rbar": float(Rbar), "rho2": rho2}


class _DispersionInterval:
    """Mixin: confidence interval of the mean by ``dispersion_interval``."""

    def _mean_confidence(self, level):
        s = self.fit_info.stats
        alpha = dispersion_interval(self.fit_info.n, s["Rbar"], s["rho2"], level)
        return ConfidenceRegion(mu=self._mean, level=level, alpha=alpha)


class WrappedCauchy(_DispersionInterval, OrientationDistribution):
    """Wrapped Cauchy distribution of directions on the circle.

    The density is ``(1 - rho^2) / (2 pi (1 + rho^2 - 2 rho cos(theta - mu)))``
    (MJ eq. 3.5.68-3.5.69), the Cauchy distribution with scale ``-ln(rho)``
    wrapped around the circle. ``rho`` is the mean resultant length; it has
    heavier tails than the von Mises distribution.

    Args:
        mean: mean direction as ``Vector2`` or angle in degrees. Default 0.
        rho: concentration parameter, ``0 <= rho < 1``. Default 0.5.

    Fitting methods are "mle" (maximum likelihood by the iteration of Kent &
    Tyler, 1988, MJ eq. 5.4.1-5.4.6) and "moments" (mean direction and ``rho =
    Rbar``). Sampling wraps Cauchy deviates. The confidence interval of the
    mean direction is ``dispersion_interval``.

    References:
        Kent, J.T. & Tyler, D.E. (1988). Maximum likelihood estimation for the
            wrapped Cauchy distribution. Journal of Applied Statistics 15,
            247-254.
        Mardia, K.V. & Jupp, P.E. (2000). Directional Statistics. Wiley.
    """

    dim = 2
    axial = False
    n_params = 2
    name = "wrapped_cauchy"
    feature = Vector2

    def __init__(self, mean=0, rho=0.5):
        self._mean = Vector2(unit_vector("mean", Vector2(mean)))
        rho = finite_param("rho", rho)
        if not 0 <= rho < 1:
            raise ValueError("rho must be in [0, 1)")
        self.rho = rho

    @property
    def params(self):
        return {"mean": self._mean, "rho": self.rho}

    @property
    def mean(self):
        return self._mean

    def logpdf_array(self, X):
        c = np.cos(_angles_from(X, self._mean))
        r = self.rho
        return np.log1p(-(r**2)) - np.log(2 * np.pi) - np.log1p(r**2 - 2 * r * c)

    def rvs_array(self, n, rng):
        theta = np.radians(self._mean.direction)
        if self.rho > 0:
            theta = theta - np.log(self.rho) * rng.standard_cauchy(n)
        else:
            theta = rng.uniform(0, 2 * np.pi, n)
        return np.column_stack([np.cos(theta), np.sin(theta)])

    @classmethod
    def _fit(cls, X, method="mle", tol=1e-12, maxiter=1000, **options):
        resultant = X.mean(axis=0)
        converged = True
        match method:
            case "moments":
                mean, rho = resultant, np.linalg.norm(resultant)
            case "mle":
                # Kent & Tyler (1988): weighted means of the data
                mu, converged = resultant, False
                for _ in range(maxiter):
                    w = 1 / (1 - X @ mu)
                    new = (w[:, None] * X).sum(axis=0) / w.sum()
                    if np.abs(new - mu).max() < tol:
                        mu, converged = new, True
                        break
                    mu = new
                length = np.linalg.norm(mu)
                rho = (1 - np.sqrt(1 - length**2)) / length if length > 0 else 0.0
                mean = mu
            case _:
                raise ValueError(f"Unknown fitting method {method!r}")
        mean = Vector2(mean) if np.linalg.norm(mean) > 0 else Vector2(1, 0)
        rho = min(float(rho), 1 - 1e-12)
        return {"mean": mean, "rho": rho}, converged, _moment_stats(X, mean)


class Cardioid(_DispersionInterval, OrientationDistribution):
    """Cardioid distribution of directions on the circle.

    The density is ``(1 + 2 rho cos(theta - mu)) / (2 pi)`` with ``0 <= rho <=
    1/2`` (MJ eq. 3.5.47); ``rho`` is the mean resultant length. It is close to
    the von Mises distribution with ``kappa = 2 rho`` for small ``rho``.

    Args:
        mean: mean direction as ``Vector2`` or angle in degrees. Default 0.
        rho: concentration parameter, ``0 <= rho <= 0.5``. Default 0.25.

    Fitting methods are "mle" (maximum likelihood, numerical) and "moments"
    (mean direction and ``rho = min(Rbar, 1/2)``). Sampling is by rejection
    from the uniform distribution. The confidence interval of the mean
    direction is ``dispersion_interval``.

    References:
        Mardia, K.V. & Jupp, P.E. (2000). Directional Statistics. Wiley.
    """

    dim = 2
    axial = False
    n_params = 2
    name = "cardioid"
    feature = Vector2

    def __init__(self, mean=0, rho=0.25):
        self._mean = Vector2(unit_vector("mean", Vector2(mean)))
        rho = finite_param("rho", rho)
        if not 0 <= rho <= 0.5:
            raise ValueError("rho must be in [0, 0.5]")
        self.rho = rho

    @property
    def params(self):
        return {"mean": self._mean, "rho": self.rho}

    @property
    def mean(self):
        return self._mean

    def logpdf_array(self, X):
        c = np.cos(_angles_from(X, self._mean))
        with np.errstate(divide="ignore"):
            return np.log1p(2 * self.rho * c) - np.log(2 * np.pi)

    def rvs_array(self, n, rng):
        mu = np.radians(self._mean.direction)
        out = np.empty(0)
        while len(out) < n:
            m = max(2 * (n - len(out)), 16)
            t = rng.uniform(-np.pi, np.pi, m)
            accept = rng.uniform(size=m) * (
                1 + 2 * self.rho
            ) < 1 + 2 * self.rho * np.cos(t)
            out = np.concatenate([out, t[accept]])
        theta = out[:n] + mu
        return np.column_stack([np.cos(theta), np.sin(theta)])

    @classmethod
    def _fit(cls, X, method="mle", **options):
        resultant = X.mean(axis=0)
        Rbar = np.linalg.norm(resultant)
        mean = Vector2(resultant) if Rbar > 0 else Vector2(1, 0)
        rho = min(Rbar, 0.5)
        converged = True
        match method:
            case "moments":
                pass
            case "mle":
                theta = np.arctan2(X[:, 1], X[:, 0])

                def nll(p):
                    r = 0.5 * special.expit(p[1])
                    return -np.sum(np.log1p(2 * r * np.cos(theta - p[0])))

                start = [
                    np.radians(mean.direction),
                    special.logit(np.clip(2 * rho, 1e-6, 1 - 1e-6)),
                ]
                res = minimize(nll, start, method="Nelder-Mead", options=_NM_OPTIONS)
                mean = Vector2(np.degrees(res.x[0]))
                rho = float(0.5 * special.expit(res.x[1]))
                converged = bool(res.success)
            case _:
                raise ValueError(f"Unknown fitting method {method!r}")
        return {"mean": mean, "rho": float(rho)}, converged, _moment_stats(X, mean)


class JonesPewsey(_DispersionInterval, OrientationDistribution):
    """Jones-Pewsey distribution of directions on the circle.

    The density is proportional to ``(1 + tanh(kappa psi) cos(theta -
    mu))^(1 / psi)`` (Jones & Pewsey, 2005; Ley & Verdebout, 2017, eq. 2.5).
    The shape parameter ``psi`` links the von Mises (``psi -> 0``), cardioid
    (``psi = 1``, ``rho = tanh(kappa) / 2``) and wrapped Cauchy (``psi = -1``,
    ``rho = tanh(kappa / 2)``) distributions; negative ``psi`` gives heavier
    tails, positive lighter ones.

    Args:
        mean: mean direction as ``Vector2`` or angle in degrees. Default 0.
        kappa: concentration parameter, ``kappa >= 0``. Default 1.
        psi: shape parameter. Default 0 (von Mises).

    The normalizer is evaluated by quadrature. The fit is the
    maximum-likelihood estimate (numerical, stable for ``|kappa psi| < 6``).
    Sampling is by rejection from the uniform distribution. The confidence
    interval of the mean direction is ``dispersion_interval``.

    References:
        Jones, M.C. & Pewsey, A. (2005). A family of symmetric distributions on
            the circle. Journal of the American Statistical Association 100,
            1422-1428.
        Ley, C. & Verdebout, T. (2017). Modern Directional Statistics, sec.
            2.2.4. CRC Press.
    """

    dim = 2
    axial = False
    n_params = 3
    name = "jones_pewsey"
    feature = Vector2

    def __init__(self, mean=0, kappa=1, psi=0):
        self._mean = Vector2(unit_vector("mean", Vector2(mean)))
        kappa = finite_param("kappa", kappa)
        if not kappa >= 0:
            raise ValueError("kappa must be non-negative")
        self.kappa = kappa
        self.psi = finite_param("psi", psi)
        self._log_norm = _jones_pewsey_log_norm(self.kappa, self.psi)

    @property
    def params(self):
        return {"mean": self._mean, "kappa": self.kappa, "psi": self.psi}

    @property
    def mean(self):
        return self._mean

    def logpdf_array(self, X):
        return (
            _jones_pewsey_kernel(_angles_from(X, self._mean), self.kappa, self.psi)
            - self._log_norm
        )

    def rvs_array(self, n, rng):
        mu = np.radians(self._mean.direction)
        # the density is largest at the mean, where the kernel is 0
        log_m = -self._log_norm
        out = np.empty(0)
        while len(out) < n:
            m = max(2 * (n - len(out)), 16)
            t = rng.uniform(-np.pi, np.pi, m)
            log_f = _jones_pewsey_kernel(t, self.kappa, self.psi) - self._log_norm
            accept = np.log(rng.uniform(size=m)) < log_f - log_m
            out = np.concatenate([out, t[accept]])
        theta = out[:n] + mu
        return np.column_stack([np.cos(theta), np.sin(theta)])

    @classmethod
    def _fit(cls, X, method="mle", **options):
        if method != "mle":
            raise ValueError(f"Unknown fitting method {method!r}")
        resultant = X.mean(axis=0)
        Rbar = np.linalg.norm(resultant)
        mean = Vector2(resultant) if Rbar > 0 else Vector2(1, 0)
        theta = np.arctan2(X[:, 1], X[:, 0])

        def nll(p):
            k, psi = np.exp(p[1]), p[2]
            kernel = _jones_pewsey_kernel(theta - p[0], k, psi)
            return -(np.sum(kernel) - len(theta) * _jones_pewsey_log_norm(k, psi))

        kappa0 = max(_vonmises_kappa_mle(min(Rbar, 1 - 1e-9)), 1e-3)
        start = [np.radians(mean.direction), np.log(kappa0), 0.0]
        res = minimize(nll, start, method="Nelder-Mead", options=_NM_OPTIONS)
        mean = Vector2(np.degrees(res.x[0]))
        params = {
            "mean": mean,
            "kappa": float(np.exp(res.x[1])),
            "psi": float(res.x[2]),
        }
        return params, bool(res.success), _moment_stats(X, mean)


def _jones_pewsey_kernel(t, kappa, psi):
    """Log of the unnormalized Jones-Pewsey density at angles `t` from the mean,
    minus its value at the mean."""
    c = np.cos(t)
    if abs(kappa * psi) < 1e-8:
        return kappa * (c - 1)
    a = np.tanh(kappa * psi)
    return (np.log1p(a * c) - np.log1p(a)) / psi


def _jones_pewsey_log_norm(kappa, psi):
    """Log of the normalizer of the Jones-Pewsey density, relative to the
    kernel of ``_jones_pewsey_kernel``."""
    from scipy.integrate import quad

    value = quad(
        lambda t: np.exp(_jones_pewsey_kernel(t, kappa, psi)), 0, np.pi, limit=200
    )[0]
    return float(np.log(2 * value))
