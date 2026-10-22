"""
Distributions on the circle.
"""

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
    Rbar``) and "approx" (the piecewise approximation of the maximum-likelihood
    estimate of Fisher, 1993, Statistical Analysis of Circular Data).
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
        match method:
            case "mle":
                kappa = _vonmises_kappa_mle(Rbar)
            case "approx":
                kappa = _vonmises_kappa_approx(Rbar)
            case _:
                raise ValueError(f"Unknown fitting method {method!r}")
        return {"mean": mean, "kappa": kappa}, True, {"Rbar": Rbar}

    def _mean_confidence(self, level):
        alpha = upton_interval(self.fit_info.n, self.fit_info.stats["Rbar"], level)
        return ConfidenceRegion(mu=self._mean, level=level, alpha=alpha)


class WrappedNormal(OrientationDistribution):
    """Wrapped normal distribution of directions on the circle.

    The normal distribution ``N(mu, sigma^2)`` wrapped around the circle, with
    ``rho = exp(-sigma^2 / 2)`` (López-Custodio, 2025, eq. 7). The density is
    evaluated as a Fourier series for wide distributions and as a sum of
    wrapped normal densities for narrow ones.

    Args:
        mean: mean direction as ``Vector2`` or angle in degrees. Default 0.
        rho: concentration parameter, the mean resultant length,
            ``0 <= rho < 1``. Default 0.5.

    Fitting methods are "mle" (maximum likelihood, numerical) and "moments"
    (mean direction and mean resultant length, López-Custodio, 2025, eq. 8).
    The confidence interval of the mean direction is ``dispersion_interval``.
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
    # Fourier series (López-Custodio, 2025, eq. 7)
    rho = np.exp(-(sigma**2) / 2)
    k = np.arange(1, int(np.ceil(np.sqrt(-np.log(1e-17) / -np.log(rho)))) + 1)
    series = (rho ** (k**2) * np.cos(theta[..., None] * k)).sum(axis=-1)
    return (1 + 2 * series) / (2 * np.pi)


class AxialVonMises(OrientationDistribution):
    """Axial von Mises distribution, the von Mises distribution of doubled angles.

    The density is ``exp(kappa * cos(2 (theta - mu))) / (2 pi I0(kappa))``.
    It is the Bingham distribution on the circle.

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

    def _mean_confidence(self, level):
        alpha = upton_interval(self.fit_info.n, self.fit_info.stats["Rbar"], level)
        return ConfidenceRegion(mu=self._mean, level=level, alpha=alpha / 2)


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
    """Piecewise approximation of the maximum-likelihood kappa (Fisher, 1993)."""
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

    Large-sample interval for any unimodal distribution (Fisher, 1993,
    Statistical Analysis of Circular Data): ``arcsin(z * sqrt(delta / N))``
    with the circular dispersion ``delta = (1 - rho2) / (2 Rbar^2)``. Returns
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
