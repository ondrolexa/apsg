"""
Distributions of axes (axial data) on the sphere.
"""

import numpy as np
from scipy import stats
from scipy.optimize import brentq, minimize

from apsg.feature._distributions._base import (
    ConfidenceRegion,
    OrientationDistribution,
    finite_param,
    unit_vector,
)
from apsg.feature._distributions._special import (
    bingham_log_norm,
    watson_log_norm,
    watson_mean_u2,
)
from apsg.feature._distributions._spherical import Uniform3, _tangent_basis
from apsg.feature._geodata import Lineation


class AxialUniform3(Uniform3):
    """Uniform distribution of axes (undirected lines) on the sphere."""

    axial = True
    name = "axial_uniform"
    feature = Lineation


class Watson(OrientationDistribution):
    """Watson distribution of axes on the sphere.

    The density is ``exp(kappa (mu.x)^2) / (4 pi M(1/2, 3/2, kappa))`` with
    Kummer's function ``M``. For ``kappa > 0`` the axes cluster about `mean`
    (bipolar form); for ``kappa < 0`` they spread in the plane perpendicular to
    `mean`, the pole of the girdle (girdle form).

    Args:
        mean: mean axis (pole of the girdle for negative kappa) as
            ``Lineation``, ``Vector3`` or array. Default (0, 0, 1).
        kappa: concentration parameter. Default 1.

    The fit is the maximum-likelihood estimate of both forms, with the mean
    axis the major (bipolar) or minor (girdle) eigenvector of the orientation
    tensor, keeping the form with the higher likelihood. The confidence cone
    is the large-sample approximation ``sin^2(alpha) = -ln(1 - level) / (N
    |kappa| |tau - tau_m|)`` with ``tau`` the eigenvalue of the mean axis and
    ``tau_m`` the mean of the other two.
    """

    dim = 3
    axial = True
    n_params = 3
    name = "watson"
    feature = Lineation

    def __init__(self, mean=(0, 0, 1), kappa=1):
        self._mean = Lineation(unit_vector("mean", mean))
        self.kappa = finite_param("kappa", kappa)

    @property
    def params(self):
        return {"mean": self._mean, "kappa": self.kappa}

    @property
    def mean(self):
        return self._mean

    def logpdf_array(self, X):
        c = X @ np.asarray(self._mean, dtype=float)
        return self.kappa * c**2 - np.log(4 * np.pi) - watson_log_norm(self.kappa)

    def rvs_array(self, n, rng):
        mu = np.asarray(self._mean, dtype=float)
        e1, e2 = _tangent_basis(mu)
        if self.kappa >= 0:
            bingham = Bingham(mu, e1, -self.kappa, -self.kappa)
        else:
            bingham = Bingham(e1, e2, 0, self.kappa)
        return bingham.rvs_array(n, rng)

    @classmethod
    def _fit(cls, X, method="mle", **options):
        if method != "mle":
            raise ValueError(f"Unknown fitting method {method!r}")
        taus, vecs = _scatter_eigen(X)
        bipolar = _watson_kappa(taus[0])
        girdle = _watson_kappa(taus[2])
        if _watson_loglik(bipolar, taus[0]) >= _watson_loglik(girdle, taus[2]):
            which, kappa = 0, bipolar
        else:
            which, kappa = 2, girdle
        stats_ = {"eigenvalues": tuple(taus), "which": which}
        return {"mean": vecs[which], "kappa": kappa}, True, stats_

    def _mean_confidence(self, level):
        taus = self.fit_info.stats["eigenvalues"]
        tau = taus[self.fit_info.stats["which"]]
        tau_minor = (sum(taus) - tau) / 2.0
        if np.isinf(self.kappa):
            alpha = 0.0
        elif abs(self.kappa) < 1e-5 or abs(tau - tau_minor) < 1e-5:
            alpha = 90.0
        else:
            val = -np.log(1.0 - level)
            val /= self.fit_info.n * abs(self.kappa) * abs(tau - tau_minor)
            alpha = float(np.degrees(np.arcsin(np.sqrt(min(val, 1.0)))))
        return ConfidenceRegion(mu=self._mean, level=level, alpha=alpha)


class Bingham(OrientationDistribution):
    """Bingham distribution of axes on the sphere.

    The density is ``exp(kappa_major (major.x)^2 + kappa_minor (minor.x)^2) /
    F`` with ``kappa_minor <= kappa_major <= 0`` (López-Custodio, 2025, eq.
    28), so that `mean` is the mode, `major` the axis of the larger spread and
    ``minor = mean x major``. A girdle has ``kappa_major`` near 0 and `minor`
    as its pole.

    Args:
        mean: mean axis as ``Lineation``, ``Vector3`` or array. Default (0, 0, 1).
        major: major axis, made perpendicular to `mean`. Default (1, 0, 0).
        kappa_major: concentration along the major axis. Default -1.
        kappa_minor: concentration along the minor axis. Default -2.

    The axes of the fit are the eigenvectors of the orientation tensor and the
    concentrations the maximum-likelihood estimates (López-Custodio, 2025, eq.
    30). The normalizer is evaluated as a one-dimensional integral. Sampling
    is by rejection from an angular central Gaussian envelope (Kent, Ganeiber
    & Mardia, 2013; López-Custodio, 2025, eq. 31-32). The confidence region is
    the large-sample ellipse of the mean axis, see ``bingham_ellipse``.
    """

    dim = 3
    axial = True
    n_params = 5
    name = "bingham"
    feature = Lineation

    def __init__(self, mean=(0, 0, 1), major=(1, 0, 0), kappa_major=-1, kappa_minor=-2):
        g1 = unit_vector("mean", mean)
        g2 = unit_vector("major", major)
        g2 = g2 - (g2 @ g1) * g1
        if np.linalg.norm(g2) < 1e-12:
            raise ValueError("major axis must not be parallel to mean")
        g2 = g2 / np.linalg.norm(g2)
        kappa_major = finite_param("kappa_major", kappa_major)
        kappa_minor = finite_param("kappa_minor", kappa_minor)
        if not kappa_minor <= kappa_major <= 0:
            raise ValueError("kappa_minor <= kappa_major <= 0 is required")
        self._axes = np.array([g1, g2, np.cross(g1, g2)])
        self.kappa_major = kappa_major
        self.kappa_minor = kappa_minor
        self._log_norm = bingham_log_norm(self.kappa_major, self.kappa_minor)

    @property
    def params(self):
        return {
            "mean": Lineation(self._axes[0]),
            "major": Lineation(self._axes[1]),
            "kappa_major": self.kappa_major,
            "kappa_minor": self.kappa_minor,
        }

    @property
    def mean(self):
        return Lineation(self._axes[0])

    @property
    def axes(self):
        """tuple: mean, major and minor axis as ``Lineation``."""
        return tuple(Lineation(g) for g in self._axes)

    def logpdf_array(self, X):
        g = X @ self._axes.T
        return (
            self.kappa_major * g[:, 1] ** 2
            + self.kappa_minor * g[:, 2] ** 2
            - self._log_norm
        )

    def rvs_array(self, n, rng):
        # exp(-x'Ax) with A = diag(0, -kappa_major, -kappa_minor) in the frame
        # of the axes, ACG envelope with Omega = I + 2A / b
        lam = np.array([0.0, -self.kappa_major, -self.kappa_minor])
        b = brentq(lambda b: np.sum(1 / (b + 2 * lam)) - 1, 1e-12, 3.0)
        omega = 1 + 2 * lam / b
        log_m = -(3 - b) / 2 + 1.5 * np.log(3 / b)
        if not (np.all(np.isfinite(omega)) and np.isfinite(log_m)):
            # rejection sampling would never accept a sample
            raise ValueError("Concentration is too large to sample from")
        out = np.empty((0, 3))
        while len(out) < n:
            m = max(2 * (n - len(out)), 16)
            Y = rng.standard_normal((m, 3)) / np.sqrt(omega)
            Y = Y / np.linalg.norm(Y, axis=1)[:, None]
            log_ratio = -(Y**2 @ lam) + 1.5 * np.log(Y**2 @ omega) - log_m
            out = np.concatenate([out, Y[np.log(rng.uniform(size=m)) < log_ratio]])
        return out[:n] @ self._axes

    @classmethod
    def _fit(cls, X, method="mle", **options):
        if method != "mle":
            raise ValueError(f"Unknown fitting method {method!r}")
        taus, vecs = _scatter_eigen(X)
        params = {"mean": vecs[0], "major": vecs[1]}
        stats_ = {
            "eigenvalues": tuple(taus),
            "axes": tuple(vecs),
            "cross_moments": axial_cross_moments(X, vecs),
        }

        def nll(p):
            k_major = -np.exp(p[0])
            k_minor = k_major - np.exp(p[1])
            return -(k_major * taus[1] + k_minor * taus[2]) + bingham_log_norm(
                k_major, k_minor
            )

        # start from the concentrated-data approximation E[u^2] = -1 / (2 k)
        k_major = -1 / (2 * max(taus[1], 1e-6))
        k_minor = min(-1 / (2 * max(taus[2], 1e-6)), k_major - 1e-3)
        start = [np.log(-k_major), np.log(k_major - k_minor)]
        res = minimize(nll, start, method="Nelder-Mead", options=_NM_OPTIONS)
        params["kappa_major"] = float(-np.exp(res.x[0]))
        params["kappa_minor"] = float(params["kappa_major"] - np.exp(res.x[1]))
        return params, bool(res.success), stats_

    def _mean_confidence(self, level):
        stats_ = self.fit_info.stats
        return bingham_ellipse(
            stats_["eigenvalues"],
            stats_["axes"],
            stats_["cross_moments"],
            self.fit_info.n,
            level,
            which=0,
        )


class ACG(OrientationDistribution):
    """Angular central Gaussian distribution of axes on the sphere.

    The distribution of ``y / |y|`` for ``y ~ N(0, Lambda)``, with density
    ``(x' Lambda^-1 x)^(-3/2) / (4 pi sqrt(|Lambda|))`` (Tyler, 1987;
    López-Custodio, 2025, eq. 33). ``Lambda`` has eigenvalues ``a1 >= a2 >=
    a3`` along `mean`, `major` and ``minor = mean x major``, scaled to sum 3.

    Args:
        mean: mean axis as ``Lineation``, ``Vector3`` or array. Default (0, 0, 1).
        major: major axis, made perpendicular to `mean`. Default (1, 0, 0).
        shape: eigenvalues ``(a1, a2, a3)``. Default (2, 0.6, 0.4).

    The fit is the maximum-likelihood estimate by the fixed-point iteration of
    Tyler (1987) (López-Custodio, 2025, eq. 35).
    """

    dim = 3
    axial = True
    n_params = 5
    name = "acg"
    feature = Lineation

    def __init__(self, mean=(0, 0, 1), major=(1, 0, 0), shape=(2, 0.6, 0.4)):
        g1 = unit_vector("mean", mean)
        g2 = unit_vector("major", major)
        g2 = g2 - (g2 @ g1) * g1
        if np.linalg.norm(g2) < 1e-12:
            raise ValueError("major axis must not be parallel to mean")
        g2 = g2 / np.linalg.norm(g2)
        a = np.asarray(shape, dtype=float)
        if a.shape == (3,) and not np.all(np.isfinite(a)):
            raise ValueError(f"shape must be finite, got {a}")
        if a.shape != (3,) or np.any(a <= 0) or not (a[0] >= a[1] >= a[2]):
            raise ValueError("shape must be three positive decreasing eigenvalues")
        self._axes = np.array([g1, g2, np.cross(g1, g2)])
        self._shape = 3 * a / a.sum()

    @classmethod
    def from_matrix(cls, matrix):
        """Create ``ACG`` from the symmetric positive definite matrix ``Lambda``."""
        a, vecs = np.linalg.eigh(np.asarray(matrix, dtype=float))
        return cls(vecs[:, 2], vecs[:, 1], a[::-1])

    @property
    def matrix(self):
        """ndarray: the matrix ``Lambda``, scaled to trace 3."""
        return self._axes.T @ np.diag(self._shape) @ self._axes

    @property
    def params(self):
        return {
            "mean": Lineation(self._axes[0]),
            "major": Lineation(self._axes[1]),
            "shape": self._shape,
        }

    @property
    def mean(self):
        return Lineation(self._axes[0])

    @property
    def axes(self):
        """tuple: mean, major and minor axis as ``Lineation``."""
        return tuple(Lineation(g) for g in self._axes)

    def logpdf_array(self, X):
        g = X @ self._axes.T
        q = g**2 @ (1 / self._shape)
        return -1.5 * np.log(q) - np.log(4 * np.pi) - 0.5 * np.sum(np.log(self._shape))

    def rvs_array(self, n, rng):
        Y = rng.standard_normal((n, 3)) * np.sqrt(self._shape)
        return (Y / np.linalg.norm(Y, axis=1)[:, None]) @ self._axes

    @classmethod
    def _fit(cls, X, method="mle", tol=1e-10, maxiter=1000, **options):
        if method != "mle":
            raise ValueError(f"Unknown fitting method {method!r}")
        L = np.eye(3)
        converged = False
        for _ in range(maxiter):
            w = 1 / np.einsum("ij,jk,ik->i", X, np.linalg.inv(L), X)
            new = 3 * (X * w[:, None]).T @ X / w.sum()
            if np.abs(new - L).max() < tol:
                converged = True
                L = new
                break
            L = new
        dist = cls.from_matrix(L)
        return dist.params, converged, {}


_NM_OPTIONS = {"xatol": 1e-9, "fatol": 1e-12, "maxiter": 2000}


def _scatter_eigen(X):
    """Eigenvalues (descending) and eigenvectors (``Lineation``) of ``X'X / N``."""
    taus, vecs = np.linalg.eigh(X.T @ X / len(X))
    return taus[::-1], [Lineation(v) for v in vecs.T[::-1]]


def _watson_kappa(tau):
    """Maximum-likelihood Watson kappa for the eigenvalue ``tau`` of the mean axis."""
    if tau >= 1 - 1e-9:
        return np.inf
    if tau <= 1e-9:
        return -np.inf
    if abs(tau - 1 / 3) < 1e-9:
        return 0.0
    if tau > 1 / 3:
        bracket = (1e-9, 2 / (1 - tau) + 10)
    else:
        bracket = (-(2 / tau + 10), -1e-9)
    return brentq(lambda k: watson_mean_u2(k) - tau, *bracket)


def _watson_loglik(kappa, tau):
    """Maximized mean log-likelihood of the Watson fit (up to a constant)."""
    if np.isinf(kappa):
        return np.inf
    return kappa * tau - watson_log_norm(kappa)


def axial_cross_moments(X, vecs):
    """Matrix of ``mean((x.v_i)^2 (x.v_j)^2)`` of unit vectors `X` for axes `vecs`."""
    P = np.asarray(X, dtype=float) @ np.asarray([np.asarray(v) for v in vecs]).T
    P2 = P**2
    return P2.T @ P2 / len(P2)


def bingham_ellipse(taus, vecs, cross_moments, n, level=0.95, which=0):
    """Confidence ellipse of an eigenvector of the orientation tensor.

    Large-sample (central limit theorem) approximation: the angle of the
    sample eigenvector ``v_i`` toward ``v_j`` is approximately normal with
    variance ``m_ij / (n (tau_i - tau_j)^2)``, with ``m_ij = mean((x.v_i)^2
    (x.v_j)^2)``. The semi-angles of the region are ``arcsin(sqrt(chi2_2(level)
    * m_ij / (n (tau_i - tau_j)^2)))``, 90 degrees for equal eigenvalues.

    Args:
        taus: eigenvalues of the orientation tensor, descending.
        vecs: corresponding eigenvectors.
        cross_moments: 3x3 matrix ``m_ij``, see ``axial_cross_moments``.
        n: sample size.
        level: confidence level. Default 0.95.
        which: index of the eigenvector the ellipse is centered on. Default 0.

    Returns:
        ConfidenceRegion: ellipse around ``vecs[which]``.
    """
    if which not in (0, 1, 2):
        raise ValueError("which must be 0, 1 or 2")
    other = [i for i in range(3) if i != which]
    chi2 = stats.chi2.ppf(level, df=2)
    gammas = []
    for j in other:
        diff = taus[which] - taus[j]
        arg = np.inf
        if diff != 0:
            arg = np.sqrt(chi2 * cross_moments[which][j] / (n * diff**2))
        gammas.append(90.0 if arg >= 1 else float(np.degrees(np.arcsin(arg))))
    return ConfidenceRegion(
        mu=vecs[which],
        level=level,
        axes=(vecs[other[0]], vecs[other[1]]),
        gamma=tuple(gammas),
    )
