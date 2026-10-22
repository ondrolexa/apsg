"""
Distributions of directions (vectorial data) on the sphere.
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
from apsg.feature._distributions._special import kent_log_norm, log_fisher_norm, log_m2
from apsg.helpers._math import acosd
from apsg.math._vector import Vector3


class Uniform3(OrientationDistribution):
    """Uniform distribution of directions on the sphere."""

    dim = 3
    axial = False
    n_params = 0
    name = "uniform"
    feature = Vector3

    def logpdf_array(self, X):
        return np.full(len(X), -np.log(4 * np.pi))

    def rvs_array(self, n, rng):
        X = rng.standard_normal((n, 3))
        return X / np.linalg.norm(X, axis=1)[:, None]

    @classmethod
    def _fit(cls, X, method="mle", **options):
        return {}, True, {}

    @property
    def params(self):
        return {}


class VonMisesFisher(OrientationDistribution):
    """von Mises-Fisher (Fisher) distribution of directions on the sphere.

    The density is ``kappa / (4 pi sinh(kappa)) exp(kappa mu.x)``
    (López-Custodio, 2025, eq. 11).

    Args:
        mean: mean direction as ``Vector3`` or array. Default (0, 0, 1).
        kappa: concentration parameter. Default 1.

    The fit is the maximum-likelihood estimate (``coth(kappa) - 1/kappa =
    Rbar``). The confidence region of the mean direction is the cone of
    Fisher (1953).
    """

    dim = 3
    axial = False
    n_params = 3
    name = "fisher"
    feature = Vector3

    def __init__(self, mean=(0, 0, 1), kappa=1):
        self._mean = Vector3(unit_vector("mean", Vector3(mean)))
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
        return self.kappa * c - log_fisher_norm(self.kappa)

    def rvs_array(self, n, rng):
        mean = np.asarray(self._mean, dtype=float)
        if self.kappa == 0:
            return Uniform3().rvs_array(n, rng)
        X = stats.vonmises_fisher(mean, self.kappa).rvs(n, random_state=rng)
        return np.atleast_2d(X)

    @classmethod
    def _fit(cls, X, method="mle", **options):
        if method != "mle":
            raise ValueError(f"Unknown fitting method {method!r}")
        resultant = X.sum(axis=0)
        R = np.linalg.norm(resultant)
        Rbar = R / len(X)
        mean = Vector3(resultant) if R > 0 else Vector3(0, 0, 1)
        return {"mean": mean, "kappa": _fisher_kappa_mle(Rbar)}, True, {"Rbar": Rbar}

    def _mean_confidence(self, level):
        alpha = fisher_cone(self.fit_info.n, self.fit_info.stats["Rbar"], level)
        return ConfidenceRegion(mu=self._mean, level=level, alpha=alpha)


class Kent(OrientationDistribution):
    """Kent (5-parameter Fisher-Bingham, FB5) distribution on the sphere.

    The density is ``exp(kappa g1.x + beta ((g2.x)^2 - (g3.x)^2)) / c(kappa,
    beta)`` (Kent, 1982; López-Custodio, 2025, eq. 16), with ``g1`` the mean
    direction, ``g2`` the major axis (direction of the largest spread) and
    ``g3 = g1 x g2``. It is unimodal for ``2 beta < kappa``.

    Args:
        mean: mean direction as ``Vector3`` or array. Default (0, 0, 1).
        major: major axis, made perpendicular to `mean`. Default (1, 0, 0).
        kappa: concentration parameter. Default 10.
        beta: ellipticity, ``beta >= 0``. Default 0.

    Fitting methods are "moments" (the moment estimates of Kent, 1982,
    López-Custodio, 2025, eq. 18-20) and "mle" (axes from the moments, kappa
    and beta by maximum likelihood with ``2 beta < kappa``).

    The normalizer is evaluated as a one-dimensional integral. Sampling draws
    ``u = g1.x`` from its marginal density ``exp(kappa u) I0(beta (1 - u^2))``
    by rejection and the azimuth about ``g1`` from its conditional, an axial
    von Mises distribution.
    """

    dim = 3
    axial = False
    n_params = 5
    name = "kent"
    feature = Vector3

    def __init__(self, mean=(0, 0, 1), major=(1, 0, 0), kappa=10, beta=0):
        g1 = unit_vector("mean", Vector3(mean))
        g2 = unit_vector("major", major)
        g2 = g2 - (g2 @ g1) * g1
        if np.linalg.norm(g2) < 1e-12:
            raise ValueError("major axis must not be parallel to mean")
        self._gamma = np.array([g1, g2 / np.linalg.norm(g2), np.zeros(3)])
        self._gamma[2] = np.cross(self._gamma[0], self._gamma[1])
        kappa = finite_param("kappa", kappa)
        beta = finite_param("beta", beta)
        if not kappa >= 0 or not beta >= 0:
            raise ValueError("kappa and beta must be non-negative")
        self.kappa = kappa
        self.beta = beta
        self._log_norm = kent_log_norm(self.kappa, self.beta)

    @property
    def params(self):
        return {
            "mean": Vector3(self._gamma[0]),
            "major": Vector3(self._gamma[1]),
            "kappa": self.kappa,
            "beta": self.beta,
        }

    @property
    def mean(self):
        return Vector3(self._gamma[0])

    @property
    def axes(self):
        """tuple: mean direction, major and minor axis as ``Vector3``."""
        return tuple(Vector3(g) for g in self._gamma)

    def logpdf_array(self, X):
        g = X @ self._gamma.T
        return (
            self.kappa * g[:, 0]
            + self.beta * (g[:, 1] ** 2 - g[:, 2] ** 2)
            - self._log_norm
        )

    def rvs_array(self, n, rng):
        u = _kent_sample_u(self.kappa, self.beta, n, rng)
        s = np.sqrt(np.clip(1 - u**2, 0, None))
        # azimuth about g1 given u: axial von Mises (of the doubled angle)
        conc = np.maximum(self.beta * s**2, 1e-12)
        phi = stats.vonmises.rvs(conc, random_state=rng) / 2
        phi = phi + np.pi * (rng.uniform(size=n) < 0.5)
        local = np.column_stack([u, s * np.cos(phi), s * np.sin(phi)])
        return local @ self._gamma

    @classmethod
    def _fit(cls, X, method="mle", **options):
        resultant = X.sum(axis=0)
        Rbar = np.linalg.norm(resultant) / len(X)
        g1 = resultant / np.linalg.norm(resultant)
        e2, e3 = _tangent_basis(g1)
        # second moments of the data in the tangent plane of the mean
        P = X @ np.column_stack([e2, e3])
        B = P.T @ P / len(X)
        lam, vec = np.linalg.eigh(B)
        major = vec[0, 1] * e2 + vec[1, 1] * e3
        r2 = lam[1] - lam[0]
        a = max(2 - 2 * Rbar - r2, 1e-12)
        b = max(2 - 2 * Rbar + r2, 1e-12)
        kappa, beta = 1 / a + 1 / b, (1 / a - 1 / b) / 2
        stats_ = {"Rbar": Rbar, "tangent_moments": tuple(lam[::-1])}
        params = {"mean": g1, "major": major, "kappa": kappa, "beta": beta}
        match method:
            case "moments":
                return params, True, stats_
            case "mle":
                G = np.array([g1, major, np.cross(g1, major)])
                g = X @ G.T
                s1 = g[:, 0].mean()
                s2 = (g[:, 1] ** 2 - g[:, 2] ** 2).mean()

                def nll(p):
                    k = np.exp(p[0])
                    b = k / 2 * special.expit(p[1])
                    return -(k * s1 + b * s2 - kent_log_norm(k, b))

                start = [
                    np.log(kappa),
                    special.logit(np.clip(2 * beta / kappa, 1e-6, 1 - 1e-6)),
                ]
                res = minimize(nll, start, method="Nelder-Mead", options=_NM_OPTIONS)
                params["kappa"] = float(np.exp(res.x[0]))
                params["beta"] = float(params["kappa"] / 2 * special.expit(res.x[1]))
                return params, bool(res.success), stats_
        raise ValueError(f"Unknown fitting method {method!r}")

    def _mean_confidence(self, level):
        stats_ = self.fit_info.stats
        return mean_direction_ellipse(
            self.mean,
            self.axes[1:],
            stats_["tangent_moments"],
            self.fit_info.n,
            stats_["Rbar"],
            level,
        )


class ESAG(OrientationDistribution):
    """Elliptically symmetric angular Gaussian distribution on the sphere.

    The distribution of ``y / |y|`` for ``y ~ N(mu, V)`` with ``V mu = mu`` and
    ``|V| = 1`` (Paine et al., 2018; López-Custodio, 2025, eq. 22-23). ``V`` has
    eigenvalue 1 along `mu`, ``1 / rho`` along the major axis and ``rho``
    along the minor axis.

    Args:
        mu: mean vector as ``Vector3`` or array; its direction is the mean
            direction and its length the concentration. Default (0, 0, 3).
        major: major axis, made perpendicular to `mu`. Default (1, 0, 0).
        rho: shape parameter, ``0 < rho <= 1``, 1 for the isotropic case.
            Default 1.

    The fit is the maximum-likelihood estimate (López-Custodio, 2025, eq. 27).
    """

    dim = 3
    axial = False
    n_params = 5
    name = "esag"
    feature = Vector3

    def __init__(self, mu=(0, 0, 3), major=(1, 0, 0), rho=1):
        g1 = unit_vector("mu", mu)
        mu = np.asarray(mu, dtype=float)
        rho = finite_param("rho", rho)
        if not 0 < rho <= 1:
            raise ValueError("rho must be in (0, 1]")
        g2 = unit_vector("major", major)
        g2 = g2 - (g2 @ g1) * g1
        if np.linalg.norm(g2) < 1e-12:
            raise ValueError("major axis must not be parallel to mu")
        g2 = g2 / np.linalg.norm(g2)
        g3 = np.cross(g1, g2)
        self._mu = mu
        self._axes = np.array([g1, g2, g3])
        self.rho = rho
        self._Vinv = (
            np.outer(g1, g1) + self.rho * np.outer(g2, g2) + np.outer(g3, g3) / self.rho
        )

    @property
    def params(self):
        return {
            "mu": Vector3(self._mu),
            "major": Vector3(self._axes[1]),
            "rho": self.rho,
        }

    @property
    def mean(self):
        return Vector3(self._axes[0])

    @property
    def axes(self):
        """tuple: mean direction, major and minor axis as ``Vector3``."""
        return tuple(Vector3(g) for g in self._axes)

    def logpdf_array(self, X):
        return _esag_logpdf(X, self._mu, self._Vinv)

    def rvs_array(self, n, rng):
        V = np.linalg.inv(self._Vinv)
        Y = rng.multivariate_normal(self._mu, V, size=n)
        return Y / np.linalg.norm(Y, axis=1)[:, None]

    @classmethod
    def _fit(cls, X, method="mle", **options):
        if method != "mle":
            raise ValueError(f"Unknown fitting method {method!r}")
        resultant = X.sum(axis=0)
        Rbar = np.linalg.norm(resultant) / len(X)
        g1 = resultant / np.linalg.norm(resultant)
        # work in a frame with the sample mean on z, away from the singular
        # direction of the parametrization of V (López-Custodio, 2025, eq. 24)
        e1, e2 = _tangent_basis(g1)
        F = np.array([e1, e2, g1])
        Y = X @ F.T

        def unpack(p):
            mu = p[:3]
            k = mu / np.linalg.norm(mu)
            j = np.cross(k, [1.0, 0, 0])
            j = j / np.linalg.norm(j)
            i = np.cross(j, k)
            g1_, g2_ = p[3], p[4]
            s = np.sqrt(g1_**2 + g2_**2 + 1)
            Vinv = (
                np.eye(3)
                + g1_ * (np.outer(i, i) - np.outer(j, j))
                + g2_ * (np.outer(i, j) + np.outer(j, i))
                + (s - 1) * (np.outer(i, i) + np.outer(j, j))
            )
            return mu, Vinv

        def nll(p):
            mu, Vinv = unpack(p)
            return -np.mean(_esag_logpdf(Y, mu, Vinv))

        kappa = _fisher_kappa_mle(Rbar)
        start = np.array([0, 0, np.sqrt(max(kappa, 1e-3)), 0, 0])
        res = minimize(nll, start, method="BFGS")
        mu, Vinv = unpack(res.x)
        lam, vec = np.linalg.eigh(Vinv)
        # eigenvalues of Vinv: rho (major), 1 (mean), 1 / rho (minor)
        major = vec[:, 0]
        rho = float(np.clip(lam[0], 1e-12, 1.0))
        params = {"mu": mu @ F, "major": major @ F, "rho": rho}
        # tangent moments about the sample mean, for the confidence region
        P = X @ np.column_stack([e1, e2])
        lam_t, vec_t = np.linalg.eigh(P.T @ P / len(X))
        stats_ = {
            "Rbar": Rbar,
            "mean_direction": Vector3(g1),
            "tangent_axes": tuple(
                Vector3(v[0] * e1 + v[1] * e2) for v in vec_t.T[::-1]
            ),
            "tangent_moments": tuple(lam_t[::-1]),
        }
        return params, bool(res.success), stats_

    def _mean_confidence(self, level):
        stats_ = self.fit_info.stats
        return mean_direction_ellipse(
            stats_["mean_direction"],
            stats_["tangent_axes"],
            stats_["tangent_moments"],
            self.fit_info.n,
            stats_["Rbar"],
            level,
        )


_NM_OPTIONS = {"xatol": 1e-9, "fatol": 1e-12, "maxiter": 2000}


def _esag_logpdf(X, mu, Vinv):
    a = np.einsum("ij,jk,ik->i", X, Vinv, X)
    t = X @ mu
    return (
        -np.log(2 * np.pi)
        - 1.5 * np.log(a)
        + 0.5 * (t**2 / a - mu @ mu)
        + log_m2(t / np.sqrt(a))
    )


def _tangent_basis(g):
    """Return two unit vectors forming a right-handed frame with unit vector `g`."""
    ref = np.eye(3)[np.argmin(np.abs(g))]
    e1 = np.cross(ref, g)
    e1 = e1 / np.linalg.norm(e1)
    return e1, np.cross(g, e1)


def _fisher_kappa_mle(Rbar):
    """Solve ``coth(kappa) - 1/kappa = Rbar`` for kappa."""
    if Rbar < 1e-12:
        return 0.0
    if Rbar >= 1 - 1e-12:
        return np.inf

    def A3(k):
        if k < 1e-4:
            return k / 3
        return 1 / np.tanh(k) - 1 / k

    return brentq(lambda k: A3(k) - Rbar, 0, 1 / (1 - Rbar) + 10, xtol=1e-14)


def _kent_sample_u(kappa, beta, n, rng):
    """Sample ``u = g1.x`` of the Kent distribution by rejection.

    The marginal density of ``u`` is ``exp(kappa u) I0(beta (1 - u^2))`` on
    [-1, 1]. For small beta the envelope is the marginal of the Fisher
    distribution, ``exp(kappa u)``; otherwise it is ``exp(kappa u + beta (1 -
    u^2))``, a normal density truncated to [-1, 1], and the acceptance
    probability is ``I0(z) exp(-z)`` with ``z = beta (1 - u^2)``.
    """
    out = np.empty(0)
    while len(out) < n:
        m = max(2 * (n - len(out)), 16)
        if beta < 0.5:
            xi = rng.uniform(size=m)
            if kappa > 1e-8:
                u = 1 + np.log(xi + (1 - xi) * np.exp(-2 * kappa)) / kappa
            else:
                u = 2 * xi - 1
            accept = special.i0(beta * (1 - u**2)) / special.i0(beta)
        else:
            loc, scale = kappa / (2 * beta), 1 / np.sqrt(2 * beta)
            a, b = (-1 - loc) / scale, (1 - loc) / scale
            u = stats.truncnorm.rvs(
                a, b, loc=loc, scale=scale, size=m, random_state=rng
            )
            accept = special.i0e(beta * (1 - u**2))
        out = np.concatenate([out, u[rng.uniform(size=m) < accept]])
    return np.clip(out[:n], -1, 1)


def fisher_cone(N, Rbar, level=0.95):
    """Half-angle (degrees) of the confidence cone of the Fisher mean direction.

    ``cos(alpha) = 1 - (N - R) / R * ((1 - level)^(-1 / (N - 1)) - 1)``
    (Fisher, 1953).
    """
    if Rbar >= 1.0 or N <= 1:
        return 0.0
    R = N * Rbar
    cos_theta = 1.0 - ((N - R) / R) * ((1 - level) ** (-1.0 / (N - 1.0)) - 1.0)
    return acosd(np.clip(cos_theta, -1.0, 1.0))


def mean_direction_ellipse(mu, axes, moments, N, Rbar, level=0.95):
    """Large-sample confidence ellipse of the mean direction.

    By the central limit theorem the tangent-plane components of the sample
    mean direction are approximately normal with covariance ``S / (N Rbar^2)``,
    where ``S`` holds the second moments of the data in the tangent plane
    (eigenvalues `moments` along `axes`). The semi-angles of the region are
    ``arcsin(sqrt(chi2_2(level) * l / (N Rbar^2)))``.

    Returns:
        ConfidenceRegion: ellipse around `mu`.
    """
    chi2 = stats.chi2.ppf(level, df=2)
    gammas = []
    for lam in moments:
        arg = np.sqrt(chi2 * lam / (N * Rbar**2)) if Rbar > 0 else np.inf
        gammas.append(90.0 if arg >= 1 else float(np.degrees(np.arcsin(arg))))
    return ConfidenceRegion(mu=mu, level=level, axes=tuple(axes), gamma=tuple(gammas))
