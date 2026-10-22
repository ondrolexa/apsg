"""
Normalizing constants and helper functions of the spherical distributions.
"""

import numpy as np
from scipy import special
from scipy.integrate import quad


def kent_log_norm(kappa, beta):
    """Log of the normalizing constant of the Kent distribution.

    Evaluates ``c(kappa, beta) = int exp(kappa u) 2 pi I0(beta (1 - u^2)) du``
    over ``u`` in [-1, 1], the integral of the Kent density over the azimuth
    about the mean direction, scaled by the maximum of the exponent so that it
    does not overflow for large concentrations.
    """
    if beta == 0:
        return log_fisher_norm(kappa)
    # maximum of kappa u + beta (1 - u^2) on [-1, 1]
    u_max = min(kappa / (2 * beta), 1.0)
    top = kappa * u_max + beta * (1 - u_max**2)

    def integrand(u):
        z = beta * (1 - u * u)
        return np.exp(kappa * u + z - top) * special.i0e(z)

    width = 1 / max(kappa, 1.0)
    points = sorted({u_max, max(u_max - 10 * width, -1.0)} - {-1.0, 1.0})
    value, _ = quad(integrand, -1, 1, points=points or None, limit=200, epsabs=0)
    return float(np.log(2 * np.pi * value) + top)


def log_fisher_norm(kappa):
    """Log of ``4 pi sinh(kappa) / kappa``, the normalizer of the Fisher density."""
    if kappa < 1e-8:
        return float(np.log(4 * np.pi))
    return float(np.log(2 * np.pi / kappa) + kappa + np.log1p(-np.exp(-2 * kappa)))


def log_m2(alpha):
    """Log of ``M2(a) = (1 + a^2) Phi(a) + a phi(a)`` (López-Custodio, 2025, eq. 23).

    For negative ``a`` it is evaluated through the scaled complementary error
    function to avoid cancellation.
    """
    alpha = np.asarray(alpha, dtype=float)
    out = np.empty_like(alpha)
    pos = alpha >= 0
    a = alpha[pos]
    out[pos] = np.log(
        (1 + a**2) * special.ndtr(a) + a * np.exp(-(a**2) / 2) / np.sqrt(2 * np.pi)
    )
    a = alpha[~pos]
    # Phi(a) / phi(a) for a < 0
    mills = np.sqrt(np.pi / 2) * special.erfcx(-a / np.sqrt(2))
    inner = np.maximum((1 + a**2) * mills + a, np.finfo(float).tiny)
    out[~pos] = -(a**2) / 2 - 0.5 * np.log(2 * np.pi) + np.log(inner)
    return out


def watson_log_norm(kappa):
    """Log of Kummer's function ``M(1/2, 3/2, k) = int_0^1 exp(k u^2) du``.

    The normalizer of the Watson density on the sphere is ``4 pi M``. It is
    evaluated through Dawson's integral (``k > 0``) and the error function
    (``k < 0``).
    """
    if abs(kappa) < 1e-8:
        return kappa / 3
    s = np.sqrt(abs(kappa))
    if kappa > 0:
        return float(kappa + np.log(special.dawsn(s) / s))
    return float(np.log(np.sqrt(np.pi) * special.erf(s) / (2 * s)))


def watson_mean_u2(kappa):
    """Expected ``(mu.x)^2`` of the Watson distribution, ``d/dk log M``."""
    if abs(kappa) < 1e-4:
        return 1 / 3 + 4 * kappa / 45
    s = np.sqrt(abs(kappa))
    if kappa > 0:
        return 1 / (2 * s * special.dawsn(s)) - 1 / (2 * kappa)
    return -1 / (2 * kappa) - np.exp(kappa) / (np.sqrt(np.pi) * s * special.erf(s))


def _bingham_integral(k_axis, k_other, power=0):
    """``2 pi int u^power exp(k_axis u^2) i0e(-k_other (1 - u^2) / 2) du`` on [-1, 1].

    With ``u`` the coordinate along one axis of a Bingham distribution with
    concentrations ``k_axis`` along it and ``k_other`` along a second one (and
    zero along the third), the integral over the azimuth is
    ``2 pi exp(a / 2) I0(a / 2)`` with ``a = k_other (1 - u^2)``, which is
    ``2 pi i0e(-a / 2)`` for ``a <= 0``.
    """

    def integrand(u):
        return (
            u**power * np.exp(k_axis * u * u) * special.i0e(-k_other * (1 - u * u) / 2)
        )

    points = None
    if k_axis < -25:
        points = [5 / np.sqrt(-k_axis)]
    value, _ = quad(integrand, 0, 1, points=points, limit=200, epsabs=0)
    return 4 * np.pi * value


def bingham_log_norm(k_major, k_minor):
    """Log of the normalizer of the Bingham density on the sphere.

    The density is ``exp(k_major (major.x)^2 + k_minor (minor.x)^2) / F`` with
    ``k_minor <= k_major <= 0``; ``F`` is integrated numerically along the
    minor axis.
    """
    return float(np.log(_bingham_integral(k_minor, k_major)))


def bingham_moments(k_major, k_minor):
    """Expected ``(major.x)^2`` and ``(minor.x)^2`` of the Bingham distribution."""
    F = _bingham_integral(k_minor, k_major)
    m_major = _bingham_integral(k_major, k_minor, power=2) / F
    m_minor = _bingham_integral(k_minor, k_major, power=2) / F
    return m_major, m_minor


def restricted_eigen(lam, L):
    """Sum and difference of the eigenvalues of ``diag(lam)`` restricted to the
    plane perpendicular to each unit row of `L`.

    Returns:
        tuple: ``l' diag(lam) l``, ``mu1 + mu2`` and ``mu1 - mu2`` per row.
    """
    q = (L**2) @ lam
    s = lam.sum() - q
    adj = np.array([lam[1] * lam[2], lam[0] * lam[2], lam[0] * lam[1]])
    p = (L**2) @ adj
    return q, s, np.sqrt(np.clip(s**2 - 4 * p, 0, None))


_PSI = np.linspace(0, 2 * np.pi, 129)[:-1]


def lisle_log_norm(kappa1, kappa2):
    """Log of the normalizer of the Lisle distribution of pairs.

    The density is ``exp(l' L l - n' L n) / c`` with respect to the uniform
    distribution of frames, ``L = diag(kappa1 + kappa2, kappa2, 0)``. For a fixed
    lineation `l` the normal runs over the great circle perpendicular to it,
    where ``n' L n = (mu1 + mu2) / 2 + (mu1 - mu2) / 2 cos(2 phi)`` (``mu`` the
    eigenvalues of ``L`` restricted to the plane), so the integral over the
    circle is ``2 pi exp(-(mu1 + mu2) / 2) I0((mu1 - mu2) / 2)``. The remaining
    integral over the sphere is evaluated along the first axis (quadrature)
    and around it (trapezoid rule, exact for this smooth periodic integrand).
    """
    lam = np.array([kappa1 + kappa2, kappa2, 0.0])
    top = lam[0]

    def along(u):
        r = np.sqrt(max(1 - u * u, 0.0))
        L = np.column_stack([np.full_like(_PSI, u), r * np.cos(_PSI), r * np.sin(_PSI)])
        q, s, d = restricted_eigen(lam, L)
        return 2 * np.pi * np.mean(np.exp(q - (s - d) / 2 - top) * special.i0e(d / 2))

    width = 1 / max(lam[0] - lam[1], 1.0)
    points = sorted({max(0.0, 1 - 30 * width)} - {0.0}) or None
    value, _ = quad(along, 0, 1, points=points, limit=200, epsabs=0)
    # both hemispheres, the circle of normals and the volume 8 pi^2 of SO(3)
    return float(np.log(2 * value * 2 * np.pi / (8 * np.pi**2)) + top)
