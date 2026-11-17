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
    """Log of ``M2(a) = (1 + a^2) Phi(a) + a phi(a)`` (Paine et al., 2018, eq. 4).

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


# Gauss-Legendre nodes and weights on [0, 1] of the Bingham normalizer
_GL_X, _GL_W = np.polynomial.legendre.leggauss(96)
_GL_X, _GL_W = (_GL_X + 1) / 2, _GL_W / 2


def _bingham_terms(k_major, k_minor, hessian=False):
    """Normalizer of the Bingham density and its derivatives.

    With ``u`` the coordinate along the minor axis (concentration ``k_minor``)
    the integral over the azimuth leaves ``F = 4 pi int_0^1 exp(k_minor u^2)
    h(z) du`` with ``h = i0e`` and ``z = -k_major (1 - u^2) / 2 >= 0``
    (``2 pi exp(-z) I0(z)`` is the integral of ``exp(k_major v^2)`` over the
    circle perpendicular to the minor axis, ``v`` the coordinate along the
    major axis). The integrand is smooth and decays like ``exp(k_minor u^2)``,
    so a fixed Gauss-Legendre rule on ``[0, min(1, 6 / sqrt(-k_minor))]`` is
    exact to rounding for any concentration (the integral and its moments
    agree with adaptive quadrature to about 1e-12).

    Returns:
        tuple: ``log F``, the gradient ``(E[(major.x)^2], E[(minor.x)^2])`` of
        ``log F`` and, if `hessian`, its Hessian (the covariance of these two
        statistics) with respect to ``(k_major, k_minor)``.
    """
    a, b = -k_minor, -k_major
    scale = 1.0 if a < 36 else 6 / np.sqrt(a)
    u, w = scale * _GL_X, scale * _GL_W
    u2 = u * u
    s = 1 - u2
    z = b * s / 2
    i0 = special.i0e(z)
    e = np.exp(-a * u2)
    h1 = special.i1e(z) - i0  # dh / dz
    # dh/dk_major = h1 * dz/dk_major = -h1 * s / 2
    F = np.sum(w * e * i0)
    g = np.array([np.sum(w * e * h1 * (-s / 2)), np.sum(w * e * u2 * i0)]) / F
    log_f = float(np.log(4 * np.pi * F))
    if not hessian:
        return log_f, g, None
    i1 = special.i1e(z)
    # d2h/dz2 = 2 i0e - 2 i1e - i1e / z, with i1e / z -> 1/2 for z -> 0
    ratio = np.where(z > 0, i1 / np.where(z > 0, z, 1.0), 0.5)
    h2 = 2 * i0 - 2 * i1 - ratio
    f_mm = np.sum(w * e * h2 * s * s / 4)
    f_mn = np.sum(w * e * u2 * h1 * (-s / 2))
    f_nn = np.sum(w * e * u2 * u2 * i0)
    H = np.array([[f_mm, f_mn], [f_mn, f_nn]]) / F - np.outer(g, g)
    return log_f, g, H


def bingham_log_norm(k_major, k_minor):
    """Log of the normalizer of the Bingham density on the sphere.

    The density is ``exp(k_major (major.x)^2 + k_minor (minor.x)^2) / F`` with
    ``k_minor <= k_major <= 0``; ``F`` is integrated numerically along the
    minor axis, see ``_bingham_terms``.
    """
    return _bingham_terms(k_major, k_minor)[0]


def bingham_moments(k_major, k_minor):
    """Expected ``(major.x)^2`` and ``(minor.x)^2`` of the Bingham distribution."""
    g = _bingham_terms(k_major, k_minor)[1]
    return float(g[0]), float(g[1])


def _newton_convex(terms, t, k, feasible, max_iter=60):
    """Minimize the convex ``-(k . t) + log_f(k)`` of an exponential family.

    Newton's method with the Hessian of `terms`, backtracking on the
    objective and the feasibility test `feasible`; it solves the likelihood
    equations ``grad log_f(k) = t``.

    Args:
        terms: ``terms(k) -> (log_f, grad, hess)`` at the point `k`.
        t: observed sufficient statistics (non-negative).
        k: feasible starting point.
        feasible: ``feasible(k) -> bool`` for the open parameter region.
        max_iter: maximum number of Newton steps.

    Returns:
        tuple: the last iterate (float array of 2) and whether it converged (the
        gradient matches `t` to a relative 1e-9 or the step is below rounding).
        A concentration beyond 1e12 (degenerate data, e.g. one great circle)
        stops the iteration unconverged.
    """
    log_f, g, H = terms(k)
    f = -(k @ t) + log_f
    for _ in range(max_iter):
        grad = g - t
        if np.all(np.abs(grad) <= 1e-13 + 1e-9 * np.abs(t)):
            return k, True
        try:
            step = -np.linalg.solve(H, grad)
        except np.linalg.LinAlgError:
            step = -grad
        lam = 1.0
        while True:
            kk = k + lam * step
            if feasible(kk):
                log_f2, g2, H2 = terms(kk)
                f2 = -(kk @ t) + log_f2
                if np.isfinite(f2) and f2 <= f + 1e-4 * lam * (grad @ step) + 1e-15:
                    break
            lam *= 0.5
            if lam < 1e-12:
                return k, False
        small = np.all(np.abs(kk - k) <= 1e-12 * (1 + np.abs(k)))
        k, f, g, H = kk, f2, g2, H2
        if small or np.max(np.abs(k)) > 1e12:
            return k, bool(small)
    return k, False


def bingham_fit_kappas(tau_major, tau_minor, start=None, max_iter=60):
    """Maximum-likelihood concentrations of a Bingham distribution.

    Solves ``E[(major.x)^2] = tau_major`` and ``E[(minor.x)^2] = tau_minor``,
    the likelihood equations with the eigenvalues of the (weighted) orientation
    tensor along the second and third axis. The negative log-likelihood
    ``-(k_major tau_major + k_minor tau_minor) + log F`` is convex, so Newton's
    method with the exact Hessian of ``_bingham_terms`` (``_newton_convex``)
    converges in a few iterations under the constraint ``k_minor < k_major < 0``.

    Args:
        tau_major: eigenvalue of the second axis.
        tau_minor: eigenvalue of the third axis.
        start: optional starting ``(k_major, k_minor)``, e.g. the previous
            estimate of an EM iteration. Default is the concentrated-data
            approximation ``k = -1 / (2 tau)``.
        max_iter: maximum number of Newton steps. Default 60.

    Returns:
        tuple: ``k_major``, ``k_minor`` and whether the iteration converged
        (False for degenerate data, e.g. on one great circle, where the
        concentration is infinite and the last iterate is returned).
    """
    t = np.array([tau_major, tau_minor], dtype=float)
    k = None
    if start is not None:
        start = np.asarray(start, dtype=float)
        if np.all(np.isfinite(start)) and start[0] < 0 and start[1] < start[0]:
            k = start
    if k is None:
        km = -1 / (2 * max(t[0], 1e-6))
        k = np.array([km, min(-1 / (2 * max(t[1], 1e-6)), km - 1e-3)])
    k, converged = _newton_convex(
        lambda kk: _bingham_terms(*kk, hessian=True),
        t,
        k,
        lambda kk: kk[0] < 0 and kk[1] < kk[0] - 1e-12,
        max_iter,
    )
    return float(k[0]), float(k[1]), converged


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


# Gauss-Legendre rules (u or t nodes, azimuth nodes) of the Lisle normalizer, chosen by
# the largest concentration so that the log normalizer is accurate to about 1e-10
_LISLE_TIERS = ((30, 32, 24), (300, 48, 32), (1000, 64, 48), (np.inf, 96, 64))
_LISLE_RULES = {}


def _lisle_rule(kappa):
    for top, n_t, n_psi in _LISLE_TIERS:
        if kappa <= top:
            break
    if (n_t, n_psi) not in _LISLE_RULES:
        xt, wt = np.polynomial.legendre.leggauss(n_t)
        xp, wp = np.polynomial.legendre.leggauss(n_psi)
        _LISLE_RULES[n_t, n_psi] = ((xt + 1) / 2, wt / 2, (xp + 1) / 2, wp / 2)
    return _LISLE_RULES[n_t, n_psi]


def _lisle_terms(kappa1, kappa2, gradient=False):
    """Log normalizer of the Lisle distribution and the statistics ``E T``.

    The density is ``exp(l' L l - n' L n) / c`` with ``L = diag(lam)``,
    ``lam = (kappa1 + kappa2, kappa2, 0)``. For a fixed lineation `l` the
    normal runs over the great circle perpendicular to it, where ``n' L n =
    (mu1 + mu2) / 2 + (mu1 - mu2) / 2 cos(2 phi)`` (``mu`` the eigenvalues of
    ``L`` restricted to the plane), so the integral over the circle is ``2 pi
    exp(-(mu1 + mu2) / 2) I0((mu1 - mu2) / 2)``. With ``q = l' L l``, ``s =
    lam1 + lam2 - q``, ``p = lam1 lam2 l3^2`` and ``d = sqrt(s^2 - 4 p)`` the
    integrand over the sphere is ``g(l) = exp(q - (s - d) / 2) i0e(d / 2)``,
    even in every coordinate, so it is integrated over one octant with
    ``l = (u, r cos psi, r sin psi)``, ``u = 1 - t^2``, ``r = sqrt(1 - u^2)``.
    The substitution ``t`` resolves the pole, where the azimuth integral changes
    on ``r^2 ~ 1 / kappa2``; ``t`` runs up to where ``exp(-kappa1 r^2)`` has
    fallen below 1e-17 and ``psi`` up to ``8 / (r sqrt(kappa2))``, beyond which
    the ridge ``exp(-kappa2 r^2 sin^2 psi)`` has vanished. Both use
    Gauss-Legendre rules (a dense-grid reference agrees to about 1e-10
    up to ``kappa = 1e4``; the previous 128-point azimuth rule was off by up to
    0.9 in the log for ``kappa1 ~ 0``, ``kappa2 >= 1e3``).

    With `gradient`, also returns ``(E T1, E T2)`` with ``T1 = l1^2 - n1^2``
    and ``T2 = T1 + l2^2 - n2^2``, the derivatives of the log normalizer with
    respect to ``kappa1`` and ``kappa2``, from the derivative of ``g``.
    """
    xt, wt, xp, wp = _lisle_rule(max(kappa1, kappa2))
    lam1, lam2 = kappa1 + kappa2, kappa2
    t_max = 1.0 if kappa1 <= 40 else np.sqrt(1 - np.sqrt(1 - 40 / kappa1))
    t = t_max * xt
    u = 1 - t * t
    w_u = 2 * t * t_max * wt
    r = np.sqrt(np.clip(1 - u * u, 0.0, None))
    if kappa2 > 0:
        psi_max = np.minimum(np.pi / 2, 8 / (r * np.sqrt(kappa2)))
    else:
        psi_max = np.full_like(r, np.pi / 2)
    psi = psi_max[:, None] * xp[None, :]
    w = (w_u * psi_max)[:, None] * wp[None, :]
    l1sq = (u * u)[:, None]
    l2sq = (r[:, None] * np.cos(psi)) ** 2
    l3sq = (r[:, None] * np.sin(psi)) ** 2
    q = lam1 * l1sq + lam2 * l2sq
    s = lam1 + lam2 - q
    d = np.sqrt(np.clip(s * s - 4 * lam1 * lam2 * l3sq, 0.0, None))
    e = np.exp(q - (s - d) / 2 - lam1)
    i0 = special.i0e(d / 2)
    total = np.sum(w * e * i0)
    # both hemispheres and octants (8), the circle of normals (2 pi) and the
    # volume 8 pi^2 of SO(3)
    log_c = float(np.log(2 * 4 * total * 2 * np.pi / (8 * np.pi**2)) + lam1)
    if not gradient:
        return log_c
    i1 = special.i1e(d / 2)
    # i1e(d / 2) / d -> 1/4 for d -> 0
    ratio = np.where(d > 1e-8, i1 / np.where(d > 1e-8, d, 1.0), 0.25)
    stats_ = []
    for dl1, dl2 in ((1.0, 0.0), (1.0, 1.0)):
        dq = dl1 * l1sq + dl2 * l2sq
        ds = dl1 + dl2 - dq
        dp = (dl1 * lam2 + lam1 * dl2) * l3sq
        dg = e * (i0 * (dq - ds / 2) + ratio * (s * ds - 2 * dp) / 2)
        stats_.append(np.sum(w * dg) / total)
    return log_c, np.array(stats_)


def lisle_log_norm(kappa1, kappa2):
    """Log of the normalizer of the Lisle distribution of pairs.

    See ``_lisle_terms``; it is 0 for ``kappa1 = kappa2 = 0``.
    """
    return _lisle_terms(kappa1, kappa2)


def _lisle_terms_hessian(k):
    """``(log c, gradient, Hessian)`` of the Lisle normalizer at ``k = (kappa1,
    kappa2)``; the Hessian is the forward difference of the analytic gradient."""
    log_c, g = _lisle_terms(*k, gradient=True)
    cols = []
    for i in range(2):
        h = 1e-6 * max(1.0, k[i])
        kk = np.array(k, dtype=float)
        kk[i] += h
        cols.append((_lisle_terms(*kk, gradient=True)[1] - g) / h)
    H = np.array(cols).T
    return log_c, g, (H + H.T) / 2


def lisle_fit_kappas(tau0, tau1, start=None, max_iter=60):
    """Maximum-likelihood concentrations of the Lisle distribution.

    The model is the exponential family with the statistics ``T1 = tau0`` and
    ``T2 = tau0 + tau1`` (the two largest eigenvalues of Lisle's tensor), so
    the likelihood equations are ``E T = (tau0, tau0 + tau1)``, solved by
    Newton's method (``_newton_convex``) with the analytic gradient of
    ``_lisle_terms`` and a finite-difference Hessian, under ``kappa > 0``.

    Args:
        tau0: largest eigenvalue of ``mean(l l' - n n')``.
        tau1: second eigenvalue.
        start: optional starting ``(kappa1, kappa2)``. Default is the
            concentrated-data approximation from ``1 - tau0`` and ``1 + tau2``.
        max_iter: maximum number of Newton steps. Default 60.

    Returns:
        tuple: ``kappa1``, ``kappa2`` and whether the iteration converged.
        Degenerate data (identical frames, ``tau0 = 1``) give a huge finite
        concentration (about 1e9) and uniform data ones near 0; the flag is
        False only if the iteration fails or the concentration passes 1e12.
    """
    t = np.array([tau0, tau0 + tau1], dtype=float)
    k = None
    if start is not None:
        start = np.asarray(start, dtype=float)
        if np.all(np.isfinite(start)) and np.all(start > 0):
            k = start
    if k is None:
        k = np.array([1 / max(1 - t[0], 1e-3), 1 / max(1 - t[1], 1e-3)], dtype=float)
    k, converged = _newton_convex(
        _lisle_terms_hessian, t, k, lambda kk: bool(np.all(kk > 0)), max_iter
    )
    return float(k[0]), float(k[1]), converged
