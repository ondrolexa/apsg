"""Mean inclination and precision from inclination-only data."""

import warnings

import numpy as np
from scipy import optimize, special, stats

__all__ = ("inclination_only_mean",)

_T63 = 0.63  # fraction of a Fisher distribution within the angle t63


def _cubic(theta):
    """Left side of McFadden & Reid (1982) eq. 19 as a function of ``theta0``."""
    n, sc, ss = len(theta), np.cos(theta).sum(), np.sin(theta).sum()

    def f(t):
        return (
            n * np.cos(t)
            + (np.sin(t) ** 2 - np.cos(t) ** 2) * sc
            - 2 * np.sin(t) * np.cos(t) * ss
        )

    return f


def _theta0_roots(theta, grid=4000):
    """Roots of McFadden & Reid (1982) eq. 19 on ``(0, pi)`` and their ``U``."""
    n = len(theta)
    f = _cubic(theta)
    x = np.linspace(1e-6, np.pi - 1e-6, grid)
    v = f(x)
    roots = [
        optimize.brentq(f, x[i], x[i + 1]) for i in np.flatnonzero(v[:-1] * v[1:] < 0)
    ]
    out = []
    for t in roots:
        C = np.cos(t - theta).sum()
        U = 0.5 * n * (1 / np.sin(t) ** 2 - C / (n - C)) if n > C else -np.inf
        out.append((t, U, C))
    return out


def _al_nll(params, theta):
    """Negative log-likelihood of colatitudes `theta` of a Fisher distribution.

    The density of the colatitude, the declination integrated out, is
    ``kappa / (2 sinh kappa) sin(t) exp(kappa cos t cos t0) I0(kappa sin t
    sin t0)`` (Arason & Levi, 2010); `params` are ``(t0, log kappa)``.
    """
    t0, lk = params
    k = np.exp(lk)
    x = k * np.sin(theta) * np.sin(t0)
    log_i0 = np.log(special.ive(0, x)) + x
    norm = np.log(k) - k - np.log(-np.expm1(-2 * k))
    return -np.sum(
        norm + np.log(np.sin(theta)) + k * np.cos(theta) * np.cos(t0) + log_i0
    )


def _arason_levi(inc, level):
    """Exact maximum likelihood fit of inclination-only data (see the caller)."""
    n = len(inc)
    theta = np.radians(90 - inc)
    best = None
    with np.errstate(all="ignore"):
        for t0 in np.radians(np.linspace(20, 160, 5)):
            for lk in (0.0, 3.0):
                fit = optimize.minimize(
                    _al_nll,
                    [t0, lk],
                    args=(theta,),
                    method="L-BFGS-B",
                    bounds=[(0, np.pi), (-8, 14)],
                )
                if np.isfinite(fit.fun) and (best is None or fit.fun < best.fun):
                    best = fit
    if best is None:  # pragma: no cover
        raise ValueError("The maximum likelihood fit failed")
    t0, kappa = best.x[0], float(np.exp(best.x[1]))
    # a likelihood no larger than that of the uniform distribution (density
    # sin(t) / 2) by less than 1e-3 cannot tell the data from it
    if -best.fun - np.sum(np.log(np.sin(theta) / 2)) < 1e-3:
        kappa = 0.0
    # confidence cone of Fisher's distribution for the resultant length R that
    # gives the precision, kappa = (n - 1) / (n - R)
    R = n - (n - 1) / kappa if kappa > 0 else -np.inf
    if R > 0:
        p = 1 - level
        cos_a = 1 - (n - R) / R * (p ** (-1 / (n - 1)) - 1)
        alpha = float(np.degrees(np.arccos(np.clip(cos_a, -1, 1))))
    else:
        alpha = float("nan")
    if kappa:
        cos_t63 = 1 + np.log(1 - _T63 * (1 - np.exp(-2 * kappa))) / kappa
    else:  # uniform distribution: the cap holding 63 % of the sphere
        cos_t63 = 1 - 2 * _T63
    return {
        "inc": float(90 - np.degrees(t0)) if kappa else float("nan"),
        "alpha": alpha,
        "k": kappa,
        "t63": float(np.degrees(np.arccos(np.clip(cos_t63, -1, 1)))),
        "loglik": float(-best.fun),
        "theta0": float(t0),
        "n": n,
        "method": "arason_levi",
    }


def inclination_only_mean(inclinations, level=0.95, method="mcfadden_reid"):
    """Mean inclination of Fisher distributed directions of unknown declination.

    `method` ``"arason_levi"`` is the exact maximum-likelihood estimate of
    Arason & Levi (2010); ``"mcfadden_reid"`` the approximation of McFadden &
    Reid (1982), described below, which is the default. The exact estimate
    maximizes the likelihood of the inclinations with ``theta = 90 - I`` and the
    colatitude density of a Fisher distribution of unknown declination (given
    in ``_al_nll``) over the mean inclination and ``kappa``, so that it needs no
    root selection, and it is also valid for the steepest directions. It
    returns ``inc``, ``k`` (the maximum-likelihood ``kappa``), ``alpha`` (the
    Fisher cone ``alpha_95`` for the resultant length ``R = N - (N - 1) /
    kappa``, NaN when ``R <= 0``), ``t63`` (the angle containing 63 % of the
    fitted distribution), ``loglik``, ``theta0``, ``n`` and ``method``. The
    estimate lies at the pole (``inc = +-90``) when the data fold back through the
    vertical, and ``kappa = 0``, ``inc = NaN`` when their likelihood hardly exceeds that of the
    uniform distribution (by less than 1e-3). The values agree with those printed for the
    examples of PuffinPlot (Lurcock & Wilson, 2012), which implements the
    method.

    The McFadden & Reid estimate follows. Approximate maximum-likelihood estimate for data
    that do not fold back through the vertical. With ``theta = 90 - I`` the
    estimate ``theta0`` is the root of the cubic (eq. 19)

        ``N cos(t) + (sin^2 t - cos^2 t) sum(cos theta_i)
        - 2 sin(t) cos(t) sum(sin theta_i) = 0``

    with ``U = N/2 (1/sin^2 t - C / (N - C)) < 0`` (eq. 19a), where ``C =
    sum(cos(theta0 - theta_i))`` and ``S = sum(sin(theta0 - theta_i))``. The
    precision is ``k = (N - 1) / (2 (N - C))`` (eq. 20, ``1/k`` is unbiased)
    and ``(N - 1) kappa / k`` is chi-square with ``N - 1`` degrees of freedom
    (eq. 21). The inclination conditional on ``C`` is ``I = 90 - theta0 +
    (180/pi) S/C`` (eq. 40) and its error angle ``cos(alpha) = 1 - (S/C)^2 / 2
    - f (N - C) / (C (N - 1))`` with `f` the ``(1 + level)/2`` quantile of
    ``F(1, N - 1)`` (eq. 42), a two-sided interval ``I +/- alpha``.

    Note:
        The recipe printed in Tauxe (2003, pp. 130-131) drops the ``-cos^2``
        term of (19) and is wrong; this implementation follows the original.
        Simulation (Fisher data, ``I0 = 60``, ``kappa = 40``, ``N = 10``)
        shows that ``90 - theta0`` is nearly unbiased (-0.2 deg) while the
        conditional ``I`` of (40) is biased by about -1.3 deg, contrary to
        the paper, so `inc` is the former and `inc_conditional` the latter.
        The error angle belongs to `inc_conditional`. Steep inclinations with
        a small ``kappa`` cannot be estimated, as the declination ambiguity
        makes inclination and ``kappa`` inseparable (the data fold back);
        directions of both polarities should not be pooled.

    Args:
        inclinations: inclinations in degrees, positive downward.
        level: confidence level. Default 0.95.
        method: ``"mcfadden_reid"`` (default) or ``"arason_levi"``.

    Returns:
        dict: ``inc`` (``90 - theta0``), ``inc_conditional`` (eq. 40),
            ``alpha`` (eq. 42), ``k``, ``kappa_ci`` (eq. 21), ``theta0``
            (radians), ``C``, ``S`` and ``n``.

    References:
        McFadden, P.L. & Reid, A.B. (1982). Analysis of palaeomagnetic
            inclination data. Geophysical Journal of the Royal Astronomical
            Society 69, 307-319.
        Arason, P. & Levi, S. (2010). Maximum likelihood solution for
            inclination-only data in paleomagnetism. Geophysical Journal
            International 182, 753-771.
    """
    if method not in ("mcfadden_reid", "arason_levi"):
        raise ValueError(
            f"method must be 'mcfadden_reid' or 'arason_levi', not {method!r}"
        )
    inc = np.asarray(inclinations, dtype=float).ravel()
    n = len(inc)
    if n < 3:
        raise ValueError("At least three inclinations are needed")
    if not np.all(np.isfinite(inc)) or np.any(np.abs(inc) > 90):
        raise ValueError("Inclinations must be finite and within [-90, 90] degrees")
    if method == "arason_levi":
        return _arason_levi(inc, level)
    if (inc > 0).any() and (inc < 0).any():
        warnings.warn(
            "Inclinations of both polarities should not be pooled "
            "(McFadden & Reid, 1982, sect. 6)",
            UserWarning,
            stacklevel=2,
        )
    theta = np.radians(90 - inc)
    valid = [r for r in _theta0_roots(theta) if r[1] < 0]
    if not valid:
        raise ValueError(
            "No root of McFadden & Reid's eq. 19 with U < 0; the data fold "
            "back through the vertical or are too dispersed"
        )
    t0, _, C = max(valid, key=lambda r: r[2])
    S = np.sin(t0 - theta).sum()
    if n - C <= 0:
        raise ValueError("The inclinations are identical, the precision is infinite")
    k = (n - 1) / (2 * (n - C))
    q = (1 - level) / 2
    f = stats.f.ppf((1 + level) / 2, 1, n - 1)
    cos_a = 1 - 0.5 * (S / C) ** 2 - f * (n - C) / (C * (n - 1))
    return {
        "inc": float(90 - np.degrees(t0)),
        "inc_conditional": float(90 - np.degrees(t0) + np.degrees(S / C)),
        "alpha": float(np.degrees(np.arccos(np.clip(cos_a, -1, 1)))),
        "k": float(k),
        "kappa_ci": (
            float(k * stats.chi2.ppf(q, n - 1) / (n - 1)),
            float(k * stats.chi2.ppf(1 - q, n - 1) / (n - 1)),
        ),
        "theta0": float(t0),
        "C": float(C),
        "S": float(S),
        "n": n,
        "method": "mcfadden_reid",
    }
