"""Boundaries of confidence regions on the sphere for ``StereoNet.confidence()``.

Each builder returns an (N, 3) array of unit vectors tracing the boundary; the
statistics themselves come from the feature sets (``fisher_statistics``,
``watson_statistics``, ``bootstrap_statistics``, ``bingham_statistics`` and
``mean_tensor``) or from fitted distributions (``Kent``, ``ESAG``).
"""

import numpy as np

from apsg.feature._distributions import ESAG, Kent
from apsg.math._vector import Axial3, Vector3

CONFIDENCE_METHODS = (
    "fisher",
    "watson",
    "bootstrap",
    "bootstrap_ellipse",
    "bingham",
    "kent",
    "esag",
    "jelinek",
    "lisle",
)

# confidence methods given by the confidence region of a fitted distribution
_FITTED_REGIONS = {"kent": Kent, "esag": ESAG}

# polar angle sampling of elliptical boundaries
_THETA = np.linspace(0, 2 * np.pi, 181)


def cone_curve(mu, alpha):
    """Circle of angular radius `alpha` (degrees) around the mean vector `mu`."""
    # a point at `alpha` from the mean, tilted down in its vertical plane
    azi, _ = mu.geo
    secant = mu.rotate(Vector3(azi + 90, 0), -alpha)
    angles = np.linspace(0, 360, 360)
    return np.array([np.asarray(secant.rotate(mu, a)) for a in angles])


def bingham_curve(stats):
    """Elliptical cone around ``stats["mu"]`` with semi-angles ``stats["gamma"]``
    (degrees) along ``stats["axes"]``, as returned by ``bingham_statistics``."""
    mu = np.asarray(stats["mu"])
    u = np.asarray(stats["axes"][0])
    v = np.asarray(stats["axes"][1])
    g0, g1 = np.radians(stats["gamma"])
    # polar equation of an ellipse with semi-axes g0, g1
    denom = np.sqrt((g1 * np.cos(_THETA)) ** 2 + (g0 * np.sin(_THETA)) ** 2)
    rho = np.divide(g0 * g1, denom, out=np.zeros_like(denom), where=denom > 0)
    return np.cos(rho)[:, None] * mu + np.sin(rho)[:, None] * (
        np.cos(_THETA)[:, None] * u + np.sin(_THETA)[:, None] * v
    )


def jelinek_curves(ellipses, which=None):
    """Confidence ellipses of principal axes (``mean_tensor()["ellipses"]``),
    all three for ``which=None`` or the one selected by ``which``, joined by
    NaN rows so they draw as separate lines.

    Jelinek's ellipses are given by half-angles in the tangent plane at ``mu``,
    so the ellipse is built there and projected back onto the sphere.
    """
    selected = ellipses if which is None else [ellipses[which]]
    curves = []
    for ell in selected:
        mu = np.asarray(ell["mu"])
        u = np.asarray(ell["axes"][0])
        v = np.asarray(ell["axes"][1])
        t0, t1 = np.tan(np.radians(ell["gamma"]))
        curve = (
            mu + (t0 * np.cos(_THETA))[:, None] * u + (t1 * np.sin(_THETA))[:, None] * v
        )
        curves.append(curve / np.linalg.norm(curve, axis=1)[:, None])
    gap = np.full((1, 3), np.nan)
    pts = curves[0]
    for curve in curves[1:]:
        pts = np.vstack([pts, gap, curve])
    return pts


def confidence_curve(
    fset,
    method,
    *,
    level=0.95,
    which=None,
    n_resamples=1000,
    normalize=False,
    anisoft=False,
):
    """Return the boundary of the confidence region of `fset` by `method`.

    Args:
        fset: ``Vector3Set`` (methods "fisher", "watson", "bootstrap",
            "bootstrap_ellipse", "bingham", "kent", "esag"), ``EllipsoidSet``/``Stress3Set`` (method
            "jelinek") or ``PairSet``/``FaultSet`` (method "lisle").
        method (str): one of ``CONFIDENCE_METHODS``.

    Keyword Args:
        level (float): confidence level. Default 0.95.
        which (int): principal axis for "bingham" (default 0) and "jelinek"
            (default ``None``, all three).
        n_resamples (int): number of resamples for "bootstrap" and
            "bootstrap_ellipse".
        normalize (bool): normalize tensors for "jelinek".
        anisoft (bool): AniSoft scaling of the region for "jelinek".
    """
    match method:
        case "fisher":
            stats = fset.fisher_statistics(level=level)
            return cone_curve(stats["mu"], stats["alpha"])
        case "watson":
            stats = fset.watson_statistics(level=level)
            return cone_curve(stats["mu"], stats["alpha"])
        case "bootstrap":
            stats = fset.bootstrap_statistics(level=level, n=n_resamples)
            return cone_curve(stats["mu"], stats["alpha"])
        case "bootstrap_ellipse":
            stats = fset.bootstrap_statistics(
                level=level, n=n_resamples, method="ellipse"
            )
            return bingham_curve(stats)
        case "bingham":
            stats = fset.bingham_statistics(
                level=level, which=0 if which is None else which
            )
            return bingham_curve(stats)
        case "kent" | "esag":
            # axial sets are flipped into one hemisphere, as by fisher_statistics
            data = fset.normalized()
            if issubclass(fset.__feature_class__, Axial3):
                data = data.halfspace()
            dist = _FITTED_REGIONS[method].fit(np.asarray(data))
            return bingham_curve(dist.mean_confidence(level).as_dict())
        case "lisle":
            stats = fset.lisle_statistics(level=level)
            gap = np.full((1, 3), np.nan)
            return np.vstack(
                [bingham_curve(stats["lin"]), gap, bingham_curve(stats["fol"])]
            )
        case "jelinek":
            ellipses = fset.mean_tensor(
                level=level, normalize=normalize, anisoft=anisoft
            )["ellipses"]
            return jelinek_curves(ellipses, which)
    raise ValueError(
        f"Unknown confidence method {method!r}, expected one of {CONFIDENCE_METHODS}"
    )
