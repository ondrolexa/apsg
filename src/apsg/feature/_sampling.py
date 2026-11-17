"""Corrections of orientation sampling bias (Terzaghi weights)."""

import numpy as np

from apsg.feature._geodata import Foliation, Lineation


def terzaghi_weights(features, sampling, *, kind=None, per="feature", max_weight=10.0):
    """Terzaghi weights correcting the orientation bias of a line or plane sample.

    A line (borehole, scanline) of direction `d` intersects a fracture with
    unit pole `n` with probability proportional to ``|cos(theta)| = |d.n|``,
    so that fractures nearly parallel to the line are undersampled (Terzaghi,
    1965). Fractures cut on a plane (outcrop, window) with unit normal `p`
    leave traces whose number is proportional to ``|sin(theta)| = |n x p|``,
    so that fractures nearly parallel to the plane are undersampled. The
    weights are the inverse of these probabilities, ``1 / |cos(theta)|`` for
    a line and ``1 / |sin(theta)|`` for a plane (Munier, 2004, eq. 1-2;
    ``theta`` is the angle between the fracture pole and the line direction
    or the plane normal). They grow without bound as the probability goes to
    zero, so they are capped at `max_weight` (5 or 10 is common practice).

    Args:
        features: feature set of fracture poles (``FoliationSet``,
            ``Vector3Set``) or of the directions the fractures are measured
            as (see Note).
        sampling: sampling geometry. A ``Lineation`` is a line, a
            ``Foliation`` a plane (its pole is the plane normal); any other
            vector needs `kind`.

    Keyword Args:
        kind: "line" or "plane". Default None, inferred from `sampling`.
        per: "feature" (default) weights every feature by its own angle
            (the trace-level correction); "mean" uses the angle of the mean
            pole of the set (the principal axis of its orientation tensor)
            for all features, as in Munier (2004), where each fracture set
            is corrected as a whole. All features get one common weight, so
            it matters for the relative counts of different sets, not for
            the statistics of a single set.
        max_weight: upper bound of the weights. Default 10. None for no cap.

    Returns:
        numpy.ndarray: one weight per feature, not normalized (the weighted
        statistics normalize them). Pass them as ``weights=`` to the fits and
        statistics of the set.

    Note:
        The features are used as they are, i.e. as the poles of the
        fractures; a ``FoliationSet`` stores poles. The correction does not
        account for the size or spacing of the sets, which Munier (2004)
        addresses by other means, and its use is a modelling decision.

    References:
        Terzaghi, R.D. (1965). Sources of error in joint surveys.
            Geotechnique 15, 287-304.
        Munier, R. (2004). Statistical analysis of fracture data, adapted for
            modelling Discrete Fracture Networks, version 2. SKB R-04-66.
    """
    if kind is None:
        if isinstance(sampling, Lineation):
            kind = "line"
        elif isinstance(sampling, Foliation):
            kind = "plane"
        else:
            raise ValueError(
                "kind must be 'line' or 'plane' for a sampling vector that is "
                "neither a Lineation nor a Foliation"
            )
    if kind not in ("line", "plane"):
        raise ValueError(f"kind must be 'line' or 'plane', got {kind!r}")
    if per not in ("feature", "mean"):
        raise ValueError(f"per must be 'feature' or 'mean', got {per!r}")
    if max_weight is not None and not max_weight >= 1:
        raise ValueError("max_weight must be at least 1")

    n = np.asarray(features, dtype=float)
    norms = np.linalg.norm(n, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    n = n / norms
    s = np.asarray(sampling, dtype=float)
    s = s / np.linalg.norm(s)

    if per == "mean":
        _, vecs = np.linalg.eigh(n.T @ n / len(n))
        n = vecs[:, -1][None, :]

    cos = np.abs(n @ s)
    prob = cos if kind == "line" else np.sqrt(np.clip(1 - cos**2, 0, 1))
    with np.errstate(divide="ignore"):
        w = 1.0 / prob
    if max_weight is not None:
        w = np.minimum(w, max_weight)
    if per == "mean":
        w = np.full(len(features), w[0])
    return w
