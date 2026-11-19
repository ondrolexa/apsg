"""Relative paleointensity and alignment of core sections.

Relative paleointensity estimates follow the three normalizations offered by
PuffinPlot (Lurcock & Wilson, 2012): the NRM over the susceptibility, and the
NRM demagnetization against the AF demagnetization or the stepwise acquisition
of an anhysteretic remanence (ARM, Levi & Banerjee, 1976). The alignment of
core sections follows its core section tools.
"""

from dataclasses import dataclass

import numpy as np
from scipy import stats

__all__ = ("RPIEstimate", "align_core_sections", "relative_paleointensity")

_RPI_METHODS = ("susceptibility", "af", "arm_acquisition")


@dataclass(frozen=True, eq=False)
class RPIEstimate:
    """Relative paleointensity of one specimen.

    Attributes:
        method: ``"susceptibility"``, ``"af"`` or ``"arm_acquisition"``.
        ratio: NRM over the normalizer; for the stepwise methods the mean of
            the ratios at the treatment levels used.
        slope: slope of the NRM against the normalizer intensities at the
            levels used (negative of the regression for ``"arm_acquisition"``,
            as the NRM decreases while the ARM grows), None for
            ``"susceptibility"``.
        r: correlation coefficient of that regression, None without one.
        r2: its square.
        levels: treatment levels used.
        nrm: NRM intensities (A/m) at these levels.
        normalizer: normalizer intensities at these levels (A/m, or the
            susceptibility for ``"susceptibility"``).
    """

    method: str
    ratio: float
    slope: float | None
    r: float | None
    r2: float | None
    levels: np.ndarray
    nrm: np.ndarray
    normalizer: np.ndarray

    def __repr__(self):
        slope = (
            "" if self.slope is None else f", slope={self.slope:.4g}, r2={self.r2:.3f}"
        )
        return f"RPIEstimate({self.method}: ratio={self.ratio:.4g}{slope}, n={len(self.levels)})"

    @property
    def n(self):
        """Number of treatment levels used."""
        return len(self.levels)


def _intensity(step):
    return float(abs(step.M))


def relative_paleointensity(
    specimen, normalizer=None, method="susceptibility", min_level=None, max_level=None
):
    """Relative paleointensity of a specimen.

    Args:
        specimen: ``Specimen`` with the NRM demagnetization series.
        normalizer: for ``"af"`` the ``Specimen`` holding the AF demagnetization
            of the ARM (or IRM) used for normalization; for
            ``"arm_acquisition"`` the specimen whose `arm` steps (ARM acquired
            at the peak AF fields in `level`) are used, `specimen` itself if
            None; ignored for ``"susceptibility"``.
        method: ``"susceptibility"`` (NRM over the bulk susceptibility `k` of
            the first step that has one), ``"af"`` (NRM against the
            demagnetized ARM or IRM at the same levels, mean ratio and slope)
            or ``"arm_acquisition"`` (NRM against the ARM acquired at the same
            levels; the slope is negated).
        min_level: lowest treatment level used by the stepwise methods.
        max_level: highest treatment level used by the stepwise methods.

    Returns:
        RPIEstimate

    Raises:
        ValueError: for a missing susceptibility or normalizer, or when no
            level is shared by the NRM and normalizer series.
    """
    if method not in _RPI_METHODS:
        raise ValueError(f"method must be one of {_RPI_METHODS}, not {method!r}")
    if not specimen.steps:
        raise ValueError(f"{specimen.name}: no steps")
    if method == "susceptibility":
        k = next((s.k for s in specimen.steps if s.k is not None), None)
        if k is None or not np.isfinite(k) or k <= 0:
            raise ValueError(
                f"{specimen.name}: no positive susceptibility `k` in steps"
            )
        nrm = _intensity(specimen.steps[0])
        return RPIEstimate(
            method,
            nrm / k,
            None,
            None,
            None,
            np.array([0.0]),
            np.array([nrm]),
            np.array([k]),
        )
    if method == "af":
        if normalizer is None:
            raise ValueError("method 'af' needs the normalizer specimen")
        series = list(normalizer.steps)
    else:
        series = [
            s for s in (normalizer or specimen).arm if s.treatment in ("arm", None)
        ]
        if not series:
            raise ValueError(f"{(normalizer or specimen).name}: no ARM steps")
    lo = -np.inf if min_level is None else min_level - 1e-10
    hi = np.inf if max_level is None else max_level + 1e-10
    levels, nrm, norm = [], [], []
    for step in specimen.steps:
        if not lo <= step.level <= hi:
            continue
        match = next((s for s in series if abs(s.level - step.level) < 1e-6), None)
        if match is None or _intensity(match) <= 0:
            continue
        levels.append(step.level)
        nrm.append(_intensity(step))
        norm.append(_intensity(match))
    if not levels:
        raise ValueError(
            f"{specimen.name}: no treatment level shared with the normalizer"
        )
    nrm, norm = np.array(nrm), np.array(norm)
    slope = r = r2 = float("nan")
    if len(levels) > 1 and np.ptp(norm) > 0 and np.ptp(nrm) > 0:
        fit = stats.linregress(norm, nrm)
        slope = -fit.slope if method == "arm_acquisition" else fit.slope
        r, r2 = fit.rvalue, fit.rvalue**2
    return RPIEstimate(
        method, float(np.mean(nrm / norm)), slope, r, r2, np.array(levels), nrm, norm
    )


# ---------------------------------------------------------------------------
# core sections
# ---------------------------------------------------------------------------


def _declination(vectors):
    """Declination in degrees of the resultant of the vectors."""
    R = np.sum([np.asarray(v, dtype=float) / abs(v) for v in vectors], axis=0)
    return float(np.degrees(np.arctan2(R[1], R[0])) % 360)


def align_core_sections(sections, name, margin=3, target="top", declination=0.0):
    """Align the declinations of the sections of an unoriented long core.

    Long cores are oriented by the section only: each section has its own
    arbitrary azimuth. Going from the top down, each section is turned about
    the vertical so that the direction of the component `name` near its top
    has the declination of the direction near the bottom of the section
    above; the first section is turned to `declination` (``target="top"``).
    With ``target="mean"`` the first section is not moved relative to
    its neighbours and the whole core is finally turned so that the mean
    declination of all specimens is `declination`. The turn is added to
    the `azimuth` of the specimens, which is the rotation of the geographic
    coordinates about the vertical.

    The direction near an end is the resultant direction of the component of the
    `margin` specimens closest to that end (by `depth`, else in list order).

    Args:
        sections: sections from the top down, as list of ``Sample`` (see
            ``Sample.by_site``) or dict of them.
        name: name of the component fitted to each specimen (see
            ``Specimen.interpret``); a line.
        margin: number of specimens at each end of a section used for its end
            direction.
        target: ``"top"`` or ``"mean"``, see above.
        declination: the declination of the top of the first section
            (``"top"``) or of the mean of the core (``"mean"``), in degrees.

    Returns:
        dict: the total turn in degrees of each section, by its name (or its
            position when `sections` is a list of unnamed samples).

    Raises:
        ValueError: when a section has no specimen with the component, or `target`
            or `margin` are invalid.
    """
    if target not in ("top", "mean"):
        raise ValueError(f"target must be 'top' or 'mean', not {target!r}")
    if margin < 1:
        raise ValueError("margin must be at least 1")
    if isinstance(sections, dict):
        items = list(sections.items())
    else:
        names = [s.name for s in sections]
        keys = names if len(set(names)) == len(names) else range(len(names))
        items = list(zip(keys, sections, strict=True))
    ordered = []
    for key, section in items:
        specimens = [s for s in section if name in s.components]
        if not specimens:
            raise ValueError(f"section {key!r}: no specimen has the component {name!r}")
        if all(s.depth is not None for s in specimens):
            specimens = sorted(specimens, key=lambda s: s.depth)
        ordered.append((key, specimens))

    def directions(specimens):
        return [s.component(name, "geo").vector for s in specimens]

    turns = {}
    align_to = declination if target == "top" else None
    for key, specimens in ordered:
        if align_to is None:  # first section, target mean: keep as is
            offset = 0.0
        else:
            offset = align_to - _declination(directions(specimens[:margin]))
        for s in specimens:
            s.azimuth = (s.azimuth + offset) % 360
        turns[key] = offset
        align_to = _declination(directions(specimens[-margin:]))
    if target == "mean":
        every = [s for _, specimens in ordered for s in specimens]
        offset = declination - _declination(directions(every))
        for key, specimens in ordered:
            for s in specimens:
                s.azimuth = (s.azimuth + offset) % 360
            turns[key] += offset
    return {k: float(v) % 360 for k, v in turns.items()}
