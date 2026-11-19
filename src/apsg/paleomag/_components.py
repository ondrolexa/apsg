"""Automatic identification of magnetization components in demagnetization data.

The search is a heuristic layer on top of the principal component fits of
``Specimen.pca``: every contiguous window of steps is fitted at once
(``window_fits``), the best window is chosen by the acceptance criteria
(``find_component``) or the Zijderveld path is partitioned into a few linear
segments by dynamic programming (``find_components``). The criteria
objects follow PmagPy (``pmag.default_criteria``: at least 4 steps, MAD <= 5 degrees, DANG <= 10
degrees; Kirschvink, 1980; Tauxe & Staudigel, 2004), but the search uses looser defaults that
were chosen by comparison with Remasoft picks (see ``SEARCH_CRITERIA``). The angle ``alpha`` between
the free and the origin-anchored fit measures how well a line decays to the
origin. The result has to be checked on the Zijderveld diagram; PmagPy has no
automatic selection of its own (its ``autointerpret`` of demag_gui is disabled).
"""

import warnings
from dataclasses import dataclass, replace

import numpy as np
import pandas as pd

from apsg.math._vector import Vector3

__all__ = ("SEARCH_CRITERIA", "ComponentCriteria")


@dataclass(frozen=True)
class ComponentCriteria:
    """Acceptance criteria of a component fit (specimen level).

    The defaults are the PmagPy legacy criteria. A limit set to None is not
    applied.

    Attributes:
        n_min: minimum number of steps.
        mad_max: maximum angular deviation in degrees.
        dang_max: maximum deviation angle (free lines only) in degrees.
        alpha_max: maximum angle between the free and the origin-anchored line
            (components with `alpha` only) in degrees.
    """

    n_min: int | None = 4
    mad_max: float | None = 5.0
    dang_max: float | None = 10.0
    alpha_max: float | None = None

    def check(self, component):
        """Return the list of the criteria the component fails (empty if it passes)."""
        failed = []
        if self.n_min is not None and component.n < self.n_min:
            failed.append(f"n = {component.n} < {self.n_min}")
        if self.mad_max is not None and not component.mad <= self.mad_max:
            failed.append(f"MAD = {component.mad:.1f} > {self.mad_max:g}")
        if (
            self.dang_max is not None
            and component.dang is not None
            and not component.dang <= self.dang_max
        ):
            failed.append(f"DANG = {component.dang:.1f} > {self.dang_max:g}")
        if (
            self.alpha_max is not None
            and component.alpha is not None
            and not component.alpha <= self.alpha_max
        ):
            failed.append(f"alpha = {component.alpha:.1f} > {self.alpha_max:g}")
        return failed

    def passes(self, component):
        """Whether the component meets all criteria."""
        return not self.check(component)


#: Criteria of the automatic search when none are given: more tolerant than the
#: PmagPy legacy limits (MAD 5, DANG 10), which only 8 % of 800 Remasoft picks of
#: weak sediments pass, and no DANG limit, since it holds for components that decay to
#: the origin only. Pass ``ComponentCriteria()`` for the legacy limits.
SEARCH_CRITERIA = ComponentCriteria(n_min=4, mad_max=10.0, dang_max=None)


# ---------------------------------------------------------------------------
# all windows at once
# ---------------------------------------------------------------------------


def _windows(X, min_steps, max_steps=None):
    """Statistics of all contiguous windows of the rows of `X` (n, 3).

    Free and origin-anchored line fits of every window ``i..j`` (inclusive) from
    cumulative sums; MAD, DANG and the angle between the two fits agree with
    ``Specimen.pca``.
    """
    n = len(X)
    if n < min_steps or n == 0:
        return None
    scale = float(np.abs(X).max())
    if scale == 0:
        return None
    X = X / scale
    pairs = [
        (i, j)
        for i in range(n)
        for j in range(i + min_steps - 1, n)
        if max_steps is None or j - i + 1 <= max_steps
    ]
    if not pairs:
        return None
    ii, jj = np.array(pairs).T
    c1 = np.vstack([np.zeros(3), np.cumsum(X, axis=0)])
    c2 = np.concatenate(
        [np.zeros((1, 3, 3)), np.cumsum(np.einsum("ni,nj->nij", X, X), axis=0)]
    )
    m = (jj - ii + 1).astype(float)
    s1, s2 = c1[jj + 1] - c1[ii], c2[jj + 1] - c2[ii]
    mean = s1 / m[:, None]
    free = s2 / m[:, None, None] - mean[:, :, None] * mean[:, None, :]
    anch = s2 / m[:, None, None]
    out = {"i": ii, "j": jj, "n": (jj - ii + 1)}
    vecs = {}
    for key, T in (("free", free), ("anch", anch)):
        w, v = np.linalg.eigh(T)
        w = np.clip(w, 0, None)
        with np.errstate(divide="ignore", invalid="ignore"):
            mad = np.degrees(np.arctan(np.sqrt((w[:, 0] + w[:, 1]) / w[:, 2])))
        out["mad" if key == "free" else "mad_anchored"] = np.where(
            np.isfinite(mad), mad, np.inf
        )
        vecs[key] = v[:, :, 2]
    ref = X[ii] - X[jj]
    flip = np.einsum("wi,wi->w", vecs["free"], ref) < 0
    vecs["free"][flip] *= -1
    flip = np.einsum("wi,wi->w", vecs["anch"], s1) < 0
    vecs["anch"][flip] *= -1
    norm = np.linalg.norm(mean, axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        cosd = np.einsum("wi,wi->w", vecs["free"], mean) / norm
    out["dang"] = np.where(
        norm > 0, np.degrees(np.arccos(np.clip(cosd, -1, 1))), np.nan
    )
    cosa = np.abs(np.einsum("wi,wi->w", vecs["free"], vecs["anch"]))
    out["alpha"] = np.degrees(np.arccos(np.clip(cosa, -1, 1)))
    out["v"] = vecs["free"]
    return out


def window_fits(specimen, min_steps=4, max_steps=None, steps=None):
    """Free and anchored line fits of every contiguous window of steps.

    Args:
        specimen: ``Specimen``.
        min_steps: smallest window (number of steps).
        max_steps: largest window, None for no limit.
        steps: selection of the steps to scan (slice, positions or mask); windows
            are contiguous in the selection.

    Returns:
        pandas.DataFrame: one row per window with ``start`` and ``end`` (positions of the first
        and last step), ``n``, the labels and levels of the ends, the direction
        ``dec`` and ``inc`` of the free fit in specimen coordinates, ``mad``, ``dang``,
        ``mad_anchored``, ``alpha`` (angle between the free and the anchored fit)
        and ``nmad`` (``mad / sqrt(n - 2)``, with ``n - 2`` at least 1).
    """
    pos = np.arange(len(specimen))[steps if steps is not None else slice(None)]
    columns = [
        "start", "end", "n", "first", "last", "level_first", "level_last",
        "dec", "inc", "mad", "dang", "mad_anchored", "alpha", "nmad",
    ]  # fmt: skip
    X = np.asarray(specimen.V, dtype=float)[pos] if len(pos) else np.zeros((0, 3))
    w = _windows(X, min_steps, max_steps)
    if w is None:
        return pd.DataFrame(columns=columns)
    labels, levels = specimen.labels, specimen.levels
    start, end = pos[w["i"]], pos[w["j"]]
    v = w["v"]
    with np.errstate(divide="ignore", invalid="ignore"):
        nmad = w["mad"] / np.sqrt(np.maximum(w["n"] - 2, 1))
    return pd.DataFrame(
        {
            "start": start,
            "end": end,
            "n": w["n"],
            "first": [labels[k] for k in start],
            "last": [labels[k] for k in end],
            "level_first": levels[start],
            "level_last": levels[end],
            "dec": np.degrees(np.arctan2(v[:, 1], v[:, 0])) % 360,
            "inc": np.degrees(np.arcsin(np.clip(v[:, 2], -1, 1))),
            "mad": w["mad"],
            "dang": w["dang"],
            "mad_anchored": w["mad_anchored"],
            "alpha": w["alpha"],
            "nmad": nmad,
        }
    )


def _accepted(table, criteria):
    """Rows of a window table that meet the criteria (DANG is ignored when missing)."""
    ok = np.ones(len(table), dtype=bool)
    if criteria.n_min is not None:
        ok &= np.asarray(table["n"] >= criteria.n_min)
    if criteria.mad_max is not None:
        ok &= np.asarray(table["mad"] <= criteria.mad_max)
    if criteria.dang_max is not None:
        ok &= ~np.asarray(table["dang"] > criteria.dang_max)
    if criteria.alpha_max is not None:
        ok &= ~np.asarray(table["alpha"] > criteria.alpha_max)
    return table[ok]


def _component(specimen, positions, coords, alpha):
    comp = specimen.pca(list(positions), kind="line", coords=coords)
    return replace(comp, alpha=float(alpha))


def find_component(
    specimen,
    criteria=None,
    score="mad",
    end="any",
    tail=0,
    alpha_max=None,
    steps=None,
    coords="geo",
):
    """The best single line component of a specimen.

    All windows of at least ``criteria.n_min`` steps are fitted (`window_fits`);
    among those meeting the criteria the best one is taken: the one with the smallest
    MAD (``score="mad"``, ties by the longer window), the longest (``"longest"``,
    ties by the lower MAD) or the smallest ``MAD / sqrt(n - 2)`` (``"nmad"``).
    ``end="last"`` accepts only windows that end at the last step, or at one of the
    last `tail` steps before it (the characteristic component decays to the highest
    steps); `alpha_max` adds the origin test: the free and the origin-anchored fit must
    differ by no more than this angle.

    The defaults come from comparison with 800 line picks made in Remasoft on
    two collections of weak sediment specimens (439 specimens): with ``score="mad"``
    the component ending at the last step agrees with the highest pick within 5 degrees
    in about 55 % of the specimens (median 4 degrees), "longest" and "nmad" do worse;
    letting the window end earlier (`tail`) or the origin test only removed
    specimens or added noise fits. Specimens with noisy last steps get no result.

    Args:
        specimen: ``Specimen``.
        criteria: ``ComponentCriteria``; ``SEARCH_CRITERIA`` (at least 4 steps, MAD <= 10)
            if None.
        score: ``"mad"``, ``"longest"`` or ``"nmad"``.
        end: ``"any"`` or ``"last"``.
        tail: with ``end="last"``, the number of final steps the window may leave out.
        alpha_max: limit of the angle free/anchored in degrees, None for no limit.
        steps: restrict the scan to a selection of steps.
        coords: coordinates of the result, ``"geo"`` (default), ``"specimen"`` or ``"tilt"``.

    Returns:
        Component or None: the fit (with `alpha`) or None if no window passes.
    """
    if score not in ("longest", "mad", "nmad"):
        raise ValueError(f"score must be 'longest', 'mad' or 'nmad', not {score!r}")
    if end not in ("any", "last"):
        raise ValueError(f"end must be 'any' or 'last', not {end!r}")
    criteria = criteria or SEARCH_CRITERIA
    if alpha_max is not None:
        criteria = replace(criteria, alpha_max=alpha_max)
    pos = np.arange(len(specimen))[steps if steps is not None else slice(None)]
    table = window_fits(specimen, min_steps=criteria.n_min or 2, steps=steps)
    table = _accepted(table, criteria)
    if end == "last" and len(table):
        table = table[table["end"] >= pos[-1] - tail]
    if not len(table):
        return None
    keys = {
        "longest": (["n", "mad"], [False, True]),
        "mad": (["mad", "n"], [True, False]),
        "nmad": (["nmad", "n"], [True, False]),
    }[score]
    best = table.sort_values(by=keys[0], ascending=keys[1], kind="stable").iloc[0]
    inside = pos[(pos >= best["start"]) & (pos <= best["end"])]
    return _component(specimen, inside, coords, best["alpha"])


def find_components(
    specimen,
    criteria=None,
    max_components=3,
    segment_penalty=0.3,
    skip_penalty=1.0,
    share_boundary=True,
    steps=None,
    coords="geo",
):
    """Partition the demagnetization path into a few linear components.

    Dynamic programming over the steps (Zijderveld breakpoints): the path is cut into
    at most `max_components` consecutive segments, each an accepted window (see
    `ComponentCriteria`); steps that no accepted segment can hold stay unassigned.
    The cost to minimize is ``skip_penalty`` per unassigned step plus, per segment,
    ``segment_penalty + MAD / mad_max``; with ``share_boundary`` a segment may start at the last
    step of the previous one, as the picks of Remasoft often do.

    Args:
        specimen: ``Specimen``.
        criteria: ``ComponentCriteria`` of each segment; ``SEARCH_CRITERIA`` (at least
            4 steps, MAD <= 10) if None.
        max_components: largest number of segments.
        segment_penalty: cost of one more segment (in units of a step).
        skip_penalty: cost of a step that belongs to no segment.
        share_boundary: allow one common step between consecutive segments.
        steps: restrict the search to a selection of steps.
        coords: coordinates of the result.

    Returns:
        list of Component: in treatment order, each with `alpha`; empty if no window passes.
    """
    criteria = criteria or SEARCH_CRITERIA
    pos = np.arange(len(specimen))[steps if steps is not None else slice(None)]
    table = _accepted(
        window_fits(specimen, min_steps=criteria.n_min or 2, steps=steps), criteria
    )
    if not len(table) or max_components < 1:
        return []
    where = {int(p): k for k, p in enumerate(pos)}  # position -> index in the selection
    seg = {}  # (i, j) in the selection -> (cost, alpha, positions)
    mad_max = criteria.mad_max or 5.0
    for row in table.itertuples():
        seg[(where[int(row.start)], where[int(row.end)])] = (
            segment_penalty + row.mad / mad_max,
            row.alpha,
        )
    n = len(pos)
    inf = float("inf")
    # state (p, k, shared): steps 0..p-1 decided, k segments used, `shared` says the
    # last segment ended at step p - 1 (that step may start the next segment)
    best = {(0, 0, False): (0.0, None)}

    def relax(state, cost, back):
        if cost < best.get(state, (inf, None))[0]:
            best[state] = (cost, back)

    for p in range(n):
        for k in range(max_components + 1):
            for shared in (False, True):
                state = (p, k, shared)
                if state not in best:
                    continue
                cost = best[state][0]
                relax((p + 1, k, False), cost + skip_penalty, (state, None))
                if k >= max_components:
                    continue
                starts = [p] + ([p - 1] if shared and share_boundary else [])
                for s in starts:
                    for j in range(s, n):
                        if (s, j) in seg:
                            relax(
                                (j + 1, k + 1, True),
                                cost + seg[(s, j)][0],
                                (state, (s, j)),
                            )
    finals = [s for s in best if s[0] == n]
    if not finals:
        return []
    state = min(finals, key=lambda s_: best[s_][0])
    chosen = []
    while best[state][1] is not None:
        prev, move = best[state][1]
        if move is not None:
            chosen.append(move)
        state = prev
    chosen.sort()
    out = []
    for i, j in chosen:
        out.append(_component(specimen, pos[i : j + 1], coords, seg[(i, j)][1]))
    return out


# ---------------------------------------------------------------------------
# samples: stacking, level-based interpretation, grading
# ---------------------------------------------------------------------------


def stack_sample(sample, coords="geo", normalize="nrm", min_specimens=3, decimals=6):
    """Stack the specimens of a sample step by step into one pseudo-specimen.

    At each treatment level present in at least `min_specimens` specimens the
    vectors of those specimens are averaged (virtual direction, after the stacking
    of remagnetization data); the stable range found on the stack can then be fitted in
    every specimen with ``Sample.interpret``.

    Args:
        sample: ``Sample``.
        coords: coordinates in which the vectors are averaged (``"geo"``, ``"tilt"``
            or ``"specimen"``).
        normalize: ``"nrm"`` divides each specimen by the intensity of its first
            step (keeps the decay), ``"unit"`` uses unit vectors (directions only),
            None the vectors in A/m.
        min_specimens: smallest number of specimens at a level.
        decimals: levels are matched when equal to this many decimals.

    Returns:
        Specimen: named ``<sample>-stack``, in `coords` (its orientation is the identity).

    Raises:
        ValueError: for an unknown `normalize` or when no level is shared.
    """
    from apsg.paleomag._specimen import DemagStep, Specimen

    if normalize not in ("nrm", "unit", None):
        raise ValueError(f"normalize must be 'nrm', 'unit' or None, not {normalize!r}")
    groups, labels = {}, {}
    for spec in sample:
        if not len(spec):
            continue
        V = np.asarray(spec.vectors(coords), dtype=float)
        if normalize == "nrm":
            nrm = np.linalg.norm(V[0])
            if nrm == 0:  # nothing to normalize by
                continue
            V = V / nrm
        elif normalize == "unit":
            norm = np.linalg.norm(V, axis=1, keepdims=True)
            V = np.divide(V, norm, out=np.zeros_like(V), where=norm > 0)
        for level, label, v in zip(spec.levels, spec.labels, V, strict=True):
            key = round(float(level), decimals)
            groups.setdefault(key, []).append(v)
            labels.setdefault(key, label)
    levels = sorted(k for k, v in groups.items() if len(v) >= min_specimens)
    if not levels:
        raise ValueError(f"no treatment level is shared by {min_specimens} specimens")
    demag = {s.demag for s in sample if s.demag is not None}
    return Specimen(
        name=f"{sample.name}-stack",
        steps=[
            DemagStep(labels[k], k, Vector3(np.mean(groups[k], axis=0))) for k in levels
        ],
        demag=demag.pop() if len(demag) == 1 else None,
        site=sample.name,
    )


def interpret_levels(
    sample, name, min_level, max_level, kind="line", anchor=False, origin=False
):
    """Fit the same range of treatment levels in every specimen of a sample.

    Returns:
        int: the number of specimens fitted; specimens with too few steps in the
        range are skipped with a warning.
    """
    fitted, skipped = 0, []
    for spec in sample:
        lv = spec.levels
        sel = (lv >= min_level) & (lv <= max_level)
        try:
            spec.interpret(name, sel, kind=kind, anchor=anchor, origin=origin)
            fitted += 1
        except ValueError:
            skipped.append(spec.name)
    if skipped:
        warnings.warn(
            f"{len(skipped)} specimens have too few steps between {min_level:g} and "
            f"{max_level:g} for {name!r}: {', '.join(skipped[:3])}"
            f"{' ...' if len(skipped) > 3 else ''}",
            stacklevel=3,
        )
    return fitted


def grade_sample(sample, name, criteria=None):
    """Grade the components `name` of a sample against acceptance criteria.

    Returns:
        pandas.DataFrame: one row per specimen that has the component, with
        ``kind``, ``n``, ``mad``, ``dang``, ``alpha``, ``passed`` and ``failed`` (the
        text of the failed criteria).
    """
    criteria = criteria or ComponentCriteria()
    rows = []
    for spec in sample:
        if name not in spec.components:
            continue
        c = spec.components[name]
        failed = criteria.check(c)
        rows.append(
            {
                "specimen": spec.name,
                "kind": c.kind,
                "n": c.n,
                "mad": c.mad,
                "dang": c.dang,
                "alpha": c.alpha,
                "passed": not failed,
                "failed": "; ".join(failed),
            }
        )
    return pd.DataFrame(
        rows,
        columns=["specimen", "kind", "n", "mad", "dang", "alpha", "passed", "failed"],
    )
