"""Plots of paleomagnetic data: demagnetization of a ``Specimen``, field tests
and mean directions of a ``Sample``.

All plots are ``PaleomagPlot`` subclasses with the protocol of ``RosePlot`` and
``FabricPlot``: keyword arguments override ``apsg_conf.paleomag`` for the
instance, and ``show()``, ``savefig()`` and ``render2fig()`` create, save or
fill a figure.
"""

from typing import ClassVar

import matplotlib.pyplot as plt
import numpy as np
from scipy.special import ive

from apsg.feature._geodata import Cone
from apsg.paleomag._inclination import (
    _cubic,
    _theta0_roots,
    inclination_only_mean,
)
from apsg.paleomag._specimen import Sample, bailey_halls_mean
from apsg.plotting._figureplot import FigurePlot
from apsg.plotting._stereonet import StereoNet

__all__ = (
    "DemagPlot",
    "FoldTestPlot",
    "InclinationOnlyPlot",
    "LinesPlanesPlot",
    "PaleomagPlot",
    "ReversalTestPlot",
    "SpecimenPlot",
    "SpecimenStereoPlot",
    "TiltComparisonPlot",
    "ZijderveldPlot",
)

_TITLES = {
    "specimen": "Specimen coordinates",
    "geo": "Geographic coordinates",
    "tilt": "Tilt-corrected coordinates",
}


def _title(specimen):
    return f"{specimen.site} {specimen.name}".strip()


def _check_coords(coords):
    if coords not in _TITLES:
        raise ValueError(f"coords must be one of {tuple(_TITLES)}, got {coords!r}")
    return coords


class PaleomagPlot(FigurePlot):
    """Base class of paleomagnetic plots, see ``FigurePlot``.

    Args:
        data: the plotted object (``Specimen`` or ``Sample``, see subclasses).

    Keyword Args:
        title, figsize, title_kws, legend_position, legend_kws: see
            ``FigurePlot``.
        level (float): confidence level of intervals and cones.
        horizontal_color, vertical_color, normal_color, reversed_color: colors,
            see ``apsg_conf.paleomag``.
        mean_color, ci_color, ci_alpha, grid: styles, see ``apsg_conf.paleomag``.

    Plots with options of their own (``coords``, ``n``, ``rng``, ...) list
    them in their documentation; unknown keywords raise ``TypeError``.
    """

    config_key = "paleomag"
    window_title = "Paleomag plot"

    def savefig(self, filename="paleomag.png", **kwargs):
        """Save the plot to a graphics file (kwargs go to ``Figure.savefig``)."""
        super().savefig(filename, **kwargs)


# -- single specimen --------------------------------------------------------
def _zijderveld_on(ax, specimen, coords, cfg):
    """Draw the orthogonal projections of `specimen` in `ax`; returns the unit."""
    N, E, Z = np.asarray(specimen.vectors(coords), dtype=float).T
    hc, vc = cfg["horizontal_color"], cfg["vertical_color"]
    ax.plot(E, N, "-", color=hc, label="Horizontal")
    ax.plot(E, N, "o", color=hc, picker=5)
    ax.plot(E, -Z, "-", color=vc, label="Vertical")
    ax.plot(E, -Z, "o", color=vc, picker=5)
    ax.plot(E[:1], N[:1], "+", color=hc, markersize=14)
    ax.plot(E[:1], -Z[:1], "+", color=vc, markersize=14)
    mx = 1.05 * max(np.abs(np.concatenate([N, E, Z])).max(), 1e-30)
    ax.set_xlim(-mx, mx)
    ax.set_ylim(-mx, mx)
    ax.set_aspect(1)
    for side in ("left", "right", "bottom", "top"):
        ax.spines[side].set_position("zero")
    ticks = ax.xaxis.get_ticklocs()
    unit = ticks[1] - ticks[0] if len(ticks) > 1 else mx
    ax.xaxis.set_ticklabels([])
    ax.yaxis.set_ticklabels([])
    labels = specimen.labels

    def onpick(event):
        if event.artist.axes is ax and len(event.ind):
            ax.set_title(f"{_title(specimen)}  step {labels[event.ind[0]]}", loc="left")
            ax.figure.canvas.draw_idle()

    ax.figure.canvas.mpl_connect("pick_event", onpick)
    return unit


def _components_on(ax, specimen, coords):
    """Mark the steps of the stored line components in the orthogonal projections and
    draw each fitted line through the centre of mass of its steps."""
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    V = np.asarray(specimen.vectors(coords), dtype=float)
    labels = {lab: i for i, lab in enumerate(specimen.labels)}
    for k, (name, comp) in enumerate(specimen.components.items()):
        idx = [labels[lab] for lab in comp.labels if lab in labels]
        if comp.kind != "line" or len(idx) < 2:
            continue
        color = colors[k % len(colors)]
        pts = V[idx]
        v = np.asarray(comp.transform(specimen.rotation(coords), coords).vector, float)
        center = np.zeros(3) if comp.anchor else pts.mean(axis=0)
        t = (pts - center) @ v
        ends = center + np.outer([t.min(), t.max()], v)
        N, E, Z = pts.T
        ax.plot(E, N, "o", mfc="none", mec=color, ms=9, label=name)
        ax.plot(E, -Z, "o", mfc="none", mec=color, ms=9)
        ax.plot(ends[:, 1], ends[:, 0], "-", color=color, lw=1.5)
        ax.plot(ends[:, 1], -ends[:, 2], "-", color=color, lw=1.5)


def _demag_on(ax, specimen):
    M = specimen.intensity
    lv = specimen.levels
    ax.plot(lv[:1], M[:1] / M.max(), "k+", markersize=14)
    ax.plot(lv, M / M.max(), "ko-")
    unit = {"thermal": "°C", "af": "mT"}.get(specimen.demag, "")
    ax.set_xlabel(f"Level [{unit}]" if unit else "Level")
    ax.set_ylabel("M/Mmax")
    ax.set_ylim(0, 1.02)
    ax.yaxis.grid()
    return M.max()


class ZijderveldPlot(PaleomagPlot):
    """Zijderveld (orthogonal vector) diagram of a ``Specimen``.

    The horizontal projection (north up, east right) is drawn with blue lines
    and markers, the vertical projection (up, east) in green; the NRM step is
    marked by a cross. Clicking a point shows its step label in the title.

    Args:
        specimen: ``Specimen``.

    Keyword Args:
        coords (str): ``"specimen"``, ``"geo"`` (default) or ``"tilt"``.
        components (bool): ring the steps of the stored line components and
            draw their fitted lines (see ``Specimen.interpret``,
            ``Specimen.interpret_auto``). Default False.
        **kwargs: the keyword arguments of ``PaleomagPlot``.
    """

    window_title = "Zijderveld plot"
    _options: ClassVar[dict] = {"coords": "geo", "components": False}

    def _default_title(self):
        return f"{_title(self.data)}\n{_TITLES[_check_coords(self._opts['coords'])]}"

    def _draw(self):
        ax = self.fig.subplots()
        coords = _check_coords(self._opts["coords"])
        self._unit = _zijderveld_on(ax, self.data, coords, self._kwargs)
        if self._opts["components"]:
            _components_on(ax, self.data, coords)
        return ax

    def _legend_defaults(self):
        # the unit is known only after drawing
        return {"title": f"Unit={self._unit:g} A/m"}


class DemagPlot(PaleomagPlot):
    """Intensity decay plot, ``M / Mmax`` against treatment level.

    Args:
        specimen: ``Specimen``.
    """

    window_title = "Demagnetization plot"

    def _default_title(self):
        return f"{_title(self.data)} (Mmax = {self.data.intensity.max():g} A/m)"

    def _draw(self):
        ax = self.fig.subplots()
        _demag_on(ax, self.data)
        return ax


class ComponentSearchPlot(PaleomagPlot):
    """MAD of every window of steps of a ``Specimen``, for the automatic search of components.

    Each point is a window from the step on the vertical axis to the step on the
    horizontal axis, coloured by the MAD of its free line fit (capped at twice the
    limit); windows over the limit of the criteria are grey. The segments chosen by
    ``Specimen.find_components`` are outlined; they show how the limit and the
    minimum number of steps shape the result.

    Args:
        specimen: ``Specimen``.

    Keyword Args:
        min_steps (int): smallest window, default 4.
        criteria (ComponentCriteria): the limits, ``SEARCH_CRITERIA`` by default.
        max_components (int): largest number of segments, default 3.
        **kwargs: the keyword arguments of ``PaleomagPlot``.
    """

    window_title = "Component search"
    _options: ClassVar[dict] = {
        "min_steps": 4,
        "criteria": None,
        "max_components": 3,
    }

    def __init__(self, specimen, **kwargs):
        super().__init__(specimen, **kwargs)
        if "figsize" not in kwargs:
            self._kwargs["figsize"] = (7, 6)

    def _default_title(self):
        return f"{_title(self.data)}: MAD of the windows"

    def _draw(self):
        from apsg.paleomag._components import SEARCH_CRITERIA

        spec = self.data
        crit = self._opts["criteria"] or SEARCH_CRITERIA
        table = spec.window_fits(min_steps=self._opts["min_steps"])
        ax = self.fig.subplots()
        lim = crit.mad_max or 10.0
        ok = table["mad"] <= lim
        ax.scatter(
            table["end"][~ok], table["start"][~ok], s=60, color="0.85", marker="s"
        )
        sc = ax.scatter(
            table["end"][ok], table["start"][ok], s=60, c=table["mad"][ok],
            cmap="viridis_r", vmin=0, vmax=lim, marker="s",
        )  # fmt: skip
        self.fig.colorbar(sc, ax=ax, label="MAD [deg]")
        colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
        comps = spec.find_components(
            crit, max_components=self._opts["max_components"], coords="specimen"
        )
        labels = {lab: i for i, lab in enumerate(spec.labels)}
        for k, comp in enumerate(comps):
            i, j = labels[comp.labels[0]], labels[comp.labels[-1]]
            ax.add_patch(
                plt.Rectangle(
                    (j - 0.5, i - 0.5), 1, 1, fill=False, lw=2.5,
                    ec=colors[k % len(colors)], label=f"C{k + 1} (MAD {comp.mad:.1f})",
                )
            )  # fmt: skip
        ticks = range(len(spec))
        ax.set_xticks(ticks, [f"{lv:g}" for lv in spec.levels], rotation=90)
        ax.set_yticks(ticks, [f"{lv:g}" for lv in spec.levels])
        ax.set_xlabel("Last step of the window")
        ax.set_ylabel("First step of the window")
        ax.set_aspect(1)
        return ax


class KappaTPlot(PaleomagPlot):
    """Susceptibility and intensity against temperature of a thermal ``Specimen``.

    The bulk susceptibility `k` of the steps (left axis) and ``M / Mmax`` (right
    axis, dashed) against the treatment level. A vertical line marks the
    `susceptibility_jump`, the growth of new magnetic minerals on heating.

    Args:
        specimen: ``Specimen`` with the susceptibility of its steps.

    Keyword Args:
        factor (float): ratio of consecutive susceptibilities of the jump, default 2.5.
        **kwargs: the keyword arguments of ``PaleomagPlot``.
    """

    window_title = "Susceptibility-temperature plot"
    _options: ClassVar[dict] = {"factor": 2.5}

    def __init__(self, specimen, **kwargs):
        super().__init__(specimen, **kwargs)
        steps = [s for s in specimen.steps if s.k is not None]
        if not steps:
            raise ValueError(f"{specimen.name}: the steps have no susceptibility")
        self._steps = steps
        self._jump = specimen.susceptibility_jump(self._opts["factor"])

    def _default_title(self):
        jump = "" if self._jump is None else f", jump at {self._jump:g}°C"
        return f"{_title(self.data)}{jump}"

    def _draw(self):
        ax = self.fig.subplots()
        lv = [s.level for s in self._steps]
        ax.plot(
            lv, [s.k for s in self._steps], "o-", color=self._kwargs["horizontal_color"]
        )
        ax.set_xlabel("Temperature [°C]")
        ax.set_ylabel("Susceptibility", color=self._kwargs["horizontal_color"])
        ax.yaxis.grid()
        M = self.data.intensity
        twin = ax.twinx()
        twin.plot(self.data.levels, M / M.max(), "k--")
        twin.set_ylabel("M/Mmax")
        twin.set_ylim(0, 1.02)
        if self._jump is not None:
            ax.axvline(self._jump, color=self._kwargs["vertical_color"], ls=":")
        return ax


def _specimen_net(specimen, coords, components, **kwargs):
    data = specimen.vectors(coords).normalized()
    s = StereoNet(**kwargs)
    s.arc(data, color="k", ls=":")
    s.vector(data[0], color="k", marker="+", ms=14)
    s.vector(data, color="k", marker="o", label="Steps")
    if components:
        for name in specimen.components:
            c = specimen.component(name, coords)
            if c.kind == "line":
                s.vector(c.vector.normalized(), marker="*", ms=12, label=name)
            else:
                s.great_circle(c.fol, label=name)
    return s


class SpecimenStereoPlot(PaleomagPlot):
    """Stereonet of the demagnetization path of a ``Specimen``.

    The steps are joined by arcs in measurement order, the NRM is marked by a
    cross; lower and upper hemisphere vectors are distinguished by filled and
    open markers. Stored components are added (lines as vectors, planes as
    great circles) unless ``components=False``.

    Args:
        specimen: ``Specimen``.

    Keyword Args:
        coords (str): ``"specimen"``, ``"geo"`` (default) or ``"tilt"``.
        components (bool): plot the stored components. Default True.
        **kwargs: the keyword arguments of ``PaleomagPlot``.
    """

    window_title = "Specimen stereonet"
    _options: ClassVar[dict] = {"coords": "geo", "components": True}

    def _default_title(self):
        return f"{_title(self.data)}\n{_TITLES[_check_coords(self._opts['coords'])]}"

    def _net(self):
        return _specimen_net(
            self.data,
            _check_coords(self._opts["coords"]),
            self._opts["components"],
            title=self._title_text() or "",
            figsize=self._kwargs["figsize"],
            legend_position=self._kwargs["legend_position"],
            legend_kws=self._kwargs["legend_kws"],
            title_kws=self._kwargs["title_kws"],
        )

    def _draw(self):
        self._net().render2fig(self.fig)

    def _render(self, owned=False):
        # the net lays out its own legend and title
        self._draw()
        self._layout = None


def _stereo_panel(fig, title, cfg, draw, owned=False):
    """Render a ``StereoNet`` filled by `draw(net)` into `fig`.

    With `owned` the net fits the figure to itself (title, legend) as
    ``StereoNet.show`` does, instead of keeping the size of `fig`.
    """
    s = StereoNet(
        title=title,
        legend_position=cfg["legend_position"],
        legend_kws=cfg["legend_kws"],
        title_kws=cfg["title_kws"],
    )
    draw(s)
    s.fig = fig
    s._render(owned=owned)
    return s


class SpecimenPlot(PaleomagPlot):
    """Overview of a ``Specimen``: intensity decay, Zijderveld diagram and
    stereonet side by side.

    Args:
        specimen: ``Specimen``.

    Keyword Args:
        coords (str): ``"specimen"``, ``"geo"`` (default) or ``"tilt"``.
        **kwargs: the keyword arguments of ``PaleomagPlot``.
    """

    window_title = "Specimen overview"
    default_figsize = (15, 5)
    _options: ClassVar[dict] = {"coords": "geo", "components": True}

    def _default_title(self):
        return f"{_title(self.data)} - {_TITLES[_check_coords(self._opts['coords'])]}"

    def _draw(self):
        coords = _check_coords(self._opts["coords"])
        f1, f2, f3 = self.fig.subfigures(1, 3)
        ax = f1.subplots()
        _demag_on(ax, self.data)
        ax.set_title(f"Mmax = {self.data.intensity.max():g} A/m")
        ax = f2.subplots()
        unit = _zijderveld_on(ax, self.data, coords, self._kwargs)
        ax.legend(title=f"Unit={unit:g} A/m", loc="lower left")
        SpecimenStereoPlot(
            self.data,
            coords=coords,
            components=self._opts["components"],
            title="",
            legend_position=self._kwargs["legend_position"],
            legend_kws=self._kwargs["legend_kws"],
        ).render2fig(f3)


# -- sample level -----------------------------------------------------------
def _need_sample(data, name, cls):
    if not isinstance(data, Sample):
        raise TypeError(f"{cls} needs a Sample, got {type(data).__name__}")
    if name is None:
        raise TypeError(f"{cls} needs the component name")


class FoldTestPlot(PaleomagPlot):
    """Fold test of the line components of a ``Sample``.

    ``kind="tauxe_watson"`` shows the largest eigenvalue ``tau1`` of the
    orientation tensor against the percentage of untilting, with the
    confidence interval of the bootstrap maxima (Tauxe & Watson, 1994) and the
    histogram of the maxima. ``kind="mcfadden"`` shows the correlation
    statistic ``xi`` against the percentage of untilting with the critical
    band (McFadden, 1990). ``kind="both"`` (default for a ``Sample``) draws
    both side by side.

    Args:
        data: ``Sample`` (the tests are run, `name` is the component name), or
            the dict returned by ``Sample.fold_test`` or
            ``mcfadden_fold_test`` (`kind` is then inferred).
        name: component name (``Sample`` only).

    Keyword Args:
        kind (str): ``"tauxe_watson"``, ``"mcfadden"`` or ``"both"``.
        n (int): bootstrap resamples. Default 1000.
        rng: None, seed or ``numpy.random.Generator``.
        fracs: percentages of untilting tried (default of the tests).
        **kwargs: the keyword arguments of ``PaleomagPlot``.
    """

    window_title = "Fold test"
    _options: ClassVar[dict] = {"kind": None, "n": 1000, "rng": None, "fracs": None}

    def __init__(self, data, name=None, **kwargs):
        super().__init__(data, **kwargs)
        kind = self._opts["kind"]
        if isinstance(data, Sample):
            _need_sample(data, name, "FoldTestPlot")
            kind = kind or "both"
            kw = {"fracs": self._opts["fracs"], "level": self._kwargs["level"]}
            self._tw = self._mf = None
            if kind in ("tauxe_watson", "both"):
                self._tw = data.fold_test(
                    name, n=self._opts["n"], rng=self._opts["rng"], **kw
                )
            if kind in ("mcfadden", "both"):
                self._mf = data.mcfadden_fold_test(name, **kw)
            self.name = name
        elif isinstance(data, dict) and "tau1" in data:
            kind, self._tw, self._mf, self.name = "tauxe_watson", data, None, name
        elif isinstance(data, dict) and "xi" in data:
            kind, self._tw, self._mf, self.name = "mcfadden", None, data, name
        else:
            raise TypeError("FoldTestPlot needs a Sample or a fold test result dict")
        if kind not in ("tauxe_watson", "mcfadden", "both"):
            raise ValueError(
                f"kind must be 'tauxe_watson', 'mcfadden' or 'both', got {kind!r}"
            )
        self.kind = kind
        if kind == "both" and "figsize" not in kwargs:
            self._kwargs["figsize"] = (11, 4.5)

    def _default_title(self):
        return f"Fold test{' ' + str(self.name) if self.name else ''}"

    def _tauxe_watson(self, fig):
        r = self._tw
        cfg = self._kwargs
        ax, hx = fig.subplots(2, 1, sharex=True, height_ratios=(3, 1))
        ax.plot(r["fracs"], r["tau1"], "k-", label=r"$\tau_1$")
        ax.axvspan(
            *r["ci"],
            color=cfg["ci_color"],
            alpha=cfg["ci_alpha"],
            label=f"{cfg['level']:.0%} interval",
        )
        ax.axvline(r["best"], color=cfg["mean_color"], ls=":", label="Maximum")
        ax.set_ylabel(r"$\tau_1$")
        ax.grid(cfg["grid"])
        step = np.diff(r["fracs"]).mean() if len(r["fracs"]) > 1 else 1.0
        edges = np.append(r["fracs"] - step / 2, r["fracs"][-1] + step / 2)
        hx.hist(r["maxima"], bins=edges, color=cfg["ci_color"])
        hx.set_xlabel("Untilting [%]")
        hx.set_ylabel("Maxima")
        return ax

    def _mcfadden(self, fig):
        r = self._mf
        cfg = self._kwargs
        ax = fig.subplots()
        ax.axhspan(
            -r["critical"],
            r["critical"],
            color=cfg["ci_color"],
            alpha=cfg["ci_alpha"],
            label=f"{r['level']:.0%} critical band",
        )
        ax.plot(r["fracs"], r["xi"], "k-", label=r"$\xi$")
        ax.axhline(0, color="k", lw=0.5)
        ax.axvline(r["best"], color=cfg["mean_color"], ls=":", label="Minimum |ξ|")
        for key, label in (("geo", "In situ"), ("tilt", "Tilt-corrected")):
            if r[key] is not None:
                x = 0 if key == "geo" else 100
                ax.plot(x, r[key], "o", label=label)
        ax.set_xlabel("Untilting [%]")
        ax.set_ylabel(r"$\xi$")
        ax.grid(cfg["grid"])
        return ax

    def _draw(self):
        if self.kind == "tauxe_watson":
            # two stacked panels: no single main axes for FigureLayout, which would
            # trim the figure to the upper panel only
            self._tauxe_watson(self.fig).legend(loc="best", fontsize=8)
            return None
        if self.kind == "mcfadden":
            return self._mcfadden(self.fig)
        f1, f2 = self.fig.subfigures(1, 2)
        for f, draw, title in (
            (f1, self._tauxe_watson, "Tauxe & Watson (1994)"),
            (f2, self._mcfadden, "McFadden (1990)"),
        ):
            ax = draw(f)
            ax.legend(loc="best", fontsize=8)
            ax.set_title(title, fontsize=10)
        return None


class TiltComparisonPlot(PaleomagPlot):
    """Line components of a ``Sample`` in geographic and tilt-corrected
    coordinates, with the mean direction and its confidence cone.

    Args:
        sample: ``Sample``.
        name: component name.

    Keyword Args:
        **kwargs: the keyword arguments of ``PaleomagPlot``.
    """

    window_title = "Tilt correction"
    default_figsize = (10, 5)

    def __init__(self, sample, name, **kwargs):
        _need_sample(sample, name, "TiltComparisonPlot")
        super().__init__(sample, **kwargs)
        self.name = name

    def _default_title(self):
        return f"{self.data.name} {self.name}".strip()

    def _draw(self):
        cfg = self._kwargs
        for sub, coords in zip(self.fig.subfigures(1, 2), ("geo", "tilt")):
            vecs = self.data.directions(self.name, coords)
            mean = self.data.mean(self.name, coords, cfg["level"])

            def draw(s, vecs=vecs, mean=mean):
                s.vector(vecs, color=cfg["normal_color"], label=f"{self.name}")
                s.vector(
                    mean["mu"].normalized(),
                    color=cfg["mean_color"],
                    marker="*",
                    ms=14,
                    label="Mean",
                )
                s.cone(
                    Cone(mean["mu"].normalized(), float(mean["alpha"])),
                    color=cfg["mean_color"],
                    label=f"{cfg['level']:.0%} cone",
                )

            _stereo_panel(sub, _TITLES[coords], cfg, draw)


class ReversalTestPlot(PaleomagPlot):
    """Bootstrap reversal test of a ``Sample`` (Tauxe, 2003, Fig. 4.17).

    The left stereonet shows the normal and reversed modes, the right plot the
    bootstrap confidence intervals of the three Cartesian components of the
    mean of the normal mode and of the flipped reversed mode. The modes share
    a mean when the intervals overlap in all components.

    Args:
        sample: ``Sample``.
        name: component name.

    Keyword Args:
        coords (str): coordinates, default ``"tilt"``.
        n (int): bootstrap resamples. Default 1000.
        rng: None, seed or ``numpy.random.Generator``.
        **kwargs: the keyword arguments of ``PaleomagPlot``.
    """

    window_title = "Reversal test"
    default_figsize = (10, 4.5)
    _options: ClassVar[dict] = {"coords": "tilt", "n": 1000, "rng": None}

    def __init__(self, sample, name, **kwargs):
        _need_sample(sample, name, "ReversalTestPlot")
        super().__init__(sample, **kwargs)
        self.name = name
        coords = _check_coords(self._opts["coords"])
        self._result = sample.reversal_test(
            name,
            coords,
            method="bootstrap",
            level=self._kwargs["level"],
            n=self._opts["n"],
            rng=self._opts["rng"],
        )

    def _default_title(self):
        verdict = "not antipodal" if self._result.reject() else "common mean"
        return f"Reversal test {self.name} ({verdict})"

    def _draw(self):
        cfg = self._kwargs
        coords = _check_coords(self._opts["coords"])
        D = np.asarray(self.data.directions(self.name, coords), dtype=float)
        axis = np.linalg.eigh(D.T @ D)[1][:, -1]
        side = D @ axis >= 0
        # the sign of the principal axis is arbitrary: call the mode pointing
        # down (positive inclination) normal
        swap = D[side].sum(axis=0)[2] < 0
        if swap:
            side = ~side
        f1, f2 = self.fig.subfigures(1, 2)
        vecs = self.data.directions(self.name, coords)

        def draw(s):
            s.vector(
                vecs[np.flatnonzero(side)], color=cfg["normal_color"], label="Normal"
            )
            s.vector(
                vecs[np.flatnonzero(~side)],
                color=cfg["reversed_color"],
                label="Reversed",
            )

        _stereo_panel(f1, _TITLES[coords], cfg, draw)
        ax = f2.subplots()
        lo, hi = self._result.extra["lower"], self._result.extra["upper"]
        colors = (cfg["normal_color"], cfg["reversed_color"])
        labels = ("Normal", "Reversed (flipped)")
        for m in range(2):
            k = 1 - m if swap else m  # row of the test result
            y = np.arange(3) + (0.15 if m == 0 else -0.15)
            ax.hlines(y, lo[k], hi[k], color=colors[m], lw=6, label=labels[m])
        ax.set_yticks(range(3), ["x1", "x2", "x3"])
        ax.invert_yaxis()
        ax.set_xlabel(f"{cfg['level']:.0%} bootstrap interval of the mean")
        ax.grid(cfg["grid"], axis="x")
        ax.legend(loc="best", fontsize=8)


class LinesPlanesPlot(PaleomagPlot):
    """Lines and great circles of a ``Sample`` with their combined mean.

    Line components are drawn as points, plane components as great circles.
    ``method="mcfadden"`` (default) shows the mean direction of ``Sample.mean``
    (Fisher for lines only, otherwise the combined mean of McFadden &
    McElhinny, 1988) with its circular cone, ``method="bailey_halls"`` the
    maximum likelihood mean of Bailey & Halls (1984) with its confidence ellipse
    (``bailey_halls_mean``, needs at least one great circle) and ``"both"``
    draws both for comparison.

    Args:
        sample: ``Sample``.
        name: component name.

    Keyword Args:
        coords (str): coordinates, default ``"tilt"``.
        method (str): ``"mcfadden"``, ``"bailey_halls"`` or ``"both"``.
        **kwargs: the keyword arguments of ``PaleomagPlot``.
    """

    window_title = "Lines and planes"
    _options: ClassVar[dict] = {"coords": "tilt", "method": "mcfadden"}

    def __init__(self, sample, name, **kwargs):
        _need_sample(sample, name, "LinesPlanesPlot")
        super().__init__(sample, **kwargs)
        self.name = name
        method = self._opts["method"]
        if method not in ("mcfadden", "bailey_halls", "both"):
            raise ValueError(
                f"method must be 'mcfadden', 'bailey_halls' or 'both', got {method!r}"
            )
        self._bh = None
        if method != "mcfadden":
            coords = _check_coords(self._opts["coords"])
            lines = np.asarray(sample.directions(name, coords), dtype=float)
            poles = np.asarray(sample.planes(name, coords), dtype=float)
            if len(poles) == 0:
                raise ValueError(
                    f"method {method!r} needs great circle components, "
                    f"{name!r} has none"
                )
            self._bh = bailey_halls_mean(
                lines.reshape(-1, 3), poles, self._kwargs["level"]
            )

    def _default_title(self):
        coords = _check_coords(self._opts["coords"])
        return f"{self.data.name} {self.name}\n{_TITLES[coords]}".strip()

    def _draw(self):
        cfg = self._kwargs
        coords = _check_coords(self._opts["coords"])
        lines = self.data.directions(self.name, coords)
        planes = self.data.planes(self.name, coords)
        method = self._opts["method"]
        mean = (
            self.data.mean(self.name, coords, cfg["level"])
            if method != "bailey_halls"
            else None
        )

        def draw(s):
            if len(planes):
                s.great_circle(planes, color="0.5", label="Planes")
            if len(lines):
                s.vector(lines, color=cfg["normal_color"], label="Lines")
            if mean is not None:
                s.vector(
                    mean["mu"].normalized(),
                    color=cfg["mean_color"],
                    marker="*",
                    ms=14,
                    label="Mean",
                )
                s.cone(
                    Cone(mean["mu"].normalized(), float(mean["alpha"])),
                    color=cfg["mean_color"],
                    label=f"{cfg['level']:.0%} cone",
                )
            if self._bh is not None:
                s.vector(
                    self._bh["mu"].normalized(),
                    color=cfg["ci_color"],
                    marker="D",
                    ms=9,
                    label="Bailey-Halls mean",
                )
                s.confidence(
                    self._bh["region"],
                    color=cfg["ci_color"],
                    label=f"{cfg['level']:.0%} ellipse",
                )

        self._net = _stereo_panel(
            self.fig, self._title_text(), cfg, draw, owned=self._owned
        )
        self._layout = self._net._layout

    _owned = False

    def _render(self, owned=False):
        self._owned = owned
        self._draw()


def _inclination_pdf(inc, theta0, k):
    """Density (per degree) of the inclination of Fisher directions about the
    pole at the angle `theta0` (radians) from the vertical, with precision
    `k` (McFadden & Reid, 1982, eq. 5, integrated over the declination)."""
    theta = np.radians(90 - np.asarray(inc, dtype=float))
    z = k * np.sin(theta) * np.sin(theta0)
    with np.errstate(divide="ignore"):
        logf = (
            np.log(k)
            + np.log(np.sin(theta))
            - k
            - np.log1p(-np.exp(-2 * k))
            + k * np.cos(theta) * np.cos(theta0)
            + np.log(ive(0, z))
            + z
        )
    return np.exp(logf) * np.pi / 180


class InclinationOnlyPlot(PaleomagPlot):
    """Mean inclination of inclination-only data (McFadden & Reid, 1982).

    Shows the histogram of the inclinations with a rug, the density of the
    inclination implied by the estimate (a Fisher distribution of precision
    ``k`` about the mean direction, declination unknown), the maximum
    likelihood inclination (line) and the interval ``inc_conditional +/-
    alpha`` of ``inclination_only_mean`` (band). For a ``Sample`` the
    inclination of the mean of the full directions is added for comparison.
    ``diagnostic=True`` adds the cubic of eq. 19 with its roots (the accepted
    one, ``U < 0``, filled), which shows whether the data fold back through the
    vertical and the estimate is ambiguous.

    Args:
        data: inclinations in degrees (positive down), or a ``Sample`` (then
            `name` is the component name).
        name: component name (``Sample`` only).

    Keyword Args:
        coords (str): coordinates of a ``Sample``, default ``"tilt"``.
        flip (bool): turn reversed directions of a ``Sample`` to their
            antipodes first, default False.
        diagnostic (bool): add the plot of eq. 19. Default False.
        bins (int): number of histogram bins, default from the sample size.
        **kwargs: the keyword arguments of ``PaleomagPlot``.
    """

    window_title = "Inclination-only mean"
    _options: ClassVar[dict] = {
        "coords": "tilt",
        "flip": False,
        "diagnostic": False,
        "bins": None,
    }

    def __init__(self, data, name=None, **kwargs):
        super().__init__(data, **kwargs)
        self._full = None
        if self._opts["diagnostic"] and "figsize" not in kwargs:
            self._kwargs["figsize"] = (10, 4.5)
        if isinstance(data, Sample):
            _need_sample(data, name, "InclinationOnlyPlot")
            D = np.asarray(
                data.directions(name, _check_coords(self._opts["coords"])), dtype=float
            )
            if self._opts["flip"]:
                D = D * np.where(D[:, 2] < 0, -1.0, 1.0)[:, None]
            inc = np.degrees(np.arcsin(np.clip(D[:, 2], -1, 1)))
            m = D.sum(axis=0)
            self._full = float(np.degrees(np.arcsin(m[2] / np.linalg.norm(m))))
        else:
            inc = np.asarray(data, dtype=float).ravel()
        self.name = name
        self._inc = inc
        self._res = inclination_only_mean(inc, self._kwargs["level"])

    def _default_title(self):
        r = self._res
        return f"Inclination only: I = {r['inc']:.1f}°, k = {r['k']:.0f}, n = {r['n']}"

    def _main(self, ax):
        cfg, r, inc = self._kwargs, self._res, self._inc
        bins = self._opts["bins"] or max(5, int(np.sqrt(len(inc))))
        ax.hist(inc, bins=bins, density=True, color=cfg["normal_color"], alpha=0.4)
        ax.plot(
            inc,
            np.zeros(len(inc)),
            "|",
            color=cfg["normal_color"],
            ms=12,
            clip_on=False,
        )
        lo = max(-90, min(inc.min(), r["inc"]) - 15)
        hi = min(90, max(inc.max(), r["inc"]) + 15)
        x = np.linspace(lo, hi, 400)
        ax.plot(
            x,
            _inclination_pdf(x, r["theta0"], r["k"]),
            "k-",
            label=f"Fisher, k = {r['k']:.0f}",
        )
        ax.axvspan(
            r["inc_conditional"] - r["alpha"],
            r["inc_conditional"] + r["alpha"],
            color=cfg["ci_color"],
            alpha=cfg["ci_alpha"],
            label=f"{cfg['level']:.0%} interval (±{r['alpha']:.1f}°)",
        )
        ax.axvline(r["inc"], color=cfg["mean_color"], label=f"Mean I = {r['inc']:.1f}°")
        if self._full is not None:
            ax.axvline(
                self._full,
                color=cfg["mean_color"],
                ls="--",
                label=f"Full directions I = {self._full:.1f}°",
            )
        ax.set_xlim(lo, hi)
        ax.set_xlabel("Inclination [°]")
        ax.set_ylabel("Density")
        ax.grid(cfg["grid"])
        return ax

    def _diagnostic(self, ax):
        cfg = self._kwargs
        theta = np.radians(90 - self._inc)
        t = np.linspace(1e-3, np.pi - 1e-3, 721)
        ax.plot(90 - np.degrees(t), _cubic(theta)(t), "k-")
        ax.axhline(0, color="k", lw=0.5)
        roots = _theta0_roots(theta)
        for accepted in (True, False):
            sel = [x for x in roots if (x[1] < 0) == accepted]
            if sel:
                ax.plot(
                    [90 - np.degrees(x[0]) for x in sel],
                    np.zeros(len(sel)),
                    "o",
                    ms=8,
                    color=cfg["mean_color"],
                    mfc=cfg["mean_color"] if accepted else "none",
                    label="Root, U < 0" if accepted else "Root, U > 0 (rejected)",
                )
        ax.set_xlabel("Mean inclination θ0 [°]")
        ax.set_ylabel("Eq. 19")
        ax.grid(cfg["grid"])
        return ax

    def _draw(self):
        if not self._opts["diagnostic"]:
            return self._main(self.fig.subplots())
        a, b = self.fig.subplots(1, 2)
        self._main(a).legend(loc="best", fontsize=8)
        self._diagnostic(b).legend(loc="best", fontsize=8)
        return None
