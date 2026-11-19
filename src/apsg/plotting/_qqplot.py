"""Q-Q plots of directions against the Fisher distribution."""

from typing import ClassVar

import numpy as np

from apsg.feature._distributions._spherical import VonMisesFisher
from apsg.feature._hypothesis import fisher_goodness_of_fit, fisher_qq_data
from apsg.math._vector import Axial3
from apsg.plotting._figureplot import FigurePlot

__all__ = ("FisherQQPlot",)

_WHICH = ("both", "colatitude", "longitude")


class FisherQQPlot(FigurePlot):
    """Q-Q plots of directions against the Fisher distribution.

    The data are rotated to their mean direction. If they are Fisher
    distributed, the colatitude statistic ``1 - cos(theta')`` follows an
    exponential distribution (rate ``k = (n - 1) / sum(1 - cos(theta'))``) and
    the longitude is uniform, so the ordered values fall on the diagonal of
    both plots when plotted against the quantiles of these distributions
    (Fisher, Lewis & Embleton, 1987, sec. 5.3.4; Tauxe, 2003, sec. 4.6). The
    titles give the statistics ``M_E`` (colatitude) and ``M_U`` (longitude) of
    ``VonMisesFisher.goodness_of_fit`` with their critical values.

    Args:
        data: ``Vector3Set`` or other feature set of directions, or an
            ``(N, 3)`` array. Axial sets (lineations, foliation poles) are
            folded to one hemisphere first (``halfspace``). Data of both
            polarities should be flipped into one mode before.

    Keyword Args:
        which (str): ``"both"`` (default), ``"colatitude"`` or ``"longitude"``.
        level (float): level of the critical values, 0.90, 0.95 (default) or
            0.99.
        color, line_color, marker, ms, grid: styles, see ``apsg_conf.qqplot``.
        title, figsize, title_kws, legend_position, legend_kws: see
            ``FigurePlot``.

    Examples:
        >>> v = vecset.random_fisher(n=50, kappa=30, mean=vec(120, 40))
        >>> FisherQQPlot(v).show()
    """

    config_key = "qqplot"
    window_title = "Fisher Q-Q plot"
    default_figsize = None
    _options: ClassVar[dict] = {"which": "both"}

    def __init__(self, data, **kwargs):
        super().__init__(data, **kwargs)
        if self._opts["which"] not in _WHICH:
            raise ValueError(f"which must be one of {_WHICH}")
        if hasattr(data, "halfspace") and issubclass(
            getattr(data, "__feature_class__", object), Axial3
        ):
            data = data.halfspace()
        self._X = VonMisesFisher.as_array(data)
        if len(self._X) < 3:
            raise ValueError("At least three directions are needed")
        self._qq = fisher_qq_data(self._X)
        self._tests = fisher_goodness_of_fit(self._X, self._kwargs["level"])
        if self._opts["which"] != "both" and "figsize" not in kwargs:
            self._kwargs["figsize"] = (5, 5)

    def _default_title(self):
        return f"Fisher Q-Q plot (n = {self._qq['n']})"

    def _panel(self, ax, observed, expected, test, xlabel, ylabel):
        cfg = self._kwargs
        lim = (0, 1.05 * max(observed.max(), expected.max()))
        ax.plot(lim, lim, "-", color=cfg["line_color"], lw=1)
        ax.plot(
            expected,
            observed,
            ls="none",
            marker=cfg["marker"],
            ms=cfg["ms"],
            color=cfg["color"],
        )
        ax.set_xlim(lim)
        ax.set_ylim(lim)
        ax.set_aspect(1)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.grid(cfg["grid"])
        verdict = "rejected" if test.reject() else "consistent"
        ax.set_title(
            f"{test.name.split()[-1]} = {test.statistic:.3f} "
            f"(critical {test.critical:.3f}), {verdict} at {cfg['level']:.0%}",
            fontsize=10,
        )

    def _draw(self):
        q = self._qq
        n = q["n"]
        p = (np.arange(1, n + 1) - 0.5) / n
        which = self._opts["which"]
        panels = []
        if which in ("both", "colatitude"):
            panels.append(
                (
                    q["x"],
                    -np.log1p(-p) / q["k"],
                    self._tests["colatitude"],
                    "Exponential quantiles",
                    "1 - cos(colatitude)",
                )
            )
        if which in ("both", "longitude"):
            panels.append(
                (
                    q["u"],
                    p,
                    self._tests["longitude"],
                    "Uniform quantiles",
                    "Longitude / 2π",
                )
            )
        axes = np.atleast_1d(self.fig.subplots(1, len(panels)))
        for ax, args in zip(axes, panels):
            self._panel(ax, *args)
