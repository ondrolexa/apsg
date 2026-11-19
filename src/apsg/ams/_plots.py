"""AMS plots: Jelinek and Flinn-type diagrams, site stereonets and bootstrap plots."""

from typing import ClassVar

import numpy as np

from apsg.ams._specimen import AMSSite
from apsg.feature import EllipsoidSet, Lineation, LineationSet, TensorBootstrap
from apsg.feature._distributions._base import ConfidenceRegion
from apsg.feature._statistics import jelinek_shape
from apsg.plotting import StereoNet
from apsg.plotting._figureplot import FigurePlot

__all__ = (
    "AMSPlot",
    "AMSStereoPlot",
    "BootstrapPlot",
    "LFPlot",
    "ModelPlot",
    "PTPlot",
    "PjTPlot",
)


class AMSPlot(FigurePlot):
    """Base class of AMS plots, see ``FigurePlot``.

    Keyword Args:
        title, figsize, title_kws, legend_position, legend_kws: see
            ``FigurePlot``.
        level (float): confidence level of regions.
        marker, ms, alpha: style of the specimens of diagrams.
        mean_marker, mean_ms: style of the mean of a site.
        k1_marker, k2_marker, k3_marker, k1_color, k2_color, k3_color: styles of
            the principal axes on stereonets.
        ellipse_ls, grid: see ``apsg_conf.ams``.
    """

    config_key = "ams"
    window_title = "AMS plot"

    def savefig(self, filename="ams.png", **kwargs):
        """Save the plot to a graphics file (kwargs go to ``Figure.savefig``)."""
        super().savefig(filename, **kwargs)


def _sites(data):
    """dict name -> AMSSite from a site, sequence or dict of sites or a tensor set."""
    if isinstance(data, AMSSite):
        return {data.name: data}
    if isinstance(data, dict):
        out = dict(data)
    elif isinstance(data, (list, tuple)):
        if not all(isinstance(s, AMSSite) for s in data):
            raise TypeError("expected AMSSite objects")
        out = {s.name: s for s in data}
    else:
        raise TypeError(
            f"expected an AMSSite, a list or dict of sites, got {type(data).__name__}"
        )
    if not out or not all(isinstance(s, AMSSite) for s in out.values()):
        raise TypeError("expected AMSSite objects")
    return out


def _parameters(data, coords):
    """List of ``(label, dict of arrays L F P Pj T)`` of the groups of `data`."""
    if isinstance(data, EllipsoidSet):
        groups = {data.name: [np.asarray(e) for e in data]}
    else:
        groups = {
            name: [s.tensor_in(coords, normalize=True) for s in site]
            for name, site in _sites(data).items()
        }
    out = []
    for name, tensors in groups.items():
        rows = [jelinek_shape(np.linalg.eigvalsh(t)) for t in tensors]
        out.append((name, {k: np.array([r[k] for r in rows]) for k in rows[0]}))
    return out


class _DiagramPlot(AMSPlot):
    """Scatter diagram of Jelinek parameters of specimens of one or more sites."""

    _options: ClassVar[dict] = {"coords": "specimen", "mean": False, "logx": None}
    xkey = ykey = ""
    xlabel = ylabel = ""
    logx_default = False

    def __init__(self, data, **kwargs):
        super().__init__(data, **kwargs)
        if self._opts["coords"] not in ("specimen", "geographic", "tectonic"):
            raise ValueError(f"unknown coords {self._opts['coords']!r}")
        self.groups = _parameters(data, self._opts["coords"])

    def _default_title(self):
        return None

    def _limits(self, ax):
        pass

    def _draw(self):
        cfg = self._kwargs
        ax = self.fig.subplots()
        for i, (name, p) in enumerate(self.groups):
            color = f"C{i}"
            x, y = self.xvalues(p), self.yvalues(p)
            ax.plot(
                x,
                y,
                cfg["marker"],
                color=color,
                ms=cfg["ms"],
                alpha=cfg["alpha"],
                ls="",
                label=name,
            )
            if self._opts["mean"]:
                ax.plot(
                    np.mean(x),
                    np.mean(y),
                    cfg["mean_marker"],
                    color=color,
                    ms=cfg["mean_ms"],
                    mec="k",
                )
        logx = self._opts["logx"]
        if self.logx_default if logx is None else logx:
            ax.set_xscale("log")
        ax.set_xlabel(self.xlabel)
        ax.set_ylabel(self.ylabel)
        ax.grid(cfg["grid"])
        self._limits(ax)
        return ax

    def xvalues(self, p):
        return p[self.xkey]

    def yvalues(self, p):
        return p[self.ykey]


class PjTPlot(_DiagramPlot):
    """Jelinek diagram: corrected degree ``Pj`` against the shape ``T``.

    ``T`` is +1 for oblate and -1 for prolate ellipsoids (Jelinek, 1981).

    Args:
        data: ``AMSSite``, a list or dict of sites (one colour each) or an
            ``EllipsoidSet`` (positive definite tensors, normalised or not).

    Keyword Args:
        coords (str): ``"specimen"`` (default), ``"geographic"`` or
            ``"tectonic"`` for the tensors of sites (the parameters do not
            depend on it, only on the tensors).
        mean (bool): add the mean of each group. Default False.
        logx (bool): logarithmic ``Pj`` axis. Default False.
        **kwargs: the keyword arguments of ``AMSPlot``.
    """

    window_title = "Pj-T"
    xkey, ykey = "Pj", "T"
    xlabel, ylabel = "$P_j$", "$T$"

    def _limits(self, ax):
        ax.set_ylim(-1.05, 1.05)
        ax.axhline(0, color="k", lw=0.5)
        lo, hi = ax.get_xlim()
        ax.set_xlim(min(1.0, lo), hi)


class LFPlot(_DiagramPlot):
    """Flinn-type diagram: foliation ``F = k2/k3`` against lineation ``L = k1/k2``.

    The line ``L = F`` separates prolate from oblate ellipsoids. Axes are
    drawn as ``L - 1`` and ``F - 1`` with the 1:1 line, so the origin is the
    isotropic tensor.

    Args and keyword arguments: see ``PjTPlot`` (``logx`` applies to ``L``).
    """

    window_title = "L-F"
    xkey, ykey = "L", "F"
    xlabel, ylabel = "$L = k_1 / k_2$", "$F = k_2 / k_3$"

    def _limits(self, ax):
        hi = max(ax.get_xlim()[1], ax.get_ylim()[1])
        ax.set_xlim(1, hi)
        ax.set_ylim(1, hi)
        ax.plot([1, hi], [1, hi], "k-", lw=0.5)
        ax.set_aspect(1)


class PTPlot(_DiagramPlot):
    """Degree ``P = k1/k3`` against the shape ``T`` (Hrouda, 1982).

    Args and keyword arguments: see ``PjTPlot``.
    """

    window_title = "P-T"
    xkey, ykey = "P", "T"
    xlabel, ylabel = "$P = k_1 / k_3$", "$T$"
    _limits = PjTPlot._limits


def _axes_net(es, cfg, ellipses, n, level, title):
    """StereoNet of the principal axes of the tensor set `es` with regions."""
    net = StereoNet(title=title)
    for i in range(3):
        k = f"k{i + 1}"
        net.point(
            LineationSet([e.eigenlins()[i] for e in es]),
            marker=cfg[f"{k}_marker"],
            color=cfg[f"{k}_color"],
            alpha=cfg["alpha"],
            label=k,
        )
    if ellipses:
        kw = {} if ellipses == "jelinek" else {"n_resamples": n}
        method = "jelinek" if ellipses == "jelinek" else "bootstrap_tensor"
        net.confidence(es, method=method, level=level, ls=cfg["ellipse_ls"], **kw)
    return net


class AMSStereoPlot(AMSPlot):
    """Principal axes of the specimens of a site with confidence regions.

    The axes ``k1`` (squares), ``k2`` (triangles) and ``k3`` (circles) of the
    normalised specimen tensors are drawn on a stereonet together with the
    regions of the axes of the mean tensor.

    Args:
        data: ``AMSSite`` or ``EllipsoidSet``.

    Keyword Args:
        coords (str): ``"specimen"``, ``"geographic"``, ``"tectonic"`` or
            ``"both"`` (geographic and tectonic side by side). Default
            ``"tectonic"``; ``EllipsoidSet`` input is drawn as given.
        ellipses (str): ``"jelinek"`` (linear perturbation, default),
            ``"bootstrap"`` or None.
        n (int): bootstrap resamples. Default 1000.
        **kwargs: the keyword arguments of ``AMSPlot``.
    """

    window_title = "AMS stereonet"
    _options: ClassVar[dict] = {"coords": "tectonic", "ellipses": "jelinek", "n": 1000}

    def __init__(self, data, **kwargs):
        super().__init__(data, **kwargs)
        if not isinstance(data, (AMSSite, EllipsoidSet)):
            raise TypeError("AMSStereoPlot needs an AMSSite or an EllipsoidSet")
        if self._opts["coords"] not in ("specimen", "geographic", "tectonic", "both"):
            raise ValueError(f"unknown coords {self._opts['coords']!r}")
        if self._opts["ellipses"] not in ("jelinek", "bootstrap", None):
            raise ValueError("ellipses must be 'jelinek', 'bootstrap' or None")
        if self._opts["coords"] == "both" and "figsize" not in kwargs:
            self._kwargs["figsize"] = (10, 4.8)

    def _default_title(self):
        return getattr(self.data, "name", None)

    def _net(self, coords, title=None):
        es = (
            self.data
            if isinstance(self.data, EllipsoidSet)
            else self.data.tensors(coords)
        )
        return _axes_net(
            es,
            self._kwargs,
            self._opts["ellipses"],
            self._opts["n"],
            self._kwargs["level"],
            title,
        )

    def _draw(self):
        coords = self._opts["coords"]
        if coords != "both":
            net = self._net(coords)
            net.render2fig(self.fig)
            return
        for sf, c in zip(self.fig.subfigures(1, 2), ("geographic", "tectonic")):
            self._net(c, title=c.capitalize()).render2fig(sf)
        return


class BootstrapPlot(AMSPlot):
    """Bootstrap of the mean tensor: axes on a stereonet and eigenvalue histograms.

    Left, the bootstrapped principal axes with the regions containing
    ``level`` of them; right, the histograms of the three bootstrapped
    eigenvalues, whose modes show the shape class of the fabric (Constable
    and Tauxe, 1990).

    Args:
        data: ``AMSSite`` (bootstrapped here), ``EllipsoidSet`` or a
            ``TensorBootstrap``.

    Keyword Args:
        coords (str): coordinates of an ``AMSSite``. Default ``"tectonic"``.
        n (int): bootstrap resamples. Default 1000.
        rng: None, seed or ``numpy.random.Generator``.
        **kwargs: the keyword arguments of ``AMSPlot``.
    """

    window_title = "AMS bootstrap"
    _options: ClassVar[dict] = {"coords": "tectonic", "n": 1000, "rng": None}

    def __init__(self, data, **kwargs):
        kwargs.setdefault("figsize", (10, 4.5))
        super().__init__(data, **kwargs)
        o, level = self._opts, self._kwargs["level"]
        if isinstance(data, TensorBootstrap):
            self.boot, self.name = data, None
        elif isinstance(data, AMSSite):
            self.boot = data.bootstrap(o["coords"], o["n"], level, o["rng"])
            self.name = data.name
        elif isinstance(data, EllipsoidSet):
            self.boot = data.bootstrap_tensor(o["n"], level, True, o["rng"])
            self.name = data.name
        else:
            raise TypeError(
                "BootstrapPlot needs an AMSSite, an EllipsoidSet or a TensorBootstrap"
            )

    def _default_title(self):
        return f"Bootstrap {self.name}" if self.name else "Bootstrap"

    def _draw(self):
        cfg, b = self._kwargs, self.boot
        f1, f2 = self.fig.subfigures(1, 2)
        net = StereoNet()
        for i in range(3):
            k = f"k{i + 1}"
            pts = LineationSet([Lineation(v) for v in b.eigenvectors[:, :, i]])
            net.point(
                pts,
                marker=cfg[f"{k}_marker"],
                color=cfg[f"{k}_color"],
                alpha=0.3,
                ms=3,
                label=k,
            )
        for e in b.ellipses:
            net.confidence(
                ConfidenceRegion(
                    mu=e["mu"], level=cfg["level"], axes=e["axes"], gamma=e["gamma"]
                ),
                ls=cfg["ellipse_ls"],
            )
        net.render2fig(f1)
        ax = f2.subplots()
        for i in range(3):
            k = f"k{i + 1}"
            ax.hist(
                b.eigenvalues[:, i],
                bins=30,
                color=cfg[f"{k}_color"],
                alpha=0.6,
                label=k,
            )
        ax.set_xlabel("eigenvalue")
        ax.set_ylabel("count")
        ax.grid(cfg["grid"])
        ax.legend()


class ModelPlot(AMSPlot):
    """Evolution of the AMS of a model with time or strain.

    Three panels share the time axis: the degree ``Pj``, the shape ``T`` and the
    plunge of ``k1`` (lineation) and ``k3`` (foliation pole); the first time
    at which ``k1`` is steeper than 45 degrees (the switch of the lineation)
    is marked.

    Args:
        data: ``AMSEvolution`` (from ``AMSModel.evolve`` or ``jeffery_evolution``)
            or a dict of them, drawn together with their names as labels.

    Keyword Args:
        switch (bool): mark the switch of the lineation. Default True.
        xlabel (str): label of the time axis. Default "time".
        **kwargs: the keyword arguments of ``AMSPlot``.
    """

    window_title = "AMS model"
    _options: ClassVar[dict] = {"switch": True, "xlabel": "time"}

    def __init__(self, data, **kwargs):
        kwargs.setdefault("figsize", (6, 6))
        super().__init__(data, **kwargs)
        from apsg.ams._model import AMSEvolution

        if isinstance(data, AMSEvolution):
            data = {None: data}
        if not isinstance(data, dict) or not all(
            isinstance(v, AMSEvolution) for v in data.values()
        ):
            raise TypeError("ModelPlot needs an AMSEvolution or a dict of them")
        self.evolutions = data

    def _draw(self):
        cfg = self._kwargs
        axes = self.fig.subplots(3, 1, sharex=True)
        for i, (name, ev) in enumerate(self.evolutions.items()):
            t = ev.table()
            c = f"C{i}"
            axes[0].plot(t["time"], t["Pj"], color=c, label=name)
            axes[1].plot(t["time"], t["T"], color=c)
            axes[2].plot(t["time"], t["k1_inc"].abs(), color=c, ls="-")
            axes[2].plot(t["time"], t["k3_inc"].abs(), color=c, ls="--")
            sw = ev.switch() if self._opts["switch"] else None
            if sw is not None:
                for ax in axes:
                    ax.axvline(sw, color=c, ls=":", lw=1)
        axes[0].set_ylabel("$P_j$")
        axes[1].set_ylabel("$T$")
        axes[1].set_ylim(-1.05, 1.05)
        axes[1].axhline(0, color="k", lw=0.5)
        axes[2].set_ylabel("plunge $k_1$ (solid), $k_3$ (dashed)")
        axes[2].set_ylim(0, 90)
        axes[2].set_xlabel(self._opts["xlabel"])
        for ax in axes:
            ax.grid(cfg["grid"])
        if any(n is not None for n in self.evolutions):
            axes[0].legend()
