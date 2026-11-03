"""Plots of paleomagnetic demagnetization data of a ``Specimen``."""

import matplotlib.pyplot as plt
import numpy as np

from apsg.config import apsg_conf
from apsg.plotting._stereonet import StereoNet

__all__ = ("demag_plot", "stereo_plot", "zijderveld_plot")

_TITLES = {
    "specimen": "Specimen coordinates",
    "geo": "Geographic coordinates",
    "tilt": "Tilt-corrected coordinates",
}


def _title(specimen):
    return f"{specimen.site} {specimen.name}".strip()


def _axes(ax):
    if ax is None:
        _fig, ax = plt.subplots(figsize=apsg_conf.figsize)
        return ax, True
    return ax, False


def zijderveld_plot(specimen, coords="geo", ax=None):
    """Zijderveld (orthogonal vector) diagram of a ``Specimen``.

    The horizontal projection (north up, east right) is drawn with blue lines
    and markers, the vertical projection (up, east) in green; the NRM step is
    marked by a cross. Clicking a point shows its step label in the title.

    Args:
        specimen: ``Specimen``.
        coords: ``"specimen"``, ``"geo"`` or ``"tilt"``.
        ax: matplotlib axes to draw in; a new figure is created and shown when
            None.

    Returns:
        matplotlib Axes.
    """
    ax, owned = _axes(ax)
    N, E, Z = np.asarray(specimen.vectors(coords), dtype=float).T
    ax.plot(E, N, "b-", label="Horizontal")
    ax.plot(E, N, "bo", picker=5)
    ax.plot(E, -Z, "g-", label="Vertical")
    ax.plot(E, -Z, "go", picker=5)
    ax.plot(E[:1], N[:1], "b+", markersize=14)
    ax.plot(E[:1], -Z[:1], "g+", markersize=14)
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
    ax.set_title(f"{_title(specimen)}\n{_TITLES[coords]}", loc="left")
    ax.legend(title=f"Unit={unit:g} A/m", loc="lower left")
    labels = specimen.labels

    def onpick(event):
        if event.artist.axes is ax and len(event.ind):
            ax.set_title(f"{_title(specimen)}  step {labels[event.ind[0]]}", loc="left")
            ax.figure.canvas.draw_idle()

    ax.figure.canvas.mpl_connect("pick_event", onpick)
    if owned:
        plt.show()
    return ax


def demag_plot(specimen, ax=None):
    """Intensity decay plot, ``M / Mmax`` against treatment level.

    Args:
        specimen: ``Specimen``.
        ax: matplotlib axes to draw in; a new figure is created and shown when
            None.

    Returns:
        matplotlib Axes.
    """
    ax, owned = _axes(ax)
    M = specimen.intensity
    lv = specimen.levels
    ax.plot(lv[:1], M[:1] / M.max(), "k+", markersize=14)
    ax.plot(lv, M / M.max(), "ko-")
    unit = {"thermal": "°C", "af": "mT"}.get(specimen.demag, "")
    ax.set_xlabel(f"Level [{unit}]" if unit else "Level")
    ax.set_ylabel("M/Mmax")
    ax.set_title(f"{_title(specimen)} (Mmax = {M.max():g} A/m)")
    ax.set_ylim(0, 1.02)
    ax.yaxis.grid()
    if owned:
        plt.show()
    return ax


def stereo_plot(specimen, coords="geo", components=True, show=True, **kwargs):
    """Stereonet of the demagnetization path of a ``Specimen``.

    The steps are joined by arcs in measurement order, the NRM is marked by a
    cross; lower and upper hemisphere vectors are distinguished by filled and
    open markers. Stored components are added (lines as vectors, planes as
    great circles) unless ``components=False``.

    Args:
        specimen: ``Specimen``.
        coords: ``"specimen"``, ``"geo"`` or ``"tilt"``.
        components: plot the stored components. Default True.
        show: show the stereonet. Default True.
        **kwargs: passed to ``StereoNet``.

    Returns:
        StereoNet: the stereonet (with ``show=False`` ready for ``show()`` or
            ``savefig()``).
    """
    data = specimen.vectors(coords).normalized()
    s = StereoNet(title=f"{_title(specimen)}\n{_TITLES[coords]}", **kwargs)
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
    if show:
        s.show()
    return s
