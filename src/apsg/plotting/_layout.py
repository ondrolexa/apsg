"""Figure layout of apsg plots: legend placement and content-fitted figure size.

Plots are drawn on figures using constrained layout, which keeps the title and an
axes legend placed outside the axes visible. ``FigureLayout`` places the legend on
one side of the main axes, choosing the number of columns so the legend is not
longer than the plot, and, for figures apsg creates itself, grows the figure by the
legend and title and trims the empty space left by aspect-locked axes.
"""

import warnings

import numpy as np
from matplotlib.transforms import Bbox

# legend position -> (loc, bbox_to_anchor) of an axes legend just outside the axes
LEGEND_POSITIONS = {
    "right": ("center left", (1, 0.5)),
    "left": ("center right", (0, 0.5)),
    "top": ("lower center", (0.5, 1)),
    "bottom": ("upper center", (0.5, 0)),
}

_LEGEND_DEFAULTS = {"prop": {"size": 11}, "scatterpoints": 1, "numpoints": 1}

# gap between the legend and the axes decorations, in legend font size units
_LEGEND_GAP = 1.0


def check_legend_position(position):
    """Raise ``ValueError`` unless ``position`` is a valid legend position."""
    if position not in LEGEND_POSITIONS:
        raise ValueError(
            f"legend_position must be one of {', '.join(LEGEND_POSITIONS)}, "
            f"not {position!r}"
        )


def _root(fig):
    return fig.get_figure(root=True)


def _renderer(fig):
    return _root(fig)._get_renderer()


def _row_major(items, ncols):
    """Reorder ``items`` so a legend, which fills its columns one after another,
    reads them row by row."""
    sizes = [len(c) for c in np.array_split(np.arange(len(items)), ncols)]
    return [items[r * ncols + c] for c in range(ncols) for r in range(sizes[c])]


def _pads(fig):
    """Constrained layout pads of ``fig`` in inches (zero without a layout
    engine)."""
    engine = _root(fig).get_layout_engine()
    pads = engine.get() if engine is not None else {}
    return pads.get("w_pad", 0), pads.get("h_pad", 0)


def fit_figure(fig, ax, max_iter=3, tol=0.01):
    """Resize ``fig`` to the main axes ``ax`` (with its decorations and legend)
    and the title, removing the empty space constrained layout leaves around
    aspect-locked axes.

    The figure's own tight bbox would not do, as the axes is centered in its
    slot and the empty space can lie between the title and the axes.
    """
    w_pad, h_pad = _pads(fig)
    for _ in range(max_iter):
        fig.draw_without_rendering()
        renderer = fig._get_renderer()
        bbox = ax.get_tightbbox(renderer)
        w = bbox.width / fig.dpi + 2 * w_pad
        h = bbox.height / fig.dpi + 2 * h_pad
        suptitle = fig._suptitle
        if suptitle is not None:
            h += suptitle.get_window_extent(renderer).height / fig.dpi + 2 * h_pad
        cw, ch = fig.get_size_inches()
        if abs(w - cw) < tol and abs(h - ch) < tol:
            break
        fig.set_size_inches(w, h)


class FigureLayout:
    """Legend and title of the main axes of an apsg plot.

    Args:
        fig: Figure or SubFigure holding ``ax``
        ax: the main axes
        handles, labels: legend entries. No legend is drawn when empty.
        position (str): "right", "left", "top" or "bottom" side of the axes
        legend_kws (dict): extra keyword arguments of ``Axes.legend``. With ``loc``
            or ``bbox_to_anchor`` the legend is placed as given (no automatic
            placement), with ``ncols`` the number of columns is kept.
        title (str): figure title (``suptitle``). Default None
        title_kws (dict): extra keyword arguments of ``suptitle``
    """

    def __init__(
        self,
        fig,
        ax,
        handles,
        labels,
        position="right",
        legend_kws=None,
        title=None,
        title_kws=None,
    ):
        check_legend_position(position)
        self.fig = fig
        self.ax = ax
        self.handles = list(handles)
        self.labels = list(labels)
        self.position = position
        self.legend_kws = dict(legend_kws or {})
        self.title = title
        self.title_kws = dict(title_kws or {})
        self.legend = None

    @property
    def manual(self):
        """True when the user placed the legend via ``legend_kws``."""
        return "loc" in self.legend_kws or "bbox_to_anchor" in self.legend_kws

    @property
    def auto(self):
        """True when the legend is placed and fitted automatically."""
        return bool(self.handles) and not self.manual

    def apply(self, owned=False):
        """Draw legend and title.

        With ``owned=True`` (a figure apsg created) the figure is resized so the
        main axes keeps its size and the legend and title fit around it.
        Otherwise the figure size is kept and the legend is fitted to the axes.
        """
        if self.handles and self.manual:
            self.legend = self.ax.legend(
                self.handles, self.labels, **{**_LEGEND_DEFAULTS, **self.legend_kws}
            )
        if owned:
            if self.auto:
                # the legend is as long as the plot; the figure then grows by it
                _root(self.fig).draw_without_rendering()
                panel = self.ax.get_window_extent(_renderer(self.fig))
                vertical = self.position in ("left", "right")
                self._place_legend(panel.height if vertical else panel.width)
            self._add_title()
            self._grow_figure()
            fit_figure(_root(self.fig), self.ax)
        else:
            self._add_title()
            if self.auto:
                self.fit_legend()

    def fit_legend(self, max_iter=3):
        """Re-place the legend for the current figure size, until the number of
        columns no longer changes.

        The legend is limited by the space the figure has along the axes (not
        by the axes itself, which shrinks as the legend grows), so it stays
        inside the figure.
        """
        if not self.auto:
            return
        if self.legend is not None:
            self.legend.remove()
            self.legend = None
        previous = None
        # a legend too large for the figure makes constrained layout give up
        # until the legend is re-placed
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore", "constrained_layout not applied", UserWarning
            )
            for _ in range(max_iter):
                _root(self.fig).draw_without_rendering()
                ncols = self._place_legend(self._slot_length())
                if ncols == previous:
                    break
                previous = ncols

    def _slot_length(self):
        """Length available to a legend centered on the axes along its side:
        the figure height below the title (left or right) or the figure width
        (top or bottom), as laid out by the last draw."""
        renderer = _renderer(self.fig)
        w_pad, h_pad = _pads(self.fig)
        dpi = _root(self.fig).dpi
        fig_box = self.fig.bbox
        panel = self.ax.get_window_extent(renderer)
        if self.position in ("left", "right"):
            top = fig_box.y1
            suptitle = getattr(self.fig, "_suptitle", None)
            if suptitle is not None:
                top -= suptitle.get_window_extent(renderer).height + 2 * h_pad * dpi
            half = min(
                (panel.y0 + panel.y1) / 2 - fig_box.y0, top - (panel.y0 + panel.y1) / 2
            )
            return 2 * half - 2 * h_pad * dpi
        half = min(
            (panel.x0 + panel.x1) / 2 - fig_box.x0,
            fig_box.x1 - (panel.x0 + panel.x1) / 2,
        )
        return 2 * half - 2 * w_pad * dpi

    def connect_resize(self):
        """Re-fit the legend whenever the figure window is resized."""
        if not self.auto:
            return

        def on_resize(event):
            self.fit_legend()
            event.canvas.draw_idle()

        _root(self.fig).canvas.mpl_connect("resize_event", on_resize)

    def _add_title(self):
        if self.title is not None:
            self.fig.suptitle(self.title, **self.title_kws)

    def _place_legend(self, available):
        """Create the legend on its side of the axes, with as many rows (left or
        right) or columns (top or bottom) as fit in ``available`` pixels, next
        to the axes as laid out by the last draw. Returns the number of
        columns."""
        renderer = _renderer(self.fig)
        if self.legend is not None:
            self.legend.remove()
            self.legend = None
        vertical = self.position in ("left", "right")
        panel = self.ax.get_window_extent(renderer)
        # decorations of the axes (tick labels, axis labels, titles) beyond the
        # axes box on the legend side; the legend goes past them
        tight = self.ax.get_tightbbox(renderer)
        overflow = {
            "right": tight.x1 - panel.x1,
            "left": panel.x0 - tight.x0,
            "top": tight.y1 - panel.y1,
            "bottom": panel.y0 - tight.y0,
        }[self.position]
        loc, anchor = LEGEND_POSITIONS[self.position]
        kws = {**_LEGEND_DEFAULTS, **self.legend_kws}
        ncols = kws.pop("ncols", None)
        gap = kws.pop("borderaxespad", _LEGEND_GAP)
        n = len(self.handles)

        def make(ncols):
            handles, labels = self.handles, self.labels
            if not vertical:
                handles = _row_major(handles, ncols)
                labels = _row_major(labels, ncols)
            legend = self.ax.legend(
                handles, labels, loc=loc, bbox_to_anchor=anchor, ncols=ncols, **kws
            )
            legend.borderaxespad = gap + max(overflow, 0) / renderer.points_to_pixels(
                legend.prop.get_size_in_points()
            )
            return legend

        if ncols is None:
            candidates = range(1, n + 1) if vertical else range(n, 0, -1)
            for ncols in candidates:
                legend = make(ncols)
                bbox = legend.get_window_extent(renderer)
                if (bbox.height if vertical else bbox.width) <= available + 0.5:
                    break
                if ncols != candidates[-1]:
                    legend.remove()
        else:
            legend = make(ncols)
        self.legend = legend
        return ncols

    def _grow_figure(self):
        """Enlarge the figure by the legend and title, so the axes keeps the size
        it had without them (``fit_figure`` then trims any excess)."""
        root = _root(self.fig)
        renderer = _renderer(self.fig)
        w_pad, h_pad = _pads(root)
        dpi = root.dpi
        w, h = root.get_size_inches()
        if self.auto:
            bbox = self.legend.get_window_extent(renderer)
            gap = self.legend.borderaxespad * renderer.points_to_pixels(
                self.legend.prop.get_size_in_points()
            )
            if self.position in ("left", "right"):
                w += (bbox.width + gap) / dpi + 2 * w_pad
            else:
                h += (bbox.height + gap) / dpi + 2 * h_pad
        suptitle = getattr(self.fig, "_suptitle", None)
        if suptitle is not None:
            h += suptitle.get_window_extent(renderer).height / dpi + 2 * h_pad
        root.set_size_inches(w, h)


class CornerInsetLocator:
    """Axes locator of an inset in the bottom right corner of ``parent``.

    ``bounds`` are (x0, y0, width, height) in fractions of the parent axes box,
    shifted right and down by the parent's decorations (e.g. tick labels) beyond
    its box on those sides. The inset moves into the margin they already take,
    which constrained layout keeps inside the figure, so an inset left out of the
    layout gets more room without being cut off.
    """

    def __init__(self, parent, bounds):
        self.parent = parent
        self.bounds = bounds

    def __call__(self, ax, renderer):
        parent = self.parent
        box = parent.get_window_extent(renderer)
        dx = dy = 0
        if renderer is not None:
            # the legend is placed past the decorations, so it doesn't count
            artists = [
                a
                for a in parent.get_default_bbox_extra_artists()
                if a is not parent.legend_
            ]
            tight = parent.get_tightbbox(renderer, bbox_extra_artists=artists)
            dx = max(tight.x1 - box.x1, 0)
            dy = max(box.y0 - tight.y0, 0)
        x0, y0, w, h = self.bounds
        bbox = Bbox.from_bounds(
            box.x0 + x0 * box.width + dx,
            box.y0 + y0 * box.height - dy,
            w * box.width,
            h * box.height,
        )
        return bbox.transformed(ax.get_figure(root=False).transSubfigure.inverted())
