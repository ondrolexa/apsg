"""Common base of the figure-level plots of one data object."""

from typing import ClassVar

import matplotlib.pyplot as plt

from apsg.config import apsg_conf
from apsg.plotting._layout import FigureLayout, check_legend_position

__all__ = ("FigurePlot",)


class FigurePlot:
    """Base class of plots drawn from one data object.

    Same protocol as ``RosePlot`` and ``FabricPlot`` (``show``, ``savefig``,
    ``render2fig``). Subclasses name the ``apsg_conf`` section holding their
    defaults in ``config_key``, list options of their own with defaults in
    ``_options`` and implement ``_draw`` (and ``_default_title``). Keyword
    arguments override the configuration for the instance; unknown ones raise
    ``TypeError``.

    Args:
        data: the plotted object, see subclasses.

    Keyword Args:
        title (str): figure title; the default depends on the plot, an empty
            string means no title.
        figsize (tuple): size of the figure (``show``/``savefig``).
        title_kws (dict): extra arguments of ``suptitle``.
        legend_position (str): "right", "left", "top" or "bottom".
        legend_kws (dict): extra arguments of ``Axes.legend``.
    """

    config_key = None
    window_title = "Plot"
    default_figsize = None
    # options of the subclass with their defaults
    _options: ClassVar[dict] = {}

    def __init__(self, data, **kwargs):
        self.data = data
        self._kwargs = self._config().copy()
        self._opts = dict(self._options)
        for key in list(kwargs):
            if key in self._opts:
                self._opts[key] = kwargs.pop(key)
        unknown = kwargs.keys() - self._kwargs.keys()
        if unknown:
            raise TypeError(
                f"{type(self).__name__} got unexpected keyword argument(s) "
                f"{', '.join(sorted(unknown))}"
            )
        if self.default_figsize is not None and "figsize" not in kwargs:
            kwargs["figsize"] = self.default_figsize
        self._config().override(self._kwargs, kwargs)
        check_legend_position(self._kwargs["legend_position"])

    @classmethod
    def _config(cls):
        return getattr(apsg_conf, cls.config_key)

    # -- to be provided by subclasses ------------------------------------
    def _default_title(self):
        return None

    def _draw(self):
        """Draw into ``self.fig``; return the main axes (legend and title are
        placed around it) or None for multi-panel figures."""
        raise NotImplementedError

    # -- common protocol --------------------------------------------------
    def init_figure(self):
        self.fig = plt.figure(
            figsize=self._kwargs["figsize"],
            dpi=apsg_conf.dpi,
            facecolor=apsg_conf.facecolor,
            layout="constrained",
        )
        if hasattr(self.fig.canvas.manager, "set_window_title"):
            self.fig.canvas.manager.set_window_title(self.window_title)  # ty: ignore

    def _title_text(self):
        title = self._kwargs["title"]
        if title is None:
            title = self._default_title()
        return title or None

    def _render(self, owned=False):
        ax = self._draw()
        title = self._title_text()
        if ax is None:
            self._layout = None
            if title is not None:
                self.fig.suptitle(title, **self._kwargs["title_kws"])
            return
        handles, labels = ax.get_legend_handles_labels()
        self._layout = FigureLayout(
            self.fig,
            ax,
            handles,
            labels,
            position=self._kwargs["legend_position"],
            legend_kws={**self._legend_defaults(), **self._kwargs["legend_kws"]},
            title=title,
            title_kws=self._kwargs["title_kws"],
        )
        self._layout.apply(owned=owned)

    def _legend_defaults(self):
        return {}

    def render2fig(self, fig):
        """Plot to an already existing figure or subfigure (its size is kept)."""
        self.fig = fig
        self._render()

    def show(self):
        """Show the plot."""
        if hasattr(self, "fig"):
            plt.close(self.fig)
        self.init_figure()
        try:
            self._render(owned=True)
        except Exception:
            plt.close(self.fig)
            raise
        if self._layout is not None:
            self._layout.connect_resize()
        plt.show()

    def savefig(self, filename="plot.png", **kwargs):
        """Save the plot to a graphics file (kwargs go to ``Figure.savefig``)."""
        if hasattr(self, "fig"):
            plt.close(self.fig)
        self.init_figure()
        try:
            self._render(owned=True)
            self.fig.savefig(filename, **kwargs)
        finally:
            plt.close(self.fig)
