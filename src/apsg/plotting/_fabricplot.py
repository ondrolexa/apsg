import pickle

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon

from apsg.config import apsg_conf
from apsg.plotting._plot_artists import FabricPlotArtistFactory
from apsg.feature import feature_from_json

__all__ = ["VollmerPlot", "RamsayPlot", "FlinnPlot", "HsuPlot"]


class FabricPlot(object):

    """
    Metaclas for Fabric plots
    """

    def __init__(self, **kwargs):
        self._kwargs = apsg_conf["fabricplot_default_kwargs"].copy()
        self._kwargs.update((k, kwargs[k]) for k in self._kwargs.keys() & kwargs.keys())
        self._artists = []

    def clear(self):
        """Clear fabric plot"""
        self._artists = []

    def _plot_artists(self):
        for artist in self._artists:
            plot_method = getattr(self, artist.fabricplot_method)
            plot_method(*artist.args, **artist.kwargs)

    def to_json(self):
        """Return fabric plot as JSON dict"""
        artists = [artist.to_json() for artist in self._artists]
        return dict(kwargs=self._kwargs, artists=artists)

    @classmethod
    def from_json(cls, json_dict):
        """Create fabric plot from JSON dict"""
        s = cls(**json_dict["kwargs"])
        s._artists = [fabricartist_from_json(artist) for artist in json_dict["artists"]]
        return s

    def save(self, filename):
        """
        Save fabric plot to pickle file

        Args:
            filename (str): name of picke file
        """
        with open(filename, "wb") as f:
            pickle.dump(self.to_json(), f, pickle.HIGHEST_PROTOCOL)

    @classmethod
    def load(cls, filename):
        """
        Load fabric plot from pickle file

        Args:
            filename (str): name of picke file
        """
        with open(filename, "rb") as f:
            data = pickle.load(f)
        return cls.from_json(data)

    def init_figure(self):
        self.fig = plt.figure(
            0,
            figsize=apsg_conf["figsize"],
            dpi=apsg_conf["dpi"],
            facecolor=apsg_conf["facecolor"],
        )
        if hasattr(self.fig.canvas.manager, "set_window_title"):
            self.fig.canvas.manager.set_window_title(self.window_title)

    def _render(self):
        self._draw_layout()
        self._plot_artists()
        h, lbls = self.ax.get_legend_handles_labels()
        if h:
            self._lgd = self.ax.legend(
                h,
                lbls,
                prop={"size": 11},
                borderaxespad=0,
                loc="center left",
                bbox_to_anchor=(1.1, 0.5),
                scatterpoints=1,
                numpoints=1,
            )
        if self._kwargs["title"] is not None:
            self.fig.suptitle(self._kwargs["title"], **self._kwargs["title_kws"])
        if self._kwargs["tight_layout"]:
            self.fig.tight_layout()

    def render2fig(self, fig):
        """
        Plot stereonet to already existing figure or subfigure

        Args:
            fig (Figure): A mtplotlib Figure artist
        """
        self.fig = fig
        self._render()

    def show(self):
        """Show deformation plot"""
        plt.close(0)  # close previously rendered figure
        self.init_figure()
        self._render()
        plt.show()

    def savefig(self, filename="fabricplot.png", **kwargs):
        """
        Save fabric plot figure to graphics file

        Keyword Args:
            filename (str): filename

        All others kwargs are passed to matplotlib `Figure.savefig`
        """
        plt.close(0)  # close previously rendered figure
        self.init_figure()
        self._render()
        self.fig.savefig(filename, **kwargs)
        plt.close(0)


class VollmerPlot(FabricPlot):

    """
    Represents the triangular fabric plot (Vollmer, 1989).

    Keyword Args:
        title (str): figure title. Default None.
        title_kws (dict): dictionary of keyword arguments passed to matplotlib suptitle
            method.
        ticks (bool): Show ticks. Default True
        n_ticks (int): Number of ticks. Default 10
        tick_size (float): Size of ticks. Default 0.2
        margin (float): Size of margin. Default 0.05
        grid (bool): Show grid. Default is True
        grid_color (str): Matplotlib color of the grid. Default "k"
        grid_style (str): Matplotlib style of the grid. Default ":"

    Examples:
        >>> l = linset.random_fisher(position=lin(120, 40))
        >>> ot = l.ortensor()
        >>> s = VollmerPlot(title="Point distribution")
        >>> s.point(ot)
        >>> s.show()
    """

    def __init__(self, *args, **kwargs):
        self.A = np.array([0, 3**0.5 / 2])
        self.B = np.array([1, 3**0.5 / 2])
        self.C = np.array([0.5, 0])
        self.Ti = np.linalg.inv(np.array([self.A - self.C, self.B - self.C]).T)
        self.window_title = "Vollmer fabric plot"
        super().__init__(**kwargs)

    def _draw_layout(self):
        self.ax = self.fig.add_subplot(111)
        self.ax.format_coord = self.format_coord
        self.ax.set_aspect("equal")
        self.ax.set_autoscale_on(False)

        triangle = np.c_[self.A, self.B, self.C, self.A]
        n = self._kwargs["n_ticks"]
        tick_size = self._kwargs["tick_size"]
        margin = self._kwargs["margin"]

        self.ax.set_axis_off()
        self.ax.set_xlim(self.A[0] - margin, self.B[0] + margin)
        self.ax.set_ylim(self.C[1] - margin, self.A[1] + margin)

        # projection triangle
        bg = Polygon([self.A, self.B, self.C], color="w", edgecolor=None)
        self.ax.add_patch(bg)
        self.ax.plot(triangle[0], triangle[1], "k", lw=2)
        self.ax.text(
            self.A[0] - 0.02, self.A[1], "P", ha="right", va="bottom", fontsize=14
        )
        self.ax.text(
            self.B[0] + 0.02, self.B[1], "G", ha="left", va="bottom", fontsize=14
        )
        self.ax.text(
            self.C[0], self.C[1] - 0.02, "R", ha="center", va="top", fontsize=14
        )

        if self._kwargs["grid"]:
            for ln in np.arange(0.1, 1, 0.1):
                self._triplot(
                    [ln, ln],
                    [0, 1 - ln],
                    [1 - ln, 0],
                    color=self._kwargs["grid_color"],
                    ls=self._kwargs["grid_style"],
                    lw=1,
                )
                self._triplot(
                    [0, 1 - ln],
                    [ln, ln],
                    [1 - ln, 0],
                    color=self._kwargs["grid_color"],
                    ls=self._kwargs["grid_style"],
                    lw=1,
                )
                self._triplot(
                    [0, 1 - ln],
                    [1 - ln, 0],
                    [ln, ln],
                    color=self._kwargs["grid_color"],
                    ls=self._kwargs["grid_style"],
                    lw=1,
                )

        # ticks
        if self._kwargs["ticks"]:
            r = np.linspace(0, 1, n + 1)
            tick = tick_size * (self.B - self.C) / n
            x = self.A[0] * (1 - r) + self.B[0] * r
            x = np.vstack((x, x + tick[0]))
            y = self.A[1] * (1 - r) + self.B[1] * r
            y = np.vstack((y, y + tick[1]))
            self.ax.plot(x, y, "k", lw=1)
            tick = tick_size * (self.C - self.A) / n
            x = self.B[0] * (1 - r) + self.C[0] * r
            x = np.vstack((x, x + tick[0]))
            y = self.B[1] * (1 - r) + self.C[1] * r
            y = np.vstack((y, y + tick[1]))
            self.ax.plot(x, y, "k", lw=1)
            tick = tick_size * (self.A - self.B) / n
            x = self.A[0] * (1 - r) + self.C[0] * r
            x = np.vstack((x, x + tick[0]))
            y = self.A[1] * (1 - r) + self.C[1] * r
            y = np.vstack((y, y + tick[1]))
            self.ax.plot(x, y, "k", lw=1)

    ########################################
    # PLOTTING METHODS                     #
    ########################################

    def point(self, *args, **kwargs):
        """Plot ellipsoid as point"""
        try:
            artist = FabricPlotArtistFactory.create_point(*args, **kwargs)
            self._artists.append(artist)
        except TypeError as err:
            print(err)

    def path(self, *args, **kwargs):
        """Plot EllipsoidSet as path"""
        try:
            artist = FabricPlotArtistFactory.create_path(*args, **kwargs)
            self._artists.append(artist)
        except TypeError as err:
            print(err)

    ########################################
    # PLOTTING ROUTINES                    #
    ########################################

    def _triplot(self, a, b, c, **kwargs):
        a = np.atleast_1d(a)
        b = np.atleast_1d(b)
        c = np.atleast_1d(c)
        x = (a * self.A[0] + b * self.B[0] + c * self.C[0]) / (a + b + c)
        y = (a * self.A[1] + b * self.B[1] + c * self.C[1]) / (a + b + c)

        self.ax.plot(x, y, **kwargs)

    def _point(self, *args, **kwargs):
        P = [arg.P for arg in args]
        G = [arg.G for arg in args]
        R = [arg.R for arg in args]
        self._triplot(P, G, R, **kwargs)

    def _path(self, *args, **kwargs):
        for arg in args:
            self._triplot(arg.P, arg.G, arg.R, **kwargs)

    def format_coord(self, x, y):
        a, b = self.Ti.dot(np.r_[x, y] - self.C)
        c = 1 - a - b
        if a < 0 or b < 0 or c < 0:
            return ""
        else:
            return "P:{:0.2f} G:{:0.2f} R:{:0.2f}".format(a, b, c)


class RamsayPlot(FabricPlot):

    """
    Represents the Ramsay deformation plot.

    Keyword Args:
        title (str): figure title. Default None.
        title_kws (dict): dictionary of keyword arguments passed to matplotlib suptitle
            method.
        ticks (bool): Show ticks. Default True
        n_ticks (int): Number of ticks. Default 10
        tick_size (float): Size of ticks. Default 0.2
        margin (float): Size of margin. Default 0.05
        grid (bool): Show grid. Default is True
        grid_color (str): Matplotlib color of the grid. Default "k"
        grid_style (str): Matplotlib style of the grid. Default ":"

    Examples:
        >>> l = linset.random_fisher(position=lin(120, 40))
        >>> ot = l.ortensor()
        >>> s = RamsayPlot(title="Point distribution")
        >>> s.point(ot)
        >>> s.show()
    """

    def __init__(self, *args, **kwargs):
        self.mx = kwargs.pop("axes_max", "auto")
        super().__init__(**kwargs)
        self.window_title = "Ramsay deformation plot"

    def _draw_layout(self):
        self.ax = self.fig.add_subplot(111)
        self.ax.format_coord = self.format_coord
        self.ax.set_aspect("equal")
        self.ax.set_autoscale_on(True)
        self.ax.spines["top"].set_color("none")
        self.ax.spines["right"].set_color("none")
        self.ax.set_xlabel(r"$\varepsilon_2-\varepsilon_3$")
        self.ax.set_ylabel(r"$\varepsilon_1-\varepsilon_2$")
        if self._kwargs["grid"]:
            self.ax.grid(True)

    def _render(self):
        super()._render()
        if self.mx == "auto":
            mx = max(self.ax.get_xlim()[1], self.ax.get_ylim()[1])
        else:
            mx = self.mx
        self.ax.set_xlim(0, mx)
        self.ax.set_ylim(0, mx)
        self.ax.plot([0, mx], [0, mx], "k", lw=0.5)
        box = self.ax.get_position()
        self.ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])

    ########################################
    # PLOTTING METHODS                     #
    ########################################

    def point(self, *args, **kwargs):
        """Plot ellipsoid as point"""
        try:
            artist = FabricPlotArtistFactory.create_point(*args, **kwargs)
            self._artists.append(artist)
        except TypeError as err:
            print(err)

    def path(self, *args, **kwargs):
        """Plot EllipsoidSet as path"""
        try:
            artist = FabricPlotArtistFactory.create_path(*args, **kwargs)
            self._artists.append(artist)
        except TypeError as err:
            print(err)

    ########################################
    # PLOTTING ROUTINES                    #
    ########################################

    def _point(self, *args, **kwargs):
        e23 = [arg.e23 for arg in args]
        e12 = [arg.e12 for arg in args]

        self.ax.plot(e23, e12, **kwargs)

    def _path(self, *args, **kwargs):
        for arg in args:
            self.ax.plot(arg.e23, arg.e12, **kwargs)

    def format_coord(self, x, y):
        k = y / x if x > 0 else 0
        d = x**2 + y**2
        return "k:{:0.2f} d:{:0.2f}".format(k, d)


class FlinnPlot(FabricPlot):

    """
    Represents the Ramsay deformation plot.

    Keyword Args:
        title (str): figure title. Default None.
        title_kws (dict): dictionary of keyword arguments passed to matplotlib suptitle
            method.
        ticks (bool): Show ticks. Default True
        n_ticks (int): Number of ticks. Default 10
        tick_size (float): Size of ticks. Default 0.2
        margin (float): Size of margin. Default 0.05
        grid (bool): Show grid. Default is True
        grid_color (str): Matplotlib color of the grid. Default "k"
        grid_style (str): Matplotlib style of the grid. Default ":"

    Examples:
        >>> l = linset.random_fisher(position=lin(120, 40))
        >>> ot = l.ortensor()
        >>> s = FlinnPlot(title="Point distribution")
        >>> s.point(ot)
        >>> s.show()
    """

    def __init__(self, *args, **kwargs):
        self.mx = kwargs.pop("axes_max", "auto")
        super().__init__(**kwargs)
        self.window_title = "Flinn deformation plot"

    def _draw_layout(self):
        self.ax = self.fig.add_subplot(111)
        self.ax.format_coord = self.format_coord
        self.ax.set_aspect("equal")
        self.ax.set_autoscale_on(True)
        self.ax.spines["top"].set_color("none")
        self.ax.spines["right"].set_color("none")
        self.ax.set_xlabel(r"$R_{YZ}$")
        self.ax.set_ylabel(r"$R_{XY}$")
        if self._kwargs["grid"]:
            self.ax.grid(True)

    def _render(self):
        super()._render()
        if self.mx == "auto":
            mx = max(self.ax.get_xlim()[1], self.ax.get_ylim()[1])
        else:
            mx = self.mx
        self.ax.set_xlim(1, mx)
        self.ax.set_ylim(1, mx)
        self.ax.plot([1, mx], [1, mx], "k", lw=0.5)
        box = self.ax.get_position()
        self.ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])

    ########################################
    # PLOTTING METHODS                     #
    ########################################

    def point(self, *args, **kwargs):
        """Plot Ellipsoid as point"""
        try:
            artist = FabricPlotArtistFactory.create_point(*args, **kwargs)
            self._artists.append(artist)
        except TypeError as err:
            print(err)

    def path(self, *args, **kwargs):
        """Plot EllipsoidSet as path"""
        try:
            artist = FabricPlotArtistFactory.create_path(*args, **kwargs)
            self._artists.append(artist)
        except TypeError as err:
            print(err)

    ########################################
    # PLOTTING ROUTINES                    #
    ########################################

    def _point(self, *args, **kwargs):
        Ryz = [arg.Ryz for arg in args]
        Rxy = [arg.Rxy for arg in args]

        self.ax.plot(Ryz, Rxy, **kwargs)

    def _path(self, *args, **kwargs):
        for arg in args:
            self.ax.plot(arg.Ryz, arg.Rxy, **kwargs)

    def format_coord(self, x, y):
        K = (y - 1) / (x - 1) if x > 1 else 0
        D = np.sqrt((x - 1) ** 2 + (y - 1) ** 2)
        return "K:{:0.2f} D:{:0.2f}".format(K, D)


class HsuPlot(FabricPlot):

    """
    Represents the Hsu fabric plot.

    Keyword Args:
        title (str): figure title. Default None.
        title_kws (dict): dictionary of keyword arguments passed to matplotlib suptitle
            method.
        ticks (bool): Show ticks. Default True
        n_ticks (int): Number of ticks. Default 10
        tick_size (float): Size of ticks. Default 0.2
        margin (float): Size of margin. Default 0.05
        grid (bool): Show grid. Default is True
        grid_color (str): Matplotlib color of the grid. Default "k"
        grid_style (str): Matplotlib style of the grid. Default ":"

    Examples:
        >>> l = linset.random_fisher(position=lin(120, 40))
        >>> ot = l.ortensor()
        >>> s = HsuPlot(title="Point distribution")
        >>> s.point(ot)
        >>> s.show()
    """

    def __init__(self, *args, **kwargs):
        self.mx = kwargs.pop("axes_max", "auto")
        super().__init__(**kwargs)
        self.window_title = "Hsu deformation plot"

    def _draw_layout(self):
        self.ax = self.fig.add_subplot(111, polar=True)
        self.ax.format_coord = self.format_coord
        self.ax.set_theta_zero_location("N")
        self.ax.set_theta_direction(-1)
        self.ax.set_thetamin(-30)
        self.ax.set_thetamax(30)
        self.ax.set_xticks([-np.pi / 6, -np.pi / 12, 0, np.pi / 12, np.pi / 6])
        self.ax.set_xticklabels([-1, -0.5, 0, 0.5, 1])
        self.ax.set_title(r"$\nu$")
        self.ax.set_ylabel(r"$\bar{\varepsilon}_s$")
        if self._kwargs["grid"]:
            self.ax.grid(True)

    ########################################
    # PLOTTING METHODS                     #
    ########################################

    def point(self, *args, **kwargs):
        """Plot Ellipsoid as point"""
        try:
            artist = FabricPlotArtistFactory.create_point(*args, **kwargs)
            self._artists.append(artist)
        except TypeError as err:
            print(err)

    def path(self, *args, **kwargs):
        """Plot EllipsoidSet as path"""
        try:
            artist = FabricPlotArtistFactory.create_path(*args, **kwargs)
            self._artists.append(artist)
        except TypeError as err:
            print(err)

    ########################################
    # PLOTTING ROUTINES                    #
    ########################################

    def _point(self, *args, **kwargs):
        lode = [arg.lode * np.pi / 6 for arg in args]
        eoct = [arg.eoct for arg in args]

        self.ax.plot(lode, eoct, **kwargs)

    def _path(self, *args, **kwargs):
        for arg in args:
            self.ax.plot(arg.lode * np.pi / 6, arg.eoct, **kwargs)

    def format_coord(self, x, y):
        return f"lode:{x * 6 / np.pi:0.2f} eoct:{y:0.2f}"


def fabricartist_from_json(obj_json):
    args = tuple([feature_from_json(arg_json) for arg_json in obj_json["args"]])
    return getattr(FabricPlotArtistFactory, obj_json["factory"])(
        *args, **obj_json["kwargs"]
    )
