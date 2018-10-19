# -*- coding: utf-8 -*-


from __future__ import division, print_function

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import warnings
import matplotlib.cbook as mcb
import matplotlib.animation as animation

try:
    import mplstereonet
except ImportError:
    pass

from .core import (
    Vec3,
    Fol,
    Lin,
    Pair,
    Fault,
    Group,
    PairSet,
    FaultSet,
    Ortensor,
    StereoGrid,
)
from .helpers import cosd, sind, l2v, p2v, getldd, getfdd, l2xy, v2l, rodrigues
from .tensors import DefGrad, Stress


__all__ = ["StereoNet", "FabricPlot", "rose"]


# ignore matplotlib deprecation warnings
warnings.filterwarnings("ignore", category=mcb.mplDeprecation)


class StereoNet(object):

    """
    ``StereoNet`` class for Schmidt net plotting.

    A stereonet is a lower hemisphere graph on to which a variety of geological
    data can be plotted.

    If args are provided plot is immediately shown. If no args are provided,
    following methods and properties could be used for additional operations.

    Args:
        any plottable APSG class (most of data classes and tensors)

    Keyword Args:
        fol_plot: default method for ``Fol`` instances. ['plane' or 'pole']
                  Default 'plane'
        title: figure title. Default ''
        figsize: Figure size
        ncols: number of subplot columns. Default 1
        ticks: show ticks. Default True
        grid: show grid lines. Default False
        gridlw: grid lines width. Default 1
        grid_style: grid lines style. Default 'k:'
        cbpad: colorbar padding. Default 0.1

        Other keyword arguments are passed to matplotlib plot.

    Example:
        # Immediate plot
        >>> StereoNet(Fol(120, 30), Lin(36, 8))
        # StereoNet API
        >>> s = StereoNet()
        >>> g = Group.randn_lin(mean=Lin(40, 20))
        >>> s.contourf(g, 8, legend=True, sigma=2)
        >>> s.line(g, 'g.', label='My data')
        >>> s.show()
    """

    def __init__(self, *args, **kwargs):
        self.ticks = kwargs.pop("ticks", True)
        self.grid = kwargs.pop("grid", False)
        self.gridlw = kwargs.pop("gridlw", 1)
        self.ncols = kwargs.pop("ncols", 1)
        self.cbpad = kwargs.pop("cbpad", 0.1)
        self.grid_style = kwargs.pop("grid_style", "k:")
        self.fol_plot = kwargs.pop("fol_plot", "plane")
        figsize = kwargs.pop("figsize", None)
        title = kwargs.pop("title", "")
        self._lgd = None
        self.active = 0
        self.artists = []
        self.fig, self.ax = plt.subplots(ncols=self.ncols, figsize=figsize)
        self.fig.canvas.set_window_title("StereoNet - schmidt projection")
        # self.fig.set_size_inches(8 * self.ncols, 6)
        self._title = self.fig.suptitle(title)
        self._axtitle = self.ncols * [None]
        self.cla()
        # optionally immidiately plot passed objects
        if args:
            for arg in args:
                kwargs["label"] = repr(arg)
                if type(arg) in [Group, PairSet, FaultSet]:
                    typ = arg.type
                else:
                    typ = type(arg)
                if typ is Lin:
                    self.line(arg, **kwargs)
                elif typ is Fol:
                    getattr(self, self.fol_plot)(arg, **kwargs)
                elif typ is Vec3:
                    self.vector(arg.aslin, **kwargs)
                elif typ is Pair:
                    self.pair(arg, **kwargs)
                elif typ is Fault:
                    self.fault(arg, **kwargs)
                elif typ is StereoGrid:
                    kwargs.pop("label", None)
                    kwargs.pop("legend", None)
                    self.contourf(arg, legend=True, **kwargs)
                elif typ in [Ortensor, DefGrad, Stress]:
                    kwargs.pop("label", None)
                    self.tensor(arg, **kwargs)
                else:
                    raise TypeError("%s argument is not supported!" % typ)
            self.show()

    def close(self):
        plt.close(self.fig)

    @property
    def closed(self):
        return not plt.fignum_exists(self.fig.number)

    def draw(self):
        if self.closed:
            print(
                "The StereoNet figure have been closed. "
                "Use new() method or create new one."
            )
        else:
            for ax in self.fig.axes:
                h, lbls = ax.get_legend_handles_labels()
                if h:
                    self._lgd = ax.legend(
                        h,
                        lbls,
                        bbox_to_anchor=(1.12, 1),
                        prop={"size": 11},
                        loc=2,
                        borderaxespad=0,
                        scatterpoints=1,
                        numpoints=1,
                    )
                    plt.subplots_adjust(right=0.75)
                else:
                    plt.subplots_adjust(right=0.9)
            plt.draw()
            # plt.pause(0.001)

    def new(self):
        """Re-initialize StereoNet figure"""
        if self.closed:
            self.__init__()

    def cla(self):
        """Clear projection"""

        def lat(a, phi):
            return self._cone(l2v(a, 0), l2v(a, phi), limit=89.9999, res=91)

        def lon(a, theta):
            return self._cone(p2v(a, theta), l2v(a, theta), limit=80, res=91)

        for ax in self.fig.axes:
            ax.cla()
            ax.format_coord = self.format_coord
            ax.set_aspect("equal")
            ax.set_autoscale_on(False)
            ax.axis([-1.05, 1.05, -1.05, 1.05])
            ax.set_axis_off()

            # Projection circle
            ax.text(0, 1.02, "N", ha="center", va="baseline", fontsize=16)
            ax.add_artist(plt.Circle((0, 0), 1, color="w", zorder=0))
            ax.add_artist(plt.Circle((0, 0), 1, color="None", ec="k", zorder=3))

            if self.grid:
                # Main cross
                ax.plot(
                    [-1, 1, np.nan, 0, 0],
                    [0, 0, np.nan, -1, 1],
                    self.grid_style,
                    zorder=3,
                    lw=self.gridlw,
                )
                # Latitudes
                lat_n = np.array([lat(0, phi) for phi in range(10, 90, 10)])
                ax.plot(
                    lat_n[:, 0, :].T,
                    lat_n[:, 1, :].T,
                    self.grid_style,
                    zorder=3,
                    lw=self.gridlw,
                )
                lat_s = np.array([lat(180, phi) for phi in range(10, 90, 10)])
                ax.plot(
                    lat_s[:, 0, :].T,
                    lat_s[:, 1, :].T,
                    self.grid_style,
                    zorder=3,
                    lw=self.gridlw,
                )
                # Longitudes
                le = np.array([lon(90, theta) for theta in range(10, 90, 10)])
                ax.plot(
                    le[:, 0, :].T,
                    le[:, 1, :].T,
                    self.grid_style,
                    zorder=3,
                    lw=self.gridlw,
                )
                lw = np.array([lon(270, theta) for theta in range(10, 90, 10)])
                ax.plot(
                    lw[:, 0, :].T,
                    lw[:, 1, :].T,
                    self.grid_style,
                    zorder=3,
                    lw=self.gridlw,
                )

            # ticks
            if self.ticks:
                a = np.arange(0, 360, 30)
                tt = np.array([0.98, 1])
                x = np.outer(tt, sind(a))
                y = np.outer(tt, cosd(a))
                ax.plot(x, y, "k", zorder=4)
            # Middle cross
            ax.plot(
                [-0.02, 0.02, np.nan, 0, 0], [0, 0, np.nan, -0.02, 0.02], "k", zorder=4
            )
        self.draw()

    def getlin(self):
        """get Lin by mouse click"""
        x, y = plt.ginput(1)[0]
        return Lin(*getldd(x, y))

    def getfol(self):
        """get Fol by mouse click"""
        x, y = plt.ginput(1)[0]
        return Fol(*getfdd(x, y))

    def getlins(self):
        """get Group of Lin by mouse"""
        pts = plt.ginput(0, mouse_add=1, mouse_pop=2, mouse_stop=3)
        return Group([Lin(*getldd(x, y)) for x, y in pts])

    def getfols(self):
        """get Group of Fol by mouse"""
        pts = plt.ginput(0, mouse_add=1, mouse_pop=2, mouse_stop=3)
        return Group([Fol(*getfdd(x, y)) for x, y in pts])

    def _cone(self, axis, vector, limit=180, res=361, split=False):
        a = np.linspace(-limit, limit, res)
        x, y = l2xy(*v2l(rodrigues(axis, vector, a)))
        if split:
            dist = np.hypot(np.diff(x), np.diff(y))
            ix = np.nonzero(dist > 1)[0]
            x = np.insert(x, ix + 1, np.nan)
            y = np.insert(y, ix + 1, np.nan)
        return x, y

    def _arrow(self, pos_lin, dir_lin=None, sense=1):
        x, y = l2xy(*pos_lin.dd)
        if dir_lin is None:
            dx, dy = -x, -y
        else:
            ax, ay = l2xy(*dir_lin.dd)
            dx, dy = -ax, -ay
        mag = np.hypot(dx, dy)
        u, v = sense * dx / mag, sense * dy / mag
        return x, y, u, v

    def arrow(self, pos_lin, dir_lin=None, sense=1, **kwargs):
        animate = kwargs.pop("animate", False)
        x, y, u, v = self._arrow(pos_lin, dir_lin, sense=1)
        a = self.fig.axes[self.active].quiver(
            x, y, u, v, width=2, headwidth=5, zorder=6, pivot="mid", units="dots"
        )
        p = self.fig.axes[self.active].scatter(x, y, color="k", s=5, zorder=6)
        if animate:
            return tuple(a + p)

    def arc(self, f1, f2, *args, **kwargs):
        assert issubclass(type(f1), Vec3) and issubclass(
            type(f2), Vec3
        ), "Arguments mustr be subclass of Vec3"
        animate = kwargs.pop("animate", False)
        p = f1 ** f2
        a = f1.angle(f2)
        st = np.linspace(0, a, 2 + int(2 * a))
        rv = [f1.rotate(p, ang) for ang in st]
        lh = [vv.flip if vv.upper else vv for vv in rv]
        x, y = l2xy(*np.array([v.dd for v in lh]).T)
        h = self.fig.axes[self.active].plot(x, y, *args, **kwargs)
        if animate:
            self.artists.append(tuple(h))
        self.draw()

    def plane(self, obj, *args, **kwargs):
        assert obj.type is Fol, "Only Fol instance could be plotted as plane."
        if "zorder" not in kwargs:
            kwargs["zorder"] = 5
        animate = kwargs.pop("animate", False)
        if isinstance(obj, Group):
            x = []
            y = []
            for azi, inc in obj.dd.T:
                xx, yy = self._cone(
                    p2v(azi, inc), l2v(azi, inc), limit=89.9999, res=cosd(inc) * 179 + 2
                )
                x = np.hstack((x, xx, np.nan))
                y = np.hstack((y, yy, np.nan))
            x = x[:-1]
            y = y[:-1]
        else:
            azi, inc = obj.dd
            x, y = self._cone(
                p2v(azi, inc), l2v(azi, inc), limit=89.9999, res=cosd(inc) * 179 + 2
            )
        h = self.fig.axes[self.active].plot(x, y, *args, **kwargs)
        if animate:
            self.artists.append(tuple(h))
        self.draw()

    def line(self, obj, *args, **kwargs):
        assert obj.type is Lin, "Only Lin instance could be plotted as line."
        if "zorder" not in kwargs:
            kwargs["zorder"] = 5
        animate = kwargs.pop("animate", False)
        # ensure point plot
        if "ls" not in kwargs and "linestyle" not in kwargs:
            kwargs["linestyle"] = "none"
        if not args:
            if "marker" not in kwargs:
                kwargs["marker"] = "o"
        x, y = l2xy(*obj.dd)
        h = self.fig.axes[self.active].plot(x, y, *args, **kwargs)
        if animate:
            self.artists.append(tuple(h))
        self.draw()

    def vector(self, obj, *args, **kwargs):
        """ This mimics plotting on upper and lower hemisphere"""
        assert issubclass(
            obj.type, Vec3
        ), "Only Vec3-like instance could be plotted as line."
        if "zorder" not in kwargs:
            kwargs["zorder"] = 5
        animate = kwargs.pop("animate", False)
        # ensure point plot
        if "ls" not in kwargs and "linestyle" not in kwargs:
            kwargs["linestyle"] = "none"
        if not args:
            if "marker" not in kwargs:
                kwargs["marker"] = "o"
        if isinstance(obj, Group):
            uh = obj.upper
            if np.any(~uh):
                x, y = l2xy(*obj[~uh].dd)
                h1 = self.fig.axes[self.active].plot(x, y, *args, **kwargs)
                kwargs.pop("label", None)
                cc = h1[0].get_color()
            else:
                cc = None
            if np.any(uh):
                kwargs["fillstyle"] = "none"
                x, y = l2xy(*obj[uh].flip.dd)
                h2 = self.fig.axes[self.active].plot(x, y, *args, **kwargs)
                if cc is not None:
                    h2[0].set_color(cc)
            if animate:
                self.artists.append(tuple(h1 + h2))
        else:
            if obj.upper:
                kwargs["fillstyle"] = "none"
                x, y = l2xy(*obj.flip.dd)
                h = self.fig.axes[self.active].plot(x, y, *args, **kwargs)
            else:
                x, y = l2xy(*obj.dd)
                h = self.fig.axes[self.active].plot(x, y, *args, **kwargs)
            if animate:
                self.artists.append(tuple(h))
        self.draw()

    def pole(self, obj, *args, **kwargs):
        assert obj.type is Fol, "Only Fol instance could be plotted as poles."
        if "zorder" not in kwargs:
            kwargs["zorder"] = 5
        animate = kwargs.pop("animate", False)
        # ensure point plot
        if "ls" not in kwargs and "linestyle" not in kwargs:
            kwargs["linestyle"] = "none"
        if not args:
            if "marker" not in kwargs:
                kwargs["marker"] = "s"
        x, y = l2xy(*obj.aslin.dd)
        h = self.fig.axes[self.active].plot(x, y, *args, **kwargs)
        if animate:
            self.artists.append(tuple(h))
        self.draw()

    def cone(self, obj, alpha, *args, **kwargs):
        assert obj.type is Lin, "Only Lin instance could be used as cone axis."
        if "zorder" not in kwargs:
            kwargs["zorder"] = 5
        animate = kwargs.pop("animate", False)
        if isinstance(obj, Group):
            x = []
            y = []
            for azi, inc in obj.dd.T:
                xx, yy = self._cone(
                    l2v(azi, inc),
                    l2v(azi, inc - alpha),
                    limit=180,
                    res=sind(alpha) * 358 + 3,
                    split=True,
                )
                x = np.hstack((x, xx, np.nan))
                y = np.hstack((y, yy, np.nan))
            x = x[:-1]
            y = y[:-1]
        else:
            azi, inc = obj.dd
            x, y = self._cone(
                l2v(azi, inc),
                l2v(azi, inc - alpha),
                limit=180,
                res=sind(alpha) * 358 + 3,
                split=True,
            )
        h = self.fig.axes[self.active].plot(x, y, *args, **kwargs)
        if animate:
            self.artists.append(tuple(h))
        self.draw()

    def pair(self, obj, *arg, **kwargs):
        """Plot a fol-lin pair"""
        assert obj.type is Pair, "Only Pair instance could be used."
        animate = kwargs.pop("animate", False)
        h1 = self.plane(obj.fol, *arg, **kwargs)
        x, y = l2xy(*obj.lin.dd)
        h2 = self.fig.axes[self.active].scatter(x, y, color="k", s=5, zorder=6)
        if animate:
            self.artists.append(tuple(h1 + h2))
        self.draw()

    def fault(self, obj, *arg, **kwargs):
        """Plot a fault-and-striae plot - Angelier plot"""
        assert obj.type is Fault, "Only Fault instance could be used."
        animate = kwargs.get("animate", False)
        self.plane(obj.fol, *arg, **kwargs)
        x, y, u, v = self._arrow(obj.lin, sense=obj.sense)
        a = self.fig.axes[self.active].quiver(
            x, y, u, v, width=2, headwidth=5, zorder=6, pivot="mid", units="dots"
        )
        p = self.fig.axes[self.active].scatter(x, y, color="k", s=5, zorder=6)
        if animate:
            self.artists[-1] = self.artists[-1] + tuple(a + p)
        self.draw()

    def hoeppner(self, obj, *arg, **kwargs):
        """Plot a tangent lineation plot - Hoeppner plot"""
        assert obj.type is Fault, "Only Fault instance could be used."
        animate = kwargs.get("animate", False)
        self.pole(obj.fol, *arg, **kwargs)
        x, y, u, v = self._arrow(obj.fvec.aslin, dir_lin=obj.lin, sense=obj.sense)
        a = self.fig.axes[self.active].quiver(
            x, y, u, v, width=2, headwidth=5, zorder=6, pivot="mid", units="dots"
        )
        p = self.fig.axes[self.active].scatter(x, y, color="k", s=5, zorder=6)
        if animate:
            self.artists[-1] = self.artists[-1] + tuple(a + p)
        self.draw()

    def tensor(self, obj, *arg, **kwargs):
        """Plot tensor pricipal planes or directions"""
        getattr(self, self.fol_plot)(obj.eigenfols[0], label=obj.name + "-E1", **kwargs)
        getattr(self, self.fol_plot)(obj.eigenfols[1], label=obj.name + "-E2", **kwargs)
        getattr(self, self.fol_plot)(obj.eigenfols[2], label=obj.name + "-E3", **kwargs)

    def contourf(self, obj, *args, **kwargs):
        clines = kwargs.pop("clines", True)
        legend = kwargs.pop("legend", False)
        if "cmap" not in kwargs and "colors" not in kwargs:
            kwargs["cmap"] = "Greys"
        if "zorder" not in kwargs:
            kwargs["zorder"] = 1
        if isinstance(obj, StereoGrid):
            d = obj
        else:
            d = StereoGrid(obj, **kwargs)
            # clean kwargs from StereoGrid keywords
            for att in ["grid", "npoints", "sigma", "weighted", "method", "trim"]:
                kwargs.pop(att, None)
        if "levels" not in kwargs:
            if len(args) == 0:
                args = (6,)
            if isinstance(args[0], int):
                mn = d.values.min()
                mx = d.values.max()
                levels = np.linspace(mn, mx, args[0])
                levels[-1] += 1e-8
                args = (levels,)
        cs = self.fig.axes[self.active].tricontourf(d.triang, d.values, *args, **kwargs)
        if clines:
            self.fig.axes[self.active].tricontour(d.triang, d.values, *args, colors="k")
        if legend:
            if self.ncols > 1:
                ab = self.fig.axes[self.active].get_position().bounds
                cbaxes = self.fig.add_axes([ab[0] + self.cbpad * ab[2], 0.1, (1 - 2 * self.cbpad) * ab[2], 0.03])
                self.fig.colorbar(cs, cax=cbaxes, orientation='horizontal')
                # add horizontal, calculate positions (divide bars and spaces)
            else:
                ab = self.fig.axes[self.active].get_position().bounds
                cbaxes = self.fig.add_axes([0.1, ab[1] + self.cbpad * ab[3], 0.03, (1 - 2 * self.cbpad) * ab[3]])
                self.fig.colorbar(cs, cax=cbaxes)
        self.draw()

    def contour(self, obj, *args, **kwargs):
        legend = kwargs.pop("legend", False)
        if "cmap" not in kwargs and "colors" not in kwargs:
            kwargs["cmap"] = "Greys"
        if "zorder" not in kwargs:
            kwargs["zorder"] = 1
        if isinstance(obj, StereoGrid):
            d = obj
        else:
            d = StereoGrid(obj, **kwargs)
            # clean kwargs from StereoGrid keywords
            for att in ["grid", "npoints", "sigma", "weighted", "method", "trim"]:
                kwargs.pop(att, None)
        if "levels" not in kwargs:
            if len(args) == 0:
                args = (6,)
            if isinstance(args[0], int):
                mn = d.values.min()
                mx = d.values.max()
                levels = np.linspace(mn, mx, args[0])
                levels[-1] += 1e-8
                args = (levels,)
        cs = self.fig.axes[self.active].tricontour(d.triang, d.values, *args, **kwargs)
        if legend:
            if self.ncols > 1:
                ab = self.fig.axes[self.active].get_position().bounds
                cbaxes = self.fig.add_axes([ab[0] + self.cbpad * ab[2], 0.1, (1 - 2 * self.cbpad) * ab[2], 0.03])
                self.fig.colorbar(cs, cax=cbaxes, orientation='horizontal')
                # add horizontal, calculate positions (divide bars and spaces)
            else:
                ab = self.fig.axes[self.active].get_position().bounds
                cbaxes = self.fig.add_axes([0.1, ab[1] + self.cbpad * ab[3], 0.03, (1 - 2 * self.cbpad) * ab[3]])
                self.fig.colorbar(cs, cax=cbaxes)
        self.draw()

    # def _add_colorbar(self, cs):
    #     divider = make_axes_locatable(self.fig.axes[self.active])
    #     cax = divider.append_axes("left", size="5%", pad=0.5)
    #     plt.colorbar(cs, cax=cax)
    #     # modify tick labels
    #     # cb = plt.colorbar(cs, cax=cax)
    #     # lbl = [item.get_text()+'S' for item in cb.ax.get_yticklabels()]
    #     # lbl[lbl.index(next(l for l in lbl if l.startswith('0')))] = 'E'
    #     # cb.set_ticklabels(lbl)

    def axtitle(self, title):
        self._axtitle[self.active] = self.fig.axes[self.active].set_title(title)
        self._axtitle[self.active].set_y(-0.09)

    def show(self):
        plt.show()

    def animate(self, **kwargs):
        blit = kwargs.pop("blit", True)
        return animation.ArtistAnimation(self.fig, self.artists, blit=blit, **kwargs)

    def savefig(self, filename="apsg_stereonet.pdf", **kwargs):
        if self._lgd is None:
            self.fig.savefig(filename, **kwargs)
        else:
            bea = (self._lgd, self._title) + tuple(self._axtitle)
            self.fig.savefig(
                filename, bbox_extra_artists=bea, bbox_inches="tight", **kwargs
            )

    def format_coord(self, x, y):
        if np.hypot(x, y) > 1:
            return ""
        else:
            v = Vec3(*getldd(x, y))
            return repr(v.asfol) + " " + repr(v.aslin)


class FabricPlot(object):

    """
    Represents the triangular fabric plot (Vollmer, 1989).
    """

    def __init__(self, *args, **kwargs):
        self.fig = plt.figure()
        self.fig.canvas.set_window_title("Vollmer fabric plot")
        self.ticks = kwargs.get("ticks", True)
        self.grid = kwargs.get("grid", True)
        self.grid_style = kwargs.get("grid_style", "k:")
        self._lgd = None
        self.A = np.asarray(kwargs.get("A", [0, 3 ** 0.5 / 2]))
        self.B = np.asarray(kwargs.get("B", [1, 3 ** 0.5 / 2]))
        self.C = np.asarray(kwargs.get("C", [0.5, 0]))
        self.Ti = np.linalg.inv(np.array([self.A - self.C, self.B - self.C]).T)
        self.cla()
        # optionally immidiately plot passed objects
        if args:
            for arg in args:
                self.plot(arg)
            self.show()

    def close(self):
        plt.close(self.fig)

    @property
    def closed(self):
        return not plt.fignum_exists(self.fig.number)

    def draw(self):
        if self.closed:
            print(
                "The FabricPlot figure have been closed. "
                "Use new() method or create new one."
            )
        else:
            h, lbls = self.ax.get_legend_handles_labels()
            if h:
                self._lgd = self.ax.legend(
                    h,
                    lbls,
                    prop={"size": 11},
                    loc=4,
                    borderaxespad=0,
                    scatterpoints=1,
                    numpoints=1,
                )
            plt.draw()
            # plt.pause(0.001)

    def new(self):
        """
        Re-initialize ``StereoNet`` figure.
        """

        if self.closed:
            self.__init__()

    def cla(self):
        """
        Clear projection.
        """

        self.fig.clear()
        self.ax = self.fig.add_subplot(111)
        self.ax.format_coord = self.format_coord
        self.ax.set_aspect("equal")
        self.ax.set_autoscale_on(False)

        triangle = np.c_[self.A, self.B, self.C, self.A]
        n = 10
        tick_size = 0.2
        margin = 0.05

        self.ax.set_axis_off()

        plt.axis(
            [
                self.A[0] - margin,
                self.B[0] + margin,
                self.C[1] - margin,
                self.A[1] + margin,
            ]
        )

        # projection triangle
        bg = plt.Polygon([self.A, self.B, self.C], color="w", edgecolor=None)

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

        if self.grid:
            for l in np.arange(0.1, 1, 0.1):
                self.triplot([l, l], [0, 1 - l], [1 - l, 0], "k:")
                self.triplot([0, 1 - l], [l, l], [1 - l, 0], "k:")
                self.triplot([0, 1 - l], [1 - l, 0], [l, l], "k:")

        # ticks
        if self.ticks:
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

        self.ax.set_title("Fabric plot")

        self.draw()

    def triplot(self, a, b, c, *args, **kwargs):

        a = np.asarray(a)
        b = np.asarray(b)
        c = np.asarray(c)
        x = (a * self.A[0] + b * self.B[0] + c * self.C[0]) / (a + b + c)
        y = (a * self.A[1] + b * self.B[1] + c * self.C[1]) / (a + b + c)

        self.ax.plot(x, y, *args, **kwargs)

        self.draw()

    def plot(self, obj, *args, **kwargs):
        if type(obj) is Group:
            obj = obj.ortensor

        if type(obj) is not Ortensor:
            raise TypeError("%s argument is not supported!" % type(obj))

        # ensure point plot
        if "ls" not in kwargs and "linestyle" not in kwargs:
            kwargs["linestyle"] = "none"

        if not args:
            if "marker" not in kwargs:
                kwargs["marker"] = "o"
        if "label" not in kwargs:
            kwargs["label"] = obj.name

        self.triplot(obj.P, obj.G, obj.R, *args, **kwargs)

        self.draw()

    def show(self):
        plt.show()

    def savefig(self, filename="apsg_fabricplot.pdf", **kwargs):
        if self._lgd is None:
            self.ax.figure.savefig(filename, **kwargs)
        else:
            self.ax.figure.savefig(
                filename, bbox_extra_artists=(self._lgd,), bbox_inches="tight", **kwargs
            )

    def format_coord(self, x, y):
        a, b = self.Ti.dot(np.r_[x, y] - self.C)
        c = 1 - a - b
        if a < 0 or b < 0 or c < 0:
            return ""
        else:
            return "P:{:0.2f} G:{:0.2f} R:{:0.2f}".format(a, b, c)


class StereoNetJK(object):

    """
    API to Joe Kington mplstereonet.
    """

    def __init__(self, *args, **kwargs):
        _, self._ax = mplstereonet.subplots(*args, **kwargs)
        self._grid_state = False
        self._cax = None
        self._lgd = None

    def draw(self):
        h, lbls = self._ax.get_legend_handles_labels()
        if h:
            self._lgd = self._ax.legend(
                h,
                lbls,
                bbox_to_anchor=(1.12, 1),
                loc=2,
                borderaxespad=0.,
                numpoints=1,
                scatterpoints=1,
            )
            plt.subplots_adjust(right=0.75)
        else:
            plt.subplots_adjust(right=0.9)
        plt.draw()

    def cla(self):
        self._ax.cla()
        self._ax.grid(self._grid_state)
        self._cax = None
        self._lgd = None
        self.draw()

    def grid(self, state=True):
        self._ax.grid(state)
        self._grid_state = state
        self.draw()

    def plane(self, obj, *args, **kwargs):
        assert obj.type is Fol, "Only Fol instance could be plotted as plane."
        strike, dip = obj.rhr
        self._ax.plane(strike, dip, *args, **kwargs)
        self.draw()

    def pole(self, obj, *args, **kwargs):
        assert obj.type is Fol, "Only Fol instance could be plotted as pole."
        strike, dip = obj.rhr
        self._ax.pole(strike, dip, *args, **kwargs)
        self.draw()

    def rake(self, obj, rake_angle, *args, **kwargs):
        assert obj.type is Fol, "Only Fol instance could be used with rake."
        strike, dip = obj.rhr
        self._ax.rake(strike, dip, rake_angle, *args, **kwargs)
        self.draw()

    def line(self, obj, *args, **kwargs):
        assert obj.type is Lin, "Only Lin instance could be plotted as line."
        bearing, plunge = obj.dd
        self._ax.line(plunge, bearing, *args, **kwargs)
        self.draw()

    def arrow(self, obj, sense, *args, **kwargs):
        assert obj.type is Lin, "Only Lin instance could be plotted as quiver."
        bearing, plunge = obj.dd
        xx, yy = mplstereonet.line(plunge, bearing)
        xx1, yy1 = mplstereonet.line(plunge - 5, bearing)
        for x, y, x1, y1 in zip(xx, yy, xx1, yy1):
            self._ax.arrow(x, y, sense * (x1 - x), sense * (y1 - y))
        self.draw()

    def cone(self, obj, angle, segments=100, bidirectional=True, **kwargs):
        assert obj.type is Lin, "Only Lin instance could be used as cone axis."
        bearing, plunge = obj.dd
        self._ax.cone(
            plunge,
            bearing,
            angle,
            segments=segments,
            bidirectional=bidirectional,
            **kwargs
        )
        self.draw()

    def density_contour(self, group, *args, **kwargs):
        assert type(group) is Group, "Only group could be used for contouring."
        if group.type is Lin:
            bearings, plunges = group.dd
            kwargs["measurement"] = "lines"
            self._cax = self._ax.density_contour(plunges, bearings, *args, **kwargs)
            plt.draw()
        elif group.type is Fol:
            strikes, dips = group.rhr
            kwargs["measurement"] = "poles"
            self._cax = self._ax.density_contour(strikes, dips, *args, **kwargs)
            plt.draw()
        else:
            raise "Only Fol or Lin group is allowed."

    def density_contourf(self, group, *args, **kwargs):
        assert type(group) is Group, "Only group could be used for contouring."
        if group.type is Lin:
            bearings, plunges = group.dd
            kwargs["measurement"] = "lines"
            self._cax = self._ax.density_contourf(plunges, bearings, *args, **kwargs)
            plt.draw()
        elif group.type is Fol:
            strikes, dips = group.rhr
            kwargs["measurement"] = "poles"
            self._cax = self._ax.density_contourf(strikes, dips, *args, **kwargs)
            plt.draw()
        else:
            raise "Only Fol or Lin group is allowed."

    def colorbar(self):
        if self._cax is not None:
            cbaxes = self._ax.figure.add_axes([0.015, 0.2, 0.02, 0.6])
            plt.colorbar(self._cax, cax=cbaxes)

    def savefig(self, filename="stereonet.pdf", **kwargs):
        if self._lgd is None:
            self._ax.figure.savefig(filename, **kwargs)
        else:
            self._ax.figure.savefig(
                filename, bbox_extra_artists=(self._lgd,), bbox_inches="tight", **kwargs
            )

    def show(self):
        plt.show()


def rose(a, bins=13, **kwargs):
    """
    Plot the rose diagram.
    """

    if isinstance(a, Group):
        a, _ = a.dd

    fig = plt.figure()
    ax = fig.add_subplot(111, polar=True)
    ax.set_theta_direction(-1)
    ax.set_theta_zero_location("N")
    arad = a * np.pi / 180
    erad = np.linspace(0, 360, bins) * np.pi / 180
    plt.hist(arad, bins=erad, **kwargs)
