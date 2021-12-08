# -*- coding: utf-8 -*-

import sys
import warnings
import pickle

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cbook as mcb
from matplotlib.patches import Circle
from scipy.stats import vonmises

from apsg.config import apsg_conf
from apsg.math._vector import Vector3
from apsg.feature._geodata import Lineation, Foliation, Pair, Fault
from apsg.feature._container import (
    FeatureSet,
    Vector3Set,
    LineationSet,
    FoliationSet,
    PairSet,
    FaultSet,
)
from apsg.plotting._stereogrid import StereoGrid
from apsg.feature._tensor import OrientationTensor3
from apsg.plotting._projection import EqualAreaProj, EqualAngleProj
from apsg.plotting._stereonet_artist import ArtistFactory

__all__ = ["StereoNet", "VollmerPlot", "RamsayPlot", "FlinnPlot", "HsuPlot", "RosePlot"]


# Ignore `matplotlib`s deprecation warnings.
warnings.filterwarnings("ignore", category=mcb.mplDeprecation)


class StereoNet:
    """
    Blasklas

    Args:
        hemisphere: which hemisphere Default 'lower'
        rotate_data: whether data should be rotated with overlay Default False
        grid_position: orientation of overlay given as ``Pair`` Default pair(0, 0, 0, 0)
        clip_pole: Default 20)
        gridstep: grid step Default 15
        resolution: number of grid lines points Default 361
        n: number of contouring grid points Default 2000
        grid_type: type of contouring grid 'gss' or 'sfs'. Default 'gss'
    """

    def __init__(self, **kwargs):
        self._kwargs = apsg_conf["stereonet_default_kwargs"].copy()
        self._kwargs.update((k, kwargs[k]) for k in self._kwargs.keys() & kwargs.keys())
        self._kwargs["title"] = kwargs.get("title", None)
        if self._kwargs["kind"].lower() in ["equal-area", "schmidt", "earea"]:
            self.proj = EqualAreaProj(**self._kwargs)
        elif self._kwargs["kind"].lower() in ["equal-angle", "wulff", "eangle"]:
            self.proj = EqualAngleProj(**self._kwargs)
        else:
            raise TypeError("Only 'Equal-area' and 'Equal-angle' implemented")
        self.angles_gc = np.linspace(
            -90 + 1e-7, 90 - 1e-7, int(self.proj.overlay_resolution / 2)
        )
        self.angles_sc = np.linspace(
            -180 + 1e-7, 180 - 1e-7, self.proj.overlay_resolution
        )
        self.grid = StereoGrid(**self._kwargs)
        self._artists = []

    def clear(self):
        self._artists = []

    def __draw_net(self):
        self.fig, self.ax = plt.subplots(figsize=apsg_conf["figsize"])
        self.ax.set_aspect(1)
        self.ax.set_axis_off()

        # overlay
        if self._kwargs["overlay"]:
            ov = self.proj.get_grid_overlay()
            for dip, d in ov["lat_e"].items():
                self.ax.plot(d["x"], d["y"], "k:", lw=1)
            for dip, d in ov["lat_w"].items():
                self.ax.plot(d["x"], d["y"], "k:", lw=1)
            for dip, d in ov["lon_n"].items():
                self.ax.plot(d["x"], d["y"], "k:", lw=1)
            for dip, d in ov["lon_s"].items():
                self.ax.plot(d["x"], d["y"], "k:", lw=1)
            if ov["main_xz"]:
                self.ax.plot(ov["main_xz"]["x"], ov["main_xz"]["y"], "k:", lw=1)
            if ov["main_yz"]:
                self.ax.plot(ov["main_yz"]["x"], ov["main_yz"]["y"], "k:", lw=1)
            if ov["main_xy"]:
                self.ax.plot(ov["main_xy"]["x"], ov["main_xy"]["y"], "k:", lw=1)
            if ov["polehole_n"]:
                self.ax.plot(ov["polehole_n"]["x"], ov["polehole_n"]["y"], "k", lw=1)
            if ov["polehole_s"]:
                self.ax.plot(ov["polehole_s"]["x"], ov["polehole_s"]["y"], "k", lw=1)
            if ov["main_x"]:
                self.ax.plot(ov["main_x"]["x"], ov["main_x"]["y"], "k", lw=2)
            if ov["main_y"]:
                self.ax.plot(ov["main_y"]["x"], ov["main_y"]["y"], "k", lw=2)
            if ov["main_z"]:
                self.ax.plot(ov["main_z"]["x"], ov["main_z"]["y"], "k", lw=2)

        # Projection circle frame
        theta = np.linspace(0, 2 * np.pi, 200)
        self.ax.plot(np.cos(theta), np.sin(theta), "k", lw=2)
        # Minor ticks
        if self._kwargs["minor_ticks"] is not None:
            ticks = np.array([1, 1.02])
            theta = np.arange(0, 2 * np.pi, np.radians(self._kwargs["minor_ticks"]))
            self.ax.plot(
                np.outer(ticks, np.cos(theta)),
                np.outer(ticks, np.sin(theta)),
                "k",
                lw=1,
            )
        # Major ticks
        if self._kwargs["major_ticks"] is not None:
            ticks = np.array([1, 1.03])
            theta = np.arange(0, 2 * np.pi, np.radians(self._kwargs["major_ticks"]))
            self.ax.plot(
                np.outer(ticks, np.cos(theta)),
                np.outer(ticks, np.sin(theta)),
                "k",
                lw=1.5,
            )
        # add clipping circle
        self.primitive = Circle(
            (0, 0),
            radius=1,
            edgecolor="black",
            fill=False,
            clip_box="None",
            label="_nolegend_",
        )
        self.ax.add_patch(self.primitive)

    def __plot_artists(self):
        for artist in self._artists:
            plot_method = getattr(self, artist.stereonet_method)
            plot_method(*artist.args, **artist.kwargs)

    def to_json(self):
        data = {}
        artists = []
        for artist in self._artists:
            for obj in artist.args:
                obj_id = id(obj)
                if obj_id not in data:
                    data[obj_id] = obj.to_json()
            artist_dict = dict(
                factory=artist.factory,
                stereonet_method=artist.stereonet_method,
                args=tuple([id(obj) for obj in artist.args]),
                kwargs=artist.kwargs.copy(),
            )
            artists.append(artist_dict)
        return dict(kwargs=self._kwargs, data=data, artists=artists,)

    @classmethod
    def from_json(cls, json_dict):
        def parse_json_data(obj_json):
            dtype_cls = getattr(sys.modules[__name__], obj_json["datatype"])
            args = []
            for arg in obj_json["args"]:
                if isinstance(arg, dict):
                    args.append([parse_json_data(jd) for jd in arg["collection"]])
                else:
                    args.append(arg)
            kwargs = obj_json.get("kwargs", {})
            return dtype_cls(*args, **kwargs)

        # parse
        s = cls(**json_dict["kwargs"])
        data = {}
        for obj_id, obj_json in json_dict["data"].items():
            data[obj_id] = parse_json_data(obj_json)
        s._artists = []
        for artist in json_dict["artists"]:
            args = tuple([data[obj_id] for obj_id in artist["args"]])
            s._artists.append(
                getattr(ArtistFactory, artist["factory"])(*args, **artist["kwargs"])
            )
        return s

    def save(self, filename):
        with open(filename, "wb") as f:
            pickle.dump(self.to_json(), f, pickle.HIGHEST_PROTOCOL)

    @classmethod
    def load(cls, filename):
        with open(filename, "rb") as f:
            data = pickle.load(f)
        return cls.from_json(data)

    def render(self):
        self.__draw_net()
        self.__plot_artists()
        self.ax.set_xlim(-1.05, 1.05)
        self.ax.set_ylim(-1.05, 1.05)
        h, labels = self.ax.get_legend_handles_labels()
        if h:
            self.ax.legend(
                h,
                labels,
                bbox_to_anchor=(1.05, 1),
                prop={"size": 11},
                loc="upper left",
                borderaxespad=0,
                scatterpoints=1,
                numpoints=1,
            )
        if self._kwargs["title"] is not None:
            self.fig.suptitle(self._kwargs["title"])
        self.fig.tight_layout()

    def show(self):
        self.render()
        plt.show()

    def savefig(self, filename="stereonet.png", **kwargs):
        self.render()
        self.fig.savefig(filename, **kwargs)
        plt.close()
        delattr(self, "ax")
        delattr(self, "fig")

    ########################################
    # PLOTTING METHODS                     #
    ########################################

    def line(self, *args, **kwargs):
        """Plot linear feature(s) as point(s)"""
        try:
            artist = ArtistFactory.create_point(*args, **kwargs)
            self._artists.append(artist)
        except TypeError as err:
            print(err)

    def pole(self, *args, **kwargs):
        """Plot pole of planar feature(s) as point(s)"""
        try:
            artist = ArtistFactory.create_pole(*args, **kwargs)
            self._artists.append(artist)
        except TypeError as err:
            print(err)

    def vector(self, *args, **kwargs):
        """Plot vector feature(s) as point(s), filled on lower and open on upper hemisphere."""
        try:
            artist = ArtistFactory.create_point(*args, **kwargs)
            self._artists.append(artist)
        except TypeError as err:
            print(err)

    def scatter(self, *args, **kwargs):
        """Plot vector-like feature(s) as point(s) using scatter"""
        try:
            artist = ArtistFactory.create_scatter(*args, **kwargs)
            self._artists.append(artist)
        except TypeError as err:
            print(err)

    def great_circle(self, *args, **kwargs):
        """Plot planar feature(s) as great circle(s)"""
        try:
            artist = ArtistFactory.create_great_circle(*args, **kwargs)
            self._artists.append(artist)
        except TypeError as err:
            print(err)

    def cone(self, *args, **kwargs):
        """Plot small circle(s) with given angle(s)"""
        try:
            artist = ArtistFactory.create_cone(*args, **kwargs)
            self._artists.append(artist)
        except TypeError as err:
            print(err)

    def pair(self, *args, **kwargs):
        """Plot pair feature(s) as great circle and point"""
        try:
            artist = ArtistFactory.create_pair(*args, **kwargs)
            self._artists.append(artist)
        except TypeError as err:
            print(err)

    def fault(self, *args, **kwargs):
        """Plot fault feature(s) as great circle and point"""
        try:
            artist = ArtistFactory.create_fault(*args, **kwargs)
            self._artists.append(artist)
        except TypeError as err:
            print(err)

    def hoeppner(self, *args, **kwargs):
        """Plot a fault-and-striae as in tangent lineation plot - Hoeppner plot."""
        try:
            artist = ArtistFactory.create_hoeppner(*args, **kwargs)
            self._artists.append(artist)
        except TypeError as err:
            print(err)

    def arrow(self, *args, **kwargs):
        """Plot arrows at position of first argument and oriented in direction of second"""
        try:
            artist = ArtistFactory.create_arrow(*args, **kwargs)
            self._artists.append(artist)
        except TypeError as err:
            print(err)

    def contourf(self, *args, **kwargs):
        """Plot filled contours."""
        try:
            artist = ArtistFactory.create_contourf(*args, **kwargs)
            # ad-hoc density calculation needed to access correct grid properties
            self.grid.calculate_density(
                args[0], sigma=kwargs.get("sigma"), trim=kwargs.get("trim")
            )
            self._artists.append(artist)
        except TypeError as err:
            print(err)

    ########################################
    # PLOTTING ROUTINES                    #
    ########################################

    def _line(self, *args, **kwargs):
        x_lower, y_lower = self.proj.project_data(*np.vstack(args).T)
        x_upper, y_upper = self.proj.project_data(*(-np.vstack(args).T))
        handles = self.ax.plot(
            np.hstack((x_lower, x_upper)), np.hstack((y_lower, y_upper)), **kwargs
        )
        for h in handles:
            h.set_clip_path(self.primitive)
        return handles

    def _vector(self, *args, **kwargs):
        x_lower, y_lower, x_upper, y_upper = self.proj.project_data_antipodal(
            *np.vstack(args).T
        )
        handles = self.ax.plot(x_lower, y_lower, **kwargs)
        for h in handles:
            h.set_clip_path(self.primitive)
        kwargs["label"] = None
        kwargs["color"] = h.get_color()
        kwargs["mfc"] = "none"
        handles = self.ax.plot(x_upper, y_upper, **kwargs)
        for h in handles:
            h.set_clip_path(self.primitive)
        return handles

    def _great_circle(self, *args, **kwargs):
        X, Y = [], []
        for arg in args:
            if self.proj.rotate_data:
                fdv = arg.transform(self.proj.R).dipvec().transform(self.proj.Ri)
            else:
                fdv = arg.dipvec()
            # iterate
            for fol, dv in zip(np.atleast_2d(arg), np.atleast_2d(fdv)):
                # plot on lower
                x, y = self.proj.project_data(
                    *np.array(
                        [Vector3(dv).rotate(Vector3(fol), a) for a in self.angles_gc]
                    ).T
                )
                X.append(np.hstack((x, np.nan)))
                Y.append(np.hstack((y, np.nan)))
                # plot on upper
                x, y = self.proj.project_data(
                    *np.array(
                        [-Vector3(dv).rotate(Vector3(fol), a) for a in self.angles_gc]
                    ).T
                )
                X.append(np.hstack((x, np.nan)))
                Y.append(np.hstack((y, np.nan)))
        handles = self.ax.plot(np.hstack(X), np.hstack(Y), **kwargs)
        for h in handles:
            h.set_clip_path(self.primitive)
        return handles

    def _scatter(self, *args, **kwargs):
        legend = kwargs.pop("legend")
        num = kwargs.pop("num")
        x_lower, y_lower, mask_lower = self.proj.project_data(
            *np.vstack(args).T, return_mask=True
        )
        x_upper, y_upper, mask_upper = self.proj.project_data(
            *(-np.vstack(args).T), return_mask=True
        )
        if kwargs["s"] is not None:
            s = np.atleast_1d(kwargs["s"])
            kwargs["s"] = np.hstack((s[mask_lower], s[mask_upper]))
        if kwargs["c"] is not None:
            c = np.atleast_1d(kwargs["c"])
            kwargs["c"] = np.hstack((c[mask_lower], c[mask_upper]))
        sc = self.ax.scatter(
            np.hstack((x_lower, x_upper)), np.hstack((y_lower, y_upper)), **kwargs,
        )
        if legend:
            self.ax.legend(
                *sc.legend_elements("sizes", num=num),
                bbox_to_anchor=(1.05, 1),
                prop={"size": 11},
                loc="upper left",
                borderaxespad=0,
                scatterpoints=1,
                numpoints=1,
            )
        sc.set_clip_path(self.primitive)

    def _cone(self, *args, **kwargs):
        X, Y = [], []
        # get scalar arguments from kwargs
        angles = kwargs.pop("angles")
        for axis, angle in zip(np.vstack(args), angles):
            if self.proj.rotate_data:
                lt = axis.transform(self.proj.R)
                azi, dip = Vector3(lt).geo
                cl_lower = Vector3(azi, dip + angle).transform(self.proj.Ri)
                cl_upper = -Vector3(azi, dip - angle).transform(self.proj.Ri)
            else:
                lt = axis
                azi, dip = Vector3(lt).geo
                cl_lower = Vector3(azi, dip + angle)
                cl_upper = -Vector3(azi, dip - angle)
            # plot on lower
            x, y = self.proj.project_data(
                *np.array([cl_lower.rotate(lt, a) for a in self.angles_sc]).T
            )
            X.append(np.hstack((x, np.nan)))
            Y.append(np.hstack((y, np.nan)))
            # plot on upper
            x, y = self.proj.project_data(
                *np.array([cl_upper.rotate(-lt, a) for a in self.angles_sc]).T
            )
            X.append(np.hstack((x, np.nan)))
            Y.append(np.hstack((y, np.nan)))
        handles = self.ax.plot(np.hstack(X), np.hstack(Y), **kwargs)
        for h in handles:
            h.set_clip_path(self.primitive)
        return handles

    def _pair(self, *args, **kwargs):
        line_marker = kwargs.pop("line_marker")
        h = self._great_circle(*[arg.fol for arg in args], **kwargs)
        self._line(
            *[arg.lin for arg in args],
            marker=line_marker,
            ls="none",
            mfc=h[0].get_color(),
            mec=h[0].get_color(),
            ms=kwargs.get("ms"),
        )

    def _fault(self, *args, **kwargs):
        h = self._great_circle(*[arg.fol for arg in args], **kwargs)
        quiver_kwargs = apsg_conf["default_quiver_kwargs"]
        quiver_kwargs["pivot"] = "tail"
        quiver_kwargs["color"] = h[0].get_color()
        for arg in args:
            self._arrow(arg.lin, sense=arg.sense, **quiver_kwargs)

    def _hoeppner(self, *args, **kwargs):
        h = self._line(*[arg.fol for arg in args], **kwargs)
        quiver_kwargs = apsg_conf["default_quiver_kwargs"]
        quiver_kwargs["color"] = h[0].get_color()
        for arg in args:
            self._arrow(arg.fol, arg.lin, sense=arg.sense, **quiver_kwargs)

    def _arrow(self, *args, **kwargs):
        x_lower, y_lower = self.proj.project_data(*np.vstack(np.atleast_2d(args[0])).T)
        x_upper, y_upper = self.proj.project_data(
            *(-np.vstack(np.atleast_2d(args[0])).T)
        )
        x = np.hstack((x_lower, x_upper))
        y = np.hstack((y_lower, y_upper))
        if len(args) > 1:
            x_lower, y_lower = self.proj.project_data(
                *np.vstack(np.atleast_2d(args[1])).T
            )
            x_upper, y_upper = self.proj.project_data(
                *(-np.vstack(np.atleast_2d(args[1])).T)
            )
            dx = np.hstack((x_lower, x_upper))
            dy = np.hstack((y_lower, y_upper))
        else:
            dx, dy = x, y
        mag = np.hypot(dx, dy)
        sense = np.atleast_1d(kwargs.pop("sense"))
        u, v = sense * dx / mag, sense * dy / mag
        self.ax.quiver(x, y, u, v, **kwargs)

    def _contourf(self, *args, **kwargs):
        sigma = kwargs.pop("sigma")
        trim = kwargs.pop("trim")
        colorbar = kwargs.pop("colorbar")
        label = kwargs.pop("label")
        clines = kwargs.pop("clines")
        linewidths = kwargs.pop("linewidths")
        linestyles = kwargs.pop("linestyles")
        if not self.grid.calculated:
            self.grid.calculate_density(args[0], sigma=sigma, trim=trim)
        dcgrid = np.asarray(self.grid.grid).T
        X, Y = self.proj.project_data(*dcgrid, clip_inside=False)
        cf = self.ax.tricontourf(X, Y, self.grid.values, **kwargs)
        for collection in cf.collections:
            collection.set_clip_path(self.primitive)
        if clines:
            kwargs["cmap"] = None
            kwargs["colors"] = "k"
            kwargs["linewidths"] = linewidths
            kwargs["linestyles"] = linestyles
            cl = self.ax.tricontour(X, Y, self.grid.values, **kwargs)
            for collection in cl.collections:
                collection.set_clip_path(self.primitive)
        if colorbar:
            self.fig.colorbar(cf, ax=self.ax, shrink=0.6)
        # plt.colorbar(cf, format="%3.2f", spacing="proportional")


class RosePlot(object):

    """
    ``RosePlot`` class for rose histogram plotting.

    Args:
        any plottable APSG class (most of data classes and tensors)

    Keyword Args:
        title: figure title. Default ''
        figsize: Figure size. Default from settings ()
        axial: Directional data are axial. Defaut True
        density: Use density instead of counts. Default False
        pdf: Plot Von Mises density function instead histogram. Default False
        kappa; Shape parameter of Von Mises pdf. Default 250
        scaled: Bins scaled by area instead value. Default False
        arrow: Bar arrowness. (0-1) Default 0.95
        rwidth: Bar width (0-1). Default 1
        ticks: show ticks. Default True
        grid: show grid lines. Default False
        grid_kw: Dict passed to Axes.grid. Default {}

        Other keyword arguments are passed to matplotlib plot.

    Examples:
        >>> g = Group.randn_fol(mean=fol(120, 0))
        >>> direction, dip  = g.rhr
        >>> RosePlot(direction)
        >>> RosePlot(direction, density=True)
        >>> RosePlot(direction, pdf=True)
        >>> s = RosePlot()
        >>> s.plot(direction, color='r')
        >>> s.show()
    """

    def __init__(self, *args, **kwargs):
        self.fig = plt.figure(figsize=kwargs.pop("figsize", apsg_conf["figsize"]))
        self.fig.canvas.set_window_title("Rose plot")
        self.bins = kwargs.get("bins", 36)
        self.axial = kwargs.get("axial", True)
        self.pdf = kwargs.get("pdf", False)
        self.kappa = kwargs.get("kappa", 250)
        self.density = kwargs.get("density", False)
        self.arrow = kwargs.get("arrow", 0.95)
        self.rwidth = kwargs.get("rwidth", 1)
        self.scaled = kwargs.get("scaled", False)
        self.title_text = kwargs.get("title", "")
        self.grid = kwargs.get("grid", True)
        self.grid_kw = kwargs.get("grid_kw", {})
        self.fill_kw = kwargs.get("fill_kw", {})
        self.cla()
        # optionally immidiately plot passed objects
        if args:
            for arg in args:
                self.plot(arg)
            self.show()

    def cla(self):
        """Clear projection."""

        self.fig.clear()
        self.ax = self.fig.add_subplot(111, polar=True)
        # self.ax.format_coord = self.format_coord
        self.ax.set_theta_direction(-1)
        self.ax.set_theta_zero_location("N")
        self.ax.grid(self.grid, **self.grid_kw)
        self.fig.suptitle(self.title_text)

    def plot(self, obj, *args, **kwargs):
        if type(obj) is FeatureSet:
            ang, _ = obj.dd
            weights = abs(obj)
            self.title_text = obj.name
        else:
            ang = np.array(obj)
            weights = None
        if "weights" in kwargs:
            weights = kwargs.pop("weights")

        if self.axial:
            ang = np.concatenate((ang % 360, (ang + 180) % 360))
            if weights is not None:
                weights = np.concatenate((weights, weights))

        if self.pdf:
            theta = np.linspace(-np.pi, np.pi, 1801)
            radii = np.zeros_like(theta)
            for a in ang:
                radii += vonmises.pdf(theta, self.kappa, loc=np.radians(a % 360))
            radii /= len(ang)
        else:
            width = 360 / self.bins
            if weights is not None:
                num, bin_edges = np.histogram(
                    ang,
                    bins=self.bins + 1,
                    range=(-width / 2, 360 + width / 2),
                    weights=weights,
                    density=self.density,
                )
            else:
                num, bin_edges = np.histogram(
                    ang,
                    bins=self.bins + 1,
                    range=(-width / 2, 360 + width / 2),
                    density=self.density,
                )
            num[0] += num[-1]
            num = num[:-1]
            theta, radii = [], []
            for cc, val in zip(np.arange(0, 360, width), num):
                theta.extend(
                    [
                        cc - width / 2,
                        cc - self.rwidth * width / 2,
                        cc,
                        cc + self.rwidth * width / 2,
                        cc + width / 2,
                    ]
                )
                radii.extend([0, val * self.arrow, val, val * self.arrow, 0])
            theta = np.deg2rad(theta)
        if self.scaled:
            radii = np.sqrt(radii)
        fill_kw = self.fill_kw.copy()
        fill_kw.update(kwargs)
        self.ax.fill(theta, radii, **fill_kw)

    def close(self):
        plt.close(self.fig)

    def show(self):
        plt.show()

    def savefig(self, filename="apsg_roseplot.pdf", **kwargs):
        self.ax.figure.savefig(filename, **kwargs)


class _FabricPlot(object):

    """
    Metaclas for Fabric plots
    """

    def close(self):
        plt.close(self.fig)

    @property
    def closed(self):
        return not plt.fignum_exists(self.fig.number)

    def draw(self):
        if self.closed:
            print(
                "The DeformationPlot figure have been closed. "
                "Use new() method or create new one."
            )
        else:
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
            plt.draw()

    def new(self):
        """
        Re-initialize figure.
        """

        if self.closed:
            self.__init__()

    def show(self):
        plt.show()

    def savefig(self, filename="apsg_fabricplot.pdf", **kwargs):
        if self._lgd is None:
            self.ax.figure.savefig(filename, **kwargs)
        else:
            self.ax.figure.savefig(
                filename, bbox_extra_artists=(self._lgd,), bbox_inches="tight", **kwargs
            )


class VollmerPlot(_FabricPlot):

    """
    Represents the triangular fabric plot (Vollmer, 1989).
    """

    def __init__(self, *args, **kwargs):
        self.fig = plt.figure(figsize=kwargs.pop("figsize", apsg_conf["figsize"]))
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

    def cla(self):
        """Clear projection."""

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
            for ln in np.arange(0.1, 1, 0.1):
                self.triplot([ln, ln], [0, 1 - ln], [1 - ln, 0], "k:")
                self.triplot([0, 1 - ln], [ln, ln], [1 - ln, 0], "k:")
                self.triplot([0, 1 - ln], [1 - ln, 0], [ln, ln], "k:")

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
        if issubclass(type(obj), Vector3Set):
            obj = obj.ortensor()

        if not isinstance(obj, OrientationTensor3):
            raise TypeError("Argument must be Vector3Set or OrientationTensor3")

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

    def path(self, objs, *args, **kwargs):
        # ensure point plot
        if "ls" not in kwargs and "linestyle" not in kwargs:
            kwargs["linestyle"] = "-"

        if not args:
            if "marker" not in kwargs:
                kwargs["marker"] = "."

        P = [obj.P for obj in objs]
        G = [obj.G for obj in objs]
        R = [obj.R for obj in objs]

        self.triplot(P, G, R, *args, **kwargs)

        self.draw()

    def format_coord(self, x, y):
        a, b = self.Ti.dot(np.r_[x, y] - self.C)
        c = 1 - a - b
        if a < 0 or b < 0 or c < 0:
            return ""
        else:
            return "P:{:0.2f} G:{:0.2f} R:{:0.2f}".format(a, b, c)


class RamsayPlot(_FabricPlot):

    """
    Represents the Ramsay deformation plot.
    """

    def __init__(self, *args, **kwargs):
        self.fig = plt.figure(figsize=kwargs.pop("figsize", apsg_conf["figsize"]))
        self.fig.canvas.set_window_title("Ramsay deformation plot")
        self.ticks = kwargs.get("ticks", True)
        self.grid = kwargs.get("grid", False)
        self.grid_style = kwargs.get("grid_style", "k:")
        self._lgd = None
        self.cla()
        # optionally immidiately plot passed objects
        if args:
            for arg in args:
                self.plot(arg)
            self.show()

    def cla(self):
        """Clear projection."""

        self.fig.clear()
        self.ax = self.fig.add_subplot(111)
        self.ax.format_coord = self.format_coord
        self.ax.set_aspect("equal")
        self.ax.set_autoscale_on(True)
        self.ax.spines["top"].set_color("none")
        self.ax.spines["right"].set_color("none")
        self.ax.set_xlabel(r"$\varepsilon_2-\varepsilon_3$")
        self.ax.set_ylabel(r"$\varepsilon_1-\varepsilon_2$")
        self.ax.grid(self.grid)

        self.ax.set_title("Ramsay plot")

        self.draw()

    def plot(self, obj, *args, **kwargs):
        if issubclass(type(obj), Vector3Set):
            obj = obj.ortensor()

        if not isinstance(obj, OrientationTensor3):
            raise TypeError("Argument must be Vector3Set or OrientationTensor3")

        # ensure point plot
        if "ls" not in kwargs and "linestyle" not in kwargs:
            kwargs["linestyle"] = "none"

        if not args:
            if "marker" not in kwargs:
                kwargs["marker"] = "o"
        if "label" not in kwargs:
            kwargs["label"] = obj.name

        self.ax.plot(obj.e23, obj.e12, *args, **kwargs)

        self.draw()

    def path(self, objs, *args, **kwargs):
        # ensure point plot
        if "ls" not in kwargs and "linestyle" not in kwargs:
            kwargs["linestyle"] = "-"

        if not args:
            if "marker" not in kwargs:
                kwargs["marker"] = "."
        # if "label" not in kwargs:
        #    kwargs["label"] = obj.name

        e23 = [obj.e23 for obj in objs]
        e12 = [obj.e12 for obj in objs]

        self.ax.plot(e23, e12, *args, **kwargs)

        self.draw()

    def show(self):
        mx = max(self.ax.get_xlim()[1], self.ax.get_ylim()[1])
        self.ax.set_xlim(0, mx)
        self.ax.set_ylim(0, mx)
        self.ax.plot([0, mx], [0, mx], "k", lw=0.5)
        box = self.ax.get_position()
        self.ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
        plt.show()

    def format_coord(self, x, y):
        k = y / x if x > 0 else 0
        d = x ** 2 + y ** 2
        return "k:{:0.2f} d:{:0.2f}".format(k, d)


class FlinnPlot(_FabricPlot):

    """
    Represents the Ramsay deformation plot.
    """

    def __init__(self, *args, **kwargs):
        self.fig = plt.figure(figsize=kwargs.pop("figsize", apsg_conf["figsize"]))
        self.fig.canvas.set_window_title("Flinn's deformation plot")
        self.ticks = kwargs.get("ticks", True)
        self.grid = kwargs.get("grid", False)
        self.grid_style = kwargs.get("grid_style", "k:")
        self._lgd = None
        self.cla()
        # optionally immidiately plot passed objects
        if args:
            for arg in args:
                self.plot(arg)
            self.show()

    def cla(self):
        """Clear projection."""

        self.fig.clear()
        self.ax = self.fig.add_subplot(111)
        self.ax.format_coord = self.format_coord
        self.ax.set_aspect("equal")
        self.ax.set_autoscale_on(True)
        self.ax.spines["top"].set_color("none")
        self.ax.spines["right"].set_color("none")
        self.ax.set_xlabel(r"$R_{YZ}$")
        self.ax.set_ylabel(r"$R_{XY}$")
        self.ax.grid(self.grid)

        self.ax.set_title("Flinn's plot")

        self.draw()

    def plot(self, obj, *args, **kwargs):
        if issubclass(type(obj), Vector3Set):
            obj = obj.ortensor()

        if not isinstance(obj, OrientationTensor3):
            raise TypeError("Argument must be Vector3Set or OrientationTensor3")

        # ensure point plot
        if "ls" not in kwargs and "linestyle" not in kwargs:
            kwargs["linestyle"] = "none"

        if not args:
            if "marker" not in kwargs:
                kwargs["marker"] = "o"
        if "label" not in kwargs:
            kwargs["label"] = obj.name

        self.ax.plot(obj.Ryz, obj.Rxy, *args, **kwargs)

        self.draw()

    def path(self, objs, *args, **kwargs):
        # ensure point plot
        if "ls" not in kwargs and "linestyle" not in kwargs:
            kwargs["linestyle"] = "-"

        if not args:
            if "marker" not in kwargs:
                kwargs["marker"] = "."
        # if "label" not in kwargs:
        #    kwargs["label"] = obj.name

        Ryz = [obj.Ryz for obj in objs]
        Rxy = [obj.Rxy for obj in objs]

        self.ax.plot(Ryz, Rxy, *args, **kwargs)

        self.draw()

    def show(self):
        mx = max(self.ax.get_xlim()[1], self.ax.get_ylim()[1])
        self.ax.set_xlim(1, mx)
        self.ax.set_ylim(1, mx)
        self.ax.plot([1, mx], [1, mx], "k", lw=0.5)
        box = self.ax.get_position()
        self.ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
        plt.show()

    def format_coord(self, x, y):
        K = (y - 1) / (x - 1) if x > 1 else 0
        D = np.sqrt((x - 1) ** 2 + (y - 1) ** 2)
        return "K:{:0.2f} D:{:0.2f}".format(K, D)


class HsuPlot(_FabricPlot):

    """
    Represents the Hsu fabric plot.
    """

    def __init__(self, *args, **kwargs):
        self.fig = plt.figure(figsize=kwargs.pop("figsize", apsg_conf["figsize"]))
        self.fig.canvas.set_window_title("Hsu fabric plot")
        self.ticks = kwargs.get("ticks", True)
        self.grid = kwargs.get("grid", True)
        self.grid_style = kwargs.get("grid_style", "k:")
        self._lgd = None
        self.cla()
        # optionally immidiately plot passed objects
        if args:
            for arg in args:
                self.plot(arg)
            self.show()

    def cla(self):
        """Clear projection."""

        self.fig.clear()
        self.ax = self.fig.add_subplot(111, polar=True)
        # self.ax.format_coord = self.format_coord
        self.ax.set_theta_zero_location("N")
        self.ax.set_theta_direction(-1)
        self.ax.set_thetamin(-30)
        self.ax.set_thetamax(30)
        self.ax.set_xticks([-np.pi / 6, -np.pi / 12, 0, np.pi / 12, np.pi / 6])
        self.ax.set_xticklabels([-1, -0.5, 0, 0.5, 1])
        self.ax.set_title(r"$\nu$")
        self.ax.set_ylabel(r"$\bar{\varepsilon}_s$")
        self.ax.grid(self.grid)

        self.draw()

    def plot(self, obj, *args, **kwargs):
        if issubclass(type(obj), Vector3Set):
            obj = obj.ortensor()

        if not isinstance(obj, OrientationTensor3):
            raise TypeError("Argument must be Vector3Set or OrientationTensor3")

        # ensure point plot
        if "ls" not in kwargs and "linestyle" not in kwargs:
            kwargs["linestyle"] = "none"

        if not args:
            if "marker" not in kwargs:
                kwargs["marker"] = "o"
        if "label" not in kwargs:
            kwargs["label"] = obj.name

        self.ax.plot(obj.lode * np.pi / 6, obj.eoct, *args, **kwargs)

        self.draw()

    def path(self, objs, *args, **kwargs):
        # ensure point plot
        if "ls" not in kwargs and "linestyle" not in kwargs:
            kwargs["linestyle"] = "-"

        if not args:
            if "marker" not in kwargs:
                kwargs["marker"] = "."
        # if "label" not in kwargs:
        #    kwargs["label"] = obj.name

        lode = [obj.lode * np.pi / 6 for obj in objs]
        eoct = [obj.eoct for obj in objs]

        self.ax.plot(lode, eoct, *args, **kwargs)

        self.draw()

    def format_coord(self, x, y):
        K = (y - 1) / (x - 1) if x > 1 else 0
        D = np.sqrt((x - 1) ** 2 + (y - 1) ** 2)
        return "K:{:0.2f} D:{:0.2f}".format(K, D)
