# -*- coding: utf-8 -*-

import pickle

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle

from apsg.config import apsg_conf
from apsg.math._vector import Vector3
from apsg.feature._geodata import Lineation, Foliation, Pair, Fault, Cone
from apsg.feature._container import (
    Vector3Set,
    LineationSet,
    FoliationSet,
    PairSet,
    FaultSet,
)
from apsg.feature import feature_from_json
from apsg.plotting._stereogrid import StereoGrid
from apsg.plotting._plot_artists import StereoNetArtistFactory

__all__ = ["StereoNet"]


class StereoNet:
    """
    Plot features on stereographic projection

    Keyword Args:
        title (str): figure title. Default None.
        title_kws (dict): dictionary of keyword arguments passed to matplotlib suptitle
            method.
        tight_layout (bool): Matplotlib figure tight_layout. Default False
        kind (str): Equal area ("equal-area", "schmidt" or "earea") or equal angle
            ("equal-angle", "wulff" or "eangle") projection. Default is "equal-area"
        hemisphere (str): "lower" or "upper". Default is "lower"
        overlay_position (tuple or Pair): Position of overlay X, Y, Z given by Pair.
            X is direction of linear element, Z is normal to planar.
            Default is (0, 0, 0, 0)
        rotate_data (bool): Whether data should be rotated together with overlay.
            Default False
        minor_ticks (None or float): Default None
        major_ticks (None or float): Default None
        overlay (bool): Whether to show overlay. Default is True
        overlay_step (float): Grid step of overlay. Default 15
        overlay_resolution (float): Resolution of overlay. Default 181
        clip_pole (float): Clipped cone around poles. Default 15
        grid_type (str): Type of contouring grid "gss" or "sfs". Default "gss"
        grid_n (int): Number of counting points in grid. Default 3000

    Examples:
        >>> l = linset.random_fisher(position=lin(120, 40))
        >>> s = StereoNet(title="Random linear features")
        >>> s.contour(l)
        >>> s.line(l)
        >>> s.show()
    """

    def __init__(self, **kwargs):
        self._kwargs = apsg_conf["stereonet_default_kwargs"].copy()
        self._kwargs.update((k, kwargs[k]) for k in self._kwargs.keys() & kwargs.keys())
        self._kwargs["title"] = kwargs.get("title", None)
        self.grid = StereoGrid(**self._kwargs)
        # alias for Projection instance
        self.proj = self.grid.proj
        self.angles_gc = np.linspace(
            -90 + 1e-7, 90 - 1e-7, int(self.proj.overlay_resolution / 2)
        )
        self.angles_sc = np.linspace(
            -180 + 1e-7, 180 - 1e-7, self.proj.overlay_resolution
        )

        self.clear()

    def clear(self):
        """Clear plot"""
        self._artists = []

    def _draw_layout(self):
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
            label="_nolegend_",
        )
        self.ax.add_patch(self.primitive)

    def _plot_artists(self):
        for artist in self._artists:
            plot_method = getattr(self, artist.stereonet_method)
            plot_method(*artist.args, **artist.kwargs)

    def to_json(self):
        """Return stereonet as JSON dict"""
        artists = [artist.to_json() for artist in self._artists]
        return dict(kwargs=self._kwargs, artists=artists)

    @classmethod
    def from_json(cls, json_dict):
        """Create stereonet from JSON dict"""
        s = cls(**json_dict["kwargs"])
        s._artists = [
            stereonetartist_from_json(artist) for artist in json_dict["artists"]
        ]
        return s

    def save(self, filename):
        """
        Save stereonet to pickle file

        Args:
            filename (str): name of picke file
        """
        with open(filename, "wb") as f:
            pickle.dump(self.to_json(), f, pickle.HIGHEST_PROTOCOL)

    @classmethod
    def load(cls, filename):
        """
        Load stereonet from pickle file

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
            self.fig.canvas.manager.set_window_title(self.proj.netname)

    def _render(self):
        self.ax = self.fig.add_subplot()
        self.ax.set_aspect(1)
        self.ax.set_axis_off()
        self._draw_layout()
        self._plot_artists()
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

    def format_coord(self, x, y):
        """Format stereonet coordinates"""
        if x is not None and y is not None:
            if (x**2 + y**2) <= 1:
                lcoord = Lineation(*self.proj.inverse_data(x, y))
                fcoord = Foliation(*self.proj.inverse_data(x, y))
                return f"{lcoord} {fcoord}"
        return ""

    def show(self):
        """Show stereonet"""
        plt.close(0)  # close previously rendered figure
        self.init_figure()
        self._render()
        self.ax.format_coord = self.format_coord
        plt.show()

    def savefig(self, filename="stereonet.png", **kwargs):
        """
        Save stereonet figure to graphics file

        Keyword Args:
            filename (str): filename

        All others kwargs are passed to matplotlib `Figure.savefig`
        """
        plt.close(0)  # close previously rendered figure
        self.init_figure()
        self._render()
        self.fig.savefig(filename, **kwargs)
        plt.close(0)

    ########################################
    # PLOTTING METHODS                     #
    ########################################

    def line(self, *args, **kwargs):
        """
        Plot linear feature(s) as point(s)

        Args:
            Vector3 or Vector3Set like feature(s)

        Keyword Args:
            alpha (scalar): Set the alpha value. Default None
            color (color): Set the color of the point. Default None
            mec (color): Set the edge color. Default None
            mfc (color): Set the face color. Default None
            mew (float): Set the marker edge width. Default 1
            ms (float): Set the marker size. Default 6
            marker (str): Marker style string. Default "o"
            ls (str): Line style string (only for multiple features).
                Default None

        """
        try:
            artist = StereoNetArtistFactory.create_point(*args, **kwargs)
            self._artists.append(artist)
        except TypeError as err:
            print(err)

    def pole(self, *args, **kwargs):
        """
        Plot pole of planar feature(s) as point(s)

        Args:
            Foliation or FoliationSet feature(s)

        Keyword Args:
            alpha (scalar): Set the alpha value. Default None
            color (color): Set the color of the point. Default None
            mec (color): Set the edge color. Default None
            mfc (color): Set the face color. Default None
            mew (float): Set the marker edge width. Default 1
            ms (float): Set the marker size. Default 6
            marker (str): Marker style string. Default "o"
            ls (str): Line style string (only for multiple features).
                Default None

        """
        try:
            artist = StereoNetArtistFactory.create_pole(*args, **kwargs)
            self._artists.append(artist)
        except TypeError as err:
            print(err)

    def vector(self, *args, **kwargs):
        """
        Plot vector feature(s) as point(s)

        Note: Markers are filled on lower and open on upper hemisphere.

        Args:
            Vector3 or Vector3Set like feature(s)

        Keyword Args:
            alpha (scalar): Set the alpha value. Default None
            color (color): Set the color of the point. Default None
            mec (color): Set the edge color. Default None
            mfc (color): Set the face color. Default None
            mew (float): Set the marker edge width. Default 1
            ms (float): Set the marker size. Default 6
            marker (str): Marker style string. Default "o"
            ls (str): Line style string (only for multiple features).
                Default None

        """
        try:
            artist = StereoNetArtistFactory.create_vector(*args, **kwargs)
            self._artists.append(artist)
        except TypeError as err:
            print(err)

    def scatter(self, *args, **kwargs):
        """
        Plot vector-like feature(s) as point(s)

        Note: This method is using scatter plot to allow variable colors
            or sizes of points

        Args:
            Vector3 or Vector3Set like feature(s)

        Keyword Args:
            s (list or array):
            c (list or array)
            alpha (scalar): Set the alpha value. Default None
            linewidths (float): The linewidth of the marker edges. Default 1.5
            marker (str): Marker style string. Default "o"
            cmap (str): Mtplotlib colormap. Default None
            legend (bool): Whether to show legend. Default False
            num (int): NUmber of legend items. Default "auto"

        """
        try:
            artist = StereoNetArtistFactory.create_scatter(*args, **kwargs)
            self._artists.append(artist)
        except TypeError as err:
            print(err)

    def great_circle(self, *args, **kwargs):
        """
        Plot planar feature(s) as great circle(s)

        Note: ``great_circle`` has also alias ``gc``

        Args:
            Foliation or FoliationSet feature(s)

        Keyword Args:
            alpha (scalar): Set the alpha value. Default None
            color (color): Set the color of the point. Default None
            ls (str): Line style string (only for multiple features).
                Default "-"
            lw (float): Set line width. Default 1.5

        """
        try:
            artist = StereoNetArtistFactory.create_great_circle(*args, **kwargs)
            self._artists.append(artist)
        except TypeError as err:
            print(err)

    gc = great_circle

    def arc(self, *args, **kwargs):
        """
        Plot arc bewtween vectors along great circle(s)

        Note: You should pass several features in connection order

        Args:
            Vector3 or Vector3Set like feature(s)

        Keyword Args:
            alpha (scalar): Set the alpha value. Default None
            color (color): Set the color of the point. Default None
            ls (str): Line style string (only for multiple features).
                Default "-"
            lw (float): Set line width. Default 1.5

        """
        try:
            artist = StereoNetArtistFactory.create_arc(*args, **kwargs)
            self._artists.append(artist)
        except TypeError as err:
            print(err)

    def cone(self, *args, **kwargs):
        """"""
        """
        Plot cone(s) as small circle(s) with given apical angle(s)

        Args:
            Cone or ConeSet feature(s)

        Keyword Args:
            alpha (scalar): Set the alpha value. Default None
            color (color): Set the color of the point. Default None
            ls (str): Line style string (only for multiple features).
                Default "-"
            lw (float): Set line width. Default 1.5

        """
        try:
            artist = StereoNetArtistFactory.create_cone(*args, **kwargs)
            self._artists.append(artist)
        except TypeError as err:
            print(err)

    def pair(self, *args, **kwargs):
        """
        Plot pair feature(s) as great circle and point

        Args:
            Pair or PairSet feature(s)

        Keyword Args:
            alpha (scalar): Set the alpha value. Default None
            color (color): Set the color of the point. Default None
            ls (str): Line style string (only for multiple features).
                Default "-"
            lw (float): Set line width. Default 1.5
            line_marker (str): Marker style string for point. Default "o"

        """
        try:
            artist = StereoNetArtistFactory.create_pair(*args, **kwargs)
            self._artists.append(artist)
        except TypeError as err:
            print(err)

    def fault(self, *args, **kwargs):
        """
        Plot fault feature(s) as great circle and arrow

        Note: Arrow is styled according to default arrow config

        Args:
            Fault or FaultSet feature(s)

        Keyword Args:
            alpha (scalar): Set the alpha value. Default None
            color (color): Set the color of the point. Default None
            ls (str): Line style string (only for multiple features).
                Default "-"
            lw (float): Set line width. Default 1.5

        """
        try:
            artist = StereoNetArtistFactory.create_fault(*args, **kwargs)
            self._artists.append(artist)
        except TypeError as err:
            print(err)

    def hoeppner(self, *args, **kwargs):
        """
        Plot fault feature(s) on Hoeppner (tangent lineation) plot

        Note: Arrow is styled according to default arrow config

        Args:
            Fault or FaultSet feature(s)

        Keyword Args:
            alpha (scalar): Set the alpha value. Default None
            color (color): Set the color of the point. Default None
            ls (str): Line style string (only for multiple features).
                Default "-"
            lw (float): Set line width. Default 1.5

        """
        try:
            artist = StereoNetArtistFactory.create_hoeppner(*args, **kwargs)
            self._artists.append(artist)
        except TypeError as err:
            print(err)

    def arrow(self, *args, **kwargs):
        """
        Plot arrow at position of first argument
        and oriented in direction of second

        Note: You should pass two features

        Args:
            Vector3 or Vector3Set like feature(s)

        Keyword Args:
            color (color): Set the color of the arrow. Default None
            width (int): Width of arrow. Default 2
            headwidth (int): Width of arrow head. Default 5
            pivot (str): Arrow pivot. Default "mid"
            units (str): Arrow size units. Default "dots"

        """
        try:
            artist = StereoNetArtistFactory.create_arrow(*args, **kwargs)
            self._artists.append(artist)
        except TypeError as err:
            print(err)

    def tensor(self, *args, **kwargs):
        """
        Plot principal planes or principal directions of tensor

        Args:
            OrientationTensor3 like feature(s)

        Keyword Args:
            planes (bool): When True, plot principal planes, otherwise principal
                directions. Default True
            alpha (scalar): Set the alpha value. Default None
            color (color): Set the color. Default is red, green, blue for s1, s2, s3
            ls (str): Line style string (only for multiple features).
                Default "-"
            lw (float): Set line width. Default 1.5
            mew (float): Set the marker edge width. Default 1
            ms (float): Set the marker size. Default 9
            marker (str): Marker style string. Default "o"

        """
        try:
            artist = StereoNetArtistFactory.create_tensor(*args, **kwargs)
            self._artists.append(artist)
        except TypeError as err:
            print(err)

    def contour(self, *args, **kwargs):
        """
        Plot filled contours using modified Kamb contouring technique with exponential
        smoothing.

        Args:
            Vector3Set like feature

        Keyword Args:
            levels (int or list): number or values of contours. Default 6
            cmap: matplotlib colormap used for filled contours. Default "Greys"
            colorbar (bool): Show colorbar. Default False
            alpha (float): transparency. Default None
            antialiased (bool): Default True
            sigma (float): If None it is automatically calculated
            sigmanorm (bool): If True scaled counts are normalized by sigma.
                Default True
            trimzero (bool): Remove values equal to 0. Default True
            clines (bool): Show contour lines instead filled contours. Default False
            linewidths (float): contour lines width
            linestyles (str): contour lines style
            show_data (bool): Show data as points. Default False
            data_kws (dict): arguments passed to point factory when `show_data` True
        """
        try:
            artist = StereoNetArtistFactory.create_contour(*args, **kwargs)
            # ad-hoc density calculation needed to access correct grid properties
            if len(args) > 0:
                self.grid.calculate_density(
                    args[0],
                    sigma=artist.kwargs.get("sigma"),
                    sigmanorm=artist.kwargs.get("sigmanorm"),
                    trimzero=artist.kwargs.get("trimzero"),
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
        if len(x_lower) > 0:
            handles = self.ax.plot(x_lower, y_lower, **kwargs)
            for h in handles:
                h.set_clip_path(self.primitive)
            u_kwargs = kwargs.copy()
            u_kwargs["label"] = "_upper"
            u_kwargs["mec"] = h.get_color()
            u_kwargs["mfc"] = "none"
            handles = self.ax.plot(x_upper, y_upper, **u_kwargs)
            for h in handles:
                h.set_clip_path(self.primitive)
        else:
            u_kwargs = kwargs.copy()
            u_kwargs["mfc"] = "none"
            handles = self.ax.plot(x_upper, y_upper, **u_kwargs)
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
                        [
                            np.asarray(Vector3(dv).rotate(Vector3(fol), a))
                            for a in self.angles_gc
                        ]
                    ).T
                )
                X.append(np.hstack((x, np.nan)))
                Y.append(np.hstack((y, np.nan)))
                # plot on upper
                x, y = self.proj.project_data(
                    *np.array(
                        [
                            -np.asarray(Vector3(dv).rotate(Vector3(fol), a))
                            for a in self.angles_gc
                        ]
                    ).T
                )
                X.append(np.hstack((x, np.nan)))
                Y.append(np.hstack((y, np.nan)))
        handles = self.ax.plot(np.hstack(X), np.hstack(Y), **kwargs)
        for h in handles:
            h.set_clip_path(self.primitive)
        return handles

    def _arc(self, *args, **kwargs):
        X_lower, Y_lower = [], []
        X_upper, Y_upper = [], []
        antipodal = any([type(arg) is Vector3 for arg in args])
        u_kwargs = kwargs.copy()
        u_kwargs["ls"] = "--"
        u_kwargs["label"] = "_upper"
        for arg1, arg2 in zip(args[:-1], args[1:]):
            steps = max(2, int(arg1.angle(arg2)))
            # plot on lower
            x_lower, y_lower, x_upper, y_upper = self.proj.project_data_antipodal(
                *np.array(
                    [np.asarray(arg1.slerp(arg2, t)) for t in np.linspace(0, 1, steps)]
                ).T
            )
            X_lower.append(np.hstack((x_lower, np.nan)))
            Y_lower.append(np.hstack((y_lower, np.nan)))
            X_upper.append(np.hstack((x_upper, np.nan)))
            Y_upper.append(np.hstack((y_upper, np.nan)))
        handles = self.ax.plot(np.hstack(X_lower), np.hstack(Y_lower), **kwargs)
        for h in handles:
            h.set_clip_path(self.primitive)
        if antipodal:
            u_kwargs["color"] = h.get_color()
            handles_2 = self.ax.plot(np.hstack(X_upper), np.hstack(Y_upper), **u_kwargs)
            for h in handles_2:
                h.set_clip_path(self.primitive)
        return handles

    def _scatter(self, *args, **kwargs):
        legend = kwargs.pop("legend")
        num = kwargs.pop("num")
        x_lower, y_lower = self.proj.project_data(*np.vstack(args).T)
        # mask_lower = ~np.isnan(x_lower)
        x_upper, y_upper = self.proj.project_data(*(-np.vstack(args).T))
        # mask_upper = ~np.isnan(x_upper)
        # x_lower, y_lower, x_upper, y_upper = self.proj.project_data_antipodal(
        #    *np.vstack(args).T
        # )
        prop = "sizes"
        if kwargs["s"] is not None:
            s = np.atleast_1d(kwargs["s"])
            # kwargs["s"] = np.hstack((s[mask_lower], s[mask_upper]))
            kwargs["s"] = np.hstack((s, s))
        if kwargs["c"] is not None:
            c = np.atleast_1d(kwargs["c"])
            # kwargs["c"] = np.hstack((c[mask_lower], c[mask_upper]))
            kwargs["c"] = np.hstack((c, c))
            prop = "colors"
        sc = self.ax.scatter(
            # np.hstack((x_lower[mask_lower], x_upper[mask_upper])),
            # np.hstack((y_lower[mask_lower], y_upper[mask_upper])),
            # **kwargs,
            np.hstack((x_lower, x_upper)),
            np.hstack((y_lower, y_upper)),
            **kwargs,
        )
        if legend:
            self.ax.legend(
                *sc.legend_elements(prop, num=num),
                bbox_to_anchor=(1.05, 1),
                prop={"size": 11},
                loc="upper left",
                borderaxespad=0,
            )
        sc.set_clip_path(self.primitive)

    # def _cone(self, *args, **kwargs):
    #     X, Y = [], []
    #     # get scalar arguments from kwargs
    #     angles = kwargs.pop("angle")
    #     for axis, angle in zip(np.vstack(args), angles):
    #         if self.proj.rotate_data:
    #             lt = axis.transform(self.proj.R)
    #             azi, dip = Vector3(lt).geo
    #             cl_lower = Vector3(azi, dip + angle).transform(self.proj.Ri)
    #             cl_upper = -Vector3(azi, dip - angle).transform(self.proj.Ri)
    #         else:
    #             lt = axis
    #             azi, dip = Vector3(lt).geo
    #             cl_lower = Vector3(azi, dip + angle)
    #             cl_upper = -Vector3(azi, dip - angle)
    #         # plot on lower
    #         x, y = self.proj.project_data(
    #             *np.array([cl_lower.rotate(lt, a) for a in self.angles_sc]).T
    #         )
    #         X.append(np.hstack((x, np.nan)))
    #         Y.append(np.hstack((y, np.nan)))
    #         # plot on upper
    #         x, y = self.proj.project_data(
    #             *np.array([cl_upper.rotate(-lt, a) for a in self.angles_sc]).T
    #         )
    #         X.append(np.hstack((x, np.nan)))
    #         Y.append(np.hstack((y, np.nan)))
    #     handles = self.ax.plot(np.hstack(X), np.hstack(Y), **kwargs)
    #     for h in handles:
    #         h.set_clip_path(self.primitive)
    #     return handles

    def _cone(self, *args, **kwargs):
        X, Y = [], []
        # get scalar arguments from kwargs
        for arg in args:
            if issubclass(type(arg), Cone):
                cones = [arg]
            else:
                cones = arg
            for c in cones:
                # plot on lower
                angles = np.linspace(0, c.revangle, max(2, abs(int(c.revangle))))
                x, y = self.proj.project_data(
                    *np.array(
                        [np.asarray(c.secant.rotate(c.axis, a)) for a in angles]
                    ).T
                )
                X.append(np.hstack((x, np.nan)))
                Y.append(np.hstack((y, np.nan)))
                # plot on upper
                x, y = self.proj.project_data(
                    *np.array(
                        [-np.asarray(c.secant.rotate(c.axis, a)) for a in angles]
                    ).T
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
        quiver_kwargs = apsg_conf["stereonet_default_arrow_kwargs"]
        quiver_kwargs["pivot"] = "tail"
        quiver_kwargs["color"] = h[0].get_color()
        for arg in args:
            self._arrow(arg.lin, sense=arg.sense, **quiver_kwargs)

    def _hoeppner(self, *args, **kwargs):
        h = self._line(*[arg.fol for arg in args], **kwargs)
        quiver_kwargs = apsg_conf["stereonet_default_arrow_kwargs"]
        quiver_kwargs["color"] = h[0].get_color()
        for arg in args:
            self._arrow(arg.fol, arg.lin, sense=arg.sense, **quiver_kwargs)

    def _arrow(self, *args, **kwargs):
        sense = kwargs.pop("sense") * np.ones(
            np.atleast_2d(np.asarray(args[0])).shape[0]
        )
        x_lower, y_lower = self.proj.project_data(
            *np.vstack(np.atleast_2d(np.asarray(args[0]))).T
        )
        x_upper, y_upper = self.proj.project_data(
            *(-np.vstack(np.atleast_2d(np.asarray(args[0]))).T)
        )
        x = np.hstack((x_lower, x_upper))
        y = np.hstack((y_lower, y_upper))
        sense = np.hstack((sense, sense))
        inside = ~np.isnan(x)
        x = x[inside]
        y = y[inside]
        sense = sense[inside]
        if len(args) > 1:
            x_lower, y_lower = self.proj.project_data(
                *np.vstack(np.atleast_2d(np.asarray(args[1]))).T
            )
            x_upper, y_upper = self.proj.project_data(
                *(-np.vstack(np.atleast_2d(np.asarray(args[1]))).T)
            )
            dx = np.hstack((x_lower, x_upper))
            dy = np.hstack((y_lower, y_upper))
            dx = dx[~np.isnan(dx)]
            dy = dy[~np.isnan(dy)]
        else:
            dx, dy = x, y
        mag = np.hypot(dx, dy)
        u, v = sense * dx / mag, sense * dy / mag
        h = self.ax.quiver(x, y, u, v, **kwargs)
        h.set_clip_path(self.primitive)

    def _tensor(self, *args, **kwargs):
        if kwargs.get("planes"):
            selkw = {
                key: kwargs[key]
                for key in kwargs.keys() & {"alpha", "ls", "lw", "label"}
            }
            fols = args[0].eigenfols
            if kwargs["color"] is None:
                del kwargs["color"]
            self._great_circle(fols[0], color=kwargs.get("color", "red"), **selkw)
            self._great_circle(fols[1], color=kwargs.get("color", "green"), **selkw)
            self._great_circle(fols[2], color=kwargs.get("color", "blue"), **selkw)
        else:
            selkw = {
                key: kwargs[key]
                for key in kwargs.keys() & {"alpha", "marker", "mew", "ms", "label"}
            }
            lins = args[0].eigenfols
            if kwargs["color"] is None:
                del kwargs["color"]
            self._line(lins[0], color=kwargs.get("color", "red"), **selkw)
            self._line(lins[1], color=kwargs.get("color", "green"), **selkw)
            self._line(lins[2], color=kwargs.get("color", "blue"), **selkw)

    def _contour(self, *args, **kwargs):
        sigma = kwargs.pop("sigma")
        trimzero = kwargs.pop("trimzero")
        sigmanorm = (kwargs.pop("sigmanorm"),)
        colorbar = kwargs.pop("colorbar")
        _ = kwargs.pop("label")
        clines = kwargs.pop("clines")
        linewidths = kwargs.pop("linewidths")
        linestyles = kwargs.pop("linestyles")
        show_data = kwargs.pop("show_data")
        data_kws = kwargs.pop("data_kws")
        if not self.grid.calculated:
            if len(args) > 0:
                self.grid.calculate_density(
                    args[0], sigma=sigma, sigmanorm=sigmanorm, trimzero=trimzero
                )
            else:
                return None
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
        if show_data:
            artist = StereoNetArtistFactory.create_point(*args[0], **data_kws)
            self._line(*artist.args, **artist.kwargs)
        if colorbar:
            self.fig.colorbar(cf, ax=self.ax, shrink=0.5, anchor=(0.0, 0.3))
        # plt.colorbar(cf, format="%3.2f", spacing="proportional")


def stereonetartist_from_json(obj_json):
    args = tuple([feature_from_json(arg_json) for arg_json in obj_json["args"]])
    return getattr(StereoNetArtistFactory, obj_json["factory"])(
        *args, **obj_json["kwargs"]
    )


def quicknet(*args, **kwargs):
    """
    Function to quickly show or save ``StereoNet`` from args

    Args:
        args: object(s) to be plotted. Instaces of ``Vector3``, ``Foliation``,
            ``Lineation``, ``Pair``, ``Fault``, ``Cone``, ``Vector3Set``,
            ``FoliationSet``, ``LineationSet``, ``PairSet`` or ``FaultSet``.

    Keyword Args:
        savefig (bool): True to save figure. Default `False`
        filename (str): filename for figure. Default `stereonet.png`
        savefig_kwargs (dict): dict passed to ``plt.savefig``
        fol_as_pole (bool): True to plot planar features as poles,
            False for plotting as great circle. Default `False`

    Example:
        >>> l = linset.random_fisher(position=lin(120, 50))
        >>> f = folset.random_fisher(position=lin(300, 40))
        >>> quicknet(f, l)
    """
    savefig = kwargs.get("savefig", False)
    filename = kwargs.get("filename", "stereonet.png")
    savefig_kwargs = kwargs.get("savefig_kwargs", {})
    fol_as_pole = kwargs.get("fol_as_pole", False)
    label = kwargs.get("label", "_nolegend_")
    s = StereoNet(**kwargs)
    for arg in args:
        if isinstance(arg, Vector3):
            if isinstance(arg, Foliation):
                if fol_as_pole:
                    s.pole(arg, label=label)
                else:
                    s.great_circle(arg, label=label)
            elif isinstance(arg, Lineation):
                s.line(arg, label=label)
            else:
                s.vector(arg, label=label)
        elif isinstance(arg, Fault):
            s.fault(arg, label=label)
        elif isinstance(arg, Pair):
            s.pair(arg, label=label)
        elif isinstance(arg, Cone):
            s.cone(arg, label=label)
        elif isinstance(arg, Vector3Set):
            if isinstance(arg, FoliationSet):
                if fol_as_pole:
                    s.pole(arg, label=label)
                else:
                    s.great_circle(arg, label=label)
            elif isinstance(arg, LineationSet):
                s.line(arg, label=label)
            else:
                s.vector(arg, label=label)
        elif isinstance(arg, FaultSet):
            s.fault(arg, label=label)
        elif isinstance(arg, PairSet):
            s.pair(arg, label=label)
        else:
            print(f"{type(arg)} not supported.")
    if savefig:
        s.savefig(filename, **savefig_kwargs)
    else:
        s.show()
