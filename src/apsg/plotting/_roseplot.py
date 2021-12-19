import sys
import pickle

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import vonmises, circmean, bootstrap

from apsg.config import apsg_conf
from apsg.math._vector import Vector2
from apsg.feature._container import Vector2Set
from apsg.plotting._plot_artists import RosePlotArtistFactory

__all__ = ["RosePlot"]


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

    def __init__(self, **kwargs):
        self._kwargs = apsg_conf["roseplot_default_kwargs"].copy()
        self._kwargs.update((k, kwargs[k]) for k in self._kwargs.keys() & kwargs.keys())
        self._artists = []

    def clear(self):
        self._artists = []

    def _draw_layout(self):
        self.fig = plt.figure(figsize=apsg_conf["figsize"])
        self.ax = self.fig.add_subplot(111, polar=True)
        # self.ax.format_coord = self.format_coord
        self.ax.set_theta_direction(-1)
        self.ax.set_theta_zero_location("N")
        self.ax.grid(self._kwargs["grid"], **self._kwargs["grid_kw"])

    def _plot_artists(self):
        for artist in self._artists:
            plot_method = getattr(self, artist.roseplot_method)
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
                roseplot_method=artist.roseplot_method,
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
                getattr(RosePlotArtistFactory, artist["factory"])(
                    *args, **artist["kwargs"]
                )
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
            self.fig.suptitle(self._kwargs["title"])
        self.fig.tight_layout()

    def show(self):
        self.render()
        plt.show()

    def savefig(self, filename="roseplot.png", **kwargs):
        self.render()
        self.fig.savefig(filename, **kwargs)
        plt.close()
        delattr(self, "ax")
        delattr(self, "fig")

    ########################################
    # PLOTTING METHODS                     #
    ########################################

    def bar(self, *args, **kwargs):
        """Plot rose histogram of angles"""
        try:
            artist = RosePlotArtistFactory.create_bar(*args, **kwargs)
            self._artists.append(artist)
        except TypeError as err:
            print(err)

    def pdf(self, *args, **kwargs):
        """Plot rose histogram of angles"""
        try:
            artist = RosePlotArtistFactory.create_pdf(*args, **kwargs)
            self._artists.append(artist)
        except TypeError as err:
            print(err)

    def muci(self, *args, **kwargs):
        """Plot circular mean with confidence interval"""
        try:
            artist = RosePlotArtistFactory.create_muci(*args, **kwargs)
            self._artists.append(artist)
        except TypeError as err:
            print(err)

    ########################################
    # PLOTTING ROUTINES                    #
    ########################################

    def _bar(self, *args, **kwargs):
        bottom = np.zeros_like(self._kwargs["bins"])
        width = 2 * np.pi / self._kwargs["bins"]
        legend = kwargs.pop("legend")
        for arg in args:
            if self._kwargs["axial"]:
                ang = np.concatenate((arg.direction % 360, (arg.direction + 180) % 360))
                weights = np.concatenate((abs(arg), abs(arg)))
            else:
                ang = arg.direction % 360
                weights = abs(arg)
            num, bin_edges = np.histogram(
                np.radians(ang),
                bins=self._kwargs["bins"] + 1,
                range=(-width / 2, 2 * np.pi + width / 2),
                weights=weights,
                density=self._kwargs["density"],
            )
            num[0] += num[-1]
            num = num[:-1]
            bin_centre = (bin_edges[1:-1] + bin_edges[:-2]) / 2
            if self._kwargs["scaled"]:
                num = np.sqrt(num)
            if legend:
                kwargs["label"] = arg.label()
                self.ax.bar(bin_centre, num, width=width, bottom=bottom, **kwargs)
            else:
                self.ax.bar(bin_centre, num, width=width, bottom=bottom, **kwargs)
            bottom = bottom + num

    def _pdf(self, *args, **kwargs):
        bottom = np.zeros_like(self._kwargs["pdf_res"])
        legend = kwargs.pop("legend")
        theta = np.linspace(-np.pi, np.pi, self._kwargs["pdf_res"])
        for arg in args:
            ang = arg.direction % 360
            weights = abs(arg)
            radii = np.zeros_like(theta)
            if self._kwargs["axial"]:
                for a in ang:
                    radii += (
                        vonmises.pdf(theta, self._kwargs["kappa"], loc=np.radians(a))
                        / 2
                    )
                    radii += (
                        vonmises.pdf(
                            theta, self._kwargs["kappa"], loc=np.radians(a + 180)
                        )
                        / 2
                    )
            else:
                for a in ang:
                    radii += vonmises.pdf(
                        theta, self._kwargs["kappa"], loc=np.radians(a)
                    )
            radii /= len(ang)
            if self._kwargs["scaled"]:
                radii = np.sqrt(radii)
            if legend:
                kwargs["label"] = arg.label()
                self.ax.fill_between(theta, bottom + radii, y2=bottom, **kwargs)
            else:
                self.ax.fill_between(theta, bottom + radii, y2=bottom, **kwargs)
            bottom = bottom + radii

    def _muci(self, *args, **kwargs):
        ang = np.radians(np.concatenate([arg.direction for arg in args]))
        conflevel = kwargs.pop("confidence_level")
        # calculate mean and CI
        if self._kwargs["axial"]:
            mu = circmean(2 * ang) / 2
            ang_shift = ang + np.pi / 2 - mu
            bci = bootstrap(
                (2 * ang_shift,), circmean, confidence_level=conflevel
            ).confidence_interval
            low = bci.low / 2 + mu - np.pi / 2
            high = bci.high / 2 + mu - np.pi / 2
        else:
            mu = circmean(ang)
            ang_shift = ang + np.pi - mu
            bci = bootstrap(
                (ang_shift,), circmean, confidence_level=conflevel
            ).confidence_interval
            low = bci.low + mu - np.pi
            high = bci.high + mu - np.pi
        radii = []
        for arg in args:
            p = 0
            if self._kwargs["axial"]:
                for a in arg.direction:
                    p += vonmises.pdf(mu, self._kwargs["kappa"], loc=np.radians(a)) / 2
                    p += (
                        vonmises.pdf(mu, self._kwargs["kappa"], loc=np.radians(a + 180))
                        / 2
                    )
            else:
                for a in arg.direction:
                    p += vonmises.pdf(mu, self._kwargs["kappa"], loc=np.radians(a))
            radii.append(p / len(arg))
        if self._kwargs["scaled"]:
            radii = np.sqrt(radii)
        mur = 1.1 * sum(radii)
        ci_angles = np.linspace(low, high, int(5 * np.degrees(high - low)))
        self.ax.plot([mu, mu + np.pi], [mur, mur], **kwargs)
        self.ax.plot(ci_angles, mur * np.ones_like(ci_angles), **kwargs)
        self.ax.plot(ci_angles + np.pi, mur * np.ones_like(ci_angles), **kwargs)
