import pickle

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import vonmises, circmean

from apsg.config import apsg_conf
from apsg.plotting._plot_artists import RosePlotArtistFactory
from apsg.feature import feature_from_json

__all__ = ["RosePlot"]


class RosePlot(object):

    """
    ``RosePlot`` class for rose histogram plotting.

    Keyword Args:
        title (str): figure title. Default None
        title_kws (dict): dictionary of keyword arguments passed to matplotlib suptitle
            method.
        bins (int): Number of bins. Default 36
        axial (bool): Directional data are axial. Defaut True
        density (bool): Use density instead of counts. Default False
        pdf (bool): Plot Von Mises density function instead histogram. Default False
        pdf_res (int): Resolution of pdf. Default 901
        kappa (float): Shape parameter of Von Mises pdf. Default 250
        scaled (bool): Bins scaled by area instead value. Default False
        ticks (bool): show ticks. Default True
        grid (bool): show grid lines. Default False
        grid_kws (dict): Dict passed to Axes.grid. Default {}

        Other keyword arguments are passed to matplotlib plot.

    Examples:
        >>> v = vec2set.random_vonmises(position=120)
        >>> p = RosePlot(grid=False)
        >>> p.pdf(v)
        >>> p.bar(v, fc='none', ec='k', lw=1)
        >>> p.muci(v)
        >>> p.show()
    """

    def __init__(self, **kwargs):
        self._kwargs = apsg_conf["roseplot_default_kwargs"].copy()
        self._kwargs.update((k, kwargs[k]) for k in self._kwargs.keys() & kwargs.keys())
        self._artists = []

    def clear(self):
        """Clear plot"""
        self._artists = []

    def _draw_layout(self):
        # self.ax.format_coord = self.format_coord
        self.ax.set_theta_direction(-1)
        self.ax.set_theta_zero_location("N")
        self.ax.grid(self._kwargs["grid"], **self._kwargs["grid_kws"])

    def _plot_artists(self):
        for artist in self._artists:
            plot_method = getattr(self, artist.roseplot_method)
            plot_method(*artist.args, **artist.kwargs)

    def to_json(self):
        """Return rose plot as JSON dict"""
        artists = [artist.to_json() for artist in self._artists]
        return dict(kwargs=self._kwargs, artists=artists)

    @classmethod
    def from_json(cls, json_dict):
        """Create rose plot from JSON dict"""
        s = cls(**json_dict["kwargs"])
        s._artists = [roseartist_from_json(artist) for artist in json_dict["artists"]]
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
            self.fig.canvas.manager.set_window_title("Rose diagram")

    def _render(self):
        self.ax = self.fig.add_subplot(111, polar=True)
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
        """Show rose plot"""
        plt.close(0)  # close previously rendered figure
        self.init_figure()
        self._render()
        plt.show()

    def savefig(self, filename="roseplot.png", **kwargs):
        """
        Save rose plot figure to graphics file

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

    def bar(self, *args, **kwargs):
        """
        Plot rose histogram of angles

        Args:
            Vector2Set feature(s)

        Keyword Args:
            alpha (scalar): Set the alpha value. Default None
            color (color): Set the color of the point. Default None
            ec (color): Patch edge color. Default None
            fc (color): Patch face color. Default None
            ls (str): Line style string (only for multiple features).
                Default "-"
            lw (float): Set line width. Default 1.5
            legend (bool): Whether to show legend. Default False

        """
        try:
            artist = RosePlotArtistFactory.create_bar(*args, **kwargs)
            self._artists.append(artist)
        except TypeError as err:
            print(err)

    def pdf(self, *args, **kwargs):
        """
        Plot Von Mises probability density function from angles

        Args:
            Vector2Set feature(s)

        Keyword Args:
            alpha (scalar): Set the alpha value. Default None
            color (color): Set the color of the point. Default None
            ec (color): Patch edge color. Default None
            fc (color): Patch face color. Default None
            ls (str): Line style string (only for multiple features).
                Default "-"
            lw (float): Set line width. Default 1.5
            legend (bool): Whether to show legend. Default False
            weight (float): Factor to scale probability density function
                Default 1

        """
        try:
            artist = RosePlotArtistFactory.create_pdf(*args, **kwargs)
            self._artists.append(artist)
        except TypeError as err:
            print(err)

    def muci(self, *args, **kwargs):
        """
        Plot circular mean with bootstrapped confidence interval

        Args:
            Vector2Set feature(s)

        Keyword Args:
            alpha (scalar): Set the alpha value. Default None
            color (color): Set the color of the point. Default None
            ls (str): Line style string (only for multiple features).
                Default "-"
            lw (float): Set line width. Default 1.5
            confidence_level (float): Confidence interval. Default 95
            n_resamples (int): Number of bootstrapped samples.
                Default 9999

        """
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
        weight = kwargs.pop("weight")
        theta = np.linspace(-np.pi, np.pi, self._kwargs["pdf_res"])
        for arg in args:
            ang = arg.direction % 360
            # weights = abs(arg)
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
            radii *= weight
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
        n_resamples = kwargs.pop("n_resamples")
        # calculate mean and CI
        if self._kwargs["axial"]:
            mu = circmean(2 * ang) / 2
            ang_shift = ang + np.pi / 2 - mu
            bsmu = [
                circmean(np.random.choice(2 * ang_shift, size=len(ang_shift)))
                for i in range(n_resamples)
            ]
            low = np.percentile(bsmu, 100 - conflevel) / 2 + mu - np.pi / 2
            high = np.percentile(bsmu, conflevel) / 2 + mu - np.pi / 2
        else:
            mu = circmean(ang)
            ang_shift = ang + np.pi - mu
            bsmu = [
                circmean(np.random.choice(ang_shift, size=len(ang_shift)))
                for i in range(n_resamples)
            ]
            low = np.percentile(bsmu, (100 - conflevel) / 2) + mu - np.pi
            high = np.percentile(bsmu, 100 - (100 - conflevel) / 2) + mu - np.pi
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


def roseartist_from_json(obj_json):
    args = tuple([feature_from_json(arg_json) for arg_json in obj_json["args"]])
    return getattr(RosePlotArtistFactory, obj_json["factory"])(
        *args, **obj_json["kwargs"]
    )
