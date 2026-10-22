import pickle
import warnings

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import circmean

from apsg.config import apsg_conf
from apsg.feature import feature_from_json
from apsg.feature._distributions import vonmises_kde
from apsg.math._vector import Axial2
from apsg.plotting._layout import FigureLayout, check_legend_position
from apsg.plotting._plot_artists import RosePlotArtistFactory
from apsg.plotting._styles import RosePlotStyle

__all__ = ["RosePlot"]


class RosePlot:
    """
    ``RosePlot`` class for rose histogram plotting.

    Keyword Args:
        title (str): figure title. Default None
        title_kws (dict): dictionary of keyword arguments passed to matplotlib suptitle
            method.
        bins (int): Number of bins. Default 36
        density (bool): Use density instead of counts. Default False
        pdf (bool): Plot Von Mises density function instead histogram. Default False
        pdf_res (int): Resolution of pdf. Default 901
        kappa (float): Shape parameter of Von Mises pdf. Default 250
        scaled (bool): Bins scaled by area instead value. Default False
        ticks (bool): show ticks. Default True
        grid (bool): show grid lines. Default False
        grid_kws (dict): Dict passed to Axes.grid. Default {}
        figsize (tuple): Size of the figure without legend and title, in inches.
            The figure grows to fit them. Default (5, 5)
        legend_position (str): Side of the plot where the legend is placed, "right",
            "left", "top" or "bottom". The number of legend columns is chosen so
            the legend is not longer than the plot. Default "right"
        legend_kws (dict): Extra keyword arguments passed to matplotlib's
            ``legend``, overriding apsg's defaults. With ``loc`` or
            ``bbox_to_anchor`` the legend is placed as given instead. Default {}
        **kwargs: Other keyword arguments are passed to matplotlib plot.

    Note: Roseplot is weighted by magnitude of Direction or Vector2 features

    Examples:
        >>> v = vec2set.random_vonmises(position=120)
        >>> p = RosePlot(grid=False)
        >>> p.pdf(v)
        >>> p.bar(v, fc='none', ec='k', lw=1)
        >>> p.muci(v)
        >>> p.show()
    """

    def __init__(self, **kwargs):
        self._kwargs = apsg_conf.roseplot.copy()
        apsg_conf.roseplot.override(self._kwargs, kwargs)
        check_legend_position(self._kwargs["legend_position"])
        self._artists = []

    def clear(self):
        """Clear plot"""
        self._artists = []

    def _draw_layout(self):
        # self.ax.format_coord = self.format_coord
        self.ax.set_theta_direction(-1)  # type: ignore
        self.ax.set_theta_zero_location("N")  # type: ignore
        if self._kwargs["grid"]:
            self.ax.grid(True, **self._kwargs["grid_kws"])
        else:
            # matplotlib's grid() enables the grid regardless of a false first
            # argument -- with a warning -- when grid_kws is non-empty, so it must
            # be omitted entirely to actually keep the grid off
            self.ax.grid(False)

    def _plot_artists(self):
        for artist in self._artists:
            plot_method = getattr(self, artist.roseplot_method)
            plot_method(*artist.args, **artist.kwargs)

    def to_json(self):
        """Return rose plot as JSON dict."""
        artists = [artist.to_json() for artist in self._artists]
        return {"kwargs": self._kwargs, "artists": artists}

    @classmethod
    def from_json(cls, json_dict):
        """Create rose plot from JSON dict."""
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
            figsize=self._kwargs["figsize"],
            dpi=apsg_conf.dpi,
            facecolor=apsg_conf.facecolor,
            layout="constrained",
        )
        if hasattr(self.fig.canvas.manager, "set_window_title"):
            self.fig.canvas.manager.set_window_title("Rose diagram")

    def _render(self, owned=False):
        self.ax = self.fig.add_subplot(111, polar=True)
        self._draw_layout()
        self._plot_artists()
        h, lbls = self.ax.get_legend_handles_labels()
        self._layout = FigureLayout(
            self.fig,
            self.ax,
            h,
            lbls,
            position=self._kwargs["legend_position"],
            legend_kws=self._kwargs["legend_kws"],
            title=self._kwargs["title"],
            title_kws=self._kwargs["title_kws"],
        )
        self._layout.apply(owned=owned)

    def render2fig(self, fig):
        """
        Plot stereonet to already existing figure or subfigure

        Args:
            fig (Figure): A mtplotlib Figure artist
        """
        self.fig = fig
        self._render()

    def show(self):
        """Show rose plot."""
        if hasattr(self, "fig"):
            plt.close(self.fig)  # close previously rendered figure
        self.init_figure()
        try:
            self._render(owned=True)
        except Exception:
            plt.close(self.fig)  # don't leak the figure of a failed render
            raise
        self._layout.connect_resize()
        plt.show()

    def savefig(self, filename="roseplot.png", **kwargs):
        """
        Save rose plot figure to graphics file

        Keyword Args:
            filename (str): filename

        All others kwargs are passed to matplotlib `Figure.savefig`
        """
        if hasattr(self, "fig"):
            plt.close(self.fig)  # close previously rendered figure
        self.init_figure()
        try:
            self._render(owned=True)
        except Exception:
            plt.close(self.fig)  # don't leak the figure of a failed render
            raise
        self.fig.savefig(filename, **kwargs)
        plt.close(self.fig)  # don't leave open for Jupyter's auto-display

    ########################################
    # STYLED PLOTTING                      #
    ########################################

    def plot(self, style, *args):
        """
        Plot features using apsg styles

        Args:
            style: apsg plotting style. See roseplot_styles
            *args: any number of features to be plotted

        Note:
            Features in args are automatically filtered by style to accept only compatible features

        """
        if not isinstance(style, RosePlotStyle):
            raise TypeError("Style must be RosePlotStyle object")

        artist = style.create_artist(*args)
        if len(artist.args) > 0:
            self._artists.append(artist)

    def _add_artist(self, factory_method, *args, **kwargs):
        try:
            self._artists.append(factory_method(*args, **kwargs))
        except TypeError as err:
            warnings.warn(str(err), stacklevel=3)

    ########################################
    # PLOTTING METHODS                     #
    ########################################

    def bar(self, *args, **kwargs):
        """
        Plot rose histogram of angles

        Args:
            *args: Vector2Set feature(s)

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
        self._add_artist(RosePlotArtistFactory.create_bar, *args, **kwargs)

    def pdf(self, *args, **kwargs):
        """
        Plot Von Mises probability density function from angles

        Args:
            *args: Vector2Set feature(s)

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
        self._add_artist(RosePlotArtistFactory.create_pdf, *args, **kwargs)

    def density(self, *args, **kwargs):
        """
        Plot the density of circular distributions

        Args:
            *args: circular distribution(s), e.g. ``VonMises.fit(data)``

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
        self._add_artist(RosePlotArtistFactory.create_density, *args, **kwargs)

    def muci(self, *args, **kwargs):
        """
        Plot circular mean with bootstrapped confidence interval

        Args:
            *args: Vector2Set feature(s)

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
        self._add_artist(RosePlotArtistFactory.create_muci, *args, **kwargs)

    ########################################
    # PLOTTING ROUTINES                    #
    ########################################

    def _bar(self, *args, **kwargs):
        bottom = np.zeros_like(self._kwargs["bins"])
        width = 2 * np.pi / self._kwargs["bins"]
        legend = kwargs.pop("legend")
        for arg in args:
            if issubclass(arg.__feature_class__, Axial2):
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
            weights = len(weights) * weights / sum(weights)
            axial = issubclass(arg.__feature_class__, Axial2)
            radii = vonmises_kde(
                theta,
                np.radians(ang),
                self._kwargs["kappa"],
                axial=axial,
                weights=weights if axial else None,
            )
            if self._kwargs["scaled"]:
                radii = np.sqrt(radii)
            if legend:
                kwargs["label"] = arg.label()
                self.ax.fill_between(theta, bottom + radii, y2=bottom, **kwargs)
            else:
                self.ax.fill_between(theta, bottom + radii, y2=bottom, **kwargs)
            bottom = bottom + radii

    def _density(self, *args, **kwargs):
        bottom = np.zeros(self._kwargs["pdf_res"])
        legend = kwargs.pop("legend")
        theta = np.linspace(-np.pi, np.pi, self._kwargs["pdf_res"])
        X = np.column_stack([np.cos(theta), np.sin(theta)])
        for dist in args:
            radii = dist.pdf(X)
            if self._kwargs["scaled"]:
                radii = np.sqrt(radii)
            if legend:
                kwargs["label"] = type(dist).__name__
            self.ax.fill_between(theta, bottom + radii, y2=bottom, **kwargs)
            bottom = bottom + radii

    def _muci(self, *args, **kwargs):
        conflevel = kwargs.pop("confidence_level")
        n_resamples = kwargs.pop("n_resamples")
        for arg in args:
            radii = []
            ang = np.radians(arg.direction)
            axial = issubclass(arg.__feature_class__, Axial2)
            if axial:
                mu = circmean(2 * ang) / 2
                ang_shift = ang + np.pi / 2 - mu
                bsmu = [
                    circmean(np.random.choice(2 * ang_shift, size=len(ang_shift)))
                    for i in range(n_resamples)
                ]
                low = np.percentile(bsmu, (100 - conflevel) / 2) / 2 + mu - np.pi / 2
                high = (
                    np.percentile(bsmu, 100 - (100 - conflevel) / 2) / 2
                    + mu
                    - np.pi / 2
                )
            else:
                mu = circmean(ang)
                ang_shift = ang + np.pi - mu
                bsmu = [
                    circmean(np.random.choice(ang_shift, size=len(ang_shift)))
                    for i in range(n_resamples)
                ]
                low = np.percentile(bsmu, (100 - conflevel) / 2) + mu - np.pi
                high = np.percentile(bsmu, 100 - (100 - conflevel) / 2) + mu - np.pi
            p = vonmises_kde(mu, ang, self._kwargs["kappa"], axial=axial)[0]
            radii.append(p)
            if self._kwargs["scaled"]:
                radii = np.sqrt(radii)
            mur = 1.1 * sum(radii)
            ci_angles = np.linspace(low, high, int(5 * np.degrees(high - low)))
            if issubclass(arg.__feature_class__, Axial2):
                self.ax.plot([mu, mu + np.pi], [mur, mur], **kwargs)
                self.ax.plot(ci_angles, mur * np.ones_like(ci_angles), **kwargs)
                self.ax.plot(ci_angles + np.pi, mur * np.ones_like(ci_angles), **kwargs)
            else:
                self.ax.plot([0, mu], [0, mur], **kwargs)
                self.ax.plot(ci_angles, mur * np.ones_like(ci_angles), **kwargs)


def roseartist_from_json(obj_json):
    args = tuple([feature_from_json(arg_json) for arg_json in obj_json["args"]])
    return getattr(RosePlotArtistFactory, obj_json["factory"])(
        *args, **obj_json["kwargs"]
    )
