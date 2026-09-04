# -*- coding: utf-8 -*-

import pickle

import matplotlib.pyplot as plt
import numpy as np

from apsg.config import apsg_conf
from apsg.feature import feature_from_json
from apsg.feature._container import (
    ArcSet,
    ConeSet,
    EllipsoidSet,
    FaultSet,
    FoliationSet,
    LineationSet,
    PairSet,
    Stress3Set,
    Vector3Set,
)
from apsg.feature._geodata import Arc, Cone, Fault, Foliation, Lineation, Pair
from apsg.feature._tensor3 import Stress3, Tensor3
from apsg.math._vector import Vector3
from apsg.plotting._plot_artists import StereoNetArtistFactory
from apsg.plotting._stereo_engine import rotation_from_axis_angle
from apsg.plotting._stereogrid import StereoGrid
from apsg.plotting._styles import StereoNetStyle

__all__ = ["StereoNet", "quicknet", "rotation_from_axis_angle"]


def _kind_to_projection(kind):
    """Resolve a ``kind`` config value to a registered matplotlib projection
    name ("schmidt" or "wulff")."""
    kind = str(kind).lower()
    if kind in ("equal-area", "schmidt", "earea"):
        return "schmidt"
    elif kind in ("equal-angle", "wulff", "eangle"):
        return "wulff"
    raise TypeError("Only 'Equal-area' and 'Equal-angle' implemented")


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
        rotation (Rotation3 or array-like): Rotation applied to the whole net.
            Default None (identity, no rotation)
        rotate_data (bool): Whether plotted data should follow `rotation` along
            with the grid. Default True
        grid (bool): Whether to show the grid. Default is True
        grid_step (float): Grid step. Default 15
        grid_color (color): Grid line color. Default "grey"
        grid_style (str): Grid line style. Default ":"
        clip_pole (float): Clipped cone around poles. Default 15
        primitive_lw (float): Line width of the primitive (outer) circle. Default 1.5
        primitive_color (color): Color of the primitive circle. Default None
            (matplotlib's own default)
        azimuth_ticks (bool): Whether to show N/E/S/W compass tick labels around
            the rim. Default False
        azimuth_ticks_kws (dict): Extra keyword arguments passed to the underlying
            ``set_azimuth_ticks``. Default {}
        legend_kws (dict): Extra keyword arguments passed to matplotlib's
            ``legend``, overriding apsg's defaults. Default {}

    Note:
        Each `contour()` call owns its own `StereoGrid` -- pass a `Vector3Set`
        to have one created and its density calculated automatically (sized
        per `apsg_conf.stereogrid`'s `type`/`n`), or pass an already-populated
        `StereoGrid` (e.g. built via `apply_func`/`angmech`) to plot it
        directly. Multiple `contour()` calls add independent layers.

    Examples:
        >>> l = linset.random_fisher(position=lin(120, 40))
        >>> s = StereoNet(title="Random linear features")
        >>> s.contour(l)
        >>> s.point(l)
        >>> s.show()
    """

    def __init__(self, **kwargs):
        self._kwargs = apsg_conf.stereonet.copy()
        self._kwargs.update((k, kwargs[k]) for k in self._kwargs.keys() & kwargs.keys())
        self._kwargs["title"] = kwargs.get("title", None)
        self._projection = _kind_to_projection(self._kwargs["kind"])
        rotation = self._kwargs["rotation"]
        self._rotation = np.eye(3) if rotation is None else np.asarray(rotation)
        self.clear()

    def clear(self):
        """Clear plot"""

        self._artists = []

    # -- rotation ---------------------------------------------------------

    @property
    def rotation(self):
        """The current 3x3 rotation matrix applied to the whole net (grid
        and, when ``rotate_data`` is True, plotted data). Set via the
        ``rotation`` constructor keyword or ``set_rotation``."""
        return self._rotation.copy()

    def set_rotation(self, matrix):
        """Set an arbitrary rotation matrix for the whole net. Accepts
        anything ``np.asarray``-coercible to a 3x3 proper rotation matrix,
        including apsg's own ``Rotation3``; use ``rotation_from_axis_angle``
        (re-exported from this module) to build one from an axis/angle.
        Pass ``None`` to reset to the identity (no rotation).

        Returns:
            None
        """
        self._rotation = np.eye(3) if matrix is None else np.asarray(matrix)
        if hasattr(self, "ax"):
            self.ax.set_rotation(self._rotation)

    def _draw_layout(self):
        self.ax.grid(
            self._kwargs["grid"],
            linestyle=self._kwargs["grid_style"],
            color=self._kwargs["grid_color"],
        )
        self.ax.set_longitude_grid(self._kwargs["grid_step"])
        self.ax.set_latitude_grid(self._kwargs["grid_step"])
        self.ax.set_clip_pole(self._kwargs["clip_pole"])
        self.ax.spines["geo"].set_linewidth(self._kwargs["primitive_lw"])
        if self._kwargs["primitive_color"] is not None:
            self.ax.spines["geo"].set_edgecolor(self._kwargs["primitive_color"])
        # the axes' own clear() already draws default N/E/S/W compass ticks
        # unconditionally; honor azimuth_ticks=False by removing them, or
        # apply azimuth_ticks_kws (angles/labels/frac/tick_frac/...) on top
        if self._kwargs["azimuth_ticks"]:
            azimuth_ticks_kws = dict(self._kwargs["azimuth_ticks_kws"])
            angles = azimuth_ticks_kws.pop("angles", [0, 90, 180, 270])
            labels = azimuth_ticks_kws.pop("labels", ["N", "E", "S", "W"])
            self.ax.set_azimuth_ticks(angles, labels, **azimuth_ticks_kws)
        else:
            self.ax._clear_azimuth_ticks()
        # keep the graticule strictly behind every plotted artist (points,
        # great circles, filled contours, ...) -- matplotlib's own default
        # otherwise draws gridlines above patch-like artists such as a
        # filled contour, letting the grid show through it
        self.ax.set_axisbelow(True)
        self.ax.patch.set_zorder(0)
        self.primitive = self.ax.patch

    def _plot_artists(self):
        for artist in self._artists:
            plot_method = getattr(self, artist.stereonet_method)
            plot_method(*artist.args, **artist.kwargs)

    def to_json(self):
        """Return stereonet as JSON dict."""

        artists = [artist.to_json() for artist in self._artists]
        return dict(kwargs=self._kwargs, artists=artists)

    @classmethod
    def from_json(cls, json_dict):
        """Create stereonet from JSON dict."""

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
        Returns:
            None: The stereonet is serialized and written to a pickle file.
        """
        with open(filename, "wb") as f:
            pickle.dump(self.to_json(), f, pickle.HIGHEST_PROTOCOL)

    @classmethod
    def load(cls, filename):
        """
        Load stereonet from pickle file

        Args:
            filename (str): name of picke file
        Returns:
            StereoNet: Loaded stereonet instance from pickle file.
        """
        with open(filename, "rb") as f:
            data = pickle.load(f)
        return cls.from_json(data)

    def init_figure(self):
        self.fig = plt.figure(
            figsize=apsg_conf.figsize,
            dpi=apsg_conf.dpi,
            facecolor=apsg_conf.facecolor,
        )
        netname = "Schmidt net" if self._projection == "schmidt" else "Wulff net"
        if hasattr(self.fig.canvas.manager, "set_window_title"):
            self.fig.canvas.manager.set_window_title(netname)

    def _render(self):
        self.ax = self.fig.add_subplot(
            projection=self._projection,
            hemisphere=self._kwargs["hemisphere"],
            rotation=self._rotation,
            rotate_data=self._kwargs["rotate_data"],
        )
        self._draw_layout()
        self._plot_artists()
        h, labels = self.ax.get_legend_handles_labels()
        if h:
            legend_kwargs = dict(
                bbox_to_anchor=(1.05, 1),
                prop={"size": 11},
                loc="upper left",
                borderaxespad=0,
                scatterpoints=1,
                numpoints=1,
            )
            legend_kwargs.update(self._kwargs["legend_kws"])
            self.ax.legend(h, labels, **legend_kwargs)
        if self._kwargs["title"] is not None:
            self.fig.suptitle(self._kwargs["title"], **self._kwargs["title_kws"])
        if self._kwargs["tight_layout"]:
            self.fig.tight_layout()

    def render2fig(self, fig):
        """
        Plot stereonet to already existing figure or subfigure

        Args:
            fig (Figure): A mtplotlib Figure artist
        Returns:
            None: The stereonet is rendered on the provided figure.
        """
        self.fig = fig
        self._render()

    def format_coord(self, x, y):
        """Format stereonet coordinates.

        Uses the same axial hemisphere convention as ``point()``/
        ``great_circle()``/``contour()`` (a 180 degree rotation for
        ``hemisphere="upper"``, via ``_hemisphere_rotate``), not the
        directional one ``vector()`` needs -- the latter is a reflection
        that flips which raw vectors land inside the primitive circle at
        all, which previously made this cursor readout blank across the
        whole net whenever ``hemisphere="upper"``. ``_data_to_axial``
        inverts ``_fold_axial_to_data`` (hemisphere and, when
        ``rotate_data`` is True, the net's own rotation) to recover the
        true feature under the cursor; ``project(..., fold=True)`` then
        re-applies that same forward transform to check membership in the
        primitive circle exactly as ``point()`` would place this vector.
        """

        if x is not None and y is not None:
            v = self.ax._data_to_axial(x, y)[0]
            X, Y = self.ax.project(v, clip_inside=False, fold=True)
            if (X - 0.5) ** 2 + (Y - 0.5) ** 2 <= 0.25:
                lcoord = Lineation(*v)
                fcoord = Foliation(*v)
                return f"{lcoord} {fcoord}"
        return ""

    def show(self):
        """Show stereonet."""

        plt.close(0)  # close previously rendered figure
        self.init_figure()
        self._render()
        self.ax.format_coord = self.format_coord  # ty: ignore
        plt.show()

    def savefig(self, filename="stereonet.png", **kwargs):
        """
        Save stereonet figure to graphics file

        Keyword Args:
            filename (str): filename

        All others kwargs are passed to matplotlib `Figure.savefig`
        Returns:
            None: The figure is saved to the specified graphics file.
        """
        plt.close(0)  # close previously rendered figure
        self.init_figure()
        self._render()
        self.fig.savefig(filename, **kwargs)
        plt.close(0)

    ########################################
    # STYLED PLOTTING                      #
    ########################################

    def plot(self, style, *args):
        """
        Plot features using apsg styles.

        Args:
            style: apsg plotting style. See stereonet_styles
            *arg: any number of features to be plotted

        Note:
            Features in args are automatically filtered by style to accept only compatible features

        Returns:
            None: Features are plotted using the provided style.
        """
        assert isinstance(style, StereoNetStyle), "Style must StereoNetStyle object"

        try:
            artist = style.create_artist(*args)
        except TypeError as err:
            print(err)
            return
        if len(artist.args) > 0:
            self._artists.append(artist)

    def _add_artist(self, factory_method, *args, **kwargs):
        try:
            self._artists.append(factory_method(*args, **kwargs))
        except TypeError as err:
            print(err)

    ########################################
    # PLOTTING METHODS                     #
    ########################################

    def point(self, *args, **kwargs):
        """
        Plot linear feature(s) or poles of planar features as point(s).

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

        Returns:
            None: Linear features or poles are plotted as points.
        """
        self._add_artist(StereoNetArtistFactory.create_point, *args, **kwargs)

    # backward compatibility
    line = pole = point

    def vector(self, *args, **kwargs):
        """
        Plot vector feature(s) as point(s).

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

        Returns:
            None: Vector features are plotted as points.
        """
        self._add_artist(StereoNetArtistFactory.create_vector, *args, **kwargs)

    def scatter(self, *args, **kwargs):
        """
        Plot vector-like feature(s) as point(s).

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

        Returns:
            None: Vector-like features are plotted as points with variable properties.
        """
        self._add_artist(StereoNetArtistFactory.create_scatter, *args, **kwargs)

    def great_circle(self, *args, **kwargs):
        """
        Plot planar feature(s) as great circle(s).

        Note: ``great_circle`` has also alias ``gc``

        Args:
            Foliation or FoliationSet feature(s)

        Keyword Args:
            alpha (scalar): Set the alpha value. Default None
            color (color): Set the color of the point. Default None
            ls (str): Line style string (only for multiple features).
                Default "-"
            lw (float): Set line width. Default 1.5

        Returns:
            None: Planar features are plotted as great circles.
        """
        self._add_artist(StereoNetArtistFactory.create_great_circle, *args, **kwargs)

    gc = great_circle

    def arc(self, *args, **kwargs):
        """
        Plot arc(s) between vectors.

        Two calling conventions are supported:

        - Pass one or more ``Arc``/``ArcSet`` instances -- each contributes its own
          independently-configured curved path (see ``Arc`` for how ``curvature``,
          ``positive`` and ``short`` control its shape).
        - Pass several raw ``Vector3`` (or ``Vector3Set``) like features in connection
          order -- consecutive pairs are connected by a plain great-circle arc
          (equivalent to ``Arc(p1, p2)`` with default ``curvature=0, positive=True,
          short=True``). This is the legacy convention and remains fully supported.

        Args:
            Arc or ArcSet instance(s), or Vector3/Vector3Set like feature(s) to
            connect pairwise in sequence.

        Keyword Args:
            kind (str): Rendering mode, "line" or "points". Default "line"
            alpha (scalar): Set the alpha value. Default None
            color (color): Set the color. Default None
            ls (str): Line style string (line mode). Default "-"
            lw (float): Set line width (line mode). Default 1.5
            marker (str): Marker style (points mode). Default "o"
            ms (int): Marker size (points mode). Default 6
            mec (color): Marker edge color (points mode). Default None
            mfc (color): Marker face color (points mode). Default None
            mew (int): Marker edge width (points mode). Default 1

        Returns:
            None: Arc(s) are plotted.
        """
        self._add_artist(StereoNetArtistFactory.create_arc, *args, **kwargs)

    def cone(self, *args, **kwargs):
        """
        Plot cone(s) as small circle(s) with given apical angle(s).

        Args:
            Cone or ConeSet feature(s)

        Keyword Args:
            alpha (scalar): Set the alpha value. Default None
            color (color): Set the color of the point. Default None
            ls (str): Line style string (only for multiple features).
                Default "-"
            lw (float): Set line width. Default 1.5

        Returns:
            None: Cones are plotted as small circles with given apical angles.
        """
        self._add_artist(StereoNetArtistFactory.create_cone, *args, **kwargs)

    def pair(self, *args, **kwargs):
        """
        Plot pair feature(s) as great circle and point.

        Args:
            Pair or PairSet feature(s)

        Keyword Args:
            alpha (scalar): Set the alpha value. Default None
            color (color): Set the color of the point. Default None
            ls (str): Line style string (only for multiple features).
                Default "-"
            lw (float): Set line width. Default 1.5
            line_marker (str): Marker style string for point. Default "o"

        Returns:
            None: Pair features are plotted as great circle and point.
        """
        self._add_artist(StereoNetArtistFactory.create_pair, *args, **kwargs)

    def fault(self, *args, **kwargs):
        """
        Plot fault feature(s) as great circle and arrow.

        Note: Arrow is styled according to default arrow config

        Args:
            Fault or FaultSet feature(s)

        Keyword Args:
            alpha (scalar): Set the alpha value. Default None
            color (color): Set the color of the point. Default None
            ls (str): Line style string (only for multiple features).
                Default "-"
            lw (float): Set line width. Default 1.5

        Returns:
            None: Fault features are plotted as great circle and arrow.
        """
        self._add_artist(StereoNetArtistFactory.create_fault, *args, **kwargs)

    def hoeppner(self, *args, **kwargs):
        """
        Plot fault feature(s) on Hoeppner (tangent lineation) plot.

        Note: Arrow is styled according to default arrow config, except its
            pivot, which is taken from apsg_conf.stereonet_hoeppner.pivot
            (default "middle")

        Args:
            Fault or FaultSet feature(s)

        Keyword Args:
            alpha (scalar): Set the alpha value. Default None
            color (color): Set the color of the point. Default None
            ls (str): Line style string (only for multiple features).
                Default "-"
            lw (float): Set line width. Default 1.5

        Returns:
            None: Fault features are plotted on Hoeppner plot.
        """
        self._add_artist(StereoNetArtistFactory.create_hoeppner, *args, **kwargs)

    def arrow(self, *args, **kwargs):
        """
        Plot arrow at position of first argument
        and oriented in direction of second.

        Note: You should pass two features

        Args:
            Vector3 or Vector3Set like feature(s)

        Keyword Args:
            color (color): Set the color of the arrow. Default None
            width (int): Width of arrow. Default 2
            headwidth (int): Width of arrow head. Default 5
            pivot (str): Arrow pivot. Default "mid"
            units (str): Arrow size units. Default "dots"

        Returns:
            None: Arrow is plotted at the specified position and direction.
        """
        self._add_artist(StereoNetArtistFactory.create_arrow, *args, **kwargs)

    def tensor(self, *args, **kwargs):
        """
        Plot principal planes or principal directions of tensor.

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

        Returns:
            None: Principal planes or directions of tensor are plotted.
        """
        self._add_artist(StereoNetArtistFactory.create_tensor, *args, **kwargs)

    def stress(self, *args, **kwargs):
        """
        Plot principal stresses of stress tensor.

        Args:
            Stress3 feature(s)

        Keyword Args:
            alpha (scalar): Set the alpha value. Default None
            color (color): Set the color. Default is red, green, blue for s1, s2, s3
            ls (str): Line style string (only for multiple features).
                Default "-"
            lw (float): Set line width. Default 1.5
            mew (float): Set the marker edge width. Default 1
            ms (float): Set the marker size. Default 12
            marker (str): Marker style string. Default "*"

        Returns:
            None: Principal stresses are plotted.
        """
        self._add_artist(StereoNetArtistFactory.create_stress, *args, **kwargs)

    def confidence(self, *args, **kwargs):
        """
        Plot confidence cone or ellipse around orientation data.

        For ``method`` in ``"fisher"``, ``"watson"`` and ``"bootstrap"`` a circular
        confidence cone (single half-angle) is plotted around the mean direction. For
        ``method="bingham"`` an elliptical confidence region (two independent semi-angles)
        is plotted around a chosen eigenvector of the orientation tensor, using the
        large-sample eigenvalue method of Fisher, Lewis & Embleton (1987). See
        ``Vector3Set.fisher_statistics``, ``Vector3Set.watson_statistics`` and
        ``Vector3Set.bingham_statistics`` for the underlying statistics; the bootstrap
        cone resamples the data with replacement and takes the ``level``-percentile of
        the angular deviation of the resampled principal eigenvectors from the sample's.

        Args:
            Vector3Set like feature(s), e.g. LineationSet or FoliationSet

        Keyword Args:
            method (str): "fisher", "bingham", "watson" or "bootstrap". Default "fisher"
            which (int): index (0, 1 or 2) of the eigenvector the ellipse is
                centered on (``method="bingham"`` only). 0 is the major eigenvector,
                2 is the minor (pole) eigenvector. Default 0.
            level (float): confidence level. Default 0.95
            n_resamples (int): number of bootstrap resamples (``method="bootstrap"``
                only). Default 1000
            alpha (scalar): Set the alpha value. Default None
            color (color): Set the color. Default None
            ls (str): Line style string. Default "--"
            lw (float): Set line width. Default 1.5

        Returns:
            None: Confidence cone or ellipse is plotted.
        """
        self._add_artist(StereoNetArtistFactory.create_confidence, *args, **kwargs)

    def contour(self, source, **kwargs):
        """
        Plot contours in multiples of uniform distribution.

        Each call owns its own ``StereoGrid``, so multiple ``contour()``
        calls on the same ``StereoNet`` add independent layers.

        Args:
            source: Vector3Set like feature -- a new ``StereoGrid`` is
                created and its density calculated immediately using
                `method`/`sigma`/`n_max` below; or an already-populated
                ``StereoGrid`` (e.g. built via ``apply_func``/``angmech``),
                plotted as-is with `method`/`sigma`/`n_max` ignored.

        Keyword Args:
            method (str): "kamb" for modified Kamb contouring technique with exponential
                smoothing or "sph" for spherical harmonics method. Default "kamb"
            levels (int or list): number or values of contours. Default 6
            cmap: matplotlib colormap. Default "Greys" when `clip` is True, or
                "RdBu" (diverging, centered on 0) when `clip` is False
            clip (bool): restrict to the positive (above-uniform) region only.
                Default True
            colorbar (bool): Show colorbar. Default False
            colorbar_kws (dict): Extra keyword arguments passed to
                ``Figure.colorbar``. Default {"shrink": 0.5, "anchor": (0.0, 0.3)}
            alpha (float): transparency. Default None
            antialiased (bool): Default True
            n_max (int): maximum harmonic degree i.e. the angular resolution. Must be
                even number (for "sph" method). Default is derived from `sigma`.
            sigma (float): controls how much to smooth, for either method. Default 3
            filled (bool): filled contours if True, contour lines if False.
                Default True
            linewidth (float): contour lines width (aliased as `lw`). Default 1
            linestyles (str): contour lines style
            line_color (color): color of the black contour-line overlay drawn on
                top of filled contours. Default "k"
        Returns:
            None: Contours in multiples of uniform distribution are plotted.
        """
        if "lw" in kwargs and "linewidth" not in kwargs:
            kwargs["linewidth"] = kwargs.pop("lw")
        artist = StereoNetArtistFactory.create_contour(source, **kwargs)
        self._artists.append(artist)

    ########################################
    # PLOTTING ROUTINES                    #
    ########################################

    def _point(self, *args, **kwargs):
        return [self.ax.point(np.vstack(args), **kwargs)]

    def _vector(self, *args, **kwargs):
        return list(self.ax.vector(np.vstack(args), **kwargs))

    def _great_circle(self, *args, **kwargs):
        return self.ax.great_circle(np.vstack(args), **kwargs)

    def _arc(self, *args, **kwargs):
        kind = kwargs.pop("kind", "line")
        if kind == "points":
            kwargs["ls"] = "none"
            kwargs.setdefault("marker", "o")
        else:
            kwargs["marker"] = "None"

        antipodal = any(type(a.p1) is Vector3 or type(a.p2) is Vector3 for a in args)
        segments = []
        for a in args:
            curve = np.array([np.asarray(v) for v in a.path()])
            segments.append(curve)
            segments.append(np.full((1, 3), np.nan))
        combined = np.vstack(segments)
        return [self.ax.path(combined, antipodal=antipodal, **kwargs)]

    def _scatter(self, *args, **kwargs):
        legend = kwargs.pop("legend")
        num = kwargs.pop("num")
        X, Y = self.ax.project(np.vstack(args), clip_inside=True, fold=True)
        if kwargs["s"] is not None:
            kwargs["s"] = np.atleast_1d(kwargs["s"])
        if kwargs["c"] is not None:
            kwargs["c"] = np.atleast_1d(kwargs["c"])
        sc = self.ax.scatter(X, Y, transform=self.ax.transAxes, **kwargs)
        if legend:
            prop = "colors" if kwargs.get("c") is not None else "sizes"
            legend_kwargs = dict(
                bbox_to_anchor=(1.05, 1),
                prop={"size": 11},
                loc="upper left",
                borderaxespad=0,
            )
            legend_kwargs.update(self._kwargs["legend_kws"])
            self.ax.legend(*sc.legend_elements(prop, num=num), **legend_kwargs)
        sc.set_clip_path(self.primitive)

    def _cone(self, *args, **kwargs):
        cones = []
        for arg in args:
            cones.extend([arg] if isinstance(arg, Cone) else list(arg))
        segments = []
        for c in cones:
            angles = np.linspace(0, c.revangle, max(2, abs(int(c.revangle))))
            curve = np.array([np.asarray(c.secant.rotate(c.axis, a)) for a in angles])
            segments.append(curve)
            segments.append(np.full((1, 3), np.nan))
            segments.append(-curve)
            segments.append(np.full((1, 3), np.nan))
        combined = np.vstack(segments)
        return [self.ax.path(combined, antipodal=False, **kwargs)]

    def _confidence(self, *args, **kwargs):
        method = kwargs.pop("method")
        which = kwargs.pop("which")
        level = kwargs.pop("level")
        n_resamples = kwargs.pop("n_resamples")
        segments = []
        for arg in args:
            if method == "bingham":
                stats = arg.bingham_statistics(level=level, which=which)
                mu = np.asarray(stats["mu"])
                u = np.asarray(stats["axes"][0])
                v = np.asarray(stats["axes"][1])
                g0, g1 = np.radians(stats["gamma"])
                theta = np.linspace(0, 2 * np.pi, 181)
                denom = np.sqrt((g1 * np.cos(theta)) ** 2 + (g0 * np.sin(theta)) ** 2)
                rho = np.divide(
                    g0 * g1, denom, out=np.zeros_like(denom), where=denom > 0
                )
                pts = np.cos(rho)[:, None] * mu + np.sin(rho)[:, None] * (
                    np.cos(theta)[:, None] * u + np.sin(theta)[:, None] * v
                )
            else:
                if method == "fisher":
                    stats = arg.fisher_statistics(level=level)
                elif method == "watson":
                    stats = arg.watson_statistics(level=level)
                elif method == "bootstrap":
                    # Bootstrap confidence cone from resampled orientation-tensor eigenvectors
                    mu = arg.ortensor().eigenvectors(0)
                    deviations = []
                    for sample in arg.bootstrap(n=n_resamples):
                        v = sample.ortensor().eigenvectors(0)
                        if v.dot(mu) < 0:
                            v = -v
                        deviations.append(mu.angle(v))
                    alpha = float(np.percentile(deviations, 100 * level))
                    stats = {"mu": mu, "alpha": alpha}
                else:
                    raise ValueError(f"Unknown confidence method {method!r}")
                azi, inc = stats["mu"].geo
                secant = Vector3(azi, inc + stats["alpha"])
                angles = np.linspace(0, 360, 360)
                pts = np.array(
                    [np.asarray(secant.rotate(stats["mu"], a)) for a in angles]
                )
            segments.append(pts)
            segments.append(np.full((1, 3), np.nan))
            segments.append(-pts)
            segments.append(np.full((1, 3), np.nan))
        combined = np.vstack(segments)
        return [self.ax.path(combined, antipodal=False, **kwargs)]

    def _pair(self, *args, **kwargs):
        line_marker = kwargs.pop("line_marker")
        h = self._great_circle(*[arg.fol for arg in args], **kwargs)
        self._point(
            *[arg.lin for arg in args],
            marker=line_marker,
            ls="none",
            color=h[0].get_color(),
            mfc=h[0].get_color(),
            mec=h[0].get_color(),
            ms=kwargs.get("ms"),
        )

    def _fault(self, *args, **kwargs):
        h = self._great_circle(*[arg.fol for arg in args], **kwargs)
        quiver_kwargs = apsg_conf.stereonet_arrow.copy()
        quiver_kwargs["pivot"] = "tail"
        quiver_kwargs["color"] = h[0].get_color()
        for arg in args:
            self._arrow(arg.lin, sense=arg.sense, **quiver_kwargs)

    def _hoeppner(self, *args, **kwargs):
        pivot = kwargs.pop("pivot")
        h = self._point(*[arg.fol for arg in args], **kwargs)
        quiver_kwargs = apsg_conf.stereonet_arrow.copy()
        quiver_kwargs["pivot"] = pivot
        quiver_kwargs["color"] = h[0].get_color()
        for arg in args:
            self._arrow(arg.fol, arg.lin, sense=arg.sense, **quiver_kwargs)

    def _arrow(self, *args, **kwargs):
        sense = kwargs.pop("sense") * np.ones(
            np.atleast_2d(np.asarray(args[0])).shape[0]
        )
        x, y = self.ax.project(np.asarray(args[0]), clip_inside=True, fold=True)
        inside = ~np.isnan(x)
        x, y, sense = x[inside], y[inside], sense[inside]
        if len(args) > 1:
            dx, dy = self.ax.project(np.asarray(args[1]), clip_inside=True, fold=True)
            dx, dy = dx[~np.isnan(dx)], dy[~np.isnan(dy)]
        else:
            dx, dy = x, y
        # dx, dy are axes-fraction coordinates, so they must be re-centered
        # on the net's true-vertical reference point (the disk center
        # (0.5, 0.5) only when the net is unrotated -- see
        # ``vertical_axes_fraction``) before use as a direction.
        cx, cy = self.ax.vertical_axes_fraction()
        dx, dy = dx - cx, dy - cy
        mag = np.hypot(dx, dy)
        u, v = sense * dx / mag, sense * dy / mag
        h = self.ax.quiver(x, y, u, v, transform=self.ax.transAxes, **kwargs)
        h.set_clip_path(self.primitive)

    def _tensor(self, *args, **kwargs):
        if kwargs.get("planes"):
            selkw = {
                key: kwargs[key]
                for key in kwargs.keys() & {"alpha", "ls", "lw", "label"}
            }
            fols = args[0].eigenfols()
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
            kwargs["ls"] = "none"
            lins = args[0].eigenlins()
            if kwargs["color"] is None:
                del kwargs["color"]
            if selkw["label"] != "_tensor":
                selkw["label"] = "S1"
                self._point(lins[0], color=kwargs.get("color", "red"), **selkw)
                selkw["label"] = "S2"
                self._point(lins[1], color=kwargs.get("color", "green"), **selkw)
                selkw["label"] = "S3"
                self._point(lins[2], color=kwargs.get("color", "blue"), **selkw)
            else:
                self._point(lins[0], color=kwargs.get("color", "red"), **selkw)
                self._point(lins[1], color=kwargs.get("color", "green"), **selkw)
                self._point(lins[2], color=kwargs.get("color", "blue"), **selkw)

    def _stress(self, *args, **kwargs):
        selkw = {
            key: kwargs[key]
            for key in kwargs.keys() & {"alpha", "marker", "mew", "ms", "label"}
        }
        lins = args[0].eigenlins()
        if kwargs["color"] is None:
            del kwargs["color"]
        if selkw["label"] != "_stress":
            selkw["label"] = "σ1"
            self._point(lins[0], color=kwargs.get("color", "red"), **selkw)
            selkw["label"] = "σ2"
            self._point(lins[1], color=kwargs.get("color", "green"), **selkw)
            selkw["label"] = "σ3"
            self._point(lins[2], color=kwargs.get("color", "blue"), **selkw)
        else:
            self._point(lins[0], color=kwargs.get("color", "red"), **selkw)
            self._point(lins[1], color=kwargs.get("color", "green"), **selkw)
            self._point(lins[2], color=kwargs.get("color", "blue"), **selkw)

    def _contour(self, grid, **kwargs):
        colorbar = kwargs.pop("colorbar")
        colorbar_kws = kwargs.pop("colorbar_kws")
        line_color = kwargs.pop("line_color")
        _ = kwargs.pop("label")
        filled = kwargs.pop("filled")
        linewidth = kwargs.pop("linewidth")
        linestyles = kwargs.pop("linestyles")
        # apsg_conf.stereonet_contour.clip defaults to True; when cmap isn't
        # given explicitly, apsg's own default is "Greys" for that clipped
        # (positive/above-uniform only) view, or "RdBu" (diverging, centered
        # on 0) for the full range when clip=False is requested.
        if kwargs.get("cmap") is None:
            kwargs["cmap"] = "Greys" if kwargs.get("clip") else "RdBu"
        if not filled:
            # linewidths/linestyles are meaningless for filled contours
            # (contourf ignores/warns about them) -- only forward for lines
            kwargs["linewidths"] = linewidth
            kwargs["linestyles"] = linestyles
        cf = self.ax.contour(grid=grid._engine, filled=filled, **kwargs)
        if filled:
            # also draw a black contour-line overlay on top of the fill,
            # reusing the fill's own resolved levels so the lines land
            # exactly on the fill boundaries
            line_kwargs = dict(kwargs)
            line_kwargs.pop("norm", None)
            line_kwargs.pop("clip", None)
            line_kwargs["cmap"] = None
            line_kwargs["levels"] = cf.levels
            line_kwargs["colors"] = line_color
            line_kwargs["linewidths"] = linewidth
            line_kwargs["linestyles"] = linestyles
            self.ax.contour(grid=grid._engine, filled=False, **line_kwargs)
        if colorbar:
            self.fig.colorbar(cf, ax=self.ax, **colorbar_kws)


def stereonetartist_from_json(obj_json):
    if obj_json["factory"] == "create_contour":
        # a contour artist's args[0] is a StereoGrid's own to_json shape
        # (no "datatype" key), not a feature -- see StereoNet_Contour.to_json
        grid = StereoGrid.from_json(obj_json["args"][0])
        return StereoNetArtistFactory.create_contour(grid, **obj_json["kwargs"])
    args = tuple([feature_from_json(arg_json) for arg_json in obj_json["args"]])
    return getattr(StereoNetArtistFactory, obj_json["factory"])(
        *args, **obj_json["kwargs"]
    )


def _quicknet_plot_one(s, arg, fol_as_pole, **kwargs):
    """Plot a single ``quicknet()`` argument on ``s`` -- dispatches by type to
    whichever ``StereoNet`` method has an artist representation for it,
    accepting a single feature or its ``*Set`` counterpart alike. Order
    matters: a subclass (``Fault`` vs ``Pair``, ``Lineation``/``Foliation``
    vs ``Vector3``, ``Stress3`` vs ``Tensor3``, ``OrientationTensor3Set`` vs
    ``EllipsoidSet``) is checked before its more general base class."""
    if isinstance(arg, (Foliation, FoliationSet)):
        (s.point if fol_as_pole else s.great_circle)(arg, **kwargs)
    elif isinstance(arg, (Lineation, LineationSet)):
        s.point(arg, **kwargs)
    elif isinstance(arg, (Fault, FaultSet)):
        s.fault(arg, **kwargs)
    elif isinstance(arg, (Pair, PairSet)):
        s.pair(arg, **kwargs)
    elif isinstance(arg, (Cone, ConeSet)):
        s.cone(arg, **kwargs)
    elif isinstance(arg, (Arc, ArcSet)):
        s.arc(arg, **kwargs)
    elif isinstance(arg, Stress3):
        s.stress(arg, **kwargs)
    elif isinstance(arg, Stress3Set):
        # .stress() only ever plots one Stress3 per call
        for item in arg:
            s.stress(item, **kwargs)
    elif isinstance(arg, Tensor3):
        s.tensor(arg, **kwargs)
    elif isinstance(arg, EllipsoidSet):
        # .tensor() only ever plots one Tensor3-like object per call;
        # also matches OrientationTensor3Set (an EllipsoidSet subclass)
        for item in arg:
            s.tensor(item, **kwargs)
    elif isinstance(arg, StereoGrid):
        s.contour(arg, **kwargs)
    elif isinstance(arg, (Vector3, Vector3Set)):
        s.vector(arg, **kwargs)
    else:
        print(f"{type(arg)} not supported.")


def quicknet(*args, **kwargs):
    """
    Function to quickly show or save ``StereoNet`` from args

    Args:
        args: object(s) to be plotted -- any type with a ``StereoNet`` artist
            representation, or its ``*Set`` counterpart: ``Vector3``,
            ``Foliation``, ``Lineation``, ``Pair``, ``Fault``, ``Cone``, ``Arc``,
            ``Ellipsoid``, ``OrientationTensor3``, ``Stress3``, or a ``StereoGrid``.
            A set whose method only accepts one object per call (``Stress3Set``,
            ``EllipsoidSet``/``OrientationTensor3Set``) is plotted one artist
            per item. ``scatter()``, ``confidence()``, ``arrow()`` and
            ``hoeppner()`` need extra arguments/semantics beyond a single
            object and remain reachable only via an explicit ``StereoNet`` call.

    Keyword Args:
        savefig (bool): True to save figure. Default `False`
        filename (str): filename for figure. Default `stereonet.png`
        savefig_kwargs (dict): dict passed to ``plt.savefig``
        fol_as_pole (bool): True to plot planar features as poles,
            False for plotting as great circle. Default `False`
        Additional kwargs are passed to StereoNet method

    Examples:
        >>> l = linset.random_fisher(position=lin(120, 50))
        >>> f = folset.random_fisher(position=lin(300, 40))
        >>> quicknet(f, l)
    Returns:
        None: Quickly shows or saves a ``StereoNet`` figure from the provided arguments.
    """
    savefig = kwargs.pop("savefig", False)
    filename = kwargs.pop("filename", "stereonet.png")
    savefig_kwargs = kwargs.pop("savefig_kwargs", {})
    fol_as_pole = kwargs.pop("fol_as_pole", False)
    kwargs["label"] = kwargs.get("label", "_nolegend_")
    s = StereoNet(**kwargs)
    for arg in args:
        _quicknet_plot_one(s, arg, fol_as_pole, **kwargs)
    if savefig:
        s.savefig(filename, **savefig_kwargs)
    else:
        s.show()
