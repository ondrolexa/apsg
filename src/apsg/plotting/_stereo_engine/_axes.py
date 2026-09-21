"""Custom matplotlib Axes for Schmidt (equal-area) and Wulff (equal-angle)
stereonets. Registered as the ``"schmidt"`` and ``"wulff"`` projections,
usable via ``plt.subplot(projection="schmidt")`` once ``apsg.plotting``
has been imported.
"""

import numpy as np
import matplotlib as mpl
import matplotlib.axis as maxis
import matplotlib.spines as mspines
import matplotlib.tri as mtri
from matplotlib.axes import Axes
from matplotlib.colors import CenteredNorm
from matplotlib.lines import Line2D
from matplotlib.patches import Circle
from matplotlib.projections import register_projection
from matplotlib.ticker import FixedLocator, Formatter, MaxNLocator, NullLocator
from matplotlib.transforms import Affine2D, BboxTransformTo

from ._stereogrid import StereoGrid
from ._transforms import (
    EqualAngleTransform,
    EqualAreaTransform,
    graticule_from_ned,
    ned_from_graticule,
    rotation_from_axis_angle,
)
from ._utils import _as_vectors, _clip_ring_to_hemisphere

__all__ = ["SchmidtNetAxes", "WulffNetAxes"]

# Matplotlib's own documented diverging colormaps (see
# https://matplotlib.org/stable/users/explain/colors/colormaps.html#diverging),
# used by `StereonetAxes.contour()` to decide whether to center the
# color normalization at 0. Anything else is treated as sequential.
_DIVERGING_CMAPS = {
    "piyg",
    "prgn",
    "brbg",
    "puor",
    "rdgy",
    "rdbu",
    "rdylbu",
    "rdylgn",
    "spectral",
    "coolwarm",
    "bwr",
    "seismic",
    "berlin",
    "managua",
    "vanimo",
}

# The literal 180-degree-about-vertical rotation "upper hemisphere" reduces
# to for axial/undirected geometry -- see StereonetAxes._hemisphere_rotate.
_HEMISPHERE_ROTATION = rotation_from_axis_angle([0, 0, 1], 180)


def _is_diverging_cmap(cmap):
    """Whether `cmap` (a colormap name or Colormap object) is one of
    matplotlib's documented diverging colormaps."""
    name = cmap.name if hasattr(cmap, "name") else str(cmap)
    name = name.lower()
    if name.endswith("_r"):
        name = name[:-2]
    return name in _DIVERGING_CMAPS


def _extend_antipodal(glon, glat, values, project_fn):
    """Used by ``contour()``: no counting-grid point ever lands exactly on
    the primitive circle (r == 0.5, all coordinates axes-fraction relative
    to center (0.5, 0.5)), so the triangulated domain's outer boundary is a
    polygon strictly inside it -- interpolation has nothing to reach out to
    the circle with, leaving a thin sliver unfilled right at the rim (most
    visible wherever a band happens to reach close to the boundary), and
    the interpolated surface is poorly conditioned in general wherever a
    steep gradient sits close to the boundary, since it has support on one
    side only.

    calculate_density's statistics (kamb/sph: an abs()/even-harmonics-only
    quantity; angmech: a sign-match dihedra count) are all antipodally
    symmetric, value(g) == value(-g) -- so every already-computed grid
    point doubles as an exact, no-interpolation-needed value for its own
    antipode. Unlike a synthetic, fixed-spacing ring sampled by nearest
    neighbor (this function's previous approach), reprojecting each point's
    *true* antipodal direction reproduces the real local point spacing on
    both sides of the boundary (points near the rim get near-rim
    companions; points near the pole get companions clear out near the far
    singularity, at r up to sqrt(2)), which is what a real density surface
    would need to be well-conditioned there -- a fixed-radius ring is far
    too coarse near a density peak close to the horizon, and its
    nearest-neighbor lookup is itself only an approximation of the true
    antipodal value.

    Note: this reprojects the already-*folded* canonical (glon, glat) --
    i.e. each point's antipode is the direction that folding discarded, not
    a second, independent fold of the same physical direction (which would
    just cancel back to the original point -- see ``_fold_axial_to_data``).

    Args:
        glon, glat: this axes' native graticule coordinates of the
            already-folded, already-finite counting-grid points (i.e.
            matching ``X, Y`` after ``_graticule_to_axes_fraction``).
        values: values at those same points.
        project_fn: ``self._graticule_to_axes_fraction``.

    Returns:
        (Xa, Ya, Va): the antipodal companions' axes-fraction coordinates
        and values (a subset of ``values``, wherever the companion's own
        projection is finite -- see ``_graticule_to_axes_fraction``'s
        antipodal-singularity masking).
    """
    v = np.column_stack(ned_from_graticule(glon, glat))
    glon_a, glat_a = graticule_from_ned(-v[:, 0], -v[:, 1], -v[:, 2])
    Xa, Ya = project_fn(glon_a, glat_a)
    finite = np.isfinite(Xa) & np.isfinite(Ya)
    return Xa[finite], Ya[finite], values[finite]


_CONTOUR_RESAMPLE_N = 300


def _resample_to_grid(X, Y, values, n=_CONTOUR_RESAMPLE_N):
    """Used by ``contour()``: resample the scattered, padded counting-grid
    points onto a regular ``n`` x ``n`` grid over axes-fraction [0, 1]^2,
    via a smooth (C1) cubic interpolation of the triangulated scatter data,
    for ``contourf``/``contour`` to draw instead of ``tricontourf``/
    ``tricontour`` drawing the raw triangulation directly.

    tricontourf's contour boundaries are piecewise-linear across whatever
    triangles the (irregularly spaced, antipodally-folded) counting-grid
    scatter happens to produce -- visible as small polygonal kinks/bulges
    wherever a triangle's own shape doesn't align with the true density
    gradient, most noticeable where a steep gradient sits close to the
    primitive circle (a fisher cluster near the horizon, say). A cubic fit
    resampled onto a fine regular grid has no such per-triangle seams.
    This is purely a smoother final rendering of the same input field --
    ``_extend_antipodal``'s antipodal padding and everything upstream of it
    is untouched, so the fragile antipodal-boundary handling stays exactly
    as correct or incorrect as it already was.

    Falls back to a linear fit if the cubic one fails outright (a
    near-degenerate triangulation, e.g. only a handful of points); still
    smoother than the raw triangulation since it's evaluated on a fine
    regular grid rather than drawn from the coarse one directly.

    Returns:
        (Xg, Yg, Vg): ``n``x``n`` coordinate grids and a (possibly masked,
        outside the input points' convex hull) value grid.
    """
    tri = mtri.Triangulation(X, Y)
    try:
        interp = mtri.CubicTriInterpolator(tri, values, kind="geom")
    except (RuntimeError, ValueError):
        interp = mtri.LinearTriInterpolator(tri, values)
    lin = np.linspace(0.0, 1.0, n)
    Xg, Yg = np.meshgrid(lin, lin)
    Vg = interp(Xg, Yg)
    return Xg, Yg, Vg


def _resolve_cmap_and_clip(kwargs):
    """Used by ``contour()``: decide ``clip`` (mutates ``kwargs["cmap"]``/
    ``kwargs["norm"]`` in place) -- see ``contour()``'s own docstring for
    the clip/cmap/norm defaulting rules this implements."""
    clip = kwargs.pop("clip", None)
    cmap_given = "cmap" in kwargs
    cmap = kwargs.get("cmap")

    if clip is None:
        clip = bool(cmap_given and not _is_diverging_cmap(cmap))

    if not cmap_given:
        cmap = "RdPu" if clip else "Spectral"
        kwargs["cmap"] = cmap

    if (
        not clip
        and _is_diverging_cmap(cmap)
        and "norm" not in kwargs
        and "vmin" not in kwargs
        and "vmax" not in kwargs
    ):
        kwargs["norm"] = CenteredNorm(vcenter=0)

    return clip


def _resolve_levels(levels, values, clip):
    """Used by ``contour()``: turn an integer ``levels`` count into actual
    level values (unchanged if ``levels`` is already a sequence)."""
    if not isinstance(levels, (int, np.integer)):
        return levels
    if clip:
        bounds = MaxNLocator(nbins=levels).tick_values(0, values.max())
        return bounds[bounds > 0]
    return MaxNLocator(nbins=levels).tick_values(values.min(), values.max())


def _avoid_plateau_at_top(levels, values, filled):
    """Used by ``contour()``: kamb/sph/angmech are often integer-valued or
    otherwise plateaued statistics, so many grid points can land exactly on
    the data's own maximum. When the top level's upper edge lands exactly
    on that value too (the common case), tricontourf can leave triangles
    whose vertices all sit exactly on that edge unfilled, punching small
    holes in the topmost band right at the density maxima. Nudge the edge
    a hair above the true max so the comparison is strict rather than
    exact."""
    if not (filled and len(levels) and levels[-1] <= values.max()):
        return levels
    top = values.max()
    return np.append(np.asarray(levels[:-1]), top + abs(top) * 1e-6 + 1e-9)


class _DegreeFormatter(Formatter):
    """Formats radian tick values as rounded whole degrees with a ° suffix."""

    def __init__(self, round_to=1.0):
        self._round_to = round_to

    def __call__(self, x, pos=None):
        degrees = round(np.rad2deg(x) / self._round_to) * self._round_to
        return f"{degrees:0.0f}\N{DEGREE SIGN}"


def _orthogonal_vector(axis):
    """A unit vector orthogonal to unit vector ``axis`` (Gram-Schmidt against
    a seed that is never (near-)parallel to axis)."""
    axis = np.asarray(axis, dtype=float)
    seed = np.array([1.0, 0.0, 0.0])
    if abs(np.dot(axis, seed)) > 0.9:
        seed = np.array([0.0, 1.0, 0.0])
    ortho = seed - np.dot(seed, axis) * axis
    return ortho / np.linalg.norm(ortho)


def _rotate_around_axis(v, axis, angles):
    """Rotate unit vector ``v`` around unit vector ``axis`` by each angle
    (radians) in ``angles`` (Rodrigues' rotation formula). Returns an
    (len(angles), 3) array."""
    v = np.asarray(v, dtype=float)
    axis = np.asarray(axis, dtype=float)
    cos_a = np.cos(angles)[:, None]
    sin_a = np.sin(angles)[:, None]
    cross = np.cross(axis, v)
    dot = np.dot(axis, v)
    return v * cos_a + cross * sin_a + axis * dot * (1.0 - cos_a)


def _validate_rotation(matrix):
    """Coerce ``matrix`` to a validated 3x3 proper-rotation array (accepts
    apsg's ``Rotation3`` directly, since it implements ``__array__``).

    Raises ``ValueError`` rather than silently correcting an invalid
    input -- unlike apsg's own ``Rotation3``, which SVD-corrects a
    det=-1 matrix back to the nearest proper rotation.
    """
    R = np.asarray(matrix, dtype=float)
    if R.shape != (3, 3):
        raise ValueError(f"rotation must be a 3x3 matrix, got shape {R.shape}")
    if not np.allclose(R.T @ R, np.eye(3), atol=1e-6):
        raise ValueError("rotation must be orthogonal (R.T @ R == I)")
    det = np.linalg.det(R)
    if not np.isclose(det, 1.0, atol=1e-6):
        raise ValueError(
            f"rotation must be a proper rotation (det(R) == 1), got det(R)="
            f"{det:.6f}. A determinant of -1 is an improper transform (e.g. "
            "a reflection or hemisphere flip), not a rotation -- see the "
            "'hemisphere' constructor argument/property for that."
        )
    return R


class StereonetAxes(Axes):
    """Shared base class for the Schmidt-net and Wulff-net projections.

    Not registered directly -- see :class:`SchmidtNetAxes` and
    :class:`WulffNetAxes`.
    """

    RESOLUTION = 60
    _base_transform = None  # set by subclasses

    def __init__(
        self, *args, hemisphere="lower", rotation=None, rotate_data=True, **kwargs
    ):
        if hemisphere not in ("lower", "upper"):
            raise ValueError("hemisphere must be 'lower' or 'upper'")
        self._hemisphere = hemisphere
        self._rotation = np.eye(3) if rotation is None else _validate_rotation(rotation)
        self._rotate_data = bool(rotate_data)
        self._clip_pole = np.pi / 2
        self._azimuth_tick_artists = []
        super().__init__(*args, **kwargs)
        self.set_aspect("equal", adjustable="box", anchor="C")
        self.clear()

    # -- hemisphere -----------------------------------------------------

    @property
    def hemisphere(self):
        """Which hemisphere ('lower' or 'upper') this axes displays."""
        return self._hemisphere

    @hemisphere.setter
    def hemisphere(self, value):
        if value not in ("lower", "upper"):
            raise ValueError("hemisphere must be 'lower' or 'upper'")
        self._hemisphere = value

    # -- rotation ---------------------------------------------------------

    @property
    def rotation(self):
        """The current 3x3 proper-rotation matrix applied to the whole net
        (grid, azimuth ticks, and -- when ``rotate_data`` is True -- plotted
        data) before projecting. A copy; use ``set_rotation``/the setter to
        change it."""
        return self._rotation.copy()

    @rotation.setter
    def rotation(self, matrix):
        self.set_rotation(matrix)

    def set_rotation(self, matrix):
        """Set the rotation matrix (see the ``rotation`` property). Accepts
        anything ``np.asarray``-coercible to a 3x3 proper rotation,
        including apsg's ``Rotation3`` directly (no apsg dependency).
        Pass ``None`` to reset to the identity (no rotation).

        Unlike hemisphere selection, this is a rotation only (SO(3),
        determinant +1) -- see the module docs for why hemisphere can't be
        expressed as a rotation.
        """
        self._rotation = np.eye(3) if matrix is None else _validate_rotation(matrix)
        if hasattr(self, "transProjection"):
            # Mutate the existing transform in place (rather than rebuilding
            # self.transData from scratch) so artists already plotted using
            # the old self.transData composite pick up the change too.
            self.transProjection.set_rotation(self._rotation)
            self._refresh_azimuth_ticks()

    @property
    def rotate_data(self):
        """Whether plotted data (point/vector/great_circle/cone) follows
        ``rotation`` (default True, a rigid whole-net rotation) or stays
        anchored to the unrotated frame while only the grid/ticks rotate
        (False -- apsg's own default, for viewing fixed data against a
        differently-oriented reference net)."""
        return self._rotate_data

    @rotate_data.setter
    def rotate_data(self, value):
        self._rotate_data = bool(value)

    def _hemisphere_rotate(self, xyz):
        """The one real (proper, SO(3)) rotation "upper hemisphere" reduces
        to for axial/undirected geometry -- 180 degrees about the vertical
        (Down) axis, applied only when this axes shows the upper hemisphere
        (identity for
        "lower"). This is apsg's single default hemisphere mechanism:
        ``point()``, ``great_circle()``, ``cone()``, ``path()`` and
        ``contour()``'s whole-sphere grid all route through it (via
        ``_fold_axial_to_data`` where a canonical single representative is
        also needed first), with no other per-method hemisphere branching
        anywhere in this class. Data-only, like the transform's own
        ``rotation``/``rotate_data`` -- it never touches the grid or ticks.

        Not used by ``vector()`` -- a genuinely directional vector has an
        actual "wrong side", so its hemisphere handling needs a full
        reflection, not a rotation; see ``_vec_to_data``.
        """
        v = _as_vectors(xyz)
        if self._hemisphere != "upper":
            return v
        return v @ _HEMISPHERE_ROTATION.T

    def _vec_to_data(self, xyz):
        """Convert NED unit vector(s) to this axes' native (glon, glat)
        data coordinates, applying the hemisphere flip. Deliberately
        rotation-agnostic -- see ``_data_transform`` for how ``rotate_data``
        is handled.

        This full antipodal negation is correct for genuinely directional
        data: a vector pointing into the hemisphere this axes doesn't show
        has no valid position at all until negated onto the one it does.
        ``vector()`` is the sole remaining caller of this rule for plotted
        data (its two candidate on-screen positions must stay fixed
        regardless of hemisphere, with only which one is styled filled vs.
        open changing -- not expressible as the rotation ``_hemisphere_rotate``
        uses for every other, axial, artist); the only other caller is
        ``project()``'s unfolded path, used by ``StereoNet.format_coord``'s
        cursor-tooltip lookup. It is *not* used for axial data -- see
        ``_fold_axial_to_data``/``_hemisphere_rotate``.
        """
        v = _as_vectors(xyz)
        if self._hemisphere == "upper":
            v = -v
        glon, glat = graticule_from_ned(v[:, 0], v[:, 1], v[:, 2])
        return glon, glat

    def _fold_axial_to_data(self, xyz):
        """Convert axial (no inherent direction) NED unit vector(s) to
        (glon, glat): fold to a canonical (z >= 0) representative, then
        apply ``_hemisphere_rotate``.

        Axial data has no "wrong side" the way a true vector does -- both
        v and -v are the same feature -- so it needs a different rule than
        ``_vec_to_data``'s full negation.
        A line plunging e.g. NE should plot in the NE part of a "lower"
        net and in the antipodal SW part of an "upper" net, at the *same*
        distance from center (same plunge magnitude) either way -- exactly
        what ``_hemisphere_rotate`` does. Applying a full 3D negation to the
        canonical (z >= 0) representative instead (matching ``_vec_to_data``)
        would both flip its plunge sign, sending it outside the domain this
        projection can validly place within the primitive circle (see the
        whole-sphere-grid comment in ``contour``), and cancel back out to
        the exact same position on both hemispheres (the original bug:
        hemisphere had no visible effect on axial data at all).

        The fold decision is made in the *effective* frame -- after
        ``rotation`` (when ``rotate_data`` is True) -- not the raw
        input frame, since that is what ends up facing up/down once the
        transform pipeline applies ``self._rotation`` downstream (via
        ``transform=self._data_transform()``). A vector already
        canonicalized to z >= 0 in its own, unrotated frame can easily
        have effective z < 0 after a net rotation tips it "up"; folding on
        the raw frame instead left such points stuck outside the
        projectable domain -- most of a rotated net's axial data simply
        vanishing -- while manually pre-rotating the data (and leaving the
        net unrotated) worked fine, since then raw and effective agreed.
        The fold is applied in the effective frame and then rotated back
        (v_eff @ rotation, rotation being orthogonal) so the pipeline's own
        rotation reproduces exactly that effective, corrected vector.
        """
        v = _as_vectors(xyz)
        rotation = self._rotation if self._rotate_data else np.eye(3)
        v_eff = v @ rotation.T
        flip = (np.sign(v_eff[:, 2]) != 1.0) & (v_eff[:, 2] != 0)
        v_eff = np.where(flip[:, None], -v_eff, v_eff)
        v_eff = self._hemisphere_rotate(v_eff)
        v = v_eff @ rotation
        return graticule_from_ned(v[:, 0], v[:, 1], v[:, 2])

    def _data_to_axial(self, glon, glat):
        """Inverse of ``_fold_axial_to_data``: given a (glon, glat) data
        coordinate, recover an axial-equivalent representative of the true,
        pre-rotation vector that would have been plotted there -- undoing
        hemisphere (``_hemisphere_rotate``, its own inverse) and, when
        ``rotate_data`` is True, the net's own ``rotation``, in exactly the
        reverse order ``_fold_axial_to_data`` applied them.

        Used by ``StereoNet.format_coord``'s cursor-tooltip lookup, so that
        hovering over a plotted point reports the feature that was actually
        plotted there, on any hemisphere/rotation combination.
        """
        glon = np.atleast_1d(glon)
        glat = np.atleast_1d(glat)
        w = np.column_stack(ned_from_graticule(glon, glat))
        rotation = self._rotation if self._rotate_data else np.eye(3)
        v_eff = self._hemisphere_rotate(w @ rotation.T)
        return v_eff @ rotation

    def _data_transform(self):
        """The transform to plot data through: the live, rotation-following
        ``self.transData`` when ``rotate_data`` is True (the default --
        matches the grid/ticks), or a separate transform frozen at the
        identity rotation when False, so that data plotted while
        ``rotate_data`` is False stays anchored to the true frame no matter
        how ``rotation`` is changed afterwards (mirrors apsg: with
        ``rotate_data=False``, apsg's own ``project_data`` never applies its
        rotation matrix to data at all)."""
        return self.transData if self._rotate_data else self._fixed_transData

    def _data_axes_fraction(self, xyz):
        """Project NED vectors directly to axes-fraction (X, Y), bypassing
        the (glon, glat) intermediate coordinate.

        Needed for ``contour`` (Delaunay-triangulated scatter data), unlike
        ``_data_transform`` (used by ``point``/``vector``/``great_circle``/
        ``cone``, whose ``Line2D`` artists apply a transform per-vertex at
        draw time with no triangulation step to go wrong): a whole-sphere
        scatter's (glon, glat) encoding has a seam at glon=+-pi and severe
        distortion near glat=+-pi/2 (the graticule poles), so triangulating
        in that raw coordinate produces spurious triangles connecting
        points that are physically close but numerically far apart (and
        vice versa). Projecting to the final (X, Y) plane first -- a plain,
        seamless Euclidean space -- before triangulating avoids this
        entirely."""
        return self._graticule_to_axes_fraction(*self._vec_to_data(xyz))

    def _graticule_to_axes_fraction(self, glon, glat):
        """The shared tail of ``_data_axes_fraction``/``project``: project
        already-computed (glon, glat) data coordinates straight to
        axes-fraction (X, Y)."""
        projection = (
            self.transProjection if self._rotate_data else self._fixed_transProjection
        )
        XY = (projection + self.transAffine).transform(np.column_stack([glon, glat]))
        return XY[:, 0], XY[:, 1]

    # -- public coordinate helpers -----------------------------------------

    def project(self, vectors, clip_inside=True, fold=False):
        """A public version of ``_data_axes_fraction``, for code that needs
        raw axes-fraction ``(X, Y)`` rather than the ``(glon, glat)`` +
        ``transform=`` idiom used by ``point``/``vector``/``great_circle``/
        ``cone`` (namely
        ``apsg.plotting.StereoNet``'s ``_arrow``/``format_coord``
        wrappers -- quiver directions and cursor-position inversion are
        not expressible as a plotted ``Line2D``).

        Args:
            vectors: NED vector(s), shape (3,) or (N, 3).
            clip_inside (bool): if True, points landing outside the unit
                circle are replaced by NaN. Default True.
            fold (bool): if True, fold axial vectors onto this axes'
                hemisphere first (see ``point``) before projecting.
                Default False (vectors are projected as given, like
                ``vector``/``great_circle``).

        Returns:
            (X, Y): axes-fraction coordinate arrays.
        """
        v = _as_vectors(vectors)
        if fold:
            X, Y = self._graticule_to_axes_fraction(*self._fold_axial_to_data(v))
        else:
            X, Y = self._data_axes_fraction(v)
        if clip_inside:
            outside = (X - 0.5) ** 2 + (Y - 0.5) ** 2 > 0.25
            X = np.where(outside, np.nan, X)
            Y = np.where(outside, np.nan, Y)
        return X, Y

    def vertical_axes_fraction(self):
        """Return the axes-fraction ``(X, Y)`` of the fixed physical vertical
        direction (NED ``(0, 0, 1)``) -- i.e. where the "+" cross mark is
        drawn (see ``_draw_azimuth_ticks``). This is the disk's geometric
        center ``(0.5, 0.5)`` only when the net is unrotated; a ``rotation``
        moves it elsewhere on the disk, same as it moves the "+" mark, since
        both use the same rotation-aware ``transProjection`` regardless of
        ``rotate_data`` (the grid -- and this reference point -- always
        reflects the net's actual rotation; only whether *data* follows it
        is controlled by ``rotate_data``).

        Returns:
            (float, float): axes-fraction (X, Y) of true vertical.
        """
        core_transform = self.transProjection + self.transAffine
        cx, cy = core_transform.transform((np.pi / 2, 0.0))
        return cx, cy

    def path(self, vectors, antipodal=False, **kwargs):
        """Plot an arbitrary, already-computed NED-vector polyline -- e.g. a
        slerp-interpolated arc, a confidence-region boundary, or a
        small-circle/cone curve with a partial revolution angle -- none of
        which are expressible
        via ``point``/``vector``/``great_circle``/``cone``'s own
        curve-generation logic.

        Args:
            vectors: NED vectors, shape (N, 3), plotted as a connected
                polyline in the given order (no resampling is done here;
                the caller supplies the sample points).
            antipodal (bool): if True, also plot the antipodal reflection
                of the whole curve as a second NaN-separated segment of
                the same line (mirrors ``cone``'s own handling of small
                circles, which are not antipodally symmetric in general).
                Default False.
            **kwargs: passed to ``self.plot``.

        Returns:
            The ``Line2D`` handle.

        Note: hemisphere is handled via ``_hemisphere_rotate`` (the default,
        axial-type rotation), not ``_vec_to_data`` -- a curve, unlike a
        single vector, has no directional "wrong side" to reflect onto.
        """
        v = self._hemisphere_rotate(vectors)
        glon, glat = graticule_from_ned(v[:, 0], v[:, 1], v[:, 2])
        if antipodal:
            glon_a, glat_a = graticule_from_ned(-v[:, 0], -v[:, 1], -v[:, 2])
            glon = np.concatenate([glon, [np.nan], glon_a])
            glat = np.concatenate([glat, [np.nan], glat_a])
        plot_kwargs = dict(transform=self._data_transform())
        plot_kwargs.update(kwargs)
        (handle,) = self.plot(glon, glat, **plot_kwargs)
        handle.set_clip_path(self.patch)
        return handle

    def polygon(self, vectors, **kwargs):
        """Fill the region bounded by an already-computed NED-vector ring
        (closed automatically, last point to first), e.g. a chain of
        slerp-interpolated arcs.

        The region is the side of the ring that does not contain the
        projection's singular point (the vector antipodal to the net's center,
        the zenith of a lower-hemisphere net). It is clipped to the displayed
        hemisphere: where the ring runs through the hidden hemisphere, its
        excursion is replaced by the part of the primitive circle that
        belongs to the region.

        Args:
            vectors: NED vectors, shape (N, 3), the boundary in the given order
                (no resampling is done here; the caller supplies dense points).
            **kwargs: passed to ``self.fill``. If neither ``color`` nor
                ``facecolor`` is given, the next color of the axes' color cycle
                is used. ``label`` goes to the first patch only.

        Returns:
            list of ``Polygon`` handles (one per separate visible piece; empty
            if nothing of the region is visible).
        """
        loops = _clip_ring_to_hemisphere(self._hemisphere_rotate(vectors))
        fill_kwargs = dict(transform=self._data_transform())
        fill_kwargs.update(kwargs)
        if not {"color", "facecolor", "fc"} & fill_kwargs.keys():
            fill_kwargs["facecolor"] = self._get_lines.get_next_color()
        handles = []
        for loop in loops:
            glon, glat = graticule_from_ned(loop[:, 0], loop[:, 1], loop[:, 2])
            (handle,) = self.fill(glon, glat, **fill_kwargs)
            handle.set_clip_path(self.patch)
            handles.append(handle)
            fill_kwargs.pop("label", None)  # one legend entry only
        return handles

    # -- matplotlib custom-projection machinery --------------------------

    def _init_axis(self):
        self.xaxis = maxis.XAxis(self, clear=False)
        self.yaxis = maxis.YAxis(self, clear=False)
        self.spines["geo"].register_axis(self.yaxis)

    def clear(self):
        super().clear()
        self.stereogrids = []
        self.set_longitude_grid(10)
        self.set_latitude_grid(10)

        self.xaxis.set_minor_locator(NullLocator())
        self.yaxis.set_minor_locator(NullLocator())
        self.xaxis.set_ticks_position("none")
        self.yaxis.set_ticks_position("none")
        self.xaxis.set_tick_params(label1On=False)
        self.yaxis.set_tick_params(label1On=False)

        self.grid(mpl.rcParams["axes.grid"])

        Axes.set_xlim(self, -np.pi, np.pi)
        Axes.set_ylim(self, -np.pi / 2.0, np.pi / 2.0)
        self.set_clip_pole(20)

        self._clear_azimuth_ticks()
        self.set_azimuth_ticks([0, 90, 180, 270], labels=["N", "E", "S", "W"])

    def _set_lim_and_transforms(self):
        self.transProjection = self._get_core_transform(self.RESOLUTION)
        self.transAffine = self._get_affine_transform()
        self.transAxes = BboxTransformTo(self.bbox)

        self.transData = self.transProjection + self.transAffine + self.transAxes

        # A second data transform, permanently frozen at the identity
        # rotation (never touched by set_rotation), used for data plotted
        # while rotate_data=False -- see _data_transform.
        self._fixed_transProjection = self._base_transform(self.RESOLUTION)
        self._fixed_transData = (
            self._fixed_transProjection + self.transAffine + self.transAxes
        )

        self._xaxis_pretransform = (
            Affine2D()
            .scale(1.0, self._clip_pole * 2.0)
            .translate(0.0, -self._clip_pole)
        )
        self._xaxis_transform = self._xaxis_pretransform + self.transData
        self._xaxis_text1_transform = (
            Affine2D().scale(1, 0) + self.transData + Affine2D().translate(0, 4)
        )
        self._xaxis_text2_transform = (
            Affine2D().scale(1, 0) + self.transData + Affine2D().translate(0, -4)
        )

        yaxis_stretch = Affine2D().scale(np.pi * 2, 1).translate(-np.pi, 0)
        yaxis_space = Affine2D().scale(1, 1.1)
        self._yaxis_transform = yaxis_stretch + self.transData
        yaxis_text_base = (
            yaxis_stretch
            + self.transProjection
            + (yaxis_space + self.transAffine + self.transAxes)
        )
        self._yaxis_text1_transform = yaxis_text_base + Affine2D().translate(-8, 0)
        self._yaxis_text2_transform = yaxis_text_base + Affine2D().translate(8, 0)

    def _get_core_transform(self, resolution):
        return self._base_transform(resolution, rotation=self._rotation)

    def _get_affine_transform(self):
        return Affine2D().scale(0.5).translate(0.5, 0.5)

    def get_xaxis_transform(self, which="grid"):
        return self._xaxis_transform

    def get_xaxis_text1_transform(self, pad):
        return self._xaxis_text1_transform, "bottom", "center"

    def get_xaxis_text2_transform(self, pad):
        return self._xaxis_text2_transform, "top", "center"

    def get_yaxis_transform(self, which="grid"):
        return self._yaxis_transform

    def get_yaxis_text1_transform(self, pad):
        return self._yaxis_text1_transform, "center", "right"

    def get_yaxis_text2_transform(self, pad):
        return self._yaxis_text2_transform, "center", "left"

    def _gen_axes_patch(self):
        return Circle((0.5, 0.5), 0.5)

    def _gen_axes_spines(self):
        return {"geo": mspines.Spine.circular_spine(self, (0.5, 0.5), 0.5)}

    def set_title(self, label, **kwargs):
        # The default title pad sits right where the azimuth rim label at
        # 0 degrees is drawn (see the "frac" default in set_azimuth_ticks);
        # push it up a bit further so the two don't overlap.
        kwargs.setdefault("pad", mpl.rcParams["axes.titlepad"] + 14)
        return super().set_title(label, **kwargs)

    def set_xlim(self, *args, **kwargs):
        raise TypeError("Changing axes limits of a stereonet is not supported.")

    set_ylim = set_xlim
    set_xbound = set_xlim
    set_ybound = set_xlim

    def set_xscale(self, *args, **kwargs):
        if args[0] != "linear":
            raise NotImplementedError

    set_yscale = set_xscale

    def get_data_ratio(self):
        """The projected circle is always 1:1, regardless of the
        (glon, glat) data-limit ratio."""
        return 1.0

    def can_zoom(self):
        return False

    def can_pan(self):
        return False

    # -- grid / ticks -----------------------------------------------------

    def set_longitude_grid(self, degrees):
        """Set the spacing (degrees) between meridian ("dip-arc") gridlines."""
        # -180 and +180 are the same meridian (both halves of the
        # unrotated primitive circle), so include exactly one of them
        # (-180) rather than excluding both, which silently dropped that
        # meridian from the grid.
        n = round(360.0 / degrees)
        grid = -180 + degrees * np.arange(n)
        self.xaxis.set_major_locator(FixedLocator(np.deg2rad(grid)))
        self.xaxis.set_major_formatter(_DegreeFormatter(degrees))

    def set_latitude_grid(self, degrees):
        """Set the spacing (degrees) between parallel ("small-circle")
        gridlines."""
        self._latitude_step = degrees
        self._update_latitude_grid()

    def _update_latitude_grid(self):
        """(Re)build the yaxis (parallel/small-circle) gridline locator,
        keeping it consistent with the current clip_pole: regular gridlines
        that would fall inside the polar cap are dropped, and a single
        gridline is added exactly on the cap boundary (when there is one)
        so the "pole hole" reads as a clean, deliberate boundary rather
        than an arbitrary cutoff."""
        step = self._latitude_step
        limit = 90.0 - self.clip_pole
        grid = np.arange(-90 + step, 90, step)
        grid = grid[np.abs(grid) < limit - 1e-9]
        if 0.0 < limit < 90.0:
            grid = np.concatenate([grid, [-limit, limit]])
        grid = np.sort(grid)
        self.yaxis.set_major_locator(FixedLocator(np.deg2rad(grid)))
        self.yaxis.set_major_formatter(_DegreeFormatter(step))

    def set_clip_pole(self, degrees):
        """Set the angular radius (degrees), around the two points where
        meridian gridlines converge (compass N and S unrotated -- any
        rotated location of the graticule's own pole otherwise), within
        which gridlines are not drawn."""
        self._clip_pole = np.deg2rad(90.0 - degrees)
        if hasattr(self, "_xaxis_pretransform"):
            self._xaxis_pretransform.clear().scale(
                1.0, self._clip_pole * 2.0
            ).translate(0.0, -self._clip_pole)
        if hasattr(self, "_latitude_step"):
            self._update_latitude_grid()

    @property
    def clip_pole(self):
        """Angular radius (degrees) around N/S within which gridlines are
        not drawn."""
        return 90.0 - np.rad2deg(self._clip_pole)

    @clip_pole.setter
    def clip_pole(self, degrees):
        self.set_clip_pole(degrees)

    def _clear_azimuth_ticks(self):
        for artist in self._azimuth_tick_artists:
            if artist.figure is not None:
                artist.remove()
        self._azimuth_tick_artists = []

    def set_azimuth_ticks(
        self, angles, labels=None, frac=1.06, tick_frac=1.02, **kwargs
    ):
        """Place compass-bearing tick labels -- and a short tick line at
        each one, pointing outward from the primitive circle's rim -- around
        the rim (0 degrees / North at top, clockwise, in the *pre-rotation*
        frame -- like a grid value, a tick's position always follows
        ``rotation`` regardless of ``rotate_data``, since ticks are net
        decoration, not data).

        This is independent of ``xaxis``/``yaxis`` (azimuth is not a native
        graticule coordinate). ``frac`` positions the labels (1.0 is on the
        rim); ``tick_frac`` positions the outer end of the tick lines (the
        inner end is always on the rim, at 1.0).
        """
        self._clear_azimuth_ticks()

        angles = list(angles)
        if labels is None:
            labels = [f"{a:g}\N{DEGREE SIGN}" for a in angles]

        # Remembered so set_rotation can refresh tick positions later --
        # these artists are static, not re-evaluated on every draw.
        self._azimuth_ticks_kwargs = dict(
            angles=angles, labels=labels, frac=frac, tick_frac=tick_frac, **kwargs
        )

        text_kwargs = dict(ha="center", va="center", clip_on=False)
        text_kwargs.update(kwargs)
        line_color = text_kwargs.get("color", "black")

        # (glon, glat) -> axes-fraction, the same rotation-aware pipeline
        # gridlines use, minus the final transAxes step.
        core_transform = self.transProjection + self.transAffine

        for angle, label in zip(angles, labels):
            theta = np.deg2rad(angle)
            v = np.array([np.cos(theta), np.sin(theta), 0.0])
            # A rotation can tip this reference direction out of the
            # horizontal plane entirely (e.g. a rotation about a
            # non-vertical axis); when that happens it no longer projects
            # anywhere near the rim (it can land arbitrarily far outside
            # the axes), so skip it rather than draw a stray tick.
            if (self._rotation @ v)[2] < -1e-9:
                continue
            glon, glat = graticule_from_ned(v[0], v[1], v[2])
            rim_x, rim_y = core_transform.transform((glon, glat))
            dx, dy = rim_x - 0.5, rim_y - 0.5

            tick_line = Line2D(
                [0.5 + dx, 0.5 + tick_frac * dx],
                [0.5 + dy, 0.5 + tick_frac * dy],
                transform=self.transAxes,
                color=line_color,
                linewidth=1,
                clip_on=False,
            )
            self.add_line(tick_line)
            self._azimuth_tick_artists.append(tick_line)

            x = 0.5 + frac * dx
            y = 0.5 + frac * dy
            text_artist = self.text(
                x, y, label, transform=self.transAxes, **text_kwargs
            )
            self._azimuth_tick_artists.append(text_artist)

        # A small "+" marking the fixed physical vertical direction
        # (NED (0, 0, 1)), exactly like the compass ticks above mark the
        # fixed horizontal reference directions -- i.e. it moves under
        # rotation, computed through the same
        # rotation-aware core_transform (see ``vertical_axes_fraction``).
        # This reduces to the exact center (0.5, 0.5) when rotation is the
        # identity. Tracked in ``_azimuth_tick_artists`` and rebuilt here so
        # it stays styled and managed exactly like the compass ticks
        # (cleared by ``_clear_azimuth_ticks``, rebuilt by
        # ``_refresh_azimuth_ticks``).
        cx, cy = self.vertical_axes_fraction()
        cross_len = (tick_frac - 1.0) * 0.5
        for (x0, y0), (x1, y1) in (
            ((cx - cross_len, cy), (cx + cross_len, cy)),
            ((cx, cy - cross_len), (cx, cy + cross_len)),
        ):
            cross_line = Line2D(
                [x0, x1],
                [y0, y1],
                transform=self.transAxes,
                color=line_color,
                linewidth=1,
                clip_on=False,
            )
            self.add_line(cross_line)
            self._azimuth_tick_artists.append(cross_line)

    def _refresh_azimuth_ticks(self):
        if hasattr(self, "_azimuth_ticks_kwargs"):
            self.set_azimuth_ticks(**self._azimuth_ticks_kwargs)

    def get_azimuth_ticklabels(self):
        return [a for a in self._azimuth_tick_artists if not isinstance(a, Line2D)]

    def get_azimuth_ticklines(self):
        return [a for a in self._azimuth_tick_artists if isinstance(a, Line2D)]

    # -- vector-based plotting (apsg-inspired) -----------------------------

    def point(self, vectors, **kwargs):
        """Plot axial data (poles/lines with no inherent direction) as
        filled points, folding each vector onto this axes' hemisphere."""
        glon, glat = self._fold_axial_to_data(vectors)
        plot_kwargs = dict(ls="none", marker="o", transform=self._data_transform())
        plot_kwargs.update(kwargs)
        (handle,) = self.plot(glon, glat, **plot_kwargs)
        handle.set_clip_path(self.patch)
        return handle

    pole = line = point

    def vector(self, vectors, **kwargs):
        """Plot true (polarity-bearing) vector data: a filled point at the
        actual position, and an open point at the antipode -- matching
        apsg's convention for vectors as opposed to axial data.

        Deliberately the one plotting method that does *not* go through
        ``_hemisphere_rotate`` (apsg's default hemisphere mechanism for
        every other, axial, artist): a genuinely directional vector has an
        actual "wrong side", so on-screen it needs ``_vec_to_data``'s full
        reflection instead of a rotation -- with the effect that this
        method's two candidate positions stay fixed regardless of
        hemisphere, and only which one is styled filled vs. open changes.
        """
        v = _as_vectors(vectors)
        glon, glat = self._vec_to_data(v)
        plot_kwargs = dict(ls="none", marker="o", transform=self._data_transform())
        plot_kwargs.update(kwargs)
        (h_filled,) = self.plot(glon, glat, **plot_kwargs)
        h_filled.set_clip_path(self.patch)

        open_kwargs = dict(plot_kwargs)
        # Both must be forced (not merely defaulted) since plot_kwargs
        # already carries an explicit "mec": None from the marker style
        # config, so a plain setdefault() is a no-op and mec silently stays
        # None; and leaving "color" unset
        # lets this second self.plot() call independently advance the color
        # cycle, both giving the open (antipodal) marker a mismatched color
        # and silently burning a cycle slot that the next unrelated call
        # would otherwise have gotten.
        open_kwargs["color"] = h_filled.get_color()
        open_kwargs["mfc"] = "none"
        open_kwargs["mec"] = open_kwargs.get("mec") or h_filled.get_color()
        open_kwargs["label"] = "_nolegend_"
        glon_a, glat_a = self._vec_to_data(-v)
        (h_open,) = self.plot(glon_a, glat_a, **open_kwargs)
        h_open.set_clip_path(self.patch)
        return h_filled, h_open

    def great_circle(self, pole_vectors, segments=200, **kwargs):
        """Plot the great circle(s) (plane traces) with the given pole
        (normal) vector(s))."""
        # Draw every pole's circle as one NaN-separated Line2D, like point()
        # draws a whole set as one artist, rather than one self.plot() call
        # per pole. That previously gave each plane its own color (each
        # call independently advancing
        # the cycle) and its own legend entry (matplotlib shows one entry
        # per labeled artist, so N circles with the same label -- itself
        # untouched by the multi-artist fix -- still made N entries).
        poles = _as_vectors(pole_vectors)
        angles = np.linspace(0.0, 2 * np.pi, segments)
        curves = []
        for pole in poles:
            ref = _orthogonal_vector(pole)
            curves.append(_rotate_around_axis(ref, pole, angles))
            curves.append(np.full((1, 3), np.nan))
        combined = np.vstack(curves)
        # Hemisphere handling uses the shared _hemisphere_rotate, minus the
        # per-point canonical fold _fold_axial_to_data also applies: a great
        # circle already spans both z signs by construction (it is not
        # axially ambiguous the way a single line is), so folding each point
        # individually would replace roughly half the curve with points from
        # its *other* half instead of revealing the true complementary arc.
        # Rotating every point alike is exactly the transform verified (see
        # the hemisphere fix's commit) to mirror the whole curve through the
        # center for "upper" -- matching point()'s NE <-> SW convention --
        # without that distortion.
        combined = self._hemisphere_rotate(combined)
        glon, glat = graticule_from_ned(combined[:, 0], combined[:, 1], combined[:, 2])
        plot_kwargs = dict(transform=self._data_transform())
        plot_kwargs.update(kwargs)
        (handle,) = self.plot(glon, glat, **plot_kwargs)
        handle.set_clip_path(self.patch)
        return [handle]

    gc = great_circle

    def cone(self, axis_vector, angle_deg, segments=200, **kwargs):
        """Plot the small circle (cone) of the given angular radius
        (degrees) around ``axis_vector``.

        Unlike a great circle, a cone is generally *not* symmetric under
        antipodal reflection, so whenever part of it crosses into the
        axes' opposite hemisphere that part is not simply retraced
        elsewhere in the same curve. To match the standard stereonet
        convention, the antipodal reflection of the whole curve is also
        plotted (as a second, NaN-separated segment of the same line) so
        the missing arc reappears on the opposite side of the primitive
        circle instead of just vanishing.

        Hemisphere is handled via the shared ``_hemisphere_rotate`` (like
        every other axial artist here), applied once to the curve -- its
        antipodal companion is just its negation, since the rotation is
        linear (``_hemisphere_rotate(-curve) == -_hemisphere_rotate(curve)``).
        """
        axis = _as_vectors(axis_vector)[0]
        ref = _orthogonal_vector(axis)
        angle = np.deg2rad(angle_deg)
        start = axis * np.cos(angle) + ref * np.sin(angle)
        angles = np.linspace(0.0, 2 * np.pi, segments)
        curve = self._hemisphere_rotate(_rotate_around_axis(start, axis, angles))
        glon, glat = graticule_from_ned(curve[:, 0], curve[:, 1], curve[:, 2])
        glon_a, glat_a = graticule_from_ned(-curve[:, 0], -curve[:, 1], -curve[:, 2])
        all_glon = np.concatenate([glon, [np.nan], glon_a])
        all_glat = np.concatenate([glat, [np.nan], glat_a])
        plot_kwargs = dict(transform=self._data_transform())
        plot_kwargs.update(kwargs)
        (handle,) = self.plot(all_glon, all_glat, **plot_kwargs)
        handle.set_clip_path(self.patch)
        return handle

    def contour(
        self,
        features=None,
        filled=True,
        grid=None,
        method="sph",
        grid_n=2000,
        grid_type="gss",
        sigma=None,
        n_max=None,
        trimzero=True,
        show_data=False,
        data_kws=None,
        **kwargs,
    ):
        """Plot a density-contour diagram of axial data (poles/lines).

        Creates a new :class:`~stereonet.StereoGrid`, computes its density
        via ``calculate_density(features, method=...)``, and draws it with
        ``tricontourf``/``tricontour``. Each call appends its
        ``StereoGrid`` to ``self.stereogrids`` rather than replacing a
        previous one, so multiple independent contour layers can coexist
        on the same axes.

        By default (``clip=False``) the *full* value range is contoured
        -- negative values (regions *less* dense than a uniform/random
        distribution would predict) are just as meaningful as positive
        ones under this statistic and are shown, not hidden -- using a
        diverging colormap (``cmap="Spectral"`` unless overridden) whose
        normalization is centered so 0 (the uniform/null expectation)
        always maps to the exact middle of the colormap
        (:class:`~matplotlib.colors.CenteredNorm`), regardless of how
        asymmetric the actual min/max happen to be.

        Pass ``clip=True`` to restrict to the positive (above-uniform)
        region only. If you pass an explicit sequential ``cmap=`` without
        also passing ``clip=``, ``clip`` is inferred to be ``True``.

        Args:
            features: array-like of shape (N, 3) -- NED unit vectors to
                compute the density of. Ignored (may be omitted) if
                ``grid`` is given.

        Keyword Args:
            filled (bool): filled contours (``tricontourf``) if True,
                contour lines (``tricontour``) if False. Default True.
            grid (StereoGrid): an already-populated grid to draw directly
                (e.g. built via ``apply_func``/``angmech``, or shared
                across multiple calls/axes) -- skips creating a new one
                and calling ``calculate_density``.
            method, sigma, n_max, trimzero: passed to
                ``StereoGrid.calculate_density`` (see there) when ``grid``
                is not given.
            grid_n, grid_type: passed to ``StereoGrid()`` when ``grid`` is
                not given.
            show_data (bool): also plot `features` as points (via
                ``self.point``). Only meaningful when ``grid`` is not
                given. Default False.
            data_kws (dict): kwargs passed to ``self.point`` when
                ``show_data`` is True.
            clip (bool): restrict to the positive (above-uniform) region
                only -- see above. Default False, or inferred True if an
                explicit sequential ``cmap=`` is given without ``clip=``.
            **kwargs: passed to ``tricontourf``/``tricontour`` (``levels``,
                ``cmap``, ``colors``, ``alpha``, ``linewidths``,
                ``linestyles``, ...).

        Returns:
            The ``ContourSet``. The ``StereoGrid`` used is available as
            ``self.stereogrids[-1]`` (or the ``grid`` passed in).
        """
        if grid is None:
            grid = StereoGrid(grid_n=grid_n, grid_type=grid_type)
            grid.calculate_density(
                features,
                method=method,
                sigma=sigma,
                n_max=n_max,
                trimzero=trimzero,
            )
        self.stereogrids.append(grid)

        # The counting grid spans the *whole* sphere, so it needs the same
        # axial hemisphere handling as point()/project(fold=True) -- see
        # _fold_axial_to_data -- rather than _data_axes_fraction's
        # full-negation rule (correct only for
        # genuinely directional data): that would fold each grid point to
        # this axes' hemisphere and then negate it right back via
        # _vec_to_data, leaving every point at its "lower" position
        # regardless of hemisphere. Left unfolded at all, about half the
        # grid projects outside the unit disk (r up to sqrt(2) at the
        # antipode) into a self-overlapping fringe that Delaunay-
        # triangulates against the in-disk points and produces spurious
        # holes near the extrema.
        glon, glat = self._fold_axial_to_data(grid.grid)
        X, Y = self._graticule_to_axes_fraction(glon, glat)
        values = grid.values
        # A handful of counting-grid points can land close enough to the
        # projection's own antipodal singularity to be masked to NaN (see
        # transform_non_affine) --
        # interpolation can't handle NaN input, so drop them here rather
        # than let a rare, individually-insignificant point crash the
        # whole contour.
        finite = np.isfinite(X) & np.isfinite(Y)
        if not finite.all():
            X, Y, values = X[finite], Y[finite], values[finite]
            glon, glat = glon[finite], glat[finite]

        Xa, Ya, Va = _extend_antipodal(
            glon, glat, values, self._graticule_to_axes_fraction
        )
        X = np.concatenate([X, Xa])
        Y = np.concatenate([Y, Ya])
        values = np.concatenate([values, Va])

        Xg, Yg, Vg = _resample_to_grid(X, Y, values)

        levels = kwargs.pop("levels", 6)
        clip = _resolve_cmap_and_clip(kwargs)
        # Levels are bounded by the *resampled* field, not the raw scatter:
        # cubic resampling can overshoot slightly past the raw data's own
        # min/max right at a sharp local peak/trough (most likely exactly
        # where a steep gradient sits close to the primitive circle, now
        # well-supported by _extend_antipodal's accurate antipodal
        # companions) -- bounding levels by the raw scatter instead would
        # let the resampled field poke a small hole through the top (or
        # bottom) level's edge, left unfilled by contourf's default
        # extend="neither".
        finite_Vg = np.ma.asarray(Vg).compressed()
        levels = _resolve_levels(levels, finite_Vg, clip)
        levels = _avoid_plateau_at_top(levels, finite_Vg, filled)

        # Axes.contourf/.contour, not self.contourf/self.contour: this
        # method's own name shadows Axes.contour, so `self.contour(...)`
        # here would recurse into this very method instead of calling
        # matplotlib's regular-grid contour renderer.
        contour_fn = Axes.contourf if filled else Axes.contour
        cs = contour_fn(
            self, Xg, Yg, Vg, levels=levels, transform=self.transAxes, **kwargs
        )
        cs.set_clip_path(self.patch)

        if show_data and features is not None:
            self.point(features, **(data_kws or {}))

        return cs


class SchmidtNetAxes(StereonetAxes):
    """Lower/upper-hemisphere equal-area ("Schmidt") stereographic net."""

    name = "schmidt"
    _base_transform = EqualAreaTransform


class WulffNetAxes(StereonetAxes):
    """Lower/upper-hemisphere equal-angle ("Wulff") stereographic net."""

    name = "wulff"
    _base_transform = EqualAngleTransform


register_projection(SchmidtNetAxes)
register_projection(WulffNetAxes)
