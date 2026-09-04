"""Core matplotlib Transform classes for stereographic net projections.

No apsg dependency.

The two projections (Schmidt/equal-area and Wulff/equal-angle) share a
single "graticule" data-coordinate system for their axes: an auxiliary
``(glon, glat)`` pair (radians) chosen so that the classic curved dip-arc
and small-circle families of a printed stereonet become literal
constant-``glon`` / constant-``glat`` lines, which lets matplotlib's normal
Locator/gridline machinery draw them.

``(glon, glat)`` is standard spherical coordinates, but with the pole
placed on the geographic North axis instead of the vertical axis::

    x = sin(glat)                  # North component
    y = cos(glat) * cos(glon)      # East component
    z = cos(glat) * sin(glon)      # Down component

``glat = +-pi/2`` maps to ``(x, y, z) = (+-1, 0, 0)``, i.e. compass N/S on
the primitive circle, for *every* glon -- so constant-glon lines (meridians)
converge there, matching the way a real net's dip arcs converge at N and S.

``(x, y, z)`` is a unit vector in NED convention (x=North, y=East, z=Down),
matching apsg's vector convention. Between the graticule embedding and the
final projection, an optional 3x3 rotation matrix ``R`` (proper, SO(3)) can
be applied to ``(x, y, z)`` -- this is how the whole net (grid included, not
just plotted data) can be rotated: see ``StereonetAxes.rotation`` in
``_axes.py``. The final projection step from (rotated) ``(x, y, z)`` to plot
coordinates ``(X, Y)`` is the standard lower-hemisphere azimuthal equal-area
/ equal-angle formula.
"""

import numpy as np
from matplotlib.path import Path
from matplotlib.transforms import Transform


def ned_from_graticule(glon, glat):
    """Convert graticule coordinates (radians) to a NED unit vector."""
    x = np.sin(glat)
    y = np.cos(glat) * np.cos(glon)
    z = np.cos(glat) * np.sin(glon)
    return x, y, z


def graticule_from_ned(x, y, z):
    """Convert a NED unit vector to graticule coordinates (radians)."""
    glat = np.arcsin(np.clip(x, -1.0, 1.0))
    glon = np.arctan2(z, y)
    return glon, glat


def rotation_from_axis_angle(axis, angle_deg):
    """A proper (SO(3)) rotation matrix for a right-hand rotation of
    ``angle_deg`` degrees around ``axis`` (NED, any nonzero vector --
    normalized internally), via Rodrigues' rotation formula.

    A standalone, apsg-independent way to build a rotation matrix suitable
    for ``StereoNet.set_rotation``/``StereonetAxes.rotation`` -- apsg's own
    ``Rotation3`` (e.g. ``Rotation3.from_pair(...)``) works equally well,
    since it exposes ``__array__``.
    """
    axis = np.asarray(axis, dtype=float)
    axis = axis / np.linalg.norm(axis)
    theta = np.deg2rad(angle_deg)
    c, s = np.cos(theta), np.sin(theta)
    x, y, z = axis
    # Rodrigues' rotation formula: R = I*cos(t) + sin(t)*[axis]_x + (1-cos(t))*axis(x)axis^T
    cross_matrix = np.array([[0, -z, y], [z, 0, -x], [-y, x, 0]])
    outer = np.outer(axis, axis)
    return c * np.eye(3) + s * cross_matrix + (1.0 - c) * outer


class StereonetTransform(Transform):
    """Abstract base for the forward (data -> plot) stereonet transforms."""

    input_dims = 2
    output_dims = 2
    is_separable = False

    def __init__(self, resolution=1, rotation=None):
        """``resolution`` is the number of steps used to interpolate each
        two-point input segment (e.g. a gridline) so it renders as a smooth
        curve in projected space. ``rotation`` is an optional 3x3 proper
        rotation matrix applied to the NED vector before projecting;
        defaults to the identity (no rotation)."""
        super().__init__()
        self._resolution = resolution
        self._rotation = np.eye(3) if rotation is None else np.asarray(rotation)

    def set_rotation(self, rotation):
        """Update the rotation matrix in place and invalidate this
        transform, so anything composed from it (``ax.transData`` and
        everything built on top, e.g. already-plotted artists) picks up
        the change on next draw without needing to be rebuilt."""
        self._rotation = np.eye(3) if rotation is None else np.asarray(rotation)
        self.invalidate()

    def transform_non_affine(self, values):
        glon = values[:, 0]
        glat = values[:, 1]
        x, y, z = ned_from_graticule(glon, glat)
        v = np.column_stack([x, y, z]) @ self._rotation.T
        X, Y = self._project(v[:, 0], v[:, 1], v[:, 2])
        # near the projection's own antipodal singularity (rotated z -> -1),
        # the azimuthal angle becomes extremely sensitive to position -- the
        # (X, Y) radius stays bounded (max sqrt(2)), but two nearby graticule
        # points can land at very different azimuths there. With gridlines
        # interpolated at a coarse, fixed resolution, two adjacent samples
        # straddling this region can be nearly antipodal on that small
        # circle, and the straight segment connecting them cuts across the
        # whole disk as a spurious chord (this is what an oblique,
        # non-vertical `rotation` can bring within reach of an otherwise-
        # ordinary curve, e.g. a meridian, that never approached it when
        # unrotated). NaN out points within a fixed safety margin of the
        # singularity so the line gets a gap there instead -- verified this
        # margin comfortably exceeds the per-step azimuthal swing for the
        # interpolation resolutions in use.
        near_antipode = (1.0 + v[:, 2]) < 1e-2
        X = np.where(near_antipode, np.nan, X)
        Y = np.where(near_antipode, np.nan, Y)
        return np.column_stack([X, Y])

    def transform_path_non_affine(self, path):
        if len(path.vertices) == 2:
            ipath = path.interpolated(self._resolution)
        else:
            ipath = path
        return Path(self.transform(ipath.vertices), ipath.codes)

    def _project(self, x, y, z):
        """Subclasses implement the (x, y, z) -> (X, Y) projection."""
        raise NotImplementedError


class InvertedStereonetTransform(Transform):
    """Abstract base for the inverse (plot -> data) stereonet transforms."""

    input_dims = 2
    output_dims = 2
    is_separable = False

    def __init__(self, resolution=1, rotation=None):
        super().__init__()
        self._resolution = resolution
        self._rotation = np.eye(3) if rotation is None else np.asarray(rotation)

    def set_rotation(self, rotation):
        """See ``StereonetTransform.set_rotation``."""
        self._rotation = np.eye(3) if rotation is None else np.asarray(rotation)
        self.invalidate()

    def transform_non_affine(self, values):
        X = values[:, 0]
        Y = values[:, 1]
        x, y, z = self._unproject(X, Y)
        v = np.column_stack([x, y, z]) @ self._rotation
        glon, glat = graticule_from_ned(v[:, 0], v[:, 1], v[:, 2])
        return np.column_stack([glon, glat])

    def _unproject(self, X, Y):
        """Subclasses implement the (X, Y) -> (x, y, z) inverse projection."""
        raise NotImplementedError


class EqualAreaTransform(StereonetTransform):
    """The Schmidt (equal-area / Lambert azimuthal) forward transform."""

    def _project(self, x, y, z):
        denom = np.where(np.isclose(1.0 + z, 0.0), 1e-12, 1.0 + z)
        sqz = np.sqrt(1.0 / denom)
        return y * sqz, x * sqz

    def inverted(self):
        return InvertedEqualAreaTransform(self._resolution, self._rotation)


class InvertedEqualAreaTransform(InvertedStereonetTransform):
    """The Schmidt (equal-area / Lambert azimuthal) inverse transform."""

    def _unproject(self, X, Y):
        X2, Y2 = X * np.sqrt(2.0), Y * np.sqrt(2.0)
        r2 = X2 * X2 + Y2 * Y2
        scale = np.sqrt(np.clip(1.0 - r2 / 4.0, 0.0, None))
        x = Y2 * scale
        y = X2 * scale
        z = 1.0 - r2 / 2.0
        return x, y, z

    def inverted(self):
        return EqualAreaTransform(self._resolution, self._rotation)


class EqualAngleTransform(StereonetTransform):
    """The Wulff (equal-angle / stereographic) forward transform."""

    def _project(self, x, y, z):
        denom = np.where(np.isclose(1.0 + z, 0.0), 1e-12, 1.0 + z)
        return y / denom, x / denom

    def inverted(self):
        return InvertedEqualAngleTransform(self._resolution, self._rotation)


class InvertedEqualAngleTransform(InvertedStereonetTransform):
    """The Wulff (equal-angle / stereographic) inverse transform."""

    def _unproject(self, X, Y):
        r2 = X * X + Y * Y
        x = 2.0 * Y / (1.0 + r2)
        y = 2.0 * X / (1.0 + r2)
        z = (1.0 - r2) / (1.0 + r2)
        return x, y, z

    def inverted(self):
        return EqualAngleTransform(self._resolution, self._rotation)
