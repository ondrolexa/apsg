import math

import numpy as np

from apsg.config import apsg_conf
from apsg.helpers._math import sind, cosd, acosd, atan2d
from apsg.helpers._notation import (
    geo2vec_linear,
    vec2geo_linear_signed,
)
from apsg.decorator._decorator import ensure_first_arg_same


class Vector:
    """
    Base class for Vector2 and Vector3
    """

    __slots__ = ("_coords",)

    def __copy__(self):
        return type(self)(self._coords)

    copy = __copy__

    def __hash__(self):
        return hash((type(self).__name__,) + self._coords)

    def __array__(self, dtype=None, copy=None):
        return np.array(self._coords, dtype=dtype)

    def to_json(self):
        return {"datatype": type(self).__name__, "args": (self._coords,)}

    @ensure_first_arg_same
    def __eq__(self, other):
        return np.allclose(self, other)

    def __ne__(self, other):
        return not self.__eq__(other)

    def __nonzero__(self):
        return any(self._coords)

    #    def __getitem__(self, key):
    #        return self._coords[key]

    #    def __iter__(self):
    #        return iter(self._coords)

    def __add__(self, other):
        return type(self)(np.add(self, other))

    __radd__ = __add__

    def __sub__(self, other):
        return type(self)(np.subtract(self, other))

    def __rsub__(self, other):
        return type(self)(np.subtract(other, self))

    def __mul__(self, other):
        return type(self)(np.multiply(self, other))

    __rmul__ = __mul__

    def __div__(self, other):
        return type(self)(np.divide(self, other))

    def __rdiv__(self, other):
        return type(self)(np.divide(other, self))

    def __floordiv__(self, other):
        return type(self)(np.floor_divide(self, other))

    def __rfloordiv__(self, other):
        return type(self)(np.floor_divide(other, self))

    def __truediv__(self, other):
        return type(self)(np.true_divide(self, other))

    def __rtruediv__(self, other):
        return type(self)(np.true_divide(other, self))

    __pos__ = __copy__

    def __abs__(self):
        return math.sqrt(sum(map(lambda x: x * x, self._coords)))

    magnitude = __abs__

    def label(self):
        """Return label"""
        return str(self)

    def is_unit(self):
        """Return true if the magnitude is 1"""
        return math.isclose(self.magnitude(), 1)

    @ensure_first_arg_same
    def angle(self, other):
        """Return the angle to the vector other"""
        return acosd(self.normalized().dot(other.normalized()))

    @ensure_first_arg_same
    def project(self, other):
        """Return vector projectio on the vector other"""
        n = other.normalized()
        return type(self)(self.dot(n) * n)

    proj = project

    @ensure_first_arg_same
    def reject(self, other):
        """Return vector rejection on the vector other"""
        return self - self.project(other)

    @ensure_first_arg_same
    def slerp(self, other, t):
        """Return a spherical linear interpolation between self and other vector

        Note that for non-unit vectors the interpolation is not uniform
        """
        a, b = Vector3(self), Vector3(other)
        theta = a.angle(b)
        return type(self)(a * sind((1 - t) * theta) + b * sind(t * theta)) / sind(theta)

    @property
    def x(self):
        """Return x-component of the vector"""
        return self._coords[0]

    @property
    def y(self):
        """Return y-component of the vector"""
        return self._coords[1]


class Vector2(Vector):
    """
    A class to represent a 2D vector.

    There are different way to create ``Vector2`` object:

    - without arguments create default ``Vector2`` (0, 0, 1)
    - with single argument `v`, where

        - `v` could be Vector2-like object
        - `v` could be string 'x' or 'y' - principal axes of coordinate system
        - `v` could be tuple of (x, y) - vector components
        - `v` could be float - unit vector with given angle to 'x' axis
    - with 2 numerical arguments defining vector components

    Args:
        ang (float): angle between 'x' axis and vector in degrees

    Example:
        >>> vec2()
        >>> vec2(1, -1)
        >>> vec2('y')
        >>> vec2(50)
        >>> v = vec2(1, -2)

    """

    __shape__ = (2,)

    def __init__(self, *args):
        if len(args) == 0:
            coords = (1, 0)
        if len(args) == 1:
            if np.asarray(args[0]).shape == Vector2.__shape__:
                coords = tuple(c.item() for c in np.asarray(args[0]))
            elif isinstance(args[0], str):
                if args[0].lower() == "x":
                    coords = (1, 0)
                elif args[0].lower() == "y":
                    coords = (0, 1)
                else:
                    raise TypeError(f"Not valid arguments for {type(self).__name__}")
            else:
                coords = cosd(args[0]), sind(args[0])
        elif len(args) == 2:
            coords = args
        else:
            raise TypeError(f"Not valid arguments for {type(self).__name__}")
        self._coords = tuple(coords)

    def __repr__(self):
        n = apsg_conf["ndigits"]
        return f"Vector2({round(self.x, n):g}, {round(self.y, n):g})"

    def __len__(self):
        return 3

    def __neg__(self):
        return type(self)(-self.x, -self.y)

    def normalized(self):
        """Returns normalized (unit length) vector"""
        d = self.magnitude()
        if d:
            return type(self)(self.x / d, self.y / d)
        return self.copy()

    uv = normalized

    shape = __shape__

    @property
    def direction(self):
        """Returns direction of the vector in degrees"""
        return atan2d(self.y, self.x)

    @ensure_first_arg_same
    def dot(self, other):
        """
        Calculate dot product with other vector.

        Args:
            other (Vector2): other vector
        """
        return self.x * other.x + self.y * other.y

    def __matmul__(self, other):
        r = np.dot(self, other)
        if np.asarray(r).shape == Vector2.__shape__:
            return type(self)(r)
        else:
            return float(r)

    def __rmatmul__(self, other):
        r = np.dot(other, self)
        if np.asarray(r).shape == Vector2.__shape__:
            return type(self)(r)
        else:
            return float(r)

    @ensure_first_arg_same
    def cross(self, other):
        """Returns the magnitude of the vector that would result from a regular 3D
        cross product of the input vectors, taking their Z values implicitly as 0
        (i.e. treating the 2D space as a plane in the 3D space). The 3D cross
        product will be perpendicular to that plane, and thus have 0 X & Y components
        (thus the scalar returned is the Z value of the 3D cross product vector).

        Note that the magnitude of the vector resulting from 3D cross product is also
        equal to the area of the parallelogram between the two vectors. In addition,
        this area is signed and can be used to determine whether rotating from V1 to V2
        moves in an counter clockwise or clockwise direction.
        """

        return self.x * other.y - self.y * other.x

    @classmethod
    def random(cls):
        """
        Random 2D vector
        """
        return cls(360 * np.random.rand())

    @ensure_first_arg_same
    def rotate(self, axis, theta):
        """Return the vector rotated through angle theta. Right hand rule applies"""
        return NotImplemented

    @classmethod
    def unit_x(cls):
        """Create unit length vector in x-direction"""
        return cls(1, 0)

    @classmethod
    def unit_y(cls):
        """Create unit length vector in y-direction"""
        return cls(0, 1)

    def transform(self, *args, **kwargs):
        """
        Return affine transformation of vector `u` by matrix `F`.

        Args:
            F: transformation matrix

        Keyword Args:
            norm: normalize transformed vectors. [True or False] Default False

        Returns:
            vector representation of affine transformation (dot product)
            of `self` by `F`

        Example:
            # Reflexion of `y` axis.
            >>> F = [[1, 0], [0, -1]]
            >>> u = vec2([1, 1])
            >>> u.transform(F)
            Vector2(1, -1)

        """
        r = Vector2(np.dot(args[0], self))
        if kwargs.get("norm", False):
            r = r.normalized()
        return type(self)(r)


class Axial2(Vector2):  # Do we need it?
    """
    A class to represent a 2D axial vector.

    Note: the angle between axial data cannot be more than 90°
    """

    @ensure_first_arg_same
    def __eq__(self, other):
        return np.allclose(self, other) or np.allclose(self, -other)

    def __add__(self, other):
        if issubclass(type(other), Vector2):
            if super().dot(other) < 0:
                other = -other
        return type(self)(np.add(self, other))

    __radd__ = __add__

    def __sub__(self, other):
        if issubclass(type(other), Vector2):
            if super().dot(other) < 0:
                other = -other
        return type(self)(np.subtract(self, other))

    def __rsub__(self, other):
        if issubclass(type(other), Vector2):
            if super().dot(other) < 0:
                other = -other
        return type(self)(np.subtract(other, self))

    def dot(self, other):
        return abs(super().dot(other))


class Vector3(Vector):
    """
    A class to represent a 3D vector.

    There are different way to create ``Vector3`` object:

    - without arguments create default ``Vector3`` (1, 0, 0)
    - with single argument `v`, where

        - `v` could be Vector3-like object
        - `v` could be string 'x', 'y' or 'z' - principal axes of coordinate system
        - `v` could be tuple of (x, y, z) - vector components
    - with 2 arguments plunge direction and plunge
    - with 3 numerical arguments defining vector components

    Args:
        azi (float): plunge direction of linear feature in degrees
        inc (float): plunge of linear feature in degrees

    Example:
        >>> vec()
        >>> vec(1,2,-1)
        >>> vec('y')
        >>> vec(120, 30)
        >>> v = vec(1, -2, 1)

    """

    __shape__ = (3,)

    def __init__(self, *args):
        if len(args) == 0:
            coords = (1, 0, 0)
        elif len(args) == 1:
            if np.asarray(args[0]).shape == Vector3.__shape__:
                coords = tuple(c.item() for c in np.asarray(args[0]))
            elif isinstance(args[0], str):
                if args[0].lower() == "x":
                    coords = (1, 0, 0)
                elif args[0].lower() == "y":
                    coords = (0, 1, 0)
                elif args[0].lower() == "z":
                    coords = (0, 0, 1)
                else:
                    raise TypeError(f"Not valid arguments for {type(self).__name__}")
            else:
                raise TypeError(f"Not valid arguments for {type(self).__name__}")
        elif len(args) == 2:
            coords = geo2vec_linear(*args)
        elif len(args) == 3:
            coords = args
        else:
            raise TypeError(f"Not valid arguments for {type(self).__name__}")
        self._coords = tuple(coords)

    @property
    def z(self):
        """Return z-component of the vector"""
        return self._coords[2]

    def __repr__(self):
        if apsg_conf["vec2geo"]:
            azi, inc = self.geo
            return f"V:{azi:.0f}/{inc:.0f}"
        else:
            n = apsg_conf["ndigits"]
            return f"Vector3({round(self.x, n):g}, {round(self.y, n):g}, {round(self.z, n):g})"

    def __len__(self):
        return 3

    def __neg__(self):
        return type(self)(-self.x, -self.y, -self.z)

    def __abs__(self):
        return math.sqrt(self.x**2 + self.y**2 + self.z**2)

    def normalized(self):
        """Returns normalized (unit length) vector"""
        d = self.magnitude()
        if d:
            return type(self)(self.x / d, self.y / d, self.z / d)
        return self.copy()

    uv = normalized

    shape = __shape__

    @ensure_first_arg_same
    def dot(self, other):
        """
        Calculate dot product with other vector.

        Args:
            other (Vector3): other vector
        """
        return self.x * other.x + self.y * other.y + self.z * other.z

    def __matmul__(self, other):
        r = np.dot(self, other)
        if np.asarray(r).shape == Vector3.__shape__:
            return type(self)(r)
        else:
            return float(r)

    def __rmatmul__(self, other):
        r = np.dot(other, self)
        if np.asarray(r).shape == Vector3.__shape__:
            return type(self)(r)
        else:
            return float(r)

    def __pow__(self, other):
        if issubclass(type(other), Vector3):
            return self.cross(other)
        else:
            return type(self)(np.power(self, other))

    @ensure_first_arg_same
    def cross(self, other):
        """
        Calculate cross product with other vector.

        Args:
            other (Vector3): other vector
        """
        return type(self)(
            self.y * other.z - self.z * other.y,
            -self.x * other.z + self.z * other.x,
            self.x * other.y - self.y * other.x,
        )

    def lower(self):
        """Change vector direction to point towards positive Z direction"""
        if self.z < 0:
            return -self
        else:
            return self

    def is_upper(self):
        """Return True if vector points towards negative Z direction"""
        return self.z < 0

    @property
    def geo(self):
        """
        Return tuple of plunge direction and signed plunge
        """
        return vec2geo_linear_signed(self)

    @classmethod
    def unit_x(cls):
        """Create unit length vector in x-direction"""
        return cls(1, 0, 0)

    @classmethod
    def unit_y(cls):
        """Create unit length vector in y-direction"""
        return cls(0, 1, 0)

    @classmethod
    def unit_z(cls):
        """Create unit length vector in z-direction"""
        return cls(0, 0, 1)

    @classmethod
    def random(cls):
        """
        Create random 3D vector
        """
        return cls(np.random.randn(3)).normalized()

    @ensure_first_arg_same
    def rotate(self, axis, theta):
        """Return the vector rotated around axis through angle theta. Right-hand rule
        applies
        """
        v = Vector3(self)  # ensure vector
        k = Vector3(axis.uv())
        return type(self)(
            cosd(theta) * v
            + sind(theta) * k.cross(v)
            + (1 - cosd(theta)) * k * (k.dot(v))
        )

    @ensure_first_arg_same
    def angle(self, other):
        """Return the angle to the vector other"""
        return acosd(np.clip(self.uv().dot(other.uv()), -1, 1))

    def transform(self, F, **kwargs):
        """
        Return affine transformation of vector `u` by matrix `F`.

        Args:
            F: transformation matrix

        Keyword Args:
            norm: normalize transformed vectors. [True or False] Default False

        Returns:
            vector representation of affine transformation (dot product)
            of `self` by `F`

        Example:
            # Reflexion of `y` axis.
            >>> F = [[1, 0, 0], [0, -1, 0], [0, 0, 1]]
            >>> u = Vector3([1, 1, 1])
            >>> u.transform(F)
            Vector3(1, -1, 1)

        """
        r = Vector3(np.dot(F, self))
        if kwargs.get("norm", False):
            r = r.normalized()
        return type(self)(r)


class Axial3(Vector3):
    """
    A class to represent a 3D axial vector.

    Note: the angle between axial data cannot be more than 90°
    """

    @ensure_first_arg_same
    def __eq__(self, other):
        return np.allclose(self, other) or np.allclose(self, -other)

    def __add__(self, other):
        if issubclass(type(other), Vector3):
            if super().dot(other) < 0:
                other = -other
        return type(self)(np.add(self, other))

    __radd__ = __add__

    def __sub__(self, other):
        if issubclass(type(other), Vector3):
            if super().dot(other) < 0:
                other = -other
        return type(self)(np.subtract(self, other))

    def __rsub__(self, other):
        if issubclass(type(other), Vector3):
            if super().dot(other) < 0:
                other = -other
        return type(self)(np.subtract(other, self))

    def dot(self, other):
        return abs(super().dot(other))
