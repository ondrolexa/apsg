import math

import numpy as np

from apsg.config import apsg_conf
from apsg.helpers import sind, cosd, tand, asind, acosd, atand, atan2d
from apsg.helpers import geo2vec_linear, vec2geo_linear
from apsg.decorator import ensure_first_arg_same

"""
TO BE ADDED
"""
class Vector:
    __slots__ = ("_coords",)
    pass

class Vector2(Vector):
    __shape__ = (2,)

    def __init__(self, *args):
        if len(args) == 1 and np.asarray(args[0]).shape == Vector2.__shape__:
            coords = args[0]
        elif len(args) == 2:
            coords = args
        else:
            raise TypeError(f'Not valid arguments for {type(self).__name__}')
        self._coords = tuple(coords)


    def __repr__(self):
        n = apsg_conf['ndigits']
        return f'Vector2({round(self.x, n):g}, {round(self.y, n):g})'


class Vector3(Vector):
    __shape__ = (3,)

    def __init__(self, *args):
        if len(args) == 0:
            coords = (0, 0, 1)
        elif len(args) == 1 and np.asarray(args[0]).shape == Vector3.__shape__:
            coords = args[0]
        elif len(args) == 2:
            coords = geo2vec_linear(*args)
        elif len(args) == 3:
            coords = args
        else:
            raise TypeError(f'Not valid arguments for {type(self).__name__}')
        self._coords = tuple(coords)

    @property
    def x(self):
        return self._coords[0]

    @property
    def y(self):
        return self._coords[1]

    @property
    def z(self):
        return self._coords[2]

    def __copy__(self):
        return type(self)(self._coords)

    copy = __copy__

    def __repr__(self):
        if apsg_conf['vec2geo']:
            azi, inc = vec2geo_linear(self)
            return f'V:{azi:.0f}/{inc:.0f}'
        else:
            n = apsg_conf['ndigits']
            return f'Vector3({round(self.x, n):g}, {round(self.y, n):g}, {round(self.z, n):g})'

    def __hash__(self):
        return hash((type(self).__name__,) + self._coords)

    def __array__(self, dtype=None):
        return np.array(self._coords, dtype=dtype)

    def __dict__(self):
        return {'datatype': type(self).__name__,
                'slots':{'_coords': self._coords}}

    @ensure_first_arg_same
    def __eq__(self, other):
        return all([math.isclose(u, v) for u, v in zip(self._coords, other._coords)])

    def __ne__(self, other):
        return not self.__eq__(other)

    def __nonzero__(self):
        return any(self._coords)

    def is_unit(self):
        return math.isclose(self.magnitude(), 1)

    def __len__(self):
        return 3

    def __getitem__(self, key):
        return self._coords[key]

    def __iter__(self):
        return iter(self._coords)

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

    def __neg__(self):
        return type(self)(-self.x, -self.y, -self.z)

    __pos__ = __copy__

    def __abs__(self):
        return math.sqrt(self.x ** 2 + self.y ** 2 + self.z ** 2)

    magnitude = __abs__

    def normalized(self):
        d = self.magnitude()
        if d:
            return type(self)(self.x / d, self.y / d, self.z / d)
        return self.copy()

    uv = normalized

    @ensure_first_arg_same
    def dot(self, other):
        return self.x * other.x + self.y * other.y + self.z * other.z

    def __matmul__(self, other):
        r = np.dot(np.array(self), other)
        if np.asarray(r).shape == Vector3.__shape__:
            return type(self)(r)
        else:
            return float(r)

    def __rmatmul__(self, other):
        r = np.dot(other, np.array(self))
        if np.asarray(r).shape == Vector3.__shape__:
            return type(self)(r)
        else:
            return float(r)

    @ensure_first_arg_same
    def cross(self, other):
        return type(self)(
            self.y * other.z - self.z * other.y,
            -self.x * other.z + self.z * other.x,
            self.x * other.y - self.y * other.x,
        )

    __pow__ = cross

    @classmethod
    def random(cls):
        """
        Random Pair
        """
        return cls(np.random.randn(3)).normalized()

    @ensure_first_arg_same
    def rotate(self, axis, theta):
        """Return the vector rotated around axis through angle theta. Right hand rule applies"""
        v = Vector3(self)  # ensure vector
        k = axis.uv()
        return type(self)(
            cosd(theta) * v
            + sind(theta) * k.cross(v)
            + (1 - cosd(theta)) * k * (k.dot(v))
        )

    @ensure_first_arg_same
    def angle(self, other):
        """Return the angle to the vector other"""
        return acosd(self.uv().dot(other.uv()))

    @ensure_first_arg_same
    def project(self, other):
        """Return one vector projected on the vector other"""
        n = other.uv()
        return type(self)(self.dot(n) * n)

    # 
    proj = project

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
            >>> F = [[1, 0, 0], [0, -1, 0], [0, 0, 1]]
            >>> u = Vector3([1, 1, 1])
            >>> u.transform(F)
            V(1.000, -1.000, 1.000)

        """
        r = Vector3(np.dot(args[0], self))
        if kwargs.get("norm", False):
            r = r.normalized()
        return type(self)(r)


class Axial3(Vector3):
    @ensure_first_arg_same
    def dot(self, other):
        return abs(self.x * other.x + self.y * other.y + self.z * other.z)
