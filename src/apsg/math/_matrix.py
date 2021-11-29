import math

import numpy as np

from apsg.config import apsg_conf
from apsg.helpers import sind, cosd, tand, asind, acosd, atand, atan2d
from apsg.helpers import is_like_vec3, is_like_matrix3
from apsg.decorator import ensure_one_arg_matrix
from apsg.math import Vec3

"""
TO BE ADDED
"""

class Matrix3:
    __slots__ = ("_coefs", "eigenvals", "_U", "_V")   # FIX

    def __init__(self, *args):
        if len(args) == 0:
            coefs = ((1, 0, 0), (0, 1, 0), (0, 0, 1))
        elif len(args) == 1 and is_like_matrix3(args[0]):
            coefs = [[float(v) for v in row] for row in args[0]]
        else:
            raise TypeError("Not valid arguments for Matrix3")
        self._coefs = (tuple(coefs[0]), tuple(coefs[1]), tuple(coefs[2]))
        U, s, V = np.linalg.svd(self._coefs)
        self.eigenvals = tuple(s)
        self._U = Vec3(U.T[0]), Vec3(U.T[1]), Vec3(U.T[2])
        self._V = Vec3(V[0]), Vec3(V[1]), Vec3(V[2])

    @classmethod
    def from_comp(cls, xx=1, xy=0, xz=0, yx=0, yy=1, yz=0, zx=0, zy=0, zz=1):
        """Return ``Matrix3`` defined by individual components. Default is identity tensor.

        Keyword Args:
          xx, xy, xz, yx, yy, yz, zx, zy, zz (float): tensor components

        Example:
          >>> F = Matrix3.from_comp(xy=1, zy=-0.5)
          >>> F
          [[ 1.   1.   0. ]
           [ 0.   1.   0. ]
           [ 0.  -0.5  1. ]]

        """

        return cls([[xx, xy, xz], [yx, yy, yz], [zx, zy, zz]])

    @classmethod
    def from_axisangle(cls, axis, theta):
        """Return ``Matrix3`` rotation matrix representing rotation around axis.

        Args:
          axis: Rotation axis as ``Vec3`` like object
          theta: Angle of rotation in degrees

        Example:
          >>> F = Matrix3.from_axis(Lin(120, 30), 45)
        """

        x, y, z = axis.normalized()
        c, s = cosd(theta), sind(theta)
        xs, ys, zs = x * s, y * s, z * s
        xc, yc, zc = x * (1 - c), y * (1 - c), z * (1 - c)
        xyc, yzc, zxc = x * yc, y * zc, z * xc

        return cls(
            [
                [x * xc + c, xyc - zs, zxc + ys],
                [xyc + zs, y * yc + c, yzc - xs],
                [zxc - ys, yzc + xs, z * zc + c],
            ]
        )

    def __copy__(self):
        return type(self)(self._coefs)

    copy = __copy__

    def __repr__(self):
        n = apsg_conf["ndigits"]
        m = [[round(e, n) for e in row] for row in self._coefs]
        return str(np.array(m))

    def __hash__(self):
        return hash(self._coefs)

    def __array__(self, dtype=None):
        return np.array(self._coefs, dtype=dtype)

    def __len__(self):
        return 3

    def __getitem__(self, key):
        return self._coefs[key]

    def __iter__(self):
        return iter(self._coefs)

    def __mul__(self, other):
        return type(self)(np.multiply(self, other))

    __rmul__ = __mul__

    def __pow__(self, n):
        return type(self)(np.linalg.matrix_power(self, n))

    @ensure_one_arg_matrix
    def __eq__(self, other):
        return bool(np.sum(abs(self - other)) < 1e-14)

    def __ne__(self, other):
        return not self.__eq__(other)

    def dot(self, other):
        return Vec3(np.dot(np.array(self), other))

    def __matmul__(self, other):
        r = np.dot(np.array(self), other)
        if is_like_matrix3(r):
            return type(self)(r)
        else:
            return Vec3(r)

    def __rmatmul__(self, other):
        r = np.dot(other, np.array(self))
        if is_like_matrix3(r):
            return type(self)(r)
        else:
            return Vec3(r)

    @property
    def I(self):
        return type(self)(np.linalg.inv(self))

    @property
    def T(self):
        return type(self)(np.array(self).T)

    def rotate(self, axis, theta=0):
        """
        Rotate tensor around axis by angle theta.

        Using rotation matrix it returns ``F = R * F * R . T``.
        """

        R = self.from_axisangle(axis, theta)
        return type(self)(R @ self @ R.T)

    @property
    def E1(self):
        """Max eigenvalue"""

        return self.eigenvals[0]

    @property
    def E2(self):
        """Middle eigenvalue"""

        return self.eigenvals[1]

    @property
    def E3(self):
        """Min eigenvalue"""

        return self.eigenvals[2]

    @property
    def eigenvects(self):
        """Return ```Group``` of principal eigenvectors as ``Vec3`` objects."""

        return self._U
