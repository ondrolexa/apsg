import numpy as np

from apsg.config import apsg_conf
from apsg.decorator._decorator import ensure_first_arg_same
from apsg.math._vector import Vector3, Vector2

"""
TO BE ADDED
"""


class Matrix:
    """Base class for Matrix2 and Matrix3"""

    __slots__ = "_coefs"

    def __init__(self):
        self._cache = {}

    def __copy__(self):
        return type(self)(self._coefs)

    copy = __copy__

    @property
    def flat_coefs(self):
        return tuple(c for row in self._coefs for c in row)

    def __repr__(self):
        n = apsg_conf["ndigits"]
        m = [[round(e, n) for e in row] for row in self._coefs]
        return str(np.array(m))

    def __hash__(self):
        return hash((type(self).__name__,) + self._coefs)

    def to_json(self):
        return {"datatype": type(self).__name__, "args": (self._coefs,)}

    def __array__(self, dtype=None):
        return np.array(self._coefs, dtype=dtype)

    def __nonzero__(self):
        return not np.allclose(self, np.zeros(self.__shape__))

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

    pos__ = __copy__

    def __getitem__(self, key):
        # need fix
        return self._coefs[key]

    def __iter__(self):
        # what we want to iterate?
        return iter(self._coefs)

    def __mul__(self, other):
        return type(self)(np.multiply(self, other))

    __rmul__ = __mul__

    def __pow__(self, n):
        return type(self)(np.linalg.matrix_power(self, n))

    @ensure_first_arg_same
    def __eq__(self, other):
        return np.allclose(self, other)

    def __ne__(self, other):
        return not self.__eq__(other)

    @property
    def I(self):
        return type(self)(np.linalg.inv(self))

    @property
    def T(self):
        return type(self)(np.array(self).T)

    @ensure_first_arg_same
    def transform(self, other):
        """
        Coordinate transformations of matrix

        Using rotation matrix it returns ``A' = R * A * R . T``.
        """
        return type(self)(other @ self @ other.T)

    @property
    def _svd(self):
        if "svd" not in self._cache:
            self._cache["svd"] = np.linalg.svd(self._coefs)
        return self._cache["svd"]

    def eigenvalues(self):
        """Return sorted tuple of principal eigenvalues"""
        return tuple(self._svd[1])

    @property
    def det(self):
        """Determinant"""

        return float(np.linalg.det(self))

    @property
    def E1(self):
        """Max eigenvalue"""

        return self.eigenvalues()[0]

    @property
    def E2(self):
        """Middle eigenvalue"""

        return self.eigenvalues()[1]

    @property
    def V1(self):
        """Max eigenvector"""

        return self.eigenvectors()[0]

    @property
    def V2(self):
        """Middle eigenvector"""

        return self.eigenvectors()[1]


class Matrix2(Matrix):
    __shape__ = (2, 2)

    def __init__(self, *args):
        super().__init__()
        if len(args) == 0:
            coefs = ((1, 0), (0, 1))
        elif len(args) == 1 and np.asarray(args[0]).shape == Matrix2.__shape__:
            coefs = [[float(v) for v in row] for row in args[0]]
        else:
            raise TypeError("Not valid arguments for Matrix2")
        self._coefs = tuple(coefs[0]), tuple(coefs[1])

    @classmethod
    def from_comp(cls, xx=1, xy=0, yx=0, yy=1):
        """Return ``Matrix2`` defined by individual components. Default is identity tensor.

        Keyword Args:
          xx, xy, yx, yy (float): tensor components

        Example:
          >>> F = Matrix2.from_comp(xy=2)
          >>> F
          [[1. 2.]
           [0. 1.]]

        """

        return cls([[xx, xy], [yx, yy]])

    def __len__(self):
        return 2

    def dot(self, other):
        return Vector2(np.dot(np.array(self), other))

    def __matmul__(self, other):
        r = np.dot(np.array(self), other)
        if np.asarray(r).shape == Matrix2.__shape__:
            return type(self)(r)
        else:
            return Vector2(r)

    def __rmatmul__(self, other):
        r = np.dot(other, np.array(self))
        if np.asarray(r).shape == Matrix2.__shape__:
            return type(self)(r)
        else:
            return Vector2(r)

    def eigenvectors(self):
        """Return tuple of principal eigenvectors as ``Vector3`` objects."""
        U = self._svd[0].T
        return Vector2(U[0]), Vector2(U[1])

    def scaled_eigenvectors(self):
        """Return tuple of principal eigenvectors as ``Vector3`` objects with
        magnitudes of eigenvalues"""
        U = self._svd[0].T
        return self.E1 * Vector2(U[0]), self.E2 * Vector2(U[1])


class Matrix3(Matrix):
    __shape__ = (3, 3)

    def __init__(self, *args):
        super().__init__()
        if len(args) == 0:
            coefs = ((1, 0, 0), (0, 1, 0), (0, 0, 1))
        elif len(args) == 1 and np.asarray(args[0]).shape == Matrix3.__shape__:
            coefs = [[float(v) for v in row] for row in args[0]]
        else:
            raise TypeError("Not valid arguments for Matrix3")
        self._coefs = tuple(coefs[0]), tuple(coefs[1]), tuple(coefs[2])

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

    def __len__(self):
        return 3

    def dot(self, other):
        return Vector3(np.dot(np.array(self), other))

    def __matmul__(self, other):
        r = np.dot(np.array(self), other)
        if np.asarray(r).shape == Matrix3.__shape__:
            return type(self)(r)
        else:
            return Vector3(r)

    def __rmatmul__(self, other):
        r = np.dot(other, np.array(self))
        if np.asarray(r).shape == Matrix3.__shape__:
            return type(self)(r)
        else:
            return Vector3(r)

    @property
    def E3(self):
        """Min eigenvalue"""

        return self.eigenvalues()[2]

    @property
    def V3(self):
        """Min eigenvector"""

        return self.eigenvectors()[2]

    def eigenvectors(self):
        """Return tuple of principal eigenvectors as ``Vector3`` objects."""
        U = self._svd[0].T
        return Vector3(U[0]), Vector3(U[1]), Vector3(U[2])

    def scaled_eigenvectors(self):
        """Return tuple of principal eigenvectors as ``Vector3`` objects with
        magnitudes of eigenvalues"""
        U = self._svd[0].T
        return self.E1 * Vector2(U[0]), self.E2 * Vector2(U[1]), self.E3 * Vector3(U[2])
