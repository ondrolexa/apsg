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
        return f"{type(self).__name__}\n{str(np.array(m))}"

    def label(self):
        return str(type(self).__name__)

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
        if isinstance(key, tuple):
            return self._coefs[key[0]][key[1]]
        else:
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
    def xx(self):
        """Return xx-element of the matrix"""
        return self._coefs[0][0]

    @property
    def xy(self):
        """Return xy-element of the matrix"""
        return self._coefs[0][1]

    @property
    def yx(self):
        """Return yx-element of the matrix"""
        return self._coefs[1][0]

    @property
    def yy(self):
        """Return yy-element of the matrix"""
        return self._coefs[1][1]

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
    def _eigh(self):
        if "eigh" not in self._cache:
            evals, evecs = np.linalg.eigh(self._coefs)
            idx = evals.argsort()[::-1]
            evals = evals[idx]
            evecs = evecs[:, idx]
            self._cache["eigh"] = evals, evecs
        return self._cache["eigh"]

    def eigenvalues(self):
        """Return sorted tuple of principal eigenvalues"""
        return self._eigh[0]

    @property
    def det(self):
        """Determinant"""

        return float(np.linalg.det(self))

    @property
    def E1(self):
        """First eigenvalue"""

        return float(self.eigenvalues()[0])

    @property
    def E2(self):
        """Second eigenvalue"""

        return float(self.eigenvalues()[1])

    @property
    def V1(self):
        """First eigenvector"""

        return self.eigenvectors()[0]

    @property
    def V2(self):
        """Second eigenvector"""

        return self.eigenvectors()[1]


class Matrix2(Matrix):
    """
    A class to represent a 2x2 matrix.

    There are different way to create ``Matrix2`` object:

    - without arguments create default identity ``Matrix2``
    - with single argument of Matrix2-like object

    Args:
        v: 2-dimensional array-like object

    Example:
        >>> Matrix2()
        Matrix2
        [[1 0]
         [0 1]]
        >>> A = Matrix2([[2, 1],[0, 0.5]])

    """

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
        """Return ``Matrix2`` defined by individual components. Default is identity
        tensor.

        Keyword Args:
            xx (float): tensor component M_xx
            xy (float): tensor component M_xy
            yx (float): tensor component M_yx
            yy (float): tensor component M_yy

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
        U = self._eigh[1].T
        return Vector2(U[0]), Vector2(U[1])

    def scaled_eigenvectors(self):
        """Return tuple of principal eigenvectors as ``Vector3`` objects with
        magnitudes of eigenvalues"""
        U = self._eigh[1].T
        return self.E1 * Vector2(U[0]), self.E2 * Vector2(U[1])


class Matrix3(Matrix):
    """
    A class to represent a 3x3 matrix.

    There are different way to create ``Matrix3`` object:

    - without arguments create default identity ``Matrix3``
    - with single argument of Matrix3-like object

    Args:
        v: 2-dimensional array-like object

    Example:
        >>> Matrix3()
        Matrix3
        [[1 0 0]
         [0 1 0]
         [0 0 1]]
        >>> A = Matrix3([[2, 1, 0], [0, 0.5, 0], [0, -0.5, 1]])

    """

    __shape__ = (3, 3)

    def __init__(self, *args):
        super().__init__()
        if len(args) == 0:
            coefs = ((1, 0, 0), (0, 1, 0), (0, 0, 1))
        elif len(args) == 1 and np.asarray(args[0]).shape == Matrix3.__shape__:
            coefs = np.asarray(args[0]).tolist()
        else:
            raise TypeError("Not valid arguments for Matrix3")
        self._coefs = tuple(coefs[0]), tuple(coefs[1]), tuple(coefs[2])

    @classmethod
    def from_comp(cls, xx=1, xy=0, xz=0, yx=0, yy=1, yz=0, zx=0, zy=0, zz=1):
        """Return ``Matrix3`` defined by individual components. Default is identity
        tensor.

        Keyword Args:
            xx (float): tensor component M_xx
            xy (float): tensor component M_xy
            xz (float): tensor component M_xz
            yx (float): tensor component M_yx
            yy (float): tensor component M_yy
            yz (float): tensor component M_yz
            zx (float): tensor component M_zx
            zy (float): tensor component M_zy
            zz (float): tensor component M_zz

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
    def xz(self):
        """Return xz-element of the matrix"""
        return self._coefs[0][2]

    @property
    def yz(self):
        """Return yz-element of the matrix"""
        return self._coefs[1][2]

    @property
    def zx(self):
        """Return zx-element of the matrix"""
        return self._coefs[2][0]

    @property
    def zy(self):
        """Return zy-element of the matrix"""
        return self._coefs[2][1]

    @property
    def zz(self):
        """Return zz-element of the matrix"""
        return self._coefs[2][2]

    @property
    def E3(self):
        """Third eigenvalue"""

        return float(self.eigenvalues()[2])

    @property
    def V3(self):
        """Third eigenvector"""

        return self.eigenvectors()[2]

    def eigenvectors(self):
        """Return tuple of principal eigenvectors as ``Vector3`` objects."""
        U = self._eigh[1].T
        return Vector3(U[0]), Vector3(U[1]), Vector3(U[2])

    def scaled_eigenvectors(self):
        """Return tuple of principal eigenvectors as ``Vector3`` objects with
        magnitudes of eigenvalues"""
        U = self._eigh[1].T
        return self.E1 * Vector3(U[0]), self.E2 * Vector3(U[1]), self.E3 * Vector3(U[2])
