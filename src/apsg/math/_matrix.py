from abc import ABC, abstractmethod

import numpy as np

from apsg.config import apsg_conf
from apsg.decorator._decorator import ensure_first_arg_same
from apsg.helpers._helper import is_jsonable
from apsg.math._vector import Vector2, Vector3


class Matrix(ABC):
    """Abstarct base class for Matrix2 and Matrix3"""

    __slots__ = ("_coefs", "_attrs", "_cache")
    __shape__ = (0, 0)

    @abstractmethod
    def __init__(self, *args, **kwargs):
        self._coefs = ((None, None, None), (None, None, None), (None, None, None))
        self._attrs = {}
        self._cache = {}

    def __copy__(self):
        return type(self)(self._coefs)

    copy = __copy__

    @property
    def flat_coefs(self):
        return tuple(c for row in self._coefs for c in row)

    def __repr__(self):
        n = apsg_conf.ndigits
        return f"{self.label()}\n{str(np.asarray(self._coefs).round(n))}"

    def label(self):
        return self.__class__.__name__

    def __hash__(self):
        return hash((self.label(),) + self._coefs)

    def to_json(self):
        return {
            "datatype": self.label(),
            "args": (self._coefs,),
            "kwargs": self._attrs,
        }

    def __array__(self, dtype=None, copy=None):
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
    def I(self):  # noqa: E743
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
    def _eig(self):
        if "eig" not in self._cache:
            evals, evecs = np.linalg.eig(np.asarray(self._coefs))
            idx = evals.argsort()[::-1]
            evals = evals[idx]
            # round very small numbers to zero
            evals[np.isclose(evals, np.zeros_like(evals))] = 0
            evecs = evecs[:, idx]
            self._cache["eig"] = evals, evecs
        return self._cache["eig"]

    def eigenvalues(self, which=None):
        """Return eigenvalues

        Args:
            which: if None returns sorted tuple of eigenvalues.
                If int returns given eigenvalue. Default None.

        """
        if which is None:
            return self._eig[0]
        else:
            return self._eig[0][which]

    @property
    def det(self):
        """Determinant"""

        return float(np.linalg.det(self))


class Matrix2(Matrix):
    """
    A class to represent a 2x2 matrix.

    There are different way to create ``Matrix2`` object:

    - without arguments create default identity ``Matrix2``
    - with single argument of Matrix2-like object

    Args:
        v: 2-dimensional array-like object

    Example:
        >>> matrix2()
        Matrix2
        [[1 0]
         [0 1]]
        >>> A = Matrix2([[2, 1],[0, 0.5]])

    """

    __slots__ = ("_coefs", "_attrs")
    __shape__ = (2, 2)

    def __init__(self, *args, **kwargs):
        super().__init__()
        if len(args) == 0:
            coefs = ((1, 0), (0, 1))
        elif len(args) == 1 and np.asarray(args[0]).shape == Matrix2.__shape__:
            coefs = [[float(v) for v in row] for row in args[0]]
        else:
            raise TypeError("Not valid arguments for Matrix2")
        self._coefs = tuple(coefs[0]), tuple(coefs[1])
        if is_jsonable(kwargs):
            self._attrs = kwargs
        else:
            raise TypeError("Provided attributes are not serializable.")

    @classmethod
    def from_comp(cls, xx=0, xy=0, yx=0, yy=0):
        """Return ``Matrix2`` defined by individual components. Default is zero
        matrix.

        Keyword Args:
            xx (float): tensor component M_xx
            xy (float): tensor component M_xy
            yx (float): tensor component M_yx
            yy (float): tensor component M_yy

        Example:
            >>> M = matrix2.from_comp(xy=2)
            >>> M
            Matrix2
            [[0. 2.]
             [0. 0.]]

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

    def eigenvectors(self, which=None):
        """Return eigenvectors as ``Vector2`` objects.

        Args:
            which: if None returns sorted tuple of eigenvectors.
                If int returns given eigenvector. Default None.
        """
        U = self._eig[1].T
        if which is None:
            return Vector2(U[0]), Vector2(U[1])
        else:
            return Vector2(U[which])

    def scaled_eigenvectors(self, which=None):
        """Return eigenvectors with magnitudes of eigenvalues as
        ``Vector2`` objects

        Args:
            which: if None returns sorted tuple of eigenvectors.
                If int returns given eigenvector. Default None.
        """
        U = self._eig[1].T
        if which is None:
            return self._eig[0] * Vector2(U[0]), self._eig[1] * Vector2(U[1])
        else:
            return self._eig[0][which] * Vector2(U[which])


class Matrix3(Matrix):
    """
    A class to represent a 3x3 matrix.

    There are different way to create ``Matrix3`` object:

    - without arguments create default identity ``Matrix3``
    - with single argument of Matrix3-like object

    Args:
        v: 2-dimensional array-like object

    Example:
        >>> matrix()
        Matrix3
        [[1 0 0]
         [0 1 0]
         [0 0 1]]
        >>> A = matrix([[2, 1, 0], [0, 0.5, 0], [0, -0.5, 1]])

    """

    __slots__ = ("_coefs", "_attrs")
    __shape__ = (3, 3)

    def __init__(self, *args, **kwargs):
        super().__init__()
        if len(args) == 0:
            coefs = ((1, 0, 0), (0, 1, 0), (0, 0, 1))
        elif len(args) == 1 and np.asarray(args[0]).shape == Matrix3.__shape__:
            coefs = np.asarray(args[0]).tolist()
        else:
            raise TypeError("Not valid arguments for Matrix3")
        self._coefs = tuple(coefs[0]), tuple(coefs[1]), tuple(coefs[2])
        if is_jsonable(kwargs):
            self._attrs = kwargs
        else:
            raise TypeError("Provided attributes are not serializable.")

    @classmethod
    def from_comp(cls, **kwargs):
        """Return ``Matrix3`` defined by individual components. Default is zero
        matrix.

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
            >>> M = matrix.from_comp(xy=1, zy=-0.5)
            >>> M
            [[ 0.   1.   0. ]
             [ 0.   0.   0. ]
             [ 0.  -0.5  0. ]]

        """
        xx = kwargs.get("xx", 0)
        xy = kwargs.get("xy", 0)
        xz = kwargs.get("xz", 0)
        yx = kwargs.get("yx", 0)
        yy = kwargs.get("yy", 0)
        yz = kwargs.get("yz", 0)
        zx = kwargs.get("zx", 0)
        zy = kwargs.get("zy", 0)
        zz = kwargs.get("zz", 0)
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
    def E1(self):
        """First eigenvalue"""

        return float(self.eigenvalues()[0])

    @property
    def E2(self):
        """Second eigenvalue"""

        return float(self.eigenvalues()[1])

    @property
    def E3(self):
        """Third eigenvalue"""

        return float(self.eigenvalues()[2])

    @property
    def V1(self):
        """First eigenvector"""

        return self.eigenvectors()[0]

    @property
    def V2(self):
        """Second eigenvector"""

        return self.eigenvectors()[1]

    @property
    def V3(self):
        """Third eigenvector"""

        return self.eigenvectors()[2]

    def eigenvectors(self, which=None):
        """Return eigenvectors as ``Vector3`` objects.

        Args:
            which: if None returns sorted tuple of eigenvectors.
                If int returns given eigenvalue. Default None.
        """
        U = self._eig[1].T
        if which is None:
            return Vector3(U[0]), Vector3(U[1]), Vector3(U[2])
        else:
            return Vector3(U[which])

    def scaled_eigenvectors(self, which=None):
        """Return eigenvectors with magnitudes of eigenvalues as
        ``Vector3`` objects

        Args:
            which: if None returns sorted tuple of eigenvectors.
                If int returns given eigenvector. Default None.
        """
        U = self._eig[1].T
        if which is None:
            return (
                self._eig[0] * Vector2(U[0]),
                self._eig[1] * Vector2(U[1]),
                self._eig[2] * Vector3(U[2]),
            )
        else:
            return self._eig[0][which] * Vector2(U[which])
