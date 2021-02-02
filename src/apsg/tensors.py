# -*- coding: utf-8 -*-


from typing import Tuple


import numpy as np

from apsg.core import Vec3, Group, Pair, PairSet, Fault
from apsg.helpers import sind, cosd, atand


__all__ = ("DefGrad", "VelGrad", "Stress", "Ortensor", "Ellipsoid")


class DefGrad(np.ndarray):
    """
    ``DefGrad`` store deformation gradient tensor derived from numpy.ndarray.

    Args:
      a (3x3 array_like): Input data, that can be converted to
          3x3 2D array. This includes lists, tuples and ndarrays.

    Returns:
      ``DefGrad`` object

    Example:
      >>> F = DefGrad(np.diag([2, 1, 0.5]))
    """

    def __new__(cls, array, name='F'):
        # casting to our class
        assert np.shape(array) == (3, 3), 'DefGrad must be 3x3 2D array'
        obj = np.asarray(array).view(cls)
        obj.name = name
        return obj

    def __repr__(self):
        return 'DefGrad:\n' + str(self)

    def __mul__(self, other):
        assert np.shape(other) == (
            3,
            3,
        ), 'DefGrad could by multiplied with 3x3 2D array'
        return DefGrad(np.dot(self, other))

    def __pow__(self, n):
        # matrix power
        return DefGrad(np.linalg.matrix_power(self, n))

    def __eq__(self, other):
        # equal
        return bool(np.sum(abs(self - other)) < 1e-14)

    def __ne__(self, other):
        # not equal
        return not self == other

    @classmethod
    def from_axis(cls, vector, theta):
        """Return ``DefGrad`` representing rotation around axis.

        Args:
          vector: Rotation axis as ``Vec3`` like object
          theta: Angle of rotation in degrees

        Example:
          >>> F = DefGrad.from_axis(Lin(120, 30), 45)
        """

        x, y, z = vector.uv
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

    @classmethod
    def from_comp(cls, xx=1, xy=0, xz=0, yx=0, yy=1, yz=0, zx=0, zy=0, zz=1):
        """Return ``DefGrad`` tensor defined by individual components. Default is identity tensor.

        Keyword Args:
          xx, xy, xz, yx, yy, yz, zx, zy, zz (float): tensor components

        Example:
          >>> F = DefGrad.from_comp(xy=1, zy=-0.5)
          >>> F
          DefGrad:
          [[ 1.   1.   0. ]
           [ 0.   1.   0. ]
           [ 0.  -0.5  1. ]]

        """

        return cls([[xx, xy, xz], [yx, yy, yz], [zx, zy, zz]])

    @classmethod
    def from_ratios(cls, Rxy=1, Ryz=1):
        """Return isochoric ``DefGrad`` tensor with axial stretches defined by strain ratios.
        Default is identity tensor.

        Keyword Args:
          Rxy, Ryz (float): strain ratios

        Example:
          >>> F = DefGrad.from_ratios(Rxy=2, Ryz=3)
          >>> F
          DefGrad:
          [[2.28942849 0.         0.        ]
           [0.         1.14471424 0.        ]
           [0.         0.         0.38157141]]

        """

        assert Rxy >= 1, 'Rxy must be greater than or equal to 1.'
        assert Ryz >= 1, 'Ryz must be greater than or equal to 1.'

        y = (Ryz / Rxy) ** (1 / 3)
        return cls.from_comp(xx=y * Rxy, yy=y, zz=y / Ryz)

    @classmethod
    def from_pair(cls, p):

        assert issubclass(type(p), Pair), 'Data must be of Pair type.'

        return cls(np.array([p.lvec, p.fvec ** p.lvec, p.fvec]).T)

    @property
    def I(self):
        """
        Returns the inverse tensor.
        """

        return DefGrad(np.linalg.inv(self))

    def rotate(self, vector, theta=0):
        """
        Rotate tensor around axis by angle theta.

        Using rotation matrix it returns ``F = R * F * R . T``.
        """

        if isinstance(vector, DefGrad):
            R = vector
        else:
            R = DefGrad.from_axis(vector, theta)

        return DefGrad(R * self * R.T)

    @property
    def eigenvals(self):
        """
        Return tuple of sorted eigenvalues.
        """

        _, vals, _ = np.linalg.svd(self)

        return tuple(vals)

    @property
    def E1(self):
        """
        Max eigenvalue
        """

        return self.eigenvals[0]

    @property
    def E2(self):
        """
        Middle eigenvalue
        """

        return self.eigenvals[1]

    @property
    def E3(self):
        """
        Min eigenvalue
        """

        return self.eigenvals[2]

    @property
    def eigenvects(self):
        """
        Return ```Group``` of principal eigenvectors as ``Vec3`` objects.
        """

        U, _, _ = np.linalg.svd(self)

        return Group([Vec3(U.T[0]), Vec3(U.T[1]), Vec3(U.T[2])])

    @property
    def eigenlins(self):
        """
        Return ```Group``` of principal eigenvectors as ``Lin`` objects.
        """

        return self.eigenvects.aslin

    @property
    def eigenfols(self):
        """
        Return ```Group``` of principal eigenvectors as ``Fol`` objects.
        """

        return self.eigenvects.asfol

    @property
    def R(self):
        """
        Return rotation part of ``DefGrad`` from polar decomposition.
        """

        from scipy.linalg import polar

        R, _ = polar(self)

        return DefGrad(R)

    @property
    def U(self):
        """
        Return stretching part of ``DefGrad`` from right polar decomposition.
        """

        from scipy.linalg import polar

        _, U = polar(self, "right")

        return DefGrad(U)

    @property
    def V(self):
        """
        Return stretching part of ``DefGrad`` from left polar decomposition.
        """

        from scipy.linalg import polar

        _, V = polar(self, "left")

        return DefGrad(V)

    @property
    def axisangle(self):
        """Return rotation part of ``DefGrad`` axis, angle tuple."""
        from scipy.linalg import polar

        R, _ = polar(self)
        w, W = np.linalg.eig(R.T)
        i = np.where(abs(np.real(w) - 1.0) < 1e-8)[0]
        if not len(i):
            raise ValueError('no unit eigenvector corresponding to eigenvalue 1')
        axis = Vec3(np.real(W[:, i[-1]]).squeeze())
        # rotation angle depending on direction
        cosa = (np.trace(R) - 1.0) / 2.0
        if abs(axis[2]) > 1e-8:
            sina = (R[1, 0] + (cosa - 1.0) * axis[0] * axis[1]) / axis[2]
        elif abs(axis[1]) > 1e-8:
            sina = (R[0, 2] + (cosa - 1.0) * axis[0] * axis[2]) / axis[1]
        else:
            sina = (R[2, 1] + (cosa - 1.0) * axis[1] * axis[2]) / axis[0]
        angle = np.rad2deg(np.arctan2(sina, cosa))
        return axis, angle

    def velgrad(self, time=1):
        """Return ``VelGrad`` for given time"""
        from scipy.linalg import logm

        return VelGrad(logm(self) / time)

    @property
    def C(self):
        """
        Return right Cauchy–Green deformation tensor or Green's deformation tensor
        """

        return Tensor(np.dot(self.T, self), name=self.name)

    @property
    def B(self):
        """
        Return left Cauchy–Green deformation tensor or Finger deformation tensor
        """

        return Tensor(np.dot(self, self.T), name=self.name)


class VelGrad(np.ndarray):
    """
    ``VelGrad`` store velocity gradient tensor derived from numpy.ndarray.

    Args:
      a (3x3 array_like): Input data, that can be converted to
          3x3 2D array. This includes lists, tuples and ndarrays.

    Returns:
      ``VelGrad`` object

    Example:
      >>> L = VelGrad(np.diag([0.1, 0, -0.1]))
    """

    def __new__(cls, array, name='L'):
        # casting to our class
        assert np.shape(array) == (3, 3), 'VelGrad must be 3x3 2D array'
        obj = np.asarray(array).view(cls)
        obj.name = name
        return obj

    def __repr__(self):
        return "VelGrad:\n" + str(self)

    def __pow__(self, n):
        # matrix power
        return VelGrad(np.linalg.matrix_power(self, n))

    def __eq__(self, other):
        # equal
        return bool(np.sum(abs(self - other)) < 1e-14)

    def __ne__(self, other):
        # not equal
        return not self == other

    @classmethod
    def from_comp(cls, xx=0, xy=0, xz=0, yx=0, yy=0, yz=0, zx=0, zy=0, zz=0):
        """
        Return ``VelGrad`` tensor. Default is zero tensor.

        Keyword Args:
          xx, xy, xz, yx, yy, yz, zx, zy, zz (float): tensor components

        Example:
          >>> L = VelGrad.from_comp(xx=0.1, zz=-0.1)
          >>> L
          VelGrad:
          [[ 0.1  0.   0. ]
           [ 0.   0.   0. ]
           [ 0.   0.  -0.1]]

        """
        return cls([[xx, xy, xz], [yx, yy, yz], [zx, zy, zz]])

    def defgrad(self, time=1, steps=1):
        """
        Return ``DefGrad`` tensor accumulated after given time.

        Keyword Args:
            time (float): time of deformation. Default 1
            steps (int): when bigger than 1, will return a list
                         of ``DefGrad`` tensors for each timestep.
        """
        from scipy.linalg import expm

        if steps > 1:
            return [DefGrad(expm(self * t)) for t in np.linspace(0, time, steps)]
        else:
            return DefGrad(expm(self * time))

    def rotate(self, vector, theta=0):
        """
        Rotate tensor around axis by angle theta.

        Using rotation matrix it returns ``F = R * F * R . T``.
        """

        if isinstance(vector, DefGrad):
            R = vector
        else:
            R = DefGrad.from_axis(vector, theta)

        return VelGrad(R * self * R.T)

    @property
    def rate(self):
        """
        Return rate of deformation tensor
        """

        return VelGrad((self + self.T) / 2)

    @property
    def spin(self):
        """
        Return spin tensor
        """

        return VelGrad((self - self.T) / 2)


class Stress(np.ndarray):
    """
    ``Stress`` store stress tensor derived from numpy.ndarray.

    Args:
      a (3x3 array_like): Input data, that can be converted to
          3x3 2D array. This includes lists, tuples and ndarrays.

    Returns:
      ``Stress`` object

    Example:
      >>> S = Stress([[-8, 0, 0],[0, -5, 0],[0, 0, -1]])
    """

    def __new__(cls, array, name='S'):
        # casting to our class

        assert np.shape(array) == (3, 3), 'Stress must be 3x3 2D array'
        assert np.allclose(
            np.asarray(array), np.asarray(array).T
        ), 'Stress tensor must be symmetrical'

        obj = np.asarray(array).view(cls)
        obj.name = name
        vals, U = np.linalg.eigh(obj)
        ix = np.argsort(vals)[::-1]
        obj.eigenvals = vals[ix]
        obj.vects = U[:, ix]
        return obj

    def __repr__(self):

        return 'Stress:\n' + str(self)

    def __pow__(self, n):
        # matrix power

        return Stress(np.linalg.matrix_power(self, n))

    def __eq__(self, other):
        # equal

        return bool(np.sum(abs(self - other)) < 1e-14)

    def __ne__(self, other):
        # not equal

        return not self == other

    @classmethod
    def from_comp(cls, xx=0, xy=0, xz=0, yy=0, yz=0, zz=0):
        """
        Return ``Stress`` tensor. Default is zero tensor.

        Note that stress tensor must be symmetrical.

        Keyword Args:
          xx, xy, xz, yy, yz, zz (float): tensor components

        Example:
          >>> S = Stress.from_comp(xx=-5, yy=-2, zz=10, xy=1)
          >>> S
          Stress:
          [[-5  1  0]
           [ 1 -2  0]
           [ 0  0 10]]

        """

        return cls([[xx, xy, xz], [xy, yy, yz], [xz, yz, zz]])

    def rotate(self, vector, theta=0):
        """
        Rotate tensor around axis by angle theta.

        Using rotation matrix it returns ``S = R * S * R . T``
        """

        if isinstance(vector, DefGrad):
            R = vector
        else:
            R = DefGrad.from_axis(vector, theta)

        return Stress(R * self * R.T)

    @property
    def mean_stress(self):
        """
        Mean stress
        """

        return self.I1 / 3

    @property
    def hydrostatic(self):
        """
        Mean hydrostatic stress tensor component
        """

        return Stress(np.diag(self.mean_stress * np.ones(3)))

    @property
    def deviatoric(self):
        """
        A stress deviator tensor component
        """

        return Stress(self - self.hydrostatic)

    @property
    def E1(self):
        """
        Max eigenvalue
        """

        return self.eigenvals[0]

    @property
    def E2(self):
        """
        Middle eigenvalue
        """

        return self.eigenvals[1]

    @property
    def E3(self):
        """
        Min eigenvalue
        """

        return self.eigenvals[2]

    @property
    def I1(self):
        """
        First invariant
        """

        return np.trace(self)

    @property
    def I2(self):
        """
        Second invariant
        """

        return (self.I1 ** 2 - np.trace(self ** 2)) / 2

    @property
    def I3(self):
        """
        Third invariant
        """

        return np.linalg.det(self)

    @property
    def diagonalized(self):
        """
        Returns Stress tensor in a coordinate system with axes oriented to the principal directions and
        orthogonal matrix R, which brings actual coordinate system to principal one.

        """
        return Stress(np.diag(self.eigenvals)), DefGrad(self.vects.T)

    @property
    def eigenvects(self):
        """
        Returns Group of three eigenvectors represented as ``Vec3``
        """
        return Group(
            [Vec3(self.vects.T[0]), Vec3(self.vects.T[1]), Vec3(self.vects.T[2])]
        )

    @property
    def eigenlins(self):
        """
        Returns Group of three eigenvectors represented as ``Lin``
        """

        return self.eigenvects.aslin

    @property
    def eigenfols(self):
        """
        Returns Group of three eigenvectors represented as ``Fol``
        """

        return self.eigenvects.asfol

    def cauchy(self, n):
        """
        Return stress vector associated with plane given by normal vector.

        Args:
          n: normal given as ``Vec3`` or ``Fol`` object

        Example:
          >>> S = Stress.from_comp(xx=-5, yy=-2, zz=10, xy=1)
          >>> S.cauchy(Fol(160, 30))
          V(-2.520, 0.812, 8.660)

        """

        return Vec3(np.dot(self, n))

    def fault(self, n):
        """
        Return ``Fault`` object derived from given by normal vector.

        Args:
          n: normal given as ``Vec3`` or ``Fol`` object

        Example:
          >>> S = Stress.from_comp(xx=-5, yy=-2, zz=10, xy=8)
          >>> S.fault(Fol(160, 30))
          F:160/30-141/29 +

        """

        return Fault.from_vecs(*self.stress_comp(n))

    def stress_comp(self, n):
        """
        Return normal and shear stress ``Vec3`` components on plane given
        by normal vector.
        """

        t = self.cauchy(n)
        sn = t.proj(n)

        return sn, t - sn

    def normal_stress(self, n):
        """
        Return normal stress component on plane given by normal vector.
        """

        return np.dot(n, self.cauchy(n))

    def shear_stress(self, n):
        """
        Return magnitude of shear stress component on plane given
        by normal vector.
        """

        return np.sqrt(self.cauchy(n) ** 2 - self.normal_stress(n) ** 2)


class Tensor(object):
    """
    Tensor is a multidimensional array of scalars.
    """

    def __init__(self, matrix, **kwargs) -> None:

        assert np.shape(matrix) == (3, 3), "Ellipsoid matrix must be 3x3 2D array"

        self._matrix = np.asarray(matrix)
        self.name = kwargs.get('name', '')
        self.scaled = kwargs.get("scaled", False)

        vc, vv = np.linalg.eigh(self._matrix)
        ix = np.argsort(vc)[::-1]

        self._evals = vc[ix]
        self._evects = vv.T[ix]

    def __repr__(self) -> str:
        return (
            '{}: {} Kind: {}\n'.format(self.__class__.__name__, self.name, self.kind)
            + '(E1:{:.4g},E2:{:.4g},E3:{:.4g})\n'.format(self.E1, self.E2, self.E3)
            + str(self._matrix)
        )

    __str__ = __repr__

    def __eq__(self, other) -> bool:
        """
        Return `True` if tensors are equal, otherwise `False`.
        """
        if not isinstance(other, self.__class__):
            return False

        return tuple(map(tuple, self._matrix)) == tuple(map(tuple, other._matrix))

    def __hash__(self) -> int:
        return hash(tuple(map(tuple, self._matrix)))

    @classmethod
    def from_comp(cls, xx=1, xy=0, xz=0, yy=1, yz=0, zz=1, **kwargs) -> 'Tensor':
        """
        Return tensor. Default is identity tensor.

        Note that tensor must be symmetrical.
        """

        return cls([[xx, xy, xz], [xy, yy, yz], [xz, yz, zz]], **kwargs)

    @classmethod
    def from_axes(cls, x=1, y=1, z=1, **kwargs) -> 'Tensor':
        """
        Return diagonal tensor defined by principal axes.
        """
        return cls([[x * x, 0, 0], [0, y * y, 0], [0, 0, z * z]], **kwargs)

    @property
    def eigenvals(self) -> Tuple[float]:
        """
        Return the eigenvalues sorted in descending order.
        """
        return tuple(self._evals)

    @property
    def eigenvects(self) -> Group:
        """
        Return the group of eigenvectors. If scaled property is `True` their
        length is scaled by eigenvalues, otherwise unit length.
        """
        if self.scaled:
            e1, e2, e3 = self.E1, self.E2, self.E3
        else:
            e1 = e2 = e3 = 1.0
        return Group(
            [
                e1 * Vec3(self._evects[0]),
                e2 * Vec3(self._evects[1]),
                e3 * Vec3(self._evects[2]),
            ]
        )

    @property
    def eigenlins(self) -> Group:
        """
        Return group of eigenvectors as Lin objects.
        """
        return self.eigenvects.aslin

    @property
    def eigenfols(self) -> Group:
        """
        Return group of eigenvectors as Fol objects.
        """
        return self.eigenvects.asfol

    @property
    def strength(self) -> float:
        """
        Return the Woodcock strength.
        """
        return self.e13

    @property
    def shape(self) -> float:
        """
        return the Woodcock shape.
        """
        return self.K

    @property
    def kind(self) -> str:
        """
        Return descriptive type of ellipsoid
        """
        nu = self.lode
        if np.allclose(self.eoct, 0):
            res = 'O'
        elif nu < -0.75:
            res = 'L'
        elif nu > 0.75:
            res = 'S'
        elif nu < -0.15:
            res = 'LLS'
        elif nu > 0.15:
            res = 'SSL'
        else:
            res = 'LS'
        return res

    @property
    def E1(self) -> float:
        """
        Return the square root of maximum eigenvalue.
        """
        return np.sqrt(self._evals[0])

    @property
    def E2(self) -> float:
        """
        Return the square root of middle eigenvalue.
        """
        return np.sqrt(self._evals[1])

    @property
    def E3(self) -> float:
        """
        Return the square root of minimum eigenvalue.
        """
        return np.sqrt(self._evals[2])

    @property
    def e1(self) -> float:
        """
        Return the maximum natural principal strain.
        """
        return np.log(self.E1)

    @property
    def e2(self) -> float:
        """
        Return the middle natural principal strain.
        """
        return np.log(self.E2)

    @property
    def e3(self) -> float:
        """
        Return the minimum natural principal strain.
        """
        return np.log(self.E3)

    @property
    def Rxy(self) -> float:
        return self.E1 / self.E2

    @property
    def Ryz(self) -> float:
        return self.E2 / self.E3

    @property
    def e12(self) -> float:
        return self.e1 - self.e2

    @property
    def e13(self) -> float:
        return self.e1 - self.e3

    @property
    def e23(self) -> float:
        return self.e2 - self.e3

    @property
    def k(self) -> float:
        """
        Return the strain symmetry.
        """
        return (self.Rxy - 1) / (self.Ryz - 1)

    @property
    def d(self) -> float:
        """
        Return the strain intensity.
        """
        return np.sqrt((self.Rxy - 1) ** 2 + (self.Ryz - 1) ** 2)

    @property
    def K(self) -> float:
        """
        Return the strain symmetry (Ramsay, 1983).
        """
        return self.e12 / self.e23 if self.e23 > 0 else np.inf

    @property
    def D(self) -> float:
        """
        return the strain intensity.
        """
        return self.e12 ** 2 + self.e23 ** 2

    @property
    def r(self) -> float:
        """
        Return the strain intensity (Watterson, 1968).
        """
        return self.Rxy + self.Ryz - 1

    @property
    def goct(self) -> float:
        """
        Return the natural octahedral unit shear (Nadai, 1963).
        """
        return 2 * np.sqrt(self.e12 ** 2 + self.e23 ** 2 + self.e13 ** 2) / 3

    @property
    def eoct(self) -> float:
        """
        Return the natural octahedral unit strain (Nadai, 1963).
        """
        return np.sqrt(3) * self.goct / 2

    @property
    def lode(self) -> float:
        """
        Return Lode parameter (Lode, 1926).
         """
        return (
            (2 * self.e2 - self.e1 - self.e3) / (self.e1 - self.e3)
            if (self.e1 - self.e3) > 0
            else 0
        )

    @property
    def I(self) -> 'Tensor':
        """
        Return the inverse tensor.
        """

        return Tensor(np.linalg.inv(self._matrix), name=self.name)

    def apply(self, x1, x2=None) -> np.array:
        """
        Apply the deformation tensor on vector(s).
        """
        if x2 is None:
            return np.dot(x1, np.dot(self._matrix, x1))
        else:
            return np.dot(x1, np.dot(self._matrix, x2))


class Ortensor(Tensor):
    """
    Represents an orientation tensor, which characterize data distribution
    using eigenvalue method. See (Watson 1966, Scheidegger 1965).

    See following methods and properties for additional operations.

    Args:
      matrix (3x3 array_like): Input data, that can be converted to
             3x3 2D matrix. This includes lists, tuples and ndarrays.
             Array could be also ``Group`` (for backward compatibility)

    Keyword Args:
      name (str): name od tensor
      scaled (bool): When True eigenvectors are scaled by eigenvalues.
                     Otherwise unit length. Default False

    Returns:
      ``Ortensor`` object

    Example:
      >>> ot = Ortensor([[8, 0, 0], [0, 2, 0], [0, 0, 1]])
      >>> ot
      Ortensor:  Kind: LLS
      (E1:8,E2:2,E3:1)
      [[8 0 0]
       [0 2 0]
       [0 0 1]]

    """

    def __init__(self, matrix, **kwargs) -> None:
        if isinstance(matrix, Group):
            if 'name' not in kwargs:
                kwargs['name'] = matrix.name
            matrix = np.dot(np.array(matrix).T, np.array(matrix)) / len(matrix)
        super(Ortensor, self).__init__(matrix, **kwargs)

    @classmethod
    def from_group(cls, g, **kwargs) -> 'Ortensor':
        """
        Return ``Ortensor`` of data in ``Group``

        Args:
            g: ``Group`` of ``Vec3``, ``Lin`` or ``Fol``

        Example:
          >>> g = Group.examples('B2')
          >>> ot = Ortensor.from_group(g)
          >>> ot
          Ortensor: B2 Kind: L
          (E1:0.9825,E2:0.01039,E3:0.007101)
          [[ 0.19780807 -0.13566589 -0.35878837]
           [-0.13566589  0.10492993  0.25970594]
           [-0.35878837  0.25970594  0.697262  ]]
          >>> ot.eigenlins.data
          [L:144/57, L:360/28, L:261/16]

        """

        if 'name' not in kwargs:
            kwargs['name'] = g.name
        return cls(np.dot(np.array(g).T, np.array(g)) / len(g), **kwargs)

    @classmethod
    def from_pairs(cls, p, **kwargs) -> 'Ortensor':
        """
        Return Lisle (19890``Ortensor`` of orthogonal data in ``PairSet``

        Lisle, R. (1989). The Statistical Analysis of Orthogonal Orientation Data. The Journal of Geology, 97(3), 360-364.

        Args:
            p: ``PairSet``

        Example:
          >>> p = PairSet([Pair(109, 82, 21, 10),
                           Pair(118, 76, 30, 11),
                           Pair(97, 86, 7, 3),
                           Pair(109, 75, 23, 14) ])
          >>> ot = Ortensor.from_pairs(p)
          >>> ot
          Ortensor: Default Kind: LS
          (E1:0.956,E2:0.00473,E3:-0.9608)
          [[ 0.7307853   0.57519626  0.08621956]
           [ 0.57519626 -0.72530456  0.22401935]
           [ 0.08621956  0.22401935 -0.00548074]]
          >>> ot.eigenfols[2]
          S:108/79
          >>> ot.eigenlins[0]
          L:20/9

        """
        assert isinstance(p, PairSet), "Data must be of PairSet type."
        if 'name' not in kwargs:
            kwargs['name'] = p.name
        Tx = np.dot(np.array(p.lin).T, np.array(p.lin)) / len(p)
        Tz = np.dot(np.array(p.fol).T, np.array(p.fol)) / len(p)
        return cls(Tx - Tz, **kwargs)

    @property
    def E1(self) -> float:
        """
        Return maximum eigenvalue.
        """

        return self._evals[0]

    @property
    def E2(self) -> float:
        """
        Return middle eigenvalue.
        """

        return self._evals[1]

    @property
    def E3(self) -> float:
        """
        Return minimum eigenvalue.
        """

        return self._evals[2]

    @property
    def P(self) -> float:
        """
        Point index (Vollmer, 1990).
        """

        return self.E1 - self.E2

    @property
    def G(self) -> float:
        """
        Girdle index (Vollmer, 1990).
        """

        return 2 * (self.E2 - self.E3)

    @property
    def R(self) -> float:
        """
        Random index (Vollmer, 1990).
        """

        return 3 * self.E3

    @property
    def B(self) -> float:
        """
        Cylindricity index (Vollmer, 1990).
        """

        return self.P + self.G

    @property
    def Intensity(self) -> float:
        """
        Intensity index (Lisle, 1985).
        """

        return 7.5 * np.sum((self._evals - 1 / 3) ** 2)

    @property
    def MADp(self) -> float:
        """
        Return approximate angular deviation from the major axis along E1.
        """

        return atand(np.sqrt((1 - self.E1) / self.E1))

    @property
    def MADo(self) -> float:
        """
        Return approximate deviation from the plane normal to E3.
        """

        return atand(np.sqrt(self.E3 / (1 - self.E3)))

    @property
    def MAD(self) -> float:
        """
        Return approximate deviation according to shape
        """

        if self.shape > 1:
            return self.MADp
        else:
            return self.MADo


class Ellipsoid(Tensor):
    """
    Ellipsoid class

    See following methods and properties for additional operations.

    Args:
      matrix (3x3 array_like): Input data, that can be converted to
             3x3 2D matrix. This includes lists, tuples and ndarrays.

    Returns:
      ``Ellipsoid`` object

    Example:
      >>> E = Ellipsoid([[8, 0, 0], [0, 2, 0], [0, 0, 1]])
      >>> E
      Ellipsoid:  Kind: LLS
      (E1:2.828,E2:1.414,E3:1)
      [[8 0 0]
       [0 2 0]
       [0 0 1]]

    """

    @classmethod
    def from_defgrad(cls, F, **kwargs) -> 'Ellipsoid':
        """
        Return ``Ellipsoid`` from ``Defgrad``.

        It is ellipsoid resulting from deformation of sphere.

        """

        return cls(np.dot(F, np.transpose(F)), **kwargs)

    def transform(self, F) -> 'Ellipsoid':
        """
        Return ``Ellipsoid`` representing result of deformation F
        """

        t_matrix = np.dot(F, np.dot(self._matrix, np.transpose(F)))
        return Ellipsoid(t_matrix, name=self.name)
