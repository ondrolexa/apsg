import math
import numpy as np
from scipy import linalg as spla

from apsg.helpers._math import sind, cosd, atand
from apsg.math._vector import Vector3
from apsg.math._matrix import Matrix3
from apsg.decorator._decorator import ensure_arguments
from apsg.feature._geodata import Lineation, Foliation, Pair, Fault


class DefGrad3(Matrix3):
    """
    ``DefGrad3`` store 3D deformation gradient tensor.

    Args:
      a (3x3 array_like): Input data, that can be converted to
          3x3 2D array. This includes lists, tuples and ndarrays.

    Returns:
      ``DefGrad`` object

    Example:
      >>> F = DefGrad3(np.diag([2, 1, 0.5]))
    """

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

        assert Rxy >= 1, "Rxy must be greater than or equal to 1."
        assert Ryz >= 1, "Ryz must be greater than or equal to 1."

        y = (Ryz / Rxy) ** (1 / 3)
        return cls.from_comp(xx=y * Rxy, yy=y, zz=y / Ryz)

    @classmethod
    @ensure_arguments(Pair)
    def from_pair(cls, p):
        return cls(np.array([p.lvec, p.fvec.cross(p.lvec), p.fvec]).T)

    @classmethod
    @ensure_arguments(Vector3)
    def from_axisangle(cls, vector, theta):
        """Return ``DefGrad3`` representing rotation around axis.

        Args:
          vector: Rotation axis as ``Vector3`` like object
          theta: Angle of rotation in degrees

        Example:
          >>> F = DefGrad3.from_axis(lin(120, 30), 45)
        """

        x, y, z = vector.uv()
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
    @ensure_arguments(Vector3, Vector3)
    def from_two_vectors(cls, v1, v2):
        """Return ``DefGrad3`` representing rotation around axis perpendicular
        to both vectors and rotate v1 to v2.

        Args:
          v1: ``Vector3`` like object
          v2: ``Vector3`` like object

        Example:
          >>> F = DefGrad3.from_two_vectors(lin(120, 30), lin(210, 60))
        """
        return cls.from_axisangle(v1.cross(v2), v1.angle(v2))

    @classmethod
    def from_two_pairs(cls, p1, p2, symmetry=False):
        """
        Return ``DefGrad3`` representing rotation of coordinates from system
        defined by ``Pair`` p1 to system defined by ``Pair`` p2.

        Lineation in pair define x axis and normal to foliation in pair define z axis

        Args:
            p1 (``Pair``): from
            p2 (``Pair``): to

        Returns:
            ``Defgrad3`` rotational matrix

        Example:
            >>> p1 = pair(58, 36, 81, 34)
            >>> p2 = pair(217,42, 162, 27)
            >>> R = DefGrad3.from_two_pairs(p1, p2)
            >>> p1.transform(R) == p2
            True

        """

        if symmetry:
            R4 = [
                cls(cls.from_pair(Pair(p2.fvec, p2.lvec)) @ cls.from_pair(p1).I),
                cls(cls.from_pair(Pair(-p2.fvec, p2.lvec)) @ cls.from_pair(p1).I),
                cls(cls.from_pair(Pair(p2.fvec, -p2.lvec)) @ cls.from_pair(p1).I),
                cls(cls.from_pair(Pair(-p2.fvec, -p2.lvec)) @ cls.from_pair(p1).I),
            ]
            axes, angles = zip(*[R.axisangle() for R in R4])
            angles = [abs(a) for a in angles]
            ix = angles.index(min(angles))
            return R4[ix]
        else:
            return cls(cls.from_pair(p2) @ cls.from_pair(p1).I)

    @property
    def R(self):
        """Return rotation part of ``DefGrad`` from polar decomposition."""
        R, _ = spla.polar(self)
        return type(self)(R)

    @property
    def U(self):
        """Return stretching part of ``DefGrad`` from right polar decomposition."""
        _, U = spla.polar(self, "right")
        return type(self)(U)

    @property
    def V(self):
        """Return stretching part of ``DefGrad`` from left polar decomposition."""
        _, V = spla.polar(self, "left")
        return type(self)(V)

    def axisangle(self):
        """Return rotation part of ``DefGrad`` axis, angle tuple."""
        R = self.R
        w, W = np.linalg.eig(R.T)
        i = np.where(abs(np.real(w) - 1.0) < 1e-8)[0]
        if not len(i):
            raise ValueError("no unit eigenvector corresponding to eigenvalue 1")
        axis = Vector3(np.real(W[:, i[-1]]).squeeze())
        # rotation angle depending on direction
        cosa = (np.trace(R) - 1.0) / 2.0
        if abs(axis[2]) > 1e-8:
            sina = (R[1][0] + (cosa - 1.0) * axis[0] * axis[1]) / axis[2]
        elif abs(axis[1]) > 1e-8:
            sina = (R[0][2] + (cosa - 1.0) * axis[0] * axis[2]) / axis[1]
        else:
            sina = (R[2][1] + (cosa - 1.0) * axis[1] * axis[2]) / axis[0]
        angle = np.rad2deg(np.arctan2(sina, cosa))
        return axis, angle

    def velgrad(self, time=1):
        """Return ``VelGrad`` for given time"""
        from scipy.linalg import logm

        return VelGrad3(logm(np.asarray(self)) / time)


class VelGrad3(Matrix3):
    """
    ``VelGrad3`` represents 3D velocity gradient tensor.

    Args:
      a (3x3 array_like): Input data, that can be converted to
          3x3 2D array. This includes lists, tuples and ndarrays.

    Returns:
      ``VelGrad3`` object

    Example:
      >>> L = VelGrad3(np.diag([0.1, 0, -0.1]))
    """

    def defgrad(self, time=1, steps=1):
        """
        Return ``DefGrad`` tensor accumulated after given time.

        Keyword Args:
            time (float): time of deformation. Default 1
            steps (int): when bigger than 1, will return a list
                         of ``DefGrad`` tensors for each timestep.
        """
        from scipy.linalg import expm

        if steps > 1:  # FIX once container for matrix will be implemented
            return [
                DefGrad3(expm(np.asarray(self) * t))
                for t in np.linspace(0, time, steps)
            ]
        else:
            return DefGrad3(expm(np.asarray(self) * time))

    def rate(self):
        """
        Return rate of deformation tensor
        """

        return type(self)((self + self.T) / 2)

    def spin(self):
        """
        Return spin tensor
        """

        return type(self)((self - self.T) / 2)


class Tensor3(Matrix3):
    def __repr__(self):
        return f"{type(self).__name__}\n" + Matrix3.__repr__(self)

    @property
    def eigenlins(self):
        """
        Return tuple of eigenvectors as ``Lineation`` objects.
        """

        return tuple(Lineation(v) for v in self.eigenvectors())

    @property
    def eigenfols(self):
        """
        Return tuple of eigenvectors as ``Foliation`` objects.
        """

        return tuple(Foliation(v) for v in self.eigenvectors())

    @property
    def pair(self):
        """
        Return ``Pair`` representing orientation of principal axes.
        """

        ev = self.eigenvectors()
        return Pair(ev[2], ev[0])


class Stress3(Tensor3):
    """
    ``Stress3`` store 3D stress tensor.

    Args:
      a (3x3 array_like): Input data, that can be converted to
          3x3 2D array. This includes lists, tuples and ndarrays.

    Returns:
      ``Stress3`` object

    Example:
      >>> S = Stress3([[-8, 0, 0],[0, -5, 0],[0, 0, -1]])
    """

    @classmethod
    def from_comp(cls, xx=0, xy=0, xz=0, yy=0, yz=0, zz=0):
        """
        Return ``Stress`` tensor. Default is zero tensor.

        Note that stress tensor must be symmetrical.

        Keyword Args:
          xx, xy, xz, yy, yz, zz (float): tensor components

        Example:
          >>> S = Stress3.from_comp(xx=-5, yy=-2, zz=10, xy=1)
          >>> S
          Stress:
          [[-5  1  0]
           [ 1 -2  0]
           [ 0  0 10]]

        """

        return cls([[xx, xy, xz], [xy, yy, yz], [xz, yz, zz]])

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

        return type(self)(np.diag(self.mean_stress * np.ones(3)))

    @property
    def deviatoric(self):
        """
        A stress deviator tensor component
        """

        return type(self)(self - self.hydrostatic)

    @property
    def I1(self):
        """
        First invariant
        """

        return float(np.trace(self))

    @property
    def I2(self):
        """
        Second invariant
        """

        return float((self.I1 ** 2 - np.trace(self ** 2)) / 2)

    @property
    def I3(self):
        """
        Third invariant
        """

        return self.det

    @property
    def diagonalized(self):
        """
        Returns diagonalized Stress tensor and orthogonal matrix R, which transforms actual
        coordinate system to the principal one.

        """
        return type(self)(np.diag(self.eigenvalues())), DefGrad3(self.eigenvectors())

    def cauchy(self, n):
        """
        Return stress vector associated with plane given by normal vector.

        Args:
          n: normal given as ``Vector3`` or ``Foliation`` object

        Example:
          >>> S = Stress.from_comp(xx=-5, yy=-2, zz=10, xy=1)
          >>> S.cauchy(fol(160, 30))
          V(-2.520, 0.812, 8.660)

        """

        return Vector3(np.dot(self, n))

    def fault(self, n):
        """
        Return ``Fault`` object derived from given by normal vector.

        Args:
          n: normal given as ``Vector3`` or ``Foliation`` object

        Example:
          >>> S = Stress.from_comp(xx=-5, yy=-2, zz=10, xy=8)
          >>> S.fault(fol(160, 30))
          F:160/30-141/29 +

        """

        return Fault(*self.stress_comp(n))

    def stress_comp(self, n):
        """
        Return normal and shear stress ``Vector3`` components on plane given
        by normal vector.
        """

        t = self.cauchy(n)
        sn = t.proj(n)

        return sn, t - sn

    def normal_stress(self, n):
        """
        Return normal stress magnitude on plane given by normal vector.
        """

        return float(np.dot(n, self.cauchy(n)))

    def shear_stress(self, n):
        """
        Return shear stress magnitude on plane given by normal vector.
        """

        sn, tau = self.stress_comp(n)
        return abs(tau)


class Ellipsoid(Tensor3):
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

    def __repr__(self) -> str:
        return (
            f"{type(self).__name__} {self.kind}\n"
            + f"(E1:{self.E1:.3g},E2:{self.E2:.3g},E3:{self.E3:.3g})\n"
            + Matrix3.__repr__(self)
        )

    @classmethod
    def from_defgrad(cls, F, form="left", **kwargs) -> "Ellipsoid":
        """
        Return deformation tensor from ``Defgrad3``.

        Kwargs:
            form: 'left' or 'B' for left Cauchy–Green deformation tensor or
                  Finger deformation tensor
                  'right' or 'C' for right Cauchy–Green deformation tensor or
                  Green's deformation tensor.
                  Default is 'left'.
        """
        if form in ("left", "B"):
            return cls(np.dot(F, np.transpose(F)), **kwargs)
        elif form in ("right", "C"):
            return cls(np.dot(np.transpose(F), F), **kwargs)
        else:
            raise TypeError("Wrong form argument")

    @classmethod
    def from_stretch(cls, x=1, y=1, z=1, **kwargs) -> "Ellipsoid":
        """
        Return diagonal tensor defined by magnitudes of principal stretches.
        """
        return cls([[x * x, 0, 0], [0, y * y, 0], [0, 0, z * z]], **kwargs)

    @property
    def kind(self) -> str:
        """
        Return descriptive type of ellipsoid
        """
        nu = self.lode
        if np.allclose(self.eoct, 0):
            res = "O"
        elif nu < -0.75:
            res = "L"
        elif nu > 0.75:
            res = "S"
        elif nu < -0.15:
            res = "LLS"
        elif nu > 0.15:
            res = "SSL"
        else:
            res = "LS"
        return res

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
    def lambda1(self) -> float:
        """
        Return the square root of maximum eigenvalue.
        """
        return math.sqrt(self.E1)

    @property
    def lambda2(self) -> float:
        """
        Return the square root of middle eigenvalue.
        """
        return math.sqrt(self.E2)

    @property
    def lambda3(self) -> float:
        """
        Return the square root of minimum eigenvalue.
        """
        return math.sqrt(self.E3)

    @property
    def e1(self) -> float:
        """
        Return the maximum natural principal strain.
        """
        return math.log(self.lambda1)

    @property
    def e2(self) -> float:
        """
        Return the middle natural principal strain.
        """
        return math.log(self.lambda2)

    @property
    def e3(self) -> float:
        """
        Return the minimum natural principal strain.
        """
        return math.log(self.lambda3)

    @property
    def Rxy(self) -> float:
        return self.lambda1 / self.lambda2

    @property
    def Ryz(self) -> float:
        return self.lambda2 / self.lambda3

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
        return math.sqrt((self.Rxy - 1) ** 2 + (self.Ryz - 1) ** 2)

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
        return 2 * math.sqrt(self.e12 ** 2 + self.e23 ** 2 + self.e13 ** 2) / 3

    @property
    def eoct(self) -> float:
        """
        Return the natural octahedral unit strain (Nadai, 1963).
        """
        return math.sqrt(3) * self.goct / 2

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


class Ortensor3(Ellipsoid):
    """
    Represents an orientation tensor, which characterize data distribution
    using eigenvalue method. See (Watson 1966, Scheidegger 1965).

    See following methods and properties for additional operations.

    Args:
      matrix (3x3 array_like): Input data, that can be converted to
             3x3 2D matrix. This includes lists, tuples and ndarrays.
             Array could be also ``Group`` (for backward compatibility)

    Returns:
      ``Ortensor3`` object

    Example:
      >>> ot = Ortensor3([[8, 0, 0], [0, 2, 0], [0, 0, 1]])
      >>> ot
      Ortensor:  Kind: LLS
      (E1:8,E2:2,E3:1)
      [[8 0 0]
       [0 2 0]
       [0 0 1]]

    """

    def __repr__(self) -> str:
        return super().__repr__()

    @classmethod
    def from_features(cls, g) -> "Ortensor3":
        """
        Return ``Ortensor`` of data in ``Group``

        Args:
            g: ``Group`` of ``Vector3``, ``Lin`` or ``Foliation``

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

        return cls(np.dot(np.array(g).T, np.array(g)) / len(g))

    @classmethod
    def from_pairs(cls, p) -> "Ortensor3":
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
        Tx = np.dot(np.array(p.lin).T, np.array(p.lin)) / len(p)
        Tz = np.dot(np.array(p.fol).T, np.array(p.fol)) / len(p)
        return cls(Tx - Tz)

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
