import math
import numpy as np
from scipy import linalg as spla

from apsg.helpers._math import sind, cosd, atand
from apsg.math._vector import Vector3
from apsg.math._matrix import Matrix3
from apsg.decorator._decorator import ensure_arguments
from apsg.feature._geodata import Lineation, Foliation, Pair, Fault


class DeformationGradient3(Matrix3):
    """
    ``DeformationGradient3`` store 3D deformation gradient tensor.

    Args:
      a (3x3 array_like): Input data, that can be converted to
          3x3 2D array. This includes lists, tuples and ndarrays.

    Returns:
      ``DeformationGradient3`` object

    Example:
      >>> F = defgrad(np.diag([2, 1, 0.5]))
    """

    @classmethod
    def from_ratios(cls, Rxy=1, Ryz=1):
        """Return isochoric ``DeformationGradient3`` tensor with axial stretches defined by strain ratios.
        Default is identity tensor.

        Keyword Args:
          Rxy, Ryz (float): strain ratios

        Example:
          >>> F = defgrad.from_ratios(Rxy=2, Ryz=3)
          >>> F
          DeformationGradient3
          [[2.289 0.    0.   ]
           [0.    1.145 0.   ]
           [0.    0.    0.382]]

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
        """Return ``DeformationGradient3`` representing rotation around axis.

        Args:
          vector: Rotation axis as ``Vector3`` like object
          theta: Angle of rotation in degrees

        Example:
          >>> F = defgrad.from_axisangle(lin(120, 30), 45)
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
        """Return ``DeformationGradient3`` representing rotation around axis perpendicular
        to both vectors and rotate v1 to v2.

        Args:
          v1: ``Vector3`` like object
          v2: ``Vector3`` like object

        Example:
          >>> F = defgrad.from_two_vectors(lin(120, 30), lin(210, 60))
        """
        return cls.from_axisangle(v1.cross(v2), v1.angle(v2))

    @classmethod
    def from_two_pairs(cls, p1, p2, symmetry=False):
        """
        Return ``DeformationGradient3`` representing rotation of coordinates from system
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
            >>> R = defgrad.from_two_pairs(p1, p2)
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
        """Return rotation part of ``DeformationGradient3`` from polar decomposition."""
        R, _ = spla.polar(self)
        return type(self)(R)

    @property
    def U(self):
        """Return stretching part of ``DeformationGradient3`` from right polar decomposition."""
        _, U = spla.polar(self, "right")
        return type(self)(U)

    @property
    def V(self):
        """Return stretching part of ``DeformationGradient3`` from left polar decomposition."""
        _, V = spla.polar(self, "left")
        return type(self)(V)

    def axisangle(self):
        """Return rotation part of ``DeformationGradient3`` axis, angle tuple."""
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
        """Return ``VelocityGradient3`` for given time"""
        from scipy.linalg import logm

        return VelocityGradient3(logm(np.asarray(self)) / time)


class VelocityGradient3(Matrix3):
    """
    ``VelocityGradient3`` represents 3D velocity gradient tensor.

    Args:
      a (3x3 array_like): Input data, that can be converted to
          3x3 2D array. This includes lists, tuples and ndarrays.

    Returns:
      ``VelocityGradient3`` object

    Example:
      >>> L = velgrad(np.diag([0.1, 0, -0.1]))
    """

    def defgrad(self, time=1, steps=1):
        """
        Return ``DeformationGradient3`` tensor accumulated after given time.

        Keyword Args:
            time (float): time of deformation. Default 1
            steps (int): when bigger than 1, will return a list
                         of ``DeformationGradient3`` tensors for each timestep.
        """
        from scipy.linalg import expm

        if steps > 1:  # FIX once container for matrix will be implemented
            return [
                DeformationGradient3(expm(np.asarray(self) * t))
                for t in np.linspace(0, time, steps)
            ]
        else:
            return DeformationGradient3(expm(np.asarray(self) * time))

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
      >>> S = stress([[-8, 0, 0],[0, -5, 0],[0, 0, -1]])
    """

    @classmethod
    def from_comp(cls, xx=0, xy=0, xz=0, yy=0, yz=0, zz=0):
        """
        Return ``Stress`` tensor. Default is zero tensor.

        Note that stress tensor must be symmetrical.

        Keyword Args:
          xx, xy, xz, yy, yz, zz (float): tensor components

        Example:
          >>> S = stress.from_comp(xx=-5, yy=-2, zz=10, xy=1)
          >>> S
          Stress3
          [[-5.  1.  0.]
           [ 1. -2.  0.]
           [ 0.  0. 10.]]

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
        return (
            type(self)(np.diag(self.eigenvalues())),
            DeformationGradient3(self.eigenvectors()),
        )

    def cauchy(self, n):
        """
        Return stress vector associated with plane given by normal vector.

        Args:
          n: normal given as ``Vector3`` or ``Foliation`` object

        Example:
          >>> S = stress.from_comp(xx=-5, yy=-2, zz=10, xy=1)
          >>> S.cauchy(fol(160, 30))
          Vector3(-2.52, 0.812, 8.66)

        """

        return Vector3(np.dot(self, n.normalized()))

    def fault(self, n):
        """
        Return ``Fault`` object derived from given by normal vector.

        Args:
          n: normal given as ``Vector3`` or ``Foliation`` object

        Example:
          >>> S = stress.from_comp(xx=-5, yy=-2, zz=10, xy=8)
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
      >>> E = ellipsoid([[8, 0, 0], [0, 2, 0], [0, 0, 1]])
      >>> E
      Ellipsoid
      [[8. 0. 0.]
       [0. 2. 0.]
       [0. 0. 1.]]
      (λ1:2.83, λ2:1.41, λ3:1)

    """

    def __repr__(self) -> str:
        return f"{Matrix3.__repr__(self)}\n(λ1:{self.lambda1:.3g}, λ2:{self.lambda2:.3g}, λ3:{self.lambda3:.3g})"

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
        return self.lambda1 / self.lambda2 if self.lambda2 != 0 else np.inf

    @property
    def Ryz(self) -> float:
        return self.lambda2 / self.lambda3 if self.lambda3 != 0 else np.inf

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
        return (self.Rxy - 1) / (self.Ryz - 1) if self.Ryz > 1 else np.inf

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

        return 7.5 * np.sum((np.array(self.eigenvalues()) - 1 / 3) ** 2)

    @property
    def aMAD_l(self) -> float:
        """
        Return approximate angular deviation from the major axis along E1.
        """

        return atand(np.sqrt((1 - self.E1) / self.E1))

    @property
    def aMAD_p(self) -> float:
        """
        Return approximate deviation from the plane normal to E3.
        """

        return atand(np.sqrt(self.E3 / (1 - self.E3)))

    @property
    def aMAD(self) -> float:
        """
        Return approximate deviation according to shape
        """

        if self.shape > 1:
            return self.aMAD_l
        else:
            return self.aMAD_p

    @property
    def MAD_l(self) -> float:
        """Return maximum angular deviation (MAD) of linearly distributed vectors. Kirschvink 1980"""
        return atand(np.sqrt((self.E2 + self.E3) / self.E1))

    @property
    def MAD_p(self) -> float:
        """Return maximum angular deviation (MAD) of planarly distributed vectors. Kirschvink 1980"""
        return atand(np.sqrt(self.E3 / self.E2 + self.E3 / self.E1))

    @property
    def MAD(self) -> float:
        """
        Return approximate deviation according to shape
        """

        if self.shape > 1:
            return self.MAD_l
        else:
            return self.MAD_p


class OrientationTensor3(Ellipsoid):
    """
    Represents an orientation tensor, which characterize data distribution
    using eigenvalue method. See (Watson 1966, Scheidegger 1965).

    See following methods and properties for additional operations.

    Args:
      matrix (3x3 array_like): Input data, that can be converted to
             3x3 2D matrix. This includes lists, tuples and ndarrays.
             Array could be also ``Group`` (for backward compatibility)

    Returns:
      ``OrientationTensor3`` object

    Example:
      >>> ot = ortensor([[8, 0, 0], [0, 2, 0], [0, 0, 1]])
      >>> ot
      OrientationTensor3
      [[8. 0. 0.]
       [0. 2. 0.]
       [0. 0. 1.]]
      (λ1:2.83, λ2:1.41, λ3:1)

    """

    @classmethod
    def from_features(cls, g) -> "OrientationTensor3":
        """
        Return ``Ortensor`` of data in ``Group``

        Args:
            g: ``Group`` of ``Vector3``, ``Lin`` or ``Foliation``

        Example:
          >>> g = linset.random_fisher(position=lin(120,50))
          >>> ot = ortensor.from_features(g)
          >>> ot
          OrientationTensor3
          [[ 0.142 -0.151 -0.212]
           [-0.151  0.326  0.37 ]
           [-0.212  0.37   0.532]]
          (λ1:0.95, λ2:0.241, λ3:0.2)
          >>> ot.eigenlins
          (L:120/49, L:216/5, L:310/40)

        """

        return cls(np.dot(np.array(g).T, np.array(g)) / len(g))

    @classmethod
    def from_pairs(cls, p) -> "OrientationTensor3":
        """
        Return Lisle (19890``Ortensor`` of orthogonal data in ``PairSet``

        Lisle, R. (1989). The Statistical Analysis of Orthogonal Orientation Data. The Journal of Geology, 97(3), 360-364.

        Args:
            p: ``PairSet``

        Example:
          >>> p = pairset([pair(109, 82, 21, 10),
                           pair(118, 76, 30, 11),
                           pair(97, 86, 7, 3),
                           pair(109, 75, 23, 14)])
          >>> ot = ortensor.from_pairs(p)
          >>> ot
          OrientationTensor3
          [[ 0.731  0.575  0.086]
           [ 0.575 -0.725  0.224]
           [ 0.086  0.224 -0.005]]
          (λ1:0.98, λ2:0.978, λ3:0.0688)

        """
        return cls(OrientationTensor3.from_features(p.lvec) - OrientationTensor3.from_features(p.fvec))
