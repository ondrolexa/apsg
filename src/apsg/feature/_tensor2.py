import math
import numpy as np
from scipy import linalg as spla

from apsg.helpers._math import sind, cosd, atan2d
from apsg.math._vector import Vector2
from apsg.math._matrix import Matrix2
from apsg.decorator._decorator import ensure_arguments


class DeformationGradient2(Matrix2):
    """
    The class to represent 2D deformation gradient tensor.

    Args:
      a (2x2 array_like): Input data, that can be converted to
          2x2 2D array. This includes lists, tuples and ndarrays.

    Returns:
      ``DeformationGradient2`` object

    Example:
      >>> F = defgrad2(np.diag([2, 0.5]))
    """

    @classmethod
    def from_ratio(cls, R=1):
        """Return isochoric ``DeformationGradient2`` tensor with axial stretches defined by strain ratio.
        Default is identity tensor.

        Keyword Args:
          R (float): strain ratio

        Example:
          >>> F = defgrad2.from_ratio(R=4)
          >> F
          DeformationGradient2
          [[2.  0. ]
           [0.  0.5]]

        """

        return cls.from_comp(xx=R ** (1 / 2), yy=R ** (-1 / 2))

    @classmethod
    def from_angle(cls, theta):
        """Return ``DeformationGradient2`` representing rotation by angle theta.

        Args:
          theta: Angle of rotation in degrees

        Example:
          >>> F = defgrad2.from_angle(45)
          >>> F
          DeformationGradient2
          [[ 0.707 -0.707]
           [ 0.707  0.707]]

        """

        c, s = cosd(theta), sind(theta)
        return cls([[c, -s], [s, c]])

    @classmethod
    @ensure_arguments(Vector2, Vector2)
    def from_two_vectors(cls, v1, v2):
        """Return ``DeformationGradient2`` representing rotation around axis perpendicular
        to both vectors and rotate v1 to v2.

        Args:
          v1: ``Vector2`` like object
          v2: ``Vector2`` like object

        Example:
          >>> F = defgrad2.from_two_vectors(vec2(1, 1), vec2(0, 1))
          >>> F
          DeformationGradient2
          [[ 0.707 -0.707]
           [ 0.707  0.707]]

        """
        return cls.from_angle(v1.angle(v2))

    @property
    def R(self):
        """Return rotation part of ``DeformationGradient2`` from polar decomposition."""
        R, _ = spla.polar(self)
        return type(self)(R)

    @property
    def U(self):
        """Return stretching part of ``DeformationGradient2`` from right polar decomposition."""
        _, U = spla.polar(self, "right")
        return type(self)(U)

    @property
    def V(self):
        """Return stretching part of ``DeformationGradient2`` from left polar decomposition."""
        _, V = spla.polar(self, "left")
        return type(self)(V)

    def angle(self):
        """Return rotation part of ``DeformationGradient2`` as angle."""
        R, _ = spla.polar(self)
        return atan2d(R[1, 0], R[0, 0])

    def velgrad(self, time=1):
        """Return ``VelocityGradient2`` for given time"""
        from scipy.linalg import logm

        return VelocityGradient2(logm(np.asarray(self)) / time)


class VelocityGradient2(Matrix2):
    """
    The class to represent 2D velocity gradient tensor.

    Args:
      a (2x2 array_like): Input data, that can be converted to
          2x2 2D array. This includes lists, tuples and ndarrays.

    Returns:
      ``VelocityGradient2`` object

    Example:
      >>> L = velgrad2(np.diag([0.1, -0.1]))

    """

    def defgrad(self, time=1, steps=1):
        """
        Return ``DeformationGradient2`` tensor accumulated after given time.

        Keyword Args:
            time (float): time of deformation. Default 1
            steps (int): when bigger than 1, will return a list
                         of ``DeformationGradient2`` tensors for each timestep.
        """
        from scipy.linalg import expm

        if steps > 1:  # FIX once container for matrix will be implemented
            return [
                DeformationGradient2(expm(np.asarray(self) * t))
                for t in np.linspace(0, time, steps)
            ]
        else:
            return DeformationGradient2(expm(np.asarray(self) * time))

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


class Tensor2(Matrix2):
    pass


class Stress2(Tensor2):
    """
    The class to represent 2D stress tensor.

    Args:
      a (2x2 array_like): Input data, that can be converted to
          2x2 2D array. This includes lists, tuples and ndarrays.

    Returns:
      ``Stress2`` object

    Example:
      >>> S = Stress2([[-8, 0, 0],[0, -5, 0],[0, 0, -1]])

    """

    @classmethod
    def from_comp(cls, xx=0, xy=0, yy=0):
        """
        Return ``Stress2`` tensor. Default is zero tensor.

        Note that stress tensor must be symmetrical.

        Keyword Args:
          xx, xy, yy (float): tensor components

        Example:
          >>> S = stress2.from_comp(xx=-5, yy=-2, xy=1)
          >>> S
          Stress2
          [[-5.  1.]
           [ 1. -2.]]
        """

        return cls([[xx, xy], [xy, yy]])

    @property
    def mean_stress(self):
        """
        Mean stress
        """

        return self.I1 / 2

    @property
    def hydrostatic(self):
        """
        Mean hydrostatic stress tensor component
        """

        return type(self)(np.diag(self.mean_stress * np.ones(2)))

    @property
    def deviatoric(self):
        """
        A stress deviator tensor component
        """

        return type(self)(self - self.hydrostatic)

    @property
    def sigma1(self):
        """
        A maximum principal stress (max compressive)
        """

        return self.E2

    @property
    def sigma2(self):
        """
        A minimum principal stress
        """

        return self.E1

    @property
    def sigma1dir(self):
        """
        Return unit length vector in direction of maximum
        principal stress (max compressive)
        """

        return self.V2

    @property
    def sigma2dir(self):
        """
        Return unit length vector in direction of minimum
        principal stress
        """

        return self.V1

    @property
    def sigma1vec(self):
        """
        Return maximum principal stress vector (max compressive)
        """

        return self.E2 * self.V2

    @property
    def sigma2vec(self):
        """
        Return minimum principal stress vector
        """

        return self.E1 * self.V1

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

        return float((self.I1**2 - np.trace(self**2)) / 2)

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
            DeformationGradient2(self.eigenvectors()),
        )

    def cauchy(self, n):
        """
        Return stress vector associated with plane given by normal vector.

        Args:
          n: normal given as ``Vector2`` object

        Example:
          >>> S = Stress.from_comp(xx=-5, yy=-2, xy=1)
          >>> S.cauchy(vec2(1,1))
          V(-2.520, 0.812, 8.660)

        """

        return Vector2(np.dot(self, n.normalized()))

    def stress_comp(self, n):
        """
        Return normal and shear stress ``Vector2`` components on plane given
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

    def signed_shear_stress(self, n):
        """
        Return signed shear stress magnitude on plane given by normal vector.
        """
        R = DeformationGradient2.from_angle(n.direction)
        return self.transform(R)[1, 0]


class Ellipse(Tensor2):
    """
    The class to represent 2D ellipse

    See following methods and properties for additional operations.

    Args:
      matrix (2x2 array_like): Input data, that can be converted to
             2x2 2D matrix. This includes lists, tuples and ndarrays.

    Returns:
      ``Ellipse`` object

    Example:
      >>> E = ellipse([[8, 0], [0, 2]])
      >>> E
      Ellipse
      [[8. 0.]
       [0. 2.]]
      (ar:2, ori:0)

    """

    def __repr__(self) -> str:
        return (
            f"{Matrix2.__repr__(self)}\n(ar:{self.ar:.3g}, ori:{self.orientation:.3g})"
        )

    @classmethod
    def from_defgrad(cls, F, form="left", **kwargs) -> "Ellipse":
        """
        Return deformation tensor from ``Defgrad2``.

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
    def from_stretch(cls, x=1, y=1, **kwargs) -> "Ellipse":
        """
        Return diagonal tensor defined by magnitudes of principal stretches.
        """
        return cls([[x * x, 0], [0, y * y]], **kwargs)

    @property
    def S1(self) -> float:
        """
        Return the maximum  principal stretch.
        """
        return math.sqrt(self.E1)

    @property
    def S2(self) -> float:
        """
        Return the minimum principal stretch.
        """
        return math.sqrt(self.E2)

    @property
    def e1(self) -> float:
        """
        Return the maximum natural principal strain.
        """
        return math.log(self.S1)

    @property
    def e2(self) -> float:
        """
        Return the minimum natural principal strain.
        """
        return math.log(self.S2)

    @property
    def ar(self) -> float:
        """
        Return the ellipse axial ratio.
        """
        return self.S1 / self.S2

    @property
    def orientation(self):
        """
        Return the orientation of the maximum eigenvector.
        """
        return self.V1.direction % 180

    @property
    def e12(self) -> float:
        """
        Return the difference between natural principal strains.
        """
        return self.e1 - self.e2


class OrientationTensor2(Ellipse):
    """
    Represents an 2D orientation tensor, which characterize data distribution
    using eigenvalue method. See (Watson 1966, Scheidegger 1965).

    See following methods and properties for additional operations.

    Args:
      matrix (2x2 array_like): Input data, that can be converted to
             2x2 2D matrix. This includes lists, tuples and ndarrays.
             Array could be also ``Group`` (for backward compatibility)

    Returns:
      ``OrientationTensor2`` object

    Example:
      >>> v = vec2set.random(n=1000)
      >>> ot = v.ortensor()
      >>> ot
      OrientationTensor2
      [[ 0.502 -0.011]
       [-0.011  0.498]]
      (ar:1.02, ori:140)

    """

    @classmethod
    def from_features(cls, g) -> "OrientationTensor2":
        """
        Return ``Ortensor`` of data in Vector2Set features

        Args:
            g (Vector2Set): Set of features

        Example:
          >>> v = vec2set.random_vonmises(position=120)
          >>> ot = v.ortensor()
          >>> ot
          OrientationTensor2
          [[ 0.377 -0.282]
           [-0.282  0.623]]
          (ar:2.05, ori:123)

        """

        return cls(np.dot(np.array(g).T, np.array(g)) / len(g))
