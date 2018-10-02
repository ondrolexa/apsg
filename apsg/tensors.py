#!/usr/bin/env python
# -*- coding: utf-8 -*-


from __future__ import division, print_function

import numpy as np

from .core import Vec3, Group, Pair, Fault
from .helpers import sind, cosd


__all__ = ("DefGrad", "VelGrad", "Stress")


class DefGrad(np.ndarray):
    """
    ``DefGrad`` store deformation gradient tensor derived from numpy.ndarray.

    Args:
      a (3x3 array_like): Input data, that can be converted to
      an 3x3 2D array. This includes lists, tuples and ndarrays.

    Returns:
      ``DefGrad`` object

    Example:
      >>> F = DefGrad(np.diag([2, 1, 0.5]))
    """

    def __new__(cls, array):
        # casting to our class
        assert np.shape(array) == (3, 3), "DefGrad must be 3x3 2D array"
        obj = np.asarray(array).view(cls)
        obj.name = "D"
        return obj

    def __repr__(self):
        return "DefGrad:\n" + str(self)

    def __mul__(self, other):
        assert np.shape(other) == (
            3,
            3,
        ), "DefGrad could by multiplied with 3x3 2D array"
        return np.dot(self, other)

    def __pow__(self, n):
        # matrix power
        return np.linalg.matrix_power(self, n)

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
        """Return ``DefGrad`` tensor. Default is identity tensor.

        Keyword Args:
          xx, xy, xz, yx, yy, yz, zx, zy, zz: tensor components

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
    def from_pair(cls, p):

        assert issubclass(type(p), Pair), "Data must be of Pair type."

        return cls(np.array([p.lvec, p.fvec ** p.lvec, p.fvec]).T)

    @property
    def I(self):
        """
        Returns the inverse tensor.
        """

        return np.linalg.inv(self)

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
            raise ValueError("no unit eigenvector corresponding to eigenvalue 1")
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


class VelGrad(np.ndarray):
    """
    ``VelGrad`` store velocity gradient tensor derived from numpy.ndarray.

    Args:
      a (3x3 array_like): Input data, that can be converted to
      an 3x3 2D array. This includes lists, tuples and ndarrays.

    Returns:
      ``VelGrad`` object

    Example:
      >>> L = VelGrad(np.diag([0.1, 0, -0.1]))
    """

    def __new__(cls, array):
        # casting to our class
        assert np.shape(array) == (3, 3), "VelGrad must be 3x3 2D array"
        obj = np.asarray(array).view(cls)
        obj.name = "L"
        return obj

    def __repr__(self):
        return "VelGrad:\n" + str(self)

    def __pow__(self, n):
        # matrix power
        return np.linalg.matrix_power(self, n)

    def __eq__(self, other):
        # equal
        return bool(np.sum(abs(self - other)) < 1e-14)

    def __ne__(self, other):
        # not equal
        return not self == other

    @classmethod
    def from_comp(cls, xx=0, xy=0, xz=0, yx=0, yy=0, yz=0, zx=0, zy=0, zz=0):
        """Return ``VelGrad`` tensor. Default is zero tensor.

        Keyword Args:
          xx, xy, xz, yx, yy, yz, zx, zy, zz: tensor components

        Example:
          >>> L = VelGrad.from_comp(xx=0.1, zz=-0.1)
          >>> L
          VelGrad:
          [[ 0.1  0.   0. ]
           [ 0.   0.   0. ]
           [ 0.   0.  -0.1]]

        """
        return cls([[xx, xy, xz], [yx, yy, yz], [zx, zy, zz]])

    def defgrad(self, time=1):
        """Return ``DefGrad`` accumulated after given time"""
        from scipy.linalg import expm

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
        """Return rate of deformation tensor"""
        return (self + self.T) / 2

    @property
    def spin(self):
        """Return spin tensor"""
        return (self - self.T) / 2


class Stress(np.ndarray):
    """
    ``Stress`` store stress tensor derived from numpy.ndarray.

    Args:
      a (3x3 array_like): Input data, that can be converted to
      an 3x3 2D array. This includes lists, tuples and ndarrays.

    Returns:
      ``Stress`` object

    Example:
      >>> S = Stress([[-8, 0, 0],[0, -5, 0],[0, 0, -1]])
    """

    def __new__(cls, array):
        # casting to our class

        assert np.shape(array) == (3, 3), "Stress must be 3x3 2D array"
        assert np.allclose(
            np.asarray(array), np.asarray(array).T
        ), "Stress tensor must be symmetrical"

        obj = np.asarray(array).view(cls)
        obj.name = "S"

        return obj

    def __repr__(self):

        return "Stress:\n" + str(self)

    def __pow__(self, n):
        # matrix power

        return np.linalg.matrix_power(self, n)

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
          xx, xy, xz, yy, yz, zz: tensor components

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
    def eigenvals(self):
        """
        Return tuple of eigenvalues
        """

        vals, _ = np.linalg.eig(self)

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

        _, U = np.linalg.eig(self)

        return Group([Vec3(U.T[0]), Vec3(U.T[1]), Vec3(U.T[2])])

    @property
    def eigenlins(self):

        return self.eigenvects.aslin

    @property
    def eigenfols(self):

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
          S = Stress.from_comp(xx=-5, yy=-2, zz=10, xy=8)
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
        Return magnitude of normal stress component on plane given
        by normal vector.
        """

        sn, tau = self.stress_comp(n)

        return abs(sn)

    def shear_stress(self, n):
        """
        Return magnitude of shear stress component on plane given
        by normal vector.
        """

        sn, tau = self.stress_comp(n)

        return abs(tau)
