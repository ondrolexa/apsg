# -*- coding: utf-8 -*-

from __future__ import division, print_function

import numpy as np
from .core import Vec3, Group, Pair, Fault
from .helpers import sind, cosd

__all__ = ['DefGrad', 'VelGrad', 'Stress']


class DefGrad(np.ndarray):
    """class to store deformation gradient tensor derived from numpy.ndarray
    """
    def __new__(cls, array):
        # casting to our class
        assert np.shape(array) == (3, 3), 'DefGrad must be 3x3 2D array'
        obj = np.asarray(array).view(cls)
        obj.name = 'D'
        return obj

    def __repr__(self):
        return 'DefGrad:\n' + str(self)

    def __mul__(self, other):
        assert np.shape(other) == (3, 3), \
            'DefGrad could by multiplied with 3x3 2D array'
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
        x, y, z = vector.uv
        c, s = cosd(theta), sind(theta)
        xs, ys, zs = x * s, y * s, z * s
        xc, yc, zc = x * (1 - c), y * (1 - c), z * (1 - c)
        xyc, yzc, zxc = x * yc, y * zc, z * xc
        return cls([
            [x * xc + c, xyc - zs, zxc + ys],
            [xyc + zs, y * yc + c, yzc - xs],
            [zxc - ys, yzc + xs, z * zc + c]])

    @classmethod
    def from_comp(cls,
                  xx=1, xy=0, xz=0,
                  yx=0, yy=1, yz=0,
                  zx=0, zy=0, zz=1):
        return cls([
            [xx, xy, xz],
            [yx, yy, yz],
            [zx, zy, zz]])

    @classmethod
    def from_pair(cls, p):
        assert issubclass(type(p), Pair), 'Data must be of Pair type.'
        return cls(np.array([p.lvec, p.fvec**p.lvec, p.fvec]).T)

    @property
    def I(self):
        return np.linalg.inv(self)

    def rotate(self, vector, theta):
        R = DefGrad.from_axis(vector, theta)
        return R * self * R.T

    @property
    def eigenvals(self):
        _, vals, _ = np.linalg.svd(self)
        return tuple(vals)

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
        U, _, _ = np.linalg.svd(self)
        return Group([Vec3(U.T[0]),
                      Vec3(U.T[1]),
                      Vec3(U.T[2])])

    @property
    def eigenlins(self):
        return self.eigenvects.aslin

    @property
    def eigenfols(self):
        return self.eigenvects.asfol

    @property
    def R(self):
        from scipy.linalg import polar
        R, _ = polar(self)
        return DefGrad(R)

    @property
    def U(self):
        from scipy.linalg import polar
        _, U = polar(self)
        return DefGrad(U)

    @property
    def V(self):
        from scipy.linalg import polar
        _, V = polar(self, 'left')
        return DefGrad(V)


class VelGrad(np.ndarray):
    """class to store velocity gradient tensor derived from numpy.ndarray
    """
    def __new__(cls, array):
        # casting to our class
        assert np.shape(array) == (3, 3), 'VelGrad must be 3x3 2D array'
        obj = np.asarray(array).view(cls)
        obj.name = 'L'
        return obj

    def __repr__(self):
        return 'VelGrad:\n' + str(self)

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
    def from_comp(cls,
                  xx=0, xy=0, xz=0,
                  yx=0, yy=0, yz=0,
                  zx=0, zy=0, zz=0):
        return cls([
            [xx, xy, xz],
            [yx, yy, yz],
            [zx, zy, zz]])

    def defgrad(self, time=1):
        from scipy.linalg import expm
        return DefGrad(expm(self * time))


class Stress(np.ndarray):
    """class to store stress tensor derived from numpy.ndarray
    """
    def __new__(cls, array):
        # casting to our class
        assert np.shape(array) == (3, 3), 'Stress must be 3x3 2D array'
        assert np.allclose(np.asarray(array), np.asarray(array).T), 'Stress tensor must be symmetrical'
        obj = np.asarray(array).view(cls)
        obj.name = 'S'
        return obj

    def __repr__(self):
        return 'Stress:\n' + str(self)

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
    def from_comp(cls,
                  xx=0, xy=0, xz=0,
                  yx=0, yy=0, yz=0,
                  zx=0, zy=0, zz=0):
        return cls([
            [xx, xy, xz],
            [yx, yy, yz],
            [zx, zy, zz]])

    def rotate(self, vector, theta):
        R = DefGrad.from_axis(vector, theta)
        return Stress(R * self * R.T)

    @property
    def eigenvals(self):
        vals, _ = np.linalg.eig(self)
        return tuple(vals)

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
        _, U = np.linalg.eig(self)
        return Group([Vec3(U.T[0]),
                      Vec3(U.T[1]),
                      Vec3(U.T[2])])

    @property
    def eigenlins(self):
        return self.eigenvects.aslin

    @property
    def eigenfols(self):
        return self.eigenvects.asfol

    def cauchy(self, n):
        """Return stress vector associated with plane given by normal vector

        Args:
          n: normal given as ``Vec3`` or ``Fol`` object

        """
        return Vec3(np.dot(self, n))

    def fault(self, n):
        """Return ``Fault`` object derived from given by normal vector

        Args:
          n: normal given as ``Vec3`` or ``Fol`` object

        """
        return Fault.from_vecs(*self.stress_comp(n))

    def stress_comp(self, n):
        """Return normal and shear stress ``Vec3`` components on plane given by normal vector"""
        t = self.cauchy(n)
        sn = t.proj(n)
        return sn, t - sn

    def normal_stress(self, n):
        """Return magnitude of normal stress component on plane given by normal vector"""
        sn, tau = self.stress_comp(n)
        return abs(sn)

    def shear_stress(self, n):
        """Return magnitude of shear stress component on plane given by normal vector"""
        sn, tau = self.stress_comp(n)
        return abs(tau)
