# -*- coding: utf-8 -*-

from __future__ import division, print_function

import numpy as np
from .core import Vec3, Group
from .helpers import sind, cosd

__all__ = ['DefGrad', 'VelGrad']


class DefGrad(np.ndarray):
    """class to store deformation gradient tensor derived from numpy.ndarray
    """
    def __new__(cls, array):
        # casting to our class
        assert np.shape(array) == (3, 3), 'DefGrad must be 3x3 2D array'
        obj = np.asarray(array).view(cls)
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
        return bool(np.sum(abs(self-other)) < 1e-14)

    def __ne__(self, other):
        # not equal
        return not self == other

    @classmethod
    def from_axis(cls, vector, theta):
        x, y, z = vector.uv
        c, s = cosd(theta), sind(theta)
        xs, ys, zs = x*s, y*s, z*s
        xc, yc, zc = x*(1-c), y*(1-c), z*(1-c)
        xyc, yzc, zxc = x*yc, y*zc, z*xc
        return cls([
            [x*xc+c, xyc-zs, zxc+ys],
            [xyc+zs, y*yc+c, yzc-xs],
            [zxc-ys, yzc+xs, z*zc+c]])

    @classmethod
    def from_comp(cls,
                  xx=1, xy=0, xz=0,
                  yx=0, yy=1, yz=0,
                  zx=0, zy=0, zz=1):
        return cls([
            [xx, xy, xz],
            [yx, yy, yz],
            [zx, zy, zz]])

    @property
    def I(self):
        return np.linalg.inv(self)

    def rotate(self, vector, theta):
        R = DefGrad.from_axis(vector, theta)
        return R*self*R.T

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
    """class to store  velocity gradient tensor derived from numpy.ndarray
    """
    def __new__(cls, array):
        # casting to our class
        assert np.shape(array) == (3, 3), 'VelGrad must be 3x3 2D array'
        obj = np.asarray(array).view(cls)
        return obj

    def __repr__(self):
        return 'VelGrad:\n' + str(self)

    def __pow__(self, n):
        # matrix power
        return np.linalg.matrix_power(self, n)

    def __eq__(self, other):
        # equal
        return bool(np.sum(abs(self-other)) < 1e-14)

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
        return DefGrad(expm(self*time))
