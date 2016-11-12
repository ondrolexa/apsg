# -*- coding: utf-8 -*-
"""
Python module to manipulate, analyze and visualize structural geology data

"""

from __future__ import division, print_function
from copy import deepcopy
import warnings

import numpy as np
from .helpers import sind, cosd, acosd, asind, atan2d, angle_metric

__all__ = ['Vec3', 'Lin', 'Fol', 'Pair', 'Fault',
           'Group', 'FaultSet', 'Ortensor', 'Cluster', 'G']

settings = dict(notation='dd', vec2dd=False)


class Vec3(np.ndarray):
    """Base class to store 3D vectors derived from numpy.ndarray

    Args:
      a (array_like): Input data, that can be converted to an array.
        This includes lists, lists of tuples, tuples, tuples of tuples, tuples
        of lists and ndarrays.

    Example:
      >>> v = Vec3([0.67, 1.2, 0.73])

    """
    def __new__(cls, array, inc=None, mag=1.0):
        """Convert the input to 3D vector.

        """
        if inc is None:
            obj = np.asarray(array).view(cls)
        else:
            obj = mag * Lin(array, inc).view(cls)
        return obj

    def __repr__(self):
        if settings['vec2dd']:
            r = 'V:{:.0f}/{:.0f}'.format(*self.dd)
        else:
            r = 'V({:.3f}, {:.3f}, {:.3f})'.format(*self)
        return r

    def __mul__(self, other):
        """Returns dot product of two vectors.

        """
        return np.dot(self, other)

    def __abs__(self):
        """Returns the 2-norm or Euclidean norm of vector.

        """
        return np.sqrt(self * self)

    def __pow__(self, other):
        """Return cross product if argument is vector or power of vector

        """
        if np.isscalar(other):
            return pow(abs(self), other)
        else:
            return self.cross(other)

    def __eq__(self, other):
        """Returns True if vectors are equal.

        """
        return bool(abs(self - other) < 1e-15)

    def __ne__(self, other):
        """Returns False if vectors are equal.

        """
        return not self == other

    @property
    def type(self):
        return type(self)

    @property
    def uv(self):
        """Returns unit vector

        Example:
          >>> u = Vec3([1,1,1])
          >>> u.uv
          V(0.577, 0.577, 0.577)

        """
        return self / abs(self)

    def cross(self, other):
        """Returns cross product of two vectors::

        Example:
          >>> v=Vec3([0,2,-2])
          >>> u.cross(v)
          V(-4.000, 2.000, 2.000)

        """
        return Vec3(np.cross(self, other))

    def angle(self, other):
        """Returns angle of two vectors in degrees::

        Example:
          >>> u.angle(v)
          90.0

        """
        if isinstance(other, Group):
            return other.angle(self)
        else:
            return acosd(np.clip(np.dot(self.uv, other.uv), -1, 1))

    def rotate(self, axis, phi):
        """Rotates vector `phi` degrees about `axis`::

        Args:
          axis: axis of rotation
          phi: angle of rotation in degrees

        Example:
          >>> v.rotate(u,60)
          V(-2.000, 2.000, -0.000)

        """
        e = Vec3(self)  # rotate all types as vectors
        k = axis.uv
        r = (cosd(phi) * e +
             sind(phi) * k.cross(e) +
             (1 - cosd(phi)) * k * (k * e))
        return r.view(type(self))

    def proj(self, other):
        """Returns projection of vector *u* onto vector *v*::

        Example:
          >>> u.proj(v)

        To project on plane use: u - u.proj(v), where v in plane normal.

        """
        r = np.dot(self, other) * other / abs(other)**2
        return r.view(type(self))

    def transform(self, F):
        """Returns affine transformation of vector *u* by matrix *F*::

        Args:
          F: Transformation matrix. Should be array-like value e.g. ``DefGrad``

        Example:
          >>> u.transform(F)

        See Also:
          ``strain.DefGrad``

        """
        return np.dot(F, self).view(type(self))

    @property
    def aslin(self):
        """Convert to ``Lin`` object.

        Example:
          >>> u = Vec3([1,1,1])
          >>> u.aslin
          L:45/35

        """
        return self.copy().view(Lin)

    @property
    def asfol(self):
        """Convert to ``Fol`` object.

        Example:
          >>> u = Vec3([1,1,1])
          >>> u.asfol
          S:225/55

        """
        return self.copy().view(Fol)

    @property
    def asvec3(self):
        """Convert to ``Vec3`` object.

        Example:
          >>> l = Lin(120,50)
          >>> l.asvec3
          V(-0.321, 0.557, 0.766)

        """
        return self.copy().view(Vec3)

    @property
    def V(self):
        """Convert to ``Vec3`` object.
        """
        return self.copy().view(Vec3)

    @property
    def dd(self):
        """ Return Return dip-direction, dip tuple

        """
        n = self.uv
        dec = atan2d(n[1], n[0]) % 360
        inc = asind(n[2])
        return dec, inc


class Lin(Vec3):
    """Class to store linear feature.

    Args:
      azi: Dip direction of linear feature in degrees
      inc: dip of linear feature in degrees

    Example:
      >>> l = Lin(120, 60)

    """
    def __new__(cls, azi, inc):
        """Create linear feature.

        """
        v = [cosd(azi) * cosd(inc),
             sind(azi) * cosd(inc),
             sind(inc)]
        return Vec3(v).view(cls)

    def __repr__(self):
        return 'L:{:.0f}/{:.0f}'.format(*self.dd)

    def __add__(self, other):
        """Sum of axial data

        """
        if self * other < 0:
            other = -other
        return super(Lin, self).__add__(other)

    def __iadd__(self, other):
        if self * other < 0:
            other = -other
        return super(Lin, self).__iadd__(other)

    def __sub__(self, other):
        """Substract axial data

        """
        if self * other < 0:
            other = -other
        return super(Lin, self).__sub__(other)

    def __isub__(self, other):
        if self * other < 0:
            other = -other
        return super(Lin, self).__isub__(other)

    def __eq__(self, other):
        """Returns True if linear features are equal.

        """
        return bool(abs(self - other) < 1e-15 or abs(self + other) < 1e-15)

    def __ne__(self, other):
        """Returns False if linear features are equal.

        """
        return not (self == other or self == -other)

    def angle(self, other):
        """Returns angle of two linear features in degrees

        Example:
          >>> u.angle(v)
          90.0

        """
        if isinstance(other, Group):
            return other.angle(self)
        else:
            return acosd(abs(np.clip(np.dot(self.uv, other.uv), -1, 1)))

    def cross(self, other):
        """Returns planar feature defined by two linear features

        Example:
          >>> l=Lin(120,10)
          >>> l.cross(Lin(160,30))
          S:196/35

        """
        if isinstance(other, Group):
            return other.cross(self)
        else:
            return np.cross(self, other).view(Fol)

    @property
    def dd(self):
        """ Return dip-direction, dip tuple

        """
        n = self.uv
        if n[2] < 0:
            n = -n
        azi = atan2d(n[1], n[0]) % 360
        inc = asind(n[2])
        return azi, inc


class Fol(Vec3):
    """Class to store planar feature.

    Args:
      azi: Dip direction of planar feature in degrees
      inc: dip of planar feature in degrees

    Example:
      >>> f = Fol(120, 60)

    """
    def __new__(cls, azi, inc):
        """Create planar feature.

        """
        if settings['notation'] == 'rhr':
            azi += 90
        v = [-cosd(azi) * sind(inc),
             -sind(azi) * sind(inc),
             cosd(inc)]
        return Vec3(v).view(cls)

    def __repr__(self):
        return 'S:{:.0f}/{:.0f}'.format(*getattr(self, settings['notation']))

    def __add__(self, other):
        """Sum of axial data

        """
        if self * other < 0:
            other = -other
        return super(Fol, self).__add__(other)

    def __iadd__(self, other):
        if self * other < 0:
            other = -other
        return super(Fol, self).__iadd__(other)

    def __sub__(self, other):
        """Substract axial data

        """
        if self * other < 0:
            other = -other
        return super(Fol, self).__sub__(other)

    def __isub__(self, other):
        if self * other < 0:
            other = -other
        return super(Fol, self).__isub__(other)

    def __eq__(self, other):
        """Returns True if planar features are equal.

        """
        return bool(abs(self - other) < 1e-15 or abs(self + other) < 1e-15)

    def __ne__(self, other):
        """Returns False if planar features are equal.

        """
        return not (self == other or self == -other)

    def angle(self, other):
        """Returns angle of two planar features in degrees

        Example:
          >>> u.angle(v)
          90.0

        """
        if isinstance(other, Group):
            return other.angle(self)
        else:
            return acosd(abs(np.clip(np.dot(self.uv, other.uv), -1, 1)))

    def cross(self, other):
        """Returns linear feature defined as intersection of two planar features

        Example:
          >>> f=Fol(60,30)
          >>> f.cross(Fol(120,40))
          L:72/29

        """
        if isinstance(other, Group):
            return other.cross(self)
        else:
            return np.cross(self, other).view(Lin)

    def transform(self, F):
        """Returns affine transformation of planar feature by matrix *F*::

        Args:
          F: Transformation matrix. Should be array-like value e.g. ``DefGrad``

        Example:
          >>> f.transform(F)

        See Also:
          ``strain.DefGrad``

        """
        return np.dot(self, np.linalg.inv(F)).view(type(self))

    @property
    def dd(self):
        """ Return dip-direction, dip tuple

        """
        n = self.uv
        if n[2] < 0:
            n = -n
        azi = (atan2d(n[1], n[0]) + 180) % 360
        inc = 90 - asind(n[2])
        return azi, inc

    @property
    def rhr(self):
        """ Return strike and dip tuple (right-hand-rule)

        """
        azi, inc = self.dd
        return (azi - 90) % 360, inc

    @property
    def dv(self):
        """Convert to dip vector ``Vec3`` object.

        Example:
          >>> f = Fol(120,50)
          >>> f.dv
          V(-0.321, 0.557, 0.766)

        """
        azi, inc = self.dd
        return Lin(azi, inc).view(Vec3)

    def rake(self, rake):
        """Convert to vector ``Vec3`` object with given rake.

        Example:
          >>> f = Fol(120,50)
          >>> f.rake(30)
          V(-0.911, -0.155, 0.383)
          >>> f.rake(30).aslin

        """
        return self.dv.rotate(self, rake - 90)


class Pair(object):
    """Class to store pair of planar and linear feature.

    When ``Pair`` object is created, both planar and linear feature are
    adjusted, so linear feature perfectly fit onto planar one. Warning
    is issued, when misfit angle is bigger than 20 degrees.

    Args:
      fazi: Dip direction of planar feature in degrees
      finc: dip of planar feature in degrees
      lazi: Dip direction of linear feature in degrees
      linc: dip of linear feature in degrees

    Example:
      >>> p = Pair(140,30,110,26)

    """
    def __init__(self, fazi, finc, lazi, linc):
        """Create ``Pair`` object.

        """
        fol = Fol(fazi, finc)
        lin = Lin(lazi, linc)
        misfit = 90 - fol.angle(lin)
        if misfit > 20:
            warnings.warn('Warning: Misfit angle is %.1f degrees.' % misfit)
        ax = fol**lin
        ang = (Vec3(lin).angle(fol) - 90) / 2
        fol = fol.rotate(ax, ang)
        lin = lin.rotate(ax, -ang)
        self.fvec = Vec3(fol)
        self.lvec = Vec3(lin)
        self.misfit = misfit

    def __repr__(self):
        vals = getattr(self.fol, settings['notation']) + self.lin.dd
        return 'P:{:.0f}/{:.0f}-{:.0f}/{:.0f}'.format(*vals)

    @classmethod
    def from_pair(cls, fol, lin):
        """Create ``Pair`` from Fol and Lin objects"""
        data = getattr(fol, settings['notation']) + lin.dd
        return cls(*data)

    def rotate(self, axis, phi):
        """Rotates ``Pair`` by `phi` degrees about `axis`::

        Args:
          axis: axis of rotation
          phi: angle of rotation in degrees

        Example:
          >>> v.rotate(u,60)
          V(-2.000, 2.000, -0.000)

        """
        rot = deepcopy(self)
        rot.fvec = self.fvec.rotate(axis, phi)
        rot.lvec = self.lvec.rotate(axis, phi)
        return rot

    @property
    def type(self):
        return type(self)

    @property
    def fol(self):
        """Returns planar feature of ``Pair`` as ``Fol``.

        """
        return self.fvec.asfol

    @property
    def lin(self):
        """Returns linear feature of ``Pair`` as ``Lin``.

        """
        return self.lvec.aslin

    def transform(self, F):
        """Returns affine transformation of ``Pair`` by matrix *F*::

        Args:
          F: Transformation matrix. Should be array-like value e.g. ``DefGrad``

        Example:
          >>> p.transform(F)

        See Also:
          ``strain.DefGrad``

        """
        t = deepcopy(self)
        t.lvec = np.dot(F, t.lvec).view(Vec3)
        t.fvec = np.dot(t.fvec, np.linalg.inv(F)).view(Vec3)
        return t


class Fault(Pair):
    """Fault class for related Fol and Lin instances with sense of movement"""
    def __init__(self, fazi, finc, lazi, linc, sense):
        assert isinstance(sense, int), \
            'Third sense parameter must be positive or negative integer'
        super(Fault, self).__init__(fazi, finc, lazi, linc)
        sense = 2 * int(sense > 0) - 1
        ax = self.fvec**self.lvec
        self.lvec = sense * self.lvec
        self.pvec = self.fvec.rotate(ax, -45 * sense)
        self.tvec = self.fvec.rotate(ax, 45 * sense)

    def __repr__(self):
        s = ['', '+', '-'][self.sense]
        vals = getattr(self.fol, settings['notation']) + self.lin.dd + (s,)
        return 'F:{:.0f}/{:.0f}-{:.0f}/{:.0f} {:s}'.format(*vals)

    @classmethod
    def from_pair(cls, fol, lin, sense):
        """Create ``Fault`` from Fol and Lin objects"""
        data = getattr(fol, settings['notation']) + lin.dd + (sense,)
        return cls(*data)

    def rotate(self, axis, phi):
        rot = deepcopy(self)
        rot.fvec = self.fvec.rotate(axis, phi)
        rot.lvec = self.lvec.rotate(axis, phi)
        rot.pvec = self.pvec.rotate(axis, phi)
        rot.tvec = self.tvec.rotate(axis, phi)
        return rot

    @property
    def sense(self):
        return 2 * int(self.fvec**self.lvec == Vec3(self.fol**self.lin)) - 1

    @property
    def p(self):
        """return P axis"""
        return self.pvec.aslin

    @property
    def t(self):
        """return T axis"""
        return self.tvec.aslin

    @property
    def m(self):
        """return kinematic M-plane"""
        return (self.fvec**self.lvec).asfol


class Group(list):
    """Group class
    Group is homogeneous group of ``Vec3``, ``Fol`` or ``Lin`` objects
    """
    def __init__(self, data, name='Default'):
        assert issubclass(type(data), list), 'Argument must be list of data.'
        assert len(data) > 0, 'Empty group is not allowed.'
        tp = type(data[0])
        assert issubclass(tp, Vec3), 'Data must be Fol, Lin or Vec3 type.'
        assert all([isinstance(e, tp) for e in data]), \
            'All data in group must be of same type.'
        super(Group, self).__init__(data)
        self.type = tp
        self.name = name

    def __repr__(self):
        return '%s\n%g %s' % (self.name, len(self), self.type.__name__)

    def __abs__(self):
        # abs returns array of euclidean norms
        return np.asarray([abs(e) for e in self])

    def __add__(self, other):
        # merge Datasets
        assert isinstance(other, Group), 'Only groups could be merged'
        assert self.type is other.type, 'Only same type groups could be merged'
        return Group(list(self) + other, name=self.name)

    def __setitem__(self, key, value):
        assert isinstance(value, self.type), \
            'item is not of type %s' % self.type.__name__
        super(Group, self).__setitem__(key, value)

    def __getitem__(self, key):
        """Group fancy indexing"""
        if isinstance(key, slice):
            key = np.arange(*key.indices(len(self)))
        if isinstance(key, list) or isinstance(key, tuple):
            key = np.asarray(key)
        if isinstance(key, np.ndarray):
            if key.dtype == 'bool':
                key = np.flatnonzero(key)
            return Group([self[i] for i in key])
        else:
            return super(Group, self).__getitem__(key)

    def __getattr__(self, attr):
        if attr in self[0].__dict__:
            try:
                res = np.array([getattr(e, attr) for e in self])
            except ValueError:
                res = [getattr(e, attr) for e in self]
            return res
        else:
            raise AttributeError

    def append(self, item):
        assert isinstance(item, self.type), \
            'item is not of type %s' % self.type.__name__
        super(Group, self).append(item)

    def extend(self, items=()):
        for item in items:
            self.append(item)

    def copy(self):
        return Group(super(Group, self).copy(), self.name)

    @property
    def data(self):
        return list(self)

    @classmethod
    def from_csv(cls, fname, typ=Lin, delimiter=',', acol=1, icol=2):
        """Create group from csv file"""
        from os.path import basename
        dt = np.loadtxt(fname, dtype=float, delimiter=delimiter).T
        return cls.from_array(dt[acol - 1], dt[icol - 1],
                              typ=typ, name=basename(fname))

    def to_csv(self, fname, delimiter=','):
        np.savetxt(fname, self.dd.T, fmt='%g', delimiter=',', header=self.name)

    @classmethod
    def from_array(cls, azis, incs, typ=Lin, name='Default'):
        """Create group from arrays of dip directions and dips"""
        data = []
        data = [typ(azi, inc) for azi, inc in zip(azis, incs)]
        return cls(data, name=name)

    @property
    def aslin(self):
        """Convert all data in Group to Lin"""
        return Group([e.aslin for e in self], name=self.name)

    @property
    def asfol(self):
        """Convert all data in Group to Fol"""
        return Group([e.asfol for e in self], name=self.name)

    @property
    def asvec3(self):
        """Convert all data in Group to Vec3"""
        return Group([e.asvec3 for e in self], name=self.name)

    @property
    def V(self):
        """Convert all data in Group to Vec3"""
        return Group([e.asvec3 for e in self], name=self.name)

    @property
    def R(self):
        """Return resultant of Group"""
        # r = deepcopy(self[0])
        # for v in self[1:]:
        #     r += v
        # return r
        #
        # As axial summing is not commutative we use vectorial
        # summing of centered data
        if self.type == Vec3:
            r = Vec3(np.sum(self, axis=0))
        else:
            irot = np.linalg.inv(self.ortensor.vects)
            if self.type == Lin:
                vals = self.centered.dd
                cg = Group.from_array(*vals, typ=Lin)
                r = (Vec3(np.sum(cg, axis=0)).aslin
                     .rotate(Lin(90, 0), -90).transform(irot))
            else:
                vals = getattr(self.centered, settings['notation'])
                cg = Group.from_array(*vals, typ=Fol)
                r = (Vec3(np.sum(cg, axis=0)).asfol
                     .rotate(Lin(90, 0), -90).transform(irot))
        return r

    @property
    def var(self):
        """Spherical variance"""
        return 1 - abs(self.R) / len(self)

    @property
    def fisher_stats(self):
        """Fisher's concentration parameter"""
        stats = {'k': np.inf, 'a95': 180.0, 'csd': 0.0}
        N = len(self)
        R = abs(self.R)
        if N != R:
            stats['k'] = (N - 1) / (N - R)
            stats['csd'] = 81 / np.sqrt(stats['k'])
        stats['a95'] = acosd(1 - ((N - R) / R) * (20**(1 / (N - 1)) - 1))
        return stats

    @property
    def delta(self):
        """cone containing ~63% of the data"""
        return acosd(abs(self.R) / len(self))

    @property
    def rdegree(self):
        """degree of preffered orientation od Group"""
        N = len(self)
        return 100 * (2 * abs(self.R) - N) / N

    def cross(self, other=None):
        """return cross products of all pairs in Group"""
        res = []
        if other is None:
            for i in range(len(self) - 1):
                for j in range(i + 1, len(self)):
                    res.append(self[i]**self[j])
        elif isinstance(other, Group):
            for e in self:
                for f in other:
                    res.append(e**f)
        elif issubclass(type(other), Vec3):
            for e in self:
                res.append(e**other)
        else:
            raise TypeError('Wrong argument type!')
        return Group(res, name=self.name)

    def rotate(self, axis, phi):
        """rotate Group"""
        return Group([e.rotate(axis, phi) for e in self], name=self.name)

    @property
    def centered(self):
        """rotate eigenvectors to axes of coordinate system
        E1(vertical), E2(east-west), E3(north-south)"""
        return self.transform(self.ortensor.vects).rotate(Lin(90, 0), 90)

    @property
    def uv(self):
        """return Group with normalized elements"""
        return Group([e.uv for e in self], name=self.name)

    def angle(self, other=None):
        """return angles of all pairs in Group"""
        res = []
        if other is None:
            for i in range(len(self) - 1):
                for j in range(i + 1, len(self)):
                    res.append(self[i].angle(self[j]))
        elif isinstance(other, Group):
            for e in self:
                for f in other:
                    res.append(e.angle(f))
        elif issubclass(type(other), Vec3):
            for e in self:
                res.append(e.angle(other))
        else:
            raise TypeError('Wrong argument type!')
        return np.array(res)

    @property
    def ortensor(self):
        """return orientation tensor of Group"""
        return Ortensor(self)

    @property
    def cluster(self):
        """return hierarchical clustering of Group"""
        return Cluster(self)

    def transform(self, F):
        """Return affine transformation of Group by matrix *F*"""
        return Group([e.transform(F) for e in self], name=self.name)

    @property
    def dd(self):
        """array of dip directions and dips of Group"""
        return np.array([d.dd for d in self]).T

    @property
    def rhr(self):
        """array of strikes and dips of Group"""
        return np.array([d.rhr for d in self]).T

    @classmethod
    def randn_lin(cls, N=100, mean=Lin(0, 90), sig=20):
        data = []
        ta, td = mean.dd
        for azi, dip in zip(180 * np.random.rand(N), sig * np.random.randn(N)):
            data.append(Lin(0, 90).rotate(Lin(azi, 0), dip))
        return cls(data).rotate(Lin(ta + 90, 0), 90 - td)

    @classmethod
    def randn_fol(cls, N=100, mean=Fol(0, 0), sig=20):
        data = []
        ta, td = mean.dd
        for azi, dip in zip(180 * np.random.rand(N), sig * np.random.randn(N)):
            data.append(Fol(0, 0).rotate(Lin(azi, 0), dip))
        return cls(data).rotate(Lin(ta - 90, 0), td)

    @classmethod
    def uniform_lin(cls, N=500):
        n = 2 * np.ceil(np.sqrt(N) / 0.564)
        azi = 0
        inc = 90
        for rho in np.linspace(0, 1, np.round(n / 2 / np.pi))[:-1]:
            theta = np.linspace(0, 360, np.round(n * rho + 1))[:-1]
            x, y = rho * sind(theta), rho * cosd(theta)
            azi = np.hstack((azi, atan2d(x, y)))
            ii = asind(np.sqrt((x * x + y * y) / 2))
            inc = np.hstack((inc, 90 - 2 * ii))
        # no antipodal
        theta = np.linspace(0, 360, n + 1)[:-1:2]
        x, y = sind(theta), cosd(theta)
        azi = np.hstack((azi, atan2d(x, y)))
        inc = np.hstack((inc, 90 - 2 * asind(np.sqrt((x * x + y * y) / 2))))
        # fix
        inc[inc < 0] = 0
        return cls.from_array(azi, inc, typ=Lin)

    @classmethod
    def uniform_fol(cls, N=500):
        l = cls.uniform_lin(N=N)
        azi, inc = l.dd
        if settings['notation'] == 'rhr':
            azi -= 90
        return cls.from_array(azi + 180, 90 - inc, typ=Fol)

    @classmethod
    def sfs_vec3(cls, N=1000):
        """Spherical Fibonacci Spiral points on a sphere.

        adopted from John Burkardt
        http://people.sc.fsu.edu/~jburkardt/
        """
        phi = (1 + np.sqrt(5)) / 2
        i2 = 2 * np.arange(N) - N + 1
        theta = 2 * np.pi * i2 / phi
        sp = i2 / N
        cp = np.sqrt((N + i2) * (N - i2)) / N
        dc = np.array([cp * np.sin(theta), cp * np.cos(theta), sp]).T
        return cls([Vec3(d) for d in dc])

    @classmethod
    def sfs_lin(cls, N=500):
        g = cls.sfs_vec3(N=2 * N)
        # no antipodal
        return cls([d.aslin for d in g if d[2] > 0])

    @classmethod
    def sfs_fol(cls, N=500):
        g = cls.sfs_vec3(N=2 * N)
        # no antipodal
        return cls([d.asfol for d in g if d[2] > 0])

    @classmethod
    def gss_vec3(cls, N=1000):
        """Golden Section Spiral points on a sphere.

        adopted http://www.softimageblog.com/archives/115
        """
        inc = np.pi * (3 - np.sqrt(5))
        off = 2 / N
        k = np.arange(N)
        y = k * off - 1 + (off / 2)
        r = np.sqrt(1 - y * y)
        phi = k * inc
        dc = np.array([np.cos(phi) * r, y, np.sin(phi) * r]).T
        return cls([Vec3(d) for d in dc])

    @classmethod
    def gss_lin(cls, N=500):
        g = cls.gss_vec3(N=2 * N)
        # no antipodal
        return cls([d.aslin for d in g if d[2] > 0])

    @classmethod
    def gss_fol(cls, N=500):
        g = cls.gss_vec3(N=2 * N)
        # no antipodal
        return cls([d.asfol for d in g if d[2] > 0])

    def to_file(self, filename='group.dat'):
        import pickle
        with open(filename, 'wb') as file:
            pickle.dump(self, file)
        print('Group saved to file %s' % filename)

    @classmethod
    def from_file(cls, filename='group.dat'):
        import pickle
        with open(filename, 'rb') as file:
            data = pickle.load(file)
        print('Group loaded from file %s' % filename)
        return cls(data, name=filename)

    def bootstrap(self, num=100, size=None):
        if size is None:
            size = len(self)
        for ix in np.random.randint(0, len(self), (num, size)):
            yield self[ix]

    @classmethod
    def examples(cls, name=None):
        if name is None:
            print('Available sample datasets:')
            print(list(Group.typs.keys()))
        else:
            return cls.from_array(
                Group.azis[name],
                Group.incs[name],
                Group.typs[name])

    # Sample datasets
    azis = {}
    incs = {}
    typs = {}
    # Embleton (1970) - Measurements of magnetic remanence in specimens
    # of Palaeozoic red-beds from Argentina.
    azis['B2'] = [122.5, 130.5, 132.5, 148.5, 140.0, 133.0, 157.5, 153.0,
                  140.0, 147.5, 142.0, 163.5, 141.0, 156.0, 139.5, 153.5,
                  151.5, 147.5, 141.0, 143.5, 131.5, 147.5, 147.0, 149.0,
                  144.0, 139.5]
    incs['B2'] = [55.5, 58.0, 44.0, 56.0, 63.0, 64.5, 53.0, 44.5, 61.5,
                  54.5, 51.0, 56.0, 59.5, 56.5, 54.0, 47.5, 61.0, 58.5,
                  57.0, 67.5, 62.5, 63.5, 55.5, 62.0, 53.5, 58.0]
    typs['B2'] = Lin
    # Cohen (1983) - Facing directions of conically folded planes.
    azis['B4'] = [269, 265, 271, 272, 268, 267, 265, 265, 265, 263, 267, 267,
                  270, 270, 265, 95, 100, 95, 90, 271, 267, 272, 270, 273,
                  271, 269, 270, 267, 266, 268, 269, 270, 269, 270, 272, 271,
                  271, 270, 273, 271, 270, 274, 275, 274, 270, 268, 97, 95,
                  90, 95, 94, 93, 93, 93, 95, 96, 100, 104, 102, 108, 99, 112,
                  110, 100, 95, 93, 91, 92, 92, 95, 89, 93, 100, 270, 261,
                  275, 276, 275, 277, 276, 273, 273, 271, 275, 277, 275, 276,
                  279, 277, 278, 280, 275, 270, 275, 276, 255, 105, 99, 253,
                  96, 93, 92, 91, 91, 90, 89, 89, 96, 105, 90, 76, 91, 91, 91,
                  90, 95, 90, 92, 92, 95, 100, 135, 98, 92, 90, 99, 175, 220,
                  266, 235, 231, 256, 272, 276, 276, 275, 273, 266, 276, 274,
                  275, 274, 272, 273, 270, 103, 95, 98, 96, 111, 96, 92, 91,
                  90, 90]
    incs['B4'] = [48, 57, 61, 59, 58, 60, 59, 58, 60, 59, 59, 53, 50, 48, 61,
                  40, 56, 67, 52, 49, 60, 47, 50, 48, 50, 53, 52, 58, 60, 60,
                  62, 61, 62, 60, 58, 59, 56, 53, 49, 49, 53, 50, 46, 45, 60,
                  68, 75, 47, 48, 50, 48, 49, 45, 41, 42, 40, 51, 70, 74, 71,
                  51, 75, 73, 60, 49, 44, 41, 51, 45, 50, 41, 44, 68, 67, 73,
                  50, 40, 60, 47, 47, 54, 60, 62, 57, 43, 53, 40, 40, 42, 44,
                  60, 65, 76, 63, 71, 80, 77, 72, 80, 60, 48, 49, 48, 46, 44,
                  43, 40, 58, 65, 39, 48, 38, 43, 42, 49, 39, 43, 44, 48, 61,
                  68, 80, 60, 49, 45, 62, 79, 79, 72, 76, 77, 71, 60, 42, 50,
                  41, 60, 73, 43, 50, 46, 51, 56, 50, 40, 73, 57, 60, 54, 71,
                  54, 50, 48, 48, 51]
    typs['B4'] = Lin
    # Powell, Cole & Cudahy (1985) - Orientations of axial-plane cleavage
    # surfaces of F1 folds in Ordovician turbidites.
    azis['B11'] = [65, 75, 233, 39, 53, 58, 50, 231, 220, 30, 59, 44, 54, 251,
                   233, 52, 26, 40, 266, 67, 61, 72, 54, 32, 238, 84, 230, 228,
                   230, 231, 40, 233, 234, 225, 234, 222, 230, 51, 46, 207,
                   221, 58, 48, 222, 10, 52, 49, 36, 225, 221, 216, 194, 228,
                   27, 226, 58, 35, 37, 235, 38, 227, 34, 225, 53, 57, 66, 45,
                   47, 54, 45, 60, 51, 42, 52, 63]

    incs['B11'] = [50, 53, 85, 82, 82, 66, 75, 85, 87, 85, 82, 88, 86, 82, 83,
                   86, 80, 78, 85, 89, 85, 85, 86, 67, 87, 86, 81, 85, 79, 86,
                   88, 84, 87, 88, 83, 82, 89, 82, 82, 67, 85, 87, 82, 82, 82,
                   75, 68, 89, 81, 87, 63, 86, 81, 81, 89, 62, 81, 88, 70, 80,
                   77, 85, 74, 90, 90, 90, 90, 90, 90, 90, 90, 90, 90, 90, 90]
    typs['B11'] = Fol
    # Powell, Cole & Cudahy (1985) - Orientations of axial-plane cleavage
    # surfaces of F1 folds in Ordovician turbidites.
    azis['B12'] = [122, 132, 141, 145, 128, 133, 130, 129, 124, 120, 137, 141,
                   151, 138, 135, 135, 156, 156, 130, 112, 116, 113, 117, 110,
                   106, 106, 98, 84, 77, 111, 122, 140, 48, 279, 19, 28, 28,
                   310, 310, 331, 326, 332, 3, 324, 308, 304, 304, 299, 293,
                   293, 306, 310, 313, 319, 320, 320, 330, 327, 312, 317, 314,
                   312, 311, 307, 311, 310, 310, 305, 305, 301, 301, 300]

    incs['B12'] = [80, 72, 63, 51, 62, 53, 53, 52, 48, 45, 44, 44, 34, 37, 38,
                   40, 25, 15, 22, 63, 35, 28, 28, 22, 33, 37, 32, 27, 24, 8,
                   6, 8, 11, 8, 6, 6, 8, 20, 21, 18, 25, 28, 32, 32, 32, 34,
                   38, 37, 44, 45, 48, 42, 47, 45, 43, 45, 50, 70, 59, 66, 65,
                   70, 66, 67, 83, 66, 69, 69, 72, 67, 69, 82]
    typs['B12'] = Fol


class FaultSet(list):
    """FaultSet class
    FaultSet is group of ``Pair`` or ``Fault`` objects
    """
    def __init__(self, data, name='Default'):
        assert issubclass(type(data), list), 'Argument must be list of data.'
        assert len(data) > 0, 'Empty FaultSet is not allowed.'
        tp = type(data[0])
        assert issubclass(tp, Pair), 'Data must be Pair or Fault type.'
        assert all([isinstance(e, tp) for e in data]), \
            'All data in FaultSet must be of same type.'
        super(FaultSet, self).__init__(data)
        self.type = tp
        self.name = name

    def __repr__(self):
        return '%s\n%g %s' % (self.name, len(self), self.type.__name__)

    def __add__(self, other):
        # merge sets
        assert isinstance(other, FaultSet), 'Only FaultSets could be merged'
        assert self.type is other.type, 'Only same type could be merged'
        return FaultSet(list(self) + other, name=self.name)

    def __setitem__(self, key, value):
        assert isinstance(value, self.type), \
            'item is not of type %s' % self.type.__name__
        super(FaultSet, self).__setitem__(key, value)

    def __getitem__(self, key):
        """FaultSet fancy indexing"""
        if isinstance(key, slice):
            key = np.arange(*key.indices(len(self)))
        if isinstance(key, list) or isinstance(key, tuple):
            key = np.asarray(key)
        if isinstance(key, np.ndarray):
            if key.dtype == 'bool':
                key = np.flatnonzero(key)
            return FaultSet([self[i] for i in key])
        else:
            return super(FaultSet, self).__getitem__(key)

    def __getattr__(self, attr):
        if attr in self[0].__dict__:
            try:
                res = np.array([getattr(e, attr) for e in self])
            except ValueError:
                res = [getattr(e, attr) for e in self]
            return res
        else:
            raise AttributeError

    def append(self, item):
        assert isinstance(item, self.type), \
            'item is not of type %s' % self.type.__name__
        super(FaultSet, self).append(item)

    def extend(self, items=()):
        for item in items:
            self.append(item)

    @property
    def data(self):
        return list(self)

    def rotate(self, axis, phi):
        """rotate Group"""
        return FaultSet([f.rotate(axis, phi) for f in self], name=self.name)

    @classmethod
    def from_csv(cls, fname, typ=Lin, delimiter=',',
                 facol=1, ficol=2, lacol=3, licol=4):
        """Read FaultSet from csv file"""
        from os.path import basename
        dt = np.loadtxt(fname, dtype=float, delimiter=delimiter).T
        return cls.from_array(dt[facol - 1], dt[ficol - 1],
                              typ=typ, name=basename(fname))

    def to_csv(self, fname, delimiter=','):
        np.savetxt(fname, self.dd.T, fmt='%g', delimiter=',', header=self.name)

    @classmethod
    def from_array(cls, azis, incs, typ=Lin, name='Default'):
        """Create dataset from arrays of dip directions and dips"""
        data = []
        for azi, inc in zip(azis, incs):
            data.append(typ(azi, inc))
        return cls(data, name=name)

    @property
    def lin(self):
        """Return Lin part of pair as Group"""
        return Group([e.lin for e in self], name=self.name)

    @property
    def fol(self):
        """Return Fol part of pair as Group"""
        return Group([e.fol for e in self], name=self.name)

    @property
    def sense(self):
        return np.array([f.sense for f in self])

    @property
    def p(self):
        """Return p-axes of pair as Group"""
        return Group([e.p for e in self], name=self.name)

    @property
    def t(self):
        """Return t-axes of pa6ir as Group"""
        return Group([e.t for e in self], name=self.name)

    @property
    def m(self):
        """Return m-planes of pair as Group"""
        return Group([e.m for e in self], name=self.name)


class Ortensor(object):
    """Ortensor class
    Ortensor is orientation tensor, which characterize data distribution
    using eigenvalue method. See (Watson 1966, Scheidegger 1965).
    """
    def __init__(self, d, **kwargs):
        assert isinstance(d, Group), 'Only group could be passed to Ortensor'
        self.M = np.dot(np.array(d).T, np.array(d))
        self.n = len(d)
        self.name = d.name
        vc, vv = np.linalg.eig(self.M)
        ix = np.argsort(vc)[::-1]
        self.vals = vc[ix]
        self.vects = vv.T[ix]
        self.norm = kwargs.get('norm', True)
        self.scaled = kwargs.get('scaled', False)

    def __repr__(self):
        if self.norm:
            n = self.n
        else:
            n = 1.0
        return 'Ortensor: %s\n' % self.name + \
            '(E1:%.4g,E2:%.4g,E3:%.4g)\n' % tuple(self.vals / n) + \
            str(self.M)

    @property
    def eigenvals(self):
        """Return tuple of eigenvalues. Normalized if norm property is True"""
        if self.norm:
            n = self.n
        else:
            n = 1.0
        return self.vals[0] / n, self.vals[1] / n, self.vals[2] / n

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
        """Return group of eigenvectors. If scaled property is True their
        length is scaled by eigenvalues, otherwise with unit length."""
        if self.scaled:
            e1, e2, e3 = self.eigenvals
        else:
            e1 = e2 = e3 = 1.0
        return Group([e1 * Vec3(self.vects[0]),
                      e2 * Vec3(self.vects[1]),
                      e3 * Vec3(self.vects[2])])

    @property
    def eigenlins(self):
        """Return group of eigenvectors as Lin objects"""
        return self.eigenvects.aslin

    @property
    def eigenfols(self):
        """Return group of eigenvectors as Fol objects"""
        return self.eigenvects.asfol

    @property
    def strength(self):
        """Woodcock strength"""
        return np.log(self.E1 / self.E3)

    @property
    def C(self):
        """Cylindricity index"""
        return self.strength

    @property
    def shape(self):
        """Woodcock shape"""
        return np.log(self.E1 / self.E2) / np.log(self.E2 / self.E3)

    @property
    def P(self):
        """Point index - Vollmer, 1990"""
        return (self.vals[0] - self.vals[1]) / self.n

    @property
    def G(self):
        """Girdle index - Vollmer, 1990"""
        return 2 * (self.vals[1] - self.vals[2]) / self.n

    @property
    def R(self):
        """Random index - Vollmer, 1990"""
        return 3 * self.vals[2] / self.n

    @property
    def B(self):
        """Cylindricity index - Vollmer, 1990"""
        return self.P + self.G

    @property
    def I(self):
        """Intensity index - Lisle, 1985"""
        return 7.5 * np.sum((self.vals / self.n - 1 / 3)**2)


class Cluster(object):
    """Clustering class
    Hierarchical clustering using scipy.cluster routines. distance
    matrix is calculated as angle beetween features, where Fol and
    Lin use axial angles while Vec3 uses direction angles."""
    def __init__(self, d, **kwargs):
        assert isinstance(d, Group), 'Only group could be clustered'
        self.data = Group(d.copy())
        self.maxclust = kwargs.get('maxclust', 2)
        self.distance = kwargs.get('angle', 40)
        self.method = kwargs.get('method', 'average')
        self.criterion = kwargs.get('criterion', 'maxclust')
        self.pdist = self.data.angle()
        self.linkage()

    def __repr__(self):
        if hasattr(self, 'groups'):
            info = 'Already %d clusters created.' % len(self.groups)
        else:
            info = 'Not yet clustered. Use cluster() method.'
        return 'Clustering object\n' + \
            'Number of data: %d\n' % len(self.data) + \
            'Linkage method: %s\n' % self.method + \
            'Criterion: %s\nSettings: maxclust=%d, angle=%.4g\n' \
            % (self.criterion, self.maxclust, self.distance) + info

    def cluster(self, **kwargs):
        """Do clustering on data
        Result is stored as tuple of Groups in groups property.

        Args:
          criterion: The criterion to use in forming flat clusters
          maxclust: number of clusters
          angle: maximum cophenetic distance(angle) in clusters
        """
        from scipy.cluster.hierarchy import fcluster
        self.maxclust = kwargs.get('maxclust', 2)
        self.distance = kwargs.get('angle', 40)
        self.criterion = kwargs.get('criterion', 'maxclust')
        idx = fcluster(self.Z, getattr(self, self.criterion),
                       criterion=self.criterion)
        self.groups = tuple(self.data[np.flatnonzero(idx == c)]
                            for c in np.unique(idx))

    def linkage(self, **kwargs):
        """Do linkage of distance matrix

        Args:
          method: The linkage algorithm to use
        """
        from scipy.cluster.hierarchy import linkage
        self.method = kwargs.get('method', 'average')
        self.Z = linkage(self.pdist, method=self.method,
                         metric=angle_metric)

    def dendrogram(self, **kwargs):
        """Show dendrogram

        See ``scipy.cluster.hierarchy.dendrogram`` for possible options
        """
        from scipy.cluster.hierarchy import dendrogram
        import matplotlib.pyplot as plt
        dendrogram(self.Z, **kwargs)
        plt.show()

    def elbow(self, no_plot=False):
        """Plot within groups variance vs. number of clusters.
        Elbow criterion could be used to determine number of clusters.
        """
        from scipy.cluster.hierarchy import fcluster
        import matplotlib.pyplot as plt
        idx = fcluster(self.Z, len(self.data), criterion='maxclust')
        nclust = list(np.arange(1, np.sqrt(idx.max() / 2) + 1, dtype=int))
        within_grp_var = []
        mean_var = []
        for n in nclust:
            idx = fcluster(self.Z, n, criterion='maxclust')
            grp = [np.flatnonzero(idx == c) for c in np.unique(idx)]
            # between_grp_var = Group([self.data[ix].R.uv for ix in grp]).var
            var = [100 * self.data[ix].var for ix in grp]
            within_grp_var.append(var)
            mean_var.append(np.mean(var))
        if not no_plot:
            plt.boxplot(within_grp_var, positions=nclust)
            plt.plot(nclust, mean_var, 'k')
            plt.xlabel('Number of clusters')
            plt.ylabel('Variance')
            plt.title('Within-groups variance vs. number of clusters')
            plt.show()
        else:
            return nclust, within_grp_var

    @property
    def R(self):
        """Return group of clusters resultants."""
        return Group([group.R for group in self.groups])


# HELPERS #
def G(s, typ=Lin, name='Default'):
    """Create group from space separated string of dip directions and dips"""
    vals = np.fromstring(s, sep=' ')
    return Group.from_array(vals[::2], vals[1::2],
                            typ=typ, name=name)
