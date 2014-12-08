# -*- coding: utf-8 -*-
"""
Python module to manipulate, analyze and visualize structural geology data

"""

from __future__ import division, print_function
from copy import deepcopy

import numpy as np
from .helpers import *

class Vec3(np.ndarray):
    """Base class to store 3D vectors derived from numpy.ndarray
    """
    def __new__(cls, array):
        # casting to our class
        obj = np.asarray(array).view(cls)
        return obj

    def __repr__(self):
        return 'V' + '(%.3f, %.3f, %.3f)' % tuple(self)

    def __mul__(self, other):
        return np.dot(self, other)

    def __abs__(self):
        # abs returns euclidian norm
        return np.sqrt(self * self)

    def __pow__(self, other):
        # cross product or power of magnitude
        if np.isscalar(other):
            return pow(abs(self), other)
        else:
            return self.cross(other)

    def __eq__(self, other):
        # equal
        return bool(abs(self - other) < 1e-15)

    def __ne__(self, other):
        # not equal
        return not self == other

    @property
    def type(self):
        return type(self)

    @property
    def uv(self):
        """Return unit vector

        >>> u = Vec3([1,1,1])
        >>> u.uv
        V(0.577, 0.577, 0.577)
        """
        return self/abs(self)

    def cross(self, other):
        """Returns cross product of two vectors::

        :param vec: vector
        :type name: Vec3
        :returns:  Vec3

        >>> v=Vec3([0,2,-2])
        >>> u.cross(v)
        V(-4.000, 2.000, 2.000)
        """
        return Vec3(np.cross(self, other))

    def angle(self, other):
        """Returns angle of two vectors in degrees::

        :param vec: vector
        :type name: Vec3
        :returns:  Vec3

        >>> u.angle(v)
        90.0
        """
        return acosd(np.dot(self.uv, other.uv))

    def rotate(self, axis, phi):
        """Rotate vector phi degrees about axis::

        :param axis: vector
        :type name: Vec3
        :param phi: angle of rotation
        :returns:  Vec3

        >>> v.rotate(u,60)
        V(-2.000, 2.000, -0.000)
        """
        e = Vec3(self)  # rotate all types as vectors
        k = axis.uv
        r = cosd(phi)*e + sind(phi)*k.cross(e) + (1-cosd(phi))*k*(k*e)
        return r.view(type(self))

    def proj(self, other):
        """Return projection of vector *u* onto vector *v*::

        :param other: vector
        :type name: Vec3
        :returns:  Vec3

        >>> u.proj(v)
        """
        r = np.dot(self, other)*other / abs(other)**2
        return r.view(type(self))

    def transform(self, F):
        """Return affine transformation of vector *u* by matrix *F*::

        :param F: matric
        :type name: numpy.array
        :returns:  Vec3

        >>> u.transform(F)
        """
        return np.dot(F, self).view(type(self))

    @property
    def aslin(self):
        """Convert vector to Lin object.

        >>> u = Vec3([1,1,1])
        >>> u.aslin
        L:45/35
        """
        res = Lin(0, 0)
        np.copyto(res, self)
        return res

    @property
    def asfol(self):
        """Convert vector to Fol object.
        
        >>> u = Vec3([1,1,1])
        >>> u.asfol
        S:225/55
        """
        res = Fol(0, 0)
        np.copyto(res, self)
        return res


class Lin(Vec3):
    """Class for linear features
    """
    def __new__(cls, azi, inc):
        # casting to our class
        v = [cosd(azi)*cosd(inc), sind(azi)*cosd(inc), sind(inc)]
        return Vec3(v).view(cls)

    def __repr__(self):
        azi, inc = self.dd
        return 'L:%d/%d' % (round(azi), round(inc))

    def __add__(self, other):
        # add axial data
        if self * other < 0:
            other = -other
        return super(Lin, self).__add__(other)

    def __iadd__(self, other):
        # add axial data
        if self * other < 0:
            other = -other
        return super(Lin, self).__iadd__(other)

    def __sub__(self, other):
        # substract axial data
        if self * other < 0:
            other = -other
        return super(Lin, self).__sub__(other)

    def __isub__(self, other):
        # substract axial data
        if self * other < 0:
            other = -other
        return super(Lin, self).__isub__(other)

    def __pow__(self, other):
        # cross product or power of magnitude
        if np.isscalar(other):
            return pow(abs(self), other)
        else:
            return super(Lin, self).cross(other).asfol

    def __eq__(self, other):
        # equal
        return bool(abs(self-other) < 1e-15 or abs(self+other) < 1e-15)

    def __ne__(self, other):
        # not equal
        return not (self == other or self == -other)

    def angle(self, lin):
        """Returns angle of two lineations in degrees

        :param lin: lineation
        :type name: Lin
        :returns:  angle

        >>> u.angle(v)
        90.0
        """
        return acosd(abs(np.dot(self.uv, lin.uv)))

    def cross(self, other):
        """Returns foliaton defined by two lineations

        :param other: vector
        :type name: Vec3, Fol, Lin
        :returns:  Fol

        >>> l=Lin(120,10)
        >>> l.cross(Lin(160,30))
        S:196/35
        """
        return np.cross(self, other).view(Fol)

    @property
    def dd(self):
        n = self.uv
        if n[2] < 0:
            n = -n
        azi = atan2d(n[1], n[0]) % 360
        inc = asind(n[2])
        return azi, inc


class Fol(Vec3):
    """Class for planar features
    """
    def __new__(cls, azi, inc):
        # casting to our class
        v = [-cosd(azi)*sind(inc), -sind(azi)*sind(inc), cosd(inc)]
        return Vec3(v).view(cls)

    def __repr__(self):
        azi, inc = self.dd
        return 'S:%d/%d' % (round(azi), round(inc))

    def __add__(self, other):
        # add axial data
        if self * other < 0:
            other = -other
        return super(Fol, self).__add__(other)

    def __iadd__(self, other):
        # add axial data
        if self * other < 0:
            other = -other
        return super(Fol, self).__iadd__(other)

    def __sub__(self, other):
        # substract axial data
        if self * other < 0:
            other = -other
        return super(Fol, self).__sub__(other)

    def __isub__(self, other):
        # substract axial data
        if self * other < 0:
            other = -other
        return super(Fol, self).__isub__(other)

    def __eq__(self, other):
        # equal
        return bool(abs(self-other) < 1e-15 or abs(self+other) < 1e-15)

    def __ne__(self, other):
        # not equal
        return not (self == other or self == -other)

    def angle(self, fol):
        """Returns angle of two foliations in degrees

        :param lin: foliation
        :type name: Fol
        :returns:  angle

        >>> u.angle(v)
        90.0
        """
        return acosd(abs(np.dot(self.uv, fol.uv)))

    def cross(self, other):
        """Returns lination defined as intersecton of two foliations

        :param other: vector
        :type name: Vec3, Fol, Lin
        :returns:  Lin

        >>> f=Fol(60,30)
        >>> f.cross(Fol(120,40))
        L:72/29
        """
        return np.cross(self, other).view(Lin)

    def transform(self, F):
        """Return affine transformation of foliation by matrix *F*::

        :param F: matric
        :type name: numpy.array
        :returns:  Fol

        >>> f.transform(F)
        """
        return np.dot(self, np.linalg.inv(F)).view(type(self))

    @property
    def dd(self):
        n = self.uv
        if n[2] < 0:
            n = -n
        azi = (atan2d(n[1], n[0]) + 180) % 360
        inc = 90 - asind(n[2])
        return azi, inc

    @property
    def rhr(self):
        azi, inc = self.dd
        return (azi - 90) % 360, inc


class Pair(object):
    """Pair class store related Fol and Lin instances"""
    def __init__(self, fol, lin):
        assert isinstance(fol, Fol), 'First fol parameter must be instance of Fol'
        assert isinstance(lin, Lin), 'Second lin parameter must be instance of Lin'
        self.fol = fol
        self.lin = lin

    def __repr__(self):
        fazi, finc = self.fol.dd
        lazi, linc = self.lin.dd
        return 'P:%d/%d-%d/%d' % (round(fazi), round(finc), round(lazi), round(linc))

    @property
    def misfit(self):
        return 90 - self.fol.angle(self.lin)

    def fit(self):
        """Correct pair of planar and linear data.
        Both planar and linear feature is rotated, so linear feature perfectly
        fit onto planar one::

            p.fit()
        """
        ax = self.fol ** self.lin
        ang = (Vec3(self.lin).angle(self.fol) - 90)/2
        self.fol = self.fol.rotate(ax, ang)
        self.lin = self.lin.rotate(ax, -ang)


class Fault(Pair):
    """Fault class store related Fol and Lin instances with sense of movement"""
    def __init__(self, fol, lin, sense):
        assert isinstance(sense, int), 'Third sense parameter must be positive or negative integer'
        super(Fault, self).__init__(fol, lin)
        misfit = 90 - self.fol.angle(self.lin) 
        assert misfit < 15, 'Lin deviate too much from Fol'
        self.sense = 2*int(sense>0) - 1

    def __repr__(self):
        fazi, finc = self.fol.dd
        lazi, linc = self.lin.dd
        return 'F:%d/%d-%d/%d %s' % (round(fazi), round(finc), round(lazi), round(linc), ['-','','+'][self.sense + 1])


class Group(list):
    """Group class
    Group is homogeneous group of Vec3, Fol or Lin
    """
    def __init__(self, data, name='Default'):
        assert issubclass(type(data), list), 'Argument must be list of data.'
        assert len(data) > 0, 'Empty group is not allowed.'
        tp = type(data[0])
        assert issubclass(tp, Vec3), \
               'Data must be Fol, Lin or Vec3 type.'
        assert all([isinstance(e, tp) for e in data]), \
               'All data in group must be of same type.'
        super(Group, self).__init__(data)
        self.type = tp
        self.name = name

    def __repr__(self):
        return '%s: %g %s' % (self.name, len(self), self.type.__name__)

    def __abs__(self):
        # abs returns array of euclidian norms
        return np.asarray([abs(e) for e in self])

    def __add__(self, other):
        # merge Datasets
        assert isinstance(other, Group), 'Only groups could be merged'
        assert self.type is other.type, 'Only same type groups could be merged'
        return Group(list(self) + other, name=self.name)

    def __setitem__(self, key, value):
        assert isinstance(value, self.type), 'item is not of type %s' % self.type.__name__
        super(Group, self).__setitem__(key, value)

    def append(self, item):
        assert isinstance(item, self.type), 'item is not of type %s' % self.type.__name__
        super(Group, self).append(item)

    def extend(self, items=()):
        for item in items:
            self.append(item)

    @property
    def data(self):
        return list(self)

    @classmethod
    def fromcsv(cls, fname, typ=Lin, delimiter=',', acol=1, icol=2):
        """Read group from csv file"""
        from os.path import basename
        dt = np.loadtxt(fname, dtype=float, delimiter=delimiter).T
        return cls.fromarray(dt[acol-1], dt[icol-1], typ=typ, name=basename(fname))

    def tocsv(self, fname, delimiter=','):
        np.savetxt(fname, self.dd.T, fmt='%g', delimiter=',', header=self.name)

    @classmethod
    def fromarray(cls, azis, incs, typ=Lin, name='Default'):
        """Create dataset from arrays of dip directions and dips"""
        data = []
        for azi, inc in zip(azis, incs):
            data.append(typ(azi, inc))
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
    def resultant(self):
        """calculate resultant vector of Group"""
        r = deepcopy(self[0])
        for v in self[1:]:
            r += v
        return r

    @property
    def rdegree(self):
        """degree of preffered orientation od Group"""
        r = self.resultant
        n = len(self)
        return 100*(2*abs(r) - n)/n

    def cross(self, other=None):
        """return cross products of all pairs in Group"""
        res = []
        if other is None:
            for i in range(len(self)-1):
                for j in range(i+1, len(self)):
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

    def center(self):
        """rotate E3 direction of Group to vertical"""
        ot = self.ortensor
        azi, inc = ot.eigenlins[2][0].dd
        return self.rotate(Lin(azi - 90, 0), 90 - inc)

    def normalized(self):
        """return Group with normalized elements"""
        return Group([e/abs(e) for e in self], name=self.name)

    def angle(self, other=None):
        """return angles of all pairs in Group"""
        res = []
        if other is None:
            for i in range(len(self)-1):
                for j in range(i+1, len(self)):
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
        for azi, dip in zip(180*np.random.rand(N), sig*np.random.randn(N)):
            data.append(Lin(0, 90).rotate(Lin(azi, 0), dip))
        return cls(data).rotate(Lin(ta+90, 0), 90-td)

    @classmethod
    def randn_fol(cls, N=100, mean=Fol(0, 0), sig=20):
        data = []
        ta, td = mean.dd
        for azi, dip in zip(180*np.random.rand(N), sig*np.random.randn(N)):
            data.append(Fol(0, 0).rotate(Lin(azi, 0), dip))
        return cls(d).rotate(Lin(ta-90, 0), td)

    @classmethod
    def uniform_lin(cls, N=500):
        n = 2*np.ceil(np.sqrt(N)/0.564)
        azi = 0
        inc = 90
        for rho in np.linspace(0, 1, np.round(n/2/np.pi))[:-1]:
            theta = np.linspace(0, 2*np.pi, np.round(n*rho + 1))[:-1]
            x, y = rho*np.sin(theta), rho*np.cos(theta)
            azi = np.hstack((azi, np.rad2deg(np.arctan2(x, y))))
            inc = np.hstack((inc, 90 - 2*np.rad2deg(np.arcsin(np.sqrt((x*x + y*y)/2)))))
        # no antipodal
        theta = np.linspace(0, 2*np.pi, n + 1)[:-1]
        x, y = np.sin(theta), np.cos(theta)
        azi = np.hstack((azi, np.rad2deg(np.arctan2(x, y))[::2]))
        inc = np.hstack((inc, 90 - 2*np.rad2deg(np.arcsin(np.sqrt((x*x + y*y)/2)))[::2]))
        # fix
        inc[inc<0] = 0
        return cls.fromarray(azi, inc, typ=Lin)

    @classmethod
    def uniform_fol(cls, N=500):
        l = cls.uniform_lin(N=N)
        azi, inc = l.dd
        return cls.fromarray(azi+180, 90-inc, typ=Fol)


class Ortensor(object):
    """Ortensor class"""
    def __init__(self, d):
        self.M = np.dot(np.array(d).T, np.array(d))
        self.n = len(d)
        vc, vv = np.linalg.eig(self.M)
        ix = np.argsort(vc)[::-1]
        self.vals = vc[ix]
        self.vects = vv.T[ix]
        e1, e2, e3 = self.vals / self.n
        self.shape = np.log(e3 / e2) / np.log(e2 / e1)
        self.strength = np.log(e3 / e1)
        self.norm = True
        self.scaled = False

    def __repr__(self):
        return 'Ortensor:\n(E1:%.4g,E2:%.4g,E3:%.4g)' % tuple(self.vals) + \
            '\n' + str(self.M)

    @property
    def eigenvals(self):
        if self.norm:
            n = self.n
        else:
            n = 1.0
        return self.vals[0] / n, self.vals[1] / n, self.vals[2] / n

    @property
    def eigenvects(self):
        if self.scaled:
            e1, e2, e3 = self.eigenvals
        else:
            e1 = e2 = e3 = 1.0
        return Group([e1 * Vec3(self.vects[0]),
                      e2 * Vec3(self.vects[1]),
                      e3 * Vec3(self.vects[2])])

    @property
    def eigenlins(self):
        v1, v2, v3 = self.eigenvects
        return Group([v1.aslin, v2.aslin, v3.aslin])

    @property
    def eigenfols(self):
        v1, v2, v3 = self.eigenvects
        return Group([v1.asfol, v2.asfol, v3.asfol])


