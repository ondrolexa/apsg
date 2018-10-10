# -*- coding: utf-8 -*-


"""
Module to manipulate, analyze and visualize structural geology data.
"""


from __future__ import division, print_function
from copy import deepcopy
import warnings
import pickle

import numpy as np
import matplotlib.pyplot as plt

from .helpers import (
    KentDistribution,
    sind,
    cosd,
    acosd,
    asind,
    atand,
    atan2d,
    angle_metric,
    l2v,
    getldd,
    _linear_inverse_kamb,
    _square_inverse_kamb,
    _schmidt_count,
    _kamb_count,
    _exponential_kamb,
)


__all__ = (
    "Vec3",
    "Lin",
    "Fol",
    "Pair",
    "Fault",
    "Group",
    "PairSet",
    "FaultSet",
    "Ortensor",
    "Cluster",
    "StereoGrid",
    "G",
    "settings",
)


# Default module settings (singleton).

settings = dict(notation="dd", vec2dd=False)


class Vec3(np.ndarray):
    """
    ``Vec3`` is base class to store 3-dimensional vectors derived from
    ``numpy.ndarray`` on which ``Lin`` and ``Fol`` classes are based.

    ``Vec3`` support most of common vector algebra using following operators:
        - ``+`` - vector addition
        - ``-`` - vector subtraction
        - ``*`` - dot product
        - ``**`` - cross product
        - ``abs`` - magnitude (length) of vector

      See following methods and properties for additional operations.

    Args:
        arr (array_like):
            Input data that or can be converted to an array.
            This includes lists, tuples, and ndarrays. When more than one
            argument is passed (i.e. `inc` is not `None`) `arr` is interpreted
            as dip direction of the vector in degrees.
        inc (float):
            `None` or dip of the vector in degrees.
        mag (float):
            The magnitude of the vector if `inc` is not `None`.

    Returns:
      ``Vec3`` object

    Example:
        >>> v = Vec3([1, 0.2, 1.6])
        # The dip direction and dip angle of vector with magnitude of 1.
        >>> v = Vec3(120, 60)
        # The dip direction and dip angle of vector with magnitude of 3.
        >>> v = Vec3(120, 60, 3)
    """

    def __new__(cls, arr, inc=None, mag=1.0):
        if inc is None:
            obj = np.asarray(arr).view(cls)
        else:
            obj = mag * Lin(arr, inc).view(cls)
        return obj

    def __repr__(self):
        if settings["vec2dd"]:
            result = "V:{:.0f}/{:.0f}".format(*self.dd)
        else:
            result = "V({:.3f}, {:.3f}, {:.3f})".format(*self)
        return result

    def __str__(self):
        return repr(self)

    def __mul__(self, other):
        """
        Return the dot product of two vectors.
        """
        return np.dot(self, other)  # What about `numpy.inner`?

    def __abs__(self):
        """
        Return the 2-norm or Euclidean norm of vector.
        """

        return np.linalg.norm(self)

    def __pow__(self, other):
        """
        Return cross product if argument is vector or power of vector.
        """
        if np.isscalar(other):
            return pow(abs(self), other)
        else:
            return self.cross(other)

    def __eq__(self, other):
        """
        Return `True` if vectors are equal, otherwise `False`.
        """
        if not isinstance(other, self.__class__):
            return False
        return self is other or abs(self - other) < 1e-15

    def __ne__(self, other):
        """
        Return `True` if vectors are not equal, otherwise `False`.

        Overrides the default implementation (unnecessary in Python 3).
        """
        return not self == other

    def __hash__(self):
        return NotImplementedError

    @property
    def type(self):
        """
        Return the type of ``self``.
        """
        return type(self)

    @property
    def upper(self):
        """
        Return `True` if z-coordinate is negative, otherwise `False`.
        """
        return np.sign(self[2]) < 0

    @property
    def flip(self):
        """
        Return a new vector with inverted `z` coordinate.
        """
        return Vec3((self[0], self[1], -self[2]))

    @property
    def uv(self):
        """
        Normalize the vector to unit length.

        Returns:
          unit vector of ``self``

        Example:
          >>> u = Vec3([1,1,1])
          >>> u.uv
          V(0.577, 0.577, 0.577)

        """

        return self / abs(self)

    def cross(self, other):
        """
        Calculate the cross product of two vectors.

        Args:
          other: other ``Vec3`` vector

        Returns:
          The cross product of `self` and `other`

        Example:
          >>> v = Vec3([0, 2, -2])
          >>> u.cross(v)
          V(-4.000, 2.000, 2.000)

        """

        return Vec3(np.cross(self, other))

    def angle(self, other):
        """
        Calculate the angle between two vectors in degrees.

        Args:
            other: other ``Vec3`` vector

        Returns:
            angle of `self` and `other` in degrees

        Example:
            >>> u.angle(v)
            90.0
        """

        if isinstance(other, Group):
            return other.angle(self)
        else:
            return acosd(np.clip(np.dot(self.uv, other.uv), -1, 1))

    def rotate(self, axis, angle):
        """
        Return rotated vector about axis.

        Args:
            axis (``Vec3``): axis of rotation
            angle (float): angle of rotation in degrees

        Returns:
            vector represenatation of `self` rotated `angle` degrees about
            vector `axis`. Rotation is clockwise along axis direction.

        Example:
            >>> v.rotate(u, 60)
            V(-2.000, 2.000, -0.000)

        """

        e = Vec3(self)  # rotate all types as vectors
        k = axis.uv
        r = cosd(angle) * e + sind(angle) * k.cross(e) + (1 - cosd(angle)) * k * (k * e)

        return r.view(type(self))

    def proj(self, other):
        """
        Return projection of vector `u` onto vector `v`.

        Args:
            other (``Vec3``): other vector

        Returns:
            vector representation of `self` projected onto 'other'

        Example:
            >> u.proj(v)

        Note:
            To project on plane use: `u - u.proj(v)`, where `v` is plane normal.

        """

        r = np.dot(self, other) * other / np.linalg.norm(other)

        return r.view(type(self))

    def H(self, other):
        """
        Return ``DefGrad`` rotational matrix H which rotate vector
        `u` to vector `v`.

        Args:
            other (``Vec3``): other vector

        Returns:
            ``Defgrad`` rotational matrix

        Example:
            >>> u.transform(u.H(v)) == v
            True

        """
        from .tensors import DefGrad

        return DefGrad(
            np.outer(self + other, (self + other).T) / (1 + self * other) - np.eye(3)
        )

    def transform(self, F, **kwargs):
        """
        Return affine transformation of vector `u` by matrix `F`.

        Args:
            F (``DefGrad`` or ``numpy.array``): transformation matrix

        Keyword Args:
            norm: normalize transformed vectors. [True or False] Default False

        Returns:
            vector representation of affine transformation (dot product)
            of `self` by `F`

        Example:
            >>> u.transform(F)

        """
        if kwargs.get("norm", False):
            res = np.dot(F, self).view(type(self)).uv
        else:
            res = np.dot(F, self).view(type(self))
        return res

    @property
    def dd(self):
        """
        Return azimuth, inclination tuple.

        Example:
          >>> azi, inc = v.dd
        """

        n = self.uv
        azi = atan2d(n[1], n[0]) % 360
        inc = asind(n[2])

        return azi, inc

    @property
    def aslin(self):
        """
        Convert `self` to ``Lin`` object.

        Example:
            >>> u = Vec3([1,1,1])
            >>> u.aslin
            L:45/35
        """
        return self.copy().view(Lin)

    @property
    def asfol(self):
        """
        Convert `self` to ``Fol`` object.

        Example:
            >>> u = Vec3([1,1,1])
            >>> u.asfol
            S:225/55
        """
        return self.copy().view(Fol)

    @property
    def asvec3(self):
        """
        Convert `self` to ``Vec3`` object.

        Example:
            >>> l = Lin(120,50)
            >>> l.asvec3
            V(-0.321, 0.557, 0.766)
        """
        return self.copy().view(Vec3)

    @property
    def V(self):
        """
        Convert `self` to ``Vec3`` object.

        Note:
            This is an alias of ``asvec3`` property.
        """
        return self.copy().view(Vec3)


class Lin(Vec3):
    """
    Represents a linear feature.

    It provides all ``Vec3`` methods and properties but behave as axial vector.

    Args:
        azi: The dip direction in degrees.
        inc: The dip angle in degrees.

    Example:
        >>> l = Lin(120, 60)
        L:120/60

    """

    def __new__(cls, azi, inc):
        v = [cosd(azi) * cosd(inc), sind(azi) * cosd(inc), sind(inc)]

        return Vec3(v).view(cls)

    def __repr__(self):
        return "L:{:.0f}/{:.0f}".format(*self.dd)

    def __add__(self, other):
        """
        Sum a `self` with `other`.
        """
        if self * other < 0:
            other = -other
        return super(Lin, self).__add__(other)

    def __iadd__(self, other):
        if self * other < 0:
            other = -other
        return super(Lin, self).__iadd__(other)

    def __sub__(self, other):
        """
        Subtract a `self` with `other`.
        """
        if self * other < 0:
            other = -other

        return super(Lin, self).__sub__(other)

    def __isub__(self, other):
        if self * other < 0:
            other = -other

        return super(Lin, self).__isub__(other)

    def __eq__(self, other):
        """
        Return `True` if linear features are equal.
        """
        return bool(abs(self - other) < 1e-15 or abs(self + other) < 1e-15)

    def __ne__(self, other):
        """
        Return `True` if linear features are not equal.
        """
        return not (self == other or self == -other)

    def dot(self, other):
        """
        Calculate the axial dot product.
        """
        return abs(np.dot(self, other))

    def cross(self, other):
        """
        Create planar feature defined by two linear features.

        Example:
            >>> l=Lin(120,10)
            >>> l.cross(Lin(160,30))
            S:196/35
        """
        return (
            other.cross(self)
            if isinstance(other, Group)
            else np.cross(self, other).view(Fol)
        )

    def angle(self, other):
        """
        Return an angle (<90) between two linear features in degrees.

        Example:
          >>> u.angle(v)
          90.0
        """
        return (
            other.angle(self)
            if isinstance(other, Group)
            else acosd(np.clip(self.uv.dot(other.uv), -1, 1))
        )

    @property
    def dd(self):
        """
        Return dip direction and dip angle tuple.
        """
        n = self.uv
        if n[2] < 0:
            n = -n
        azi = atan2d(n[1], n[0]) % 360
        inc = asind(n[2])

        return azi, inc


class Fol(Vec3):
    """
    Represents a planar feature.

    It provides all ``Vec3`` methods and properties but plane normal behave
    as axial vector.

    Args:
      azi: The dip direction in degrees.
      inc: The dip angle in degrees.

    Example:
      >>> f = Fol(120, 60)
      F:120/60

    """

    def __new__(cls, azi, inc):
        """
        Create a planar feature.
        """

        if settings["notation"] == "rhr":
            azi += 90
        v = [-cosd(azi) * sind(inc), -sind(azi) * sind(inc), cosd(inc)]

        return Vec3(v).view(cls)

    def __repr__(self):
        return "S:{:.0f}/{:.0f}".format(*getattr(self, settings["notation"]))

    def __add__(self, other):
        """
        Sum of axial data.
        """

        if self * other < 0:
            other = -other

        return super(Fol, self).__add__(other)

    def __iadd__(self, other):
        if self * other < 0:
            other = -other

        return super(Fol, self).__iadd__(other)

    def __sub__(self, other):
        """
        Subtract the axial data.
        """

        if self * other < 0:
            other = -other

        return super(Fol, self).__sub__(other)

    def __isub__(self, other):
        if self * other < 0:
            other = -other

        return super(Fol, self).__isub__(other)

    def __eq__(self, other):
        """
        Return `True` if planar features are equal, otherwise `False`.
        """

        return bool(abs(self - other) < 1e-15 or abs(self + other) < 1e-15)

    def __ne__(self, other):
        """
        Return `False` if planar features are equal, otherwise `True`.
        """

        return not (self == other or self == -other)

    def angle(self, other):
        """
        Return angle of two planar features in degrees.

        Example:
            >>> u.angle(v)
            90.0

        """
        if isinstance(other, Group):
            return other.angle(self)
        else:
            return acosd(np.clip(self.uv.dot(other.uv), -1, 1))

    def cross(self, other):
        """
        Return linear feature defined as intersection of two planar features.

        Example:
            >>> f=Fol(60,30)
            >>> f.cross(Fol(120,40))
            L:72/29

        """
        if isinstance(other, Group):
            return other.cross(self)
        else:
            return np.cross(self, other).view(Lin)

    def dot(self, other):
        """
        Axial dot product.
        """

        return abs(np.dot(self, other))

    def transform(self, F, **kwargs):
        """
        Return affine transformation of planar feature by matrix `F`.

        Args:
          F (``DefGrad`` or ``numpy.array``): transformation matrix

        Keyword Args:
          norm: normalize transformed vectors. True or False. Dafault False

        Returns:
          representation of affine transformation (dot product) of `self`
          by `F`

        Example:
          >>> f.transform(F)

        """
        if kwargs.get("norm", False):
            res = np.dot(self, np.linalg.inv(F)).view(type(self)).uv
        else:
            res = np.dot(self, np.linalg.inv(F)).view(type(self))
        return res

    @property
    def dd(self):
        """
        Return dip-direction, dip tuple.
        """

        n = self.uv
        if n[2] < 0:
            n = -n
        azi = (atan2d(n[1], n[0]) + 180) % 360
        inc = 90 - asind(n[2])

        return azi, inc

    @property
    def rhr(self):
        """
        Return strike and dip tuple (right-hand-rule).
        """
        azi, inc = self.dd

        return (azi - 90) % 360, inc

    @property
    def dv(self):
        """
        Return a dip ``Vec3`` object.

        Example:
            >>> f = Fol(120,50)
            >>> f.dv
            V(-0.321, 0.557, 0.766)

        """
        azi, inc = self.dd

        return Lin(azi, inc).view(Vec3)

    def rake(self, rake):
        """
        Return a ``Vec3`` object with given rake.

        Example:
            >>> f = Fol(120,50)
            >>> f.rake(30)
            V(0.589, 0.711, 0.383)
            >>> f.rake(30).aslin
            L:50/23

        """

        return self.dv.rotate(self, rake - 90)


class Pair(object):
    """
    The class to store pair of planar and linear feature.

    When ``Pair`` object is created, both planar and linear feature are
    adjusted, so linear feature perfectly fit onto planar one. Warning
    is issued, when misfit angle is bigger than 20 degrees.

    Args:
        fazi (float): Dip direction of planar feature in degrees
        finc (float): dip of planar feature in degrees
        lazi (float): Dip direction of linear feature in degrees
        linc (float): dip of linear feature in degrees

    Example:
        >>> p = Pair(140, 30, 110, 26)

    """

    def __init__(self, fazi, finc, lazi, linc):
        fol = Fol(fazi, finc)
        lin = Lin(lazi, linc)
        misfit = 90 - fol.angle(lin)
        if misfit > 20:
            warnings.warn("Warning: Misfit angle is %.1f degrees." % misfit)
        ax = fol ** lin
        ang = (Vec3(lin).angle(fol) - 90) / 2
        fol = fol.rotate(ax, ang)
        lin = lin.rotate(ax, -ang)
        self.fvec = Vec3(fol)
        self.lvec = Vec3(lin)
        self.misfit = misfit

    def __repr__(self):
        vals = getattr(self.fol, settings["notation"]) + self.lin.dd
        return "P:{:.0f}/{:.0f}-{:.0f}/{:.0f}".format(*vals)

    @classmethod
    def from_pair(cls, fol, lin):
        """
        Create ``Pair`` from ``Fol`` and ``Lin`` objects.

        Example:
            >>> f = Fol(140, 30)
            >>> l = Lin(110, 26)
            >>> p = Pair.from_pair(f, l)
        """
        data = getattr(fol, settings["notation"]) + lin.dd
        return cls(*data)

    def rotate(self, axis, phi):
        """Rotates ``Pair`` by angle `phi` about `axis`.

        Args:
            axis (``Vec3``): axis of rotation
            phi (float): angle of rotation in degrees

        Example:
            >>> p = Pair(140, 30, 110, 26)
            >>> p.rotate(Lin(40, 50), 120)
            P:210/83-287/60

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
        """
        Return a planar feature of ``Pair`` as ``Fol``.
        """
        return self.fvec.asfol

    @property
    def lin(self):
        """
        Return a linear feature of ``Pair`` as ``Lin``.
        """
        return self.lvec.aslin

    @property
    def rax(self):
        """
        Return an oriented vector perpendicular to both ``Fol`` and ``Lin``.
        """
        return self.fvec ** self.lvec

    def transform(self, F, **kwargs):
        """Return an affine transformation of ``Pair`` by matrix `F`.

        Args:
            F (``DefGrad`` or ``numpy.array``): transformation matrix

        Keyword Args:
            norm: normalize transformed vectors. True or False. Dafault False

        Returns:
            representation of affine transformation (dot product) of `self`
            by `F`

        Example:
            >>> p.transform(F)

        """
        t = deepcopy(self)
        if kwargs.get("norm", False):
            t.lvec = np.dot(F, t.lvec).view(Vec3).uv
            t.fvec = np.dot(t.fvec, np.linalg.inv(F)).view(Vec3).uv
        else:
            t.lvec = np.dot(F, t.lvec).view(Vec3)
            t.fvec = np.dot(t.fvec, np.linalg.inv(F)).view(Vec3)
        return t


class Fault(Pair):

    """Fault class for related ``Fol`` and ``Lin`` instances with sense
    of movement.

    When ``Fault`` object is created, both planar and linear feature are
    adjusted, so linear feature perfectly fit onto planar one. Warning
    is issued, when misfit angle is bigger than 20 degrees.

    Args:
        fazi (float): dip direction of planar feature in degrees
        finc (float): dip of planar feature in degrees
        lazi (float): dip direction of linear feature in degrees
        linc (float): dip of linear feature in degrees
        sense (float): sense of movement +/-1 hanging-wall up/down

    Example:
        >>> p = Fault(140, 30, 110, 26, -1)

    """

    def __init__(self, fazi, finc, lazi, linc, sense):
        assert np.sign(sense) != 0, "Sense parameter must be positive or negative"
        super(Fault, self).__init__(fazi, finc, lazi, linc)
        self.lvec = np.sign(sense) * self.lvec

    def __repr__(self):
        s = ["", "+", "-"][self.sense]
        vals = getattr(self.fol, settings["notation"]) + self.lin.dd + (s,)
        return "F:{:.0f}/{:.0f}-{:.0f}/{:.0f} {:s}".format(*vals)

    @classmethod
    def from_pair(cls, fol, lin, sense):
        """Create ``Fault`` with given sense from ``Fol`` and ``Lin`` objects"""
        data = getattr(fol, settings["notation"]) + lin.dd + (sense,)
        return cls(*data)

    @classmethod
    def from_vecs(cls, fvec, lvec):
        """Create ``Fault`` from two ortogonal ``Vec3`` objects

        Args:
          fvec: vector normal to fault plane
          lvec: vector parallel to movement

        """
        orax = fvec ** lvec
        rax = Vec3(*fvec.aslin.dd) ** Vec3(*lvec.dd)
        sense = 1 - 2 * (orax == rax)
        data = getattr(fvec.asfol, settings["notation"]) + lvec.dd + (sense,)
        return cls(*data)

    def rotate(self, axis, phi):
        """Rotates ``Fault`` by `phi` degrees about `axis`.

        Args:
          axis: axis of rotation
          phi: angle of rotation in degrees

        Example:
          >>> f = Fault(140, 30, 110, 26, -1)
          >>> f.rotate(Lin(220, 10), 60)
          F:300/31-301/31 +

        """
        rot = deepcopy(self)
        rot.fvec = self.fvec.rotate(axis, phi)
        rot.lvec = self.lvec.rotate(axis, phi)
        return rot

    @property
    def sense(self):
        """Return sense of movement (+/-1)"""
        # return 2 * int(self.fvec**self.lvec == Vec3(self.fol**self.lin)) - 1
        orax = self.fvec.uv ** self.lvec.uv
        rax = Vec3(*self.fol.aslin.dd) ** Vec3(*self.lin.dd)
        return 2 * (orax == rax) - 1

    @property
    def pvec(self):
        """Return P axis as ``Vec3``"""
        return self.fvec.rotate(self.rax, -45)

    @property
    def tvec(self):
        """Return T-axis as ``Vec3``."""
        return self.fvec.rotate(self.rax, 45)

    @property
    def p(self):
        """Return P-axis as ``Lin``"""
        return self.pvec.aslin

    @property
    def t(self):
        """Return T-axis as ``Lin``"""
        return self.tvec.aslin

    @property
    def m(self):
        """Return kinematic M-plane as ``Fol``"""
        return (self.fvec ** self.lvec).asfol

    @property
    def d(self):
        """Return dihedra plane as ``Fol``"""
        return (self.rax ** self.fvec).asfol


class Group(list):
    """
    Represents a homogeneous group of ``Vec3``, ``Fol`` or ``Lin`` objects.

    ``Group`` provide append and extend methods as well as list indexing
    to get or set individual items. It also supports following operators:
        - ``+`` - merge groups
        - ``**`` - mutual cross product
        - ``abs`` - array of magnitudes (lengths) of all objects

    See following methods and properties for additional operations.

    Args:
        data (list): list of ``Vec3``, ``Fol`` or ``Lin`` objects
        name (str): Name of group

    Returns:
        ``Group`` object

    Example:
        >>> g = Group([Lin(120, 20), Lin(151, 23), Lin(137, 28)])
    """

    def __init__(self, data, name="Default"):
        assert issubclass(type(data), list), "Argument must be list of data."
        assert len(data) > 0, "Empty group is not allowed."
        tp = type(data[0])
        assert issubclass(tp, Vec3), "Data must be Fol, Lin or Vec3 type."
        assert all(
            [isinstance(e, tp) for e in data]
        ), "All data in group must be of same type."
        super(Group, self).__init__(data)
        self.type = tp
        self.name = name

    def __repr__(self):
        return "G:%g %s (%s)" % (len(self), self.type.__name__, self.name)

    def __abs__(self):
        # abs returns array of euclidean norms
        return np.asarray([abs(e) for e in self])

    def __add__(self, other):
        # merge Datasets
        assert isinstance(other, Group), "Only groups could be merged"
        assert self.type is other.type, "Only same type groups could be merged"
        return Group(list(self) + other, name=self.name)

    def __pow__(self, other):
        """Return all mutual cross products of two ``Group`` objects

        """
        if np.isscalar(other):
            return pow(abs(self), other)
        else:
            return self.cross(other)

    def __setitem__(self, key, value):
        assert isinstance(value, self.type), (
            "item is not of type %s" % self.type.__name__
        )
        super(Group, self).__setitem__(key, value)

    def __getitem__(self, key):
        """Group fancy indexing"""
        if isinstance(key, slice):
            key = np.arange(*key.indices(len(self)))
        if isinstance(key, list) or isinstance(key, tuple):
            key = np.asarray(key)
        if isinstance(key, np.ndarray):
            if key.dtype == "bool":
                key = np.flatnonzero(key)
            return Group([self[i] for i in key])
        else:
            return super(Group, self).__getitem__(key)

    def append(self, item):
        assert isinstance(item, self.type), (
            "item is not of type %s" % self.type.__name__
        )
        super(Group, self).append(item)

    def extend(self, items=()):
        for item in items:
            self.append(item)

    def copy(self):
        return Group(deepcopy(self.data), name=self.name)

    @property
    def upper(self):
        """
        Return boolean array of z-coordinate negative test
        """

        return np.asarray([e.upper for e in self])

    @property
    def flip(self):
        """Return ``Group`` object with inverted z-coordinate."""
        return Group([e.flip for e in self], name=self.name)

    @property
    def data(self):
        """Return list of objects in ``Group``."""
        return list(self)

    @classmethod
    def from_csv(cls, fname, typ=Lin, delimiter=",", acol=1, icol=2):
        """Create ``Group`` object from csv file

        Args:
          fname: csv filename

        Keyword Args:
          typ: Type of objects. Default ``Lin``
          delimiter (str): values delimiter. Default ','
          acol (int): azimuth column. Default 1
          icol (int): inclination column. Default 2

        Example:
          >>> g = Group.from_csv('file.csv', typ=Fol, acol=2, icol=3)

        """
        from os.path import basename

        dt = np.loadtxt(fname, dtype=float, delimiter=delimiter).T
        return cls.from_array(dt[acol - 1], dt[icol - 1], typ=typ, name=basename(fname))

    def to_csv(self, fname, delimiter=",", rounded=False):
        """Save ``Group`` object to csv file

        Args:
          fname: csv filename

        Keyword Args:
          delimiter (str): values delimiter. Default ','
          rounded (bool): round values to integer. Default False

        """
        if rounded:
            data = np.round(self.dd.T).astype(int)
        else:
            data = self.dd.T
        np.savetxt(fname, data, fmt="%g", delimiter=",", header=self.name)

    @classmethod
    def from_array(cls, azis, incs, typ=Lin, name="Default"):
        """Create ``Group`` object from arrays of dip directions and dips

        Args:
          azis: list or array of dip directions
          incs: list or array of inclinations

        Keyword Args:
          typ: type of data. ``Fol`` or ``Lin``
          name: name of ``Group`` object. Default is 'Default'

        Example:
          >>> f = Fault(140, 30, 110, 26, -1)
        """
        data = []
        data = [typ(azi, inc) for azi, inc in zip(azis, incs)]
        return cls(data, name=name)

    @property
    def aslin(self):
        """Return ``Group`` object with all data converted to ``Lin``."""
        return Group([e.aslin for e in self], name=self.name)

    @property
    def asfol(self):
        """Return ``Group`` object with all data converted to ``Fol``."""
        return Group([e.asfol for e in self], name=self.name)

    @property
    def asvec3(self):
        """Return ``Group`` object with all data converted to ``Vec3``."""
        return Group([e.asvec3 for e in self], name=self.name)

    @property
    def V(self):
        """Return ``Group`` object with all data converted to ``Vec3``."""
        return Group([e.asvec3 for e in self], name=self.name)

    @property
    def R(self):
        """Return resultant of data in ``Group`` object.

        Resultant is of same type as ``Group``. Note that ``Fol`` and ``Lin``
        are axial in nature so resultant can give other result than
        expected. For most cases is should not be problem as it is
        calculated as resultant of centered data. Anyway for axial
        data orientation tensor analysis will give you right answer.

        As axial summing is not commutative we use vectorial summing of
        centered data for Fol and Lin
        """
        if self.type == Vec3:
            r = Vec3(np.sum(self, axis=0))
        elif self.type == Lin:
            _, _, u = np.linalg.svd(self.ortensor.cov)
            # centered
            cntr = self.transform(u).rotate(Lin(90, 0), 90)
            # all points Z-ward
            cg = Group.from_array(*cntr.dd, typ=Vec3)
            r = cg.R.aslin.rotate(Lin(90, 0), -90).transform(u.T)
        elif self.type == Fol:
            _, _, u = np.linalg.svd(self.ortensor.cov)
            # centered
            cntr = self.transform(u).rotate(Lin(90, 0), 90)
            # all points Z-ward
            cg = Group.from_array(*cntr.aslin.dd, typ=Vec3)
            r = cg.R.asfol.rotate(Lin(90, 0), -90).transform(u.T)
        else:
            raise TypeError("Wrong argument type! Only Vec3, Lin and Fol!")
        return r

    @property
    def var(self):
        """Spherical variance based on resultant length (Mardia 1972).

        var = 1 - |R| / n
        """
        return 1 - abs(self.R) / len(self)

    @property
    def totvar(self):
        """Return total variance based on projections onto resultant

        totvar = sum(|x - R|^2) / 2n

        Note that difference between totvar and var is measure of difference
        between sample and population mean
        """
        return 1 - np.mean(self.dot(self.R.uv))

    @property
    def fisher_stats(self):
        """Fisher's statistics.

        fisher_stats property returns dictionary with `k`, `csd` and
        `a95` keywords.
        """
        stats = {"k": np.inf, "a95": 180.0, "csd": 0.0}
        N = len(self)
        R = abs(self.R)
        if N != R:
            stats["k"] = (N - 1) / (N - R)
            stats["csd"] = 81 / np.sqrt(stats["k"])
        stats["a95"] = acosd(1 - ((N - R) / R) * (20 ** (1 / (N - 1)) - 1))
        return stats

    @property
    def delta(self):
        """Cone angle containing ~63% of the data in degrees."""
        return acosd(abs(self.R) / len(self))

    @property
    def rdegree(self):
        """Degree of preffered orientation of data in ``Group`` object.

        D = 100 * (2 * |R| - n) / n
        """
        N = len(self)
        return 100 * (2 * abs(self.R) - N) / N

    def cross(self, other=None):
        """Return cross products of all data in ``Group`` object

        Without arguments it returns cross product of all pairs in dataset.
        If argument is group or single data object all mutual cross products
        are returned.
        """
        res = []
        if other is None:
            for i in range(len(self) - 1):
                for j in range(i + 1, len(self)):
                    res.append(self[i] ** self[j])
        elif isinstance(other, Group):
            for e in self:
                for f in other:
                    res.append(e ** f)
        elif issubclass(type(other), Vec3):
            for e in self:
                res.append(e ** other)
        else:
            raise TypeError("Wrong argument type!")
        return Group(res, name=self.name)

    def rotate(self, axis, phi):
        """Rotate ``Group`` object `phi` degress about `axis`."""
        return Group([e.rotate(axis, phi) for e in self], name=self.name)

    @property
    def centered(self):
        """Rotate ``Group`` object to position that eigenvectors are parallel
        to axes of coordinate system: E1(vertical), E2(east-west),
        E3(north-south)

        """
        _, _, u = np.linalg.svd(self.ortensor.cov)
        return self.transform(u).rotate(Lin(90, 0), 90)

    @property
    def halfspace(self):
        """Change orientation of vectors in Group, so all have angle<=90 with
        resultant.

        """
        v = self.asvec3
        alldone = np.all(v.angle(v.R) <= 90)
        while not alldone:
            ang = v.angle(v.R)
            for ix, do in enumerate(ang > 90):
                if do:
                    v[ix] = -v[ix]
                alldone = np.all(v.angle(v.R) <= 90)
        if self.type == Lin:
            v = v.aslin
        if self.type == Fol:
            v = v.asfol
        return v

    @property
    def uv(self):
        """Return ``Group`` object with normalized (unit length) elements."""
        return Group([e.uv for e in self], name=self.name)

    def angle(self, other=None):
        """Return angles of all data in ``Group`` object

        Without arguments it returns angles of all pairs in dataset.
        If argument is group or single data object all mutual angles
        are returned.
        """
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
            raise TypeError("Wrong argument type!")
        return np.array(res)

    def proj(self, vec):
        """Return projections of all data in ``Group`` onto vector.

        """
        return Group([e.proj(vec) for e in self], name=self.name)

    def dot(self, vec):
        """Return array of dot products of all data in ``Group`` with vector.

        """
        return np.array([e.dot(vec) for e in self])

    @property
    def ortensor(self):
        """Return orientation tensor ``Ortensor`` of ``Group``."""
        return Ortensor(self)

    @property
    def cluster(self):
        """Return hierarchical clustering ``Cluster`` of ``Group``."""
        return Cluster(self)

    def transform(self, F, **kwargs):
        """Return affine transformation of ``Group`` by matrix 'F'.

        Args:
          F: Transformation matrix. Should be array-like value e.g. ``DefGrad``

        Keyword Args:
          norm: normalize transformed vectors. True or False. Dafault False

        """
        return Group([e.transform(F, **kwargs) for e in self], name=self.name)

    @property
    def dd(self):
        """Return array of dip directions and dips of ``Group``"""
        return np.array([d.dd for d in self]).T

    @property
    def rhr(self):
        """Return array of strikes and dips of ``Group``"""
        return np.array([d.rhr for d in self]).T

    @classmethod
    def randn_lin(cls, N=100, mean=Lin(0, 90), sig=20, name="Default"):
        """Method to create ``Group`` of normaly distributed random ``Lin`` objects.

        Keyword Args:
          N: number of objects to be generated
          mean: mean orientation given as ``Lin``. Default Lin(0, 90)
          sig: sigma of normal distribution. Default 20
          name: name of dataset. Default is 'Default'

        Example:
          >>> g = Group.randn_lin(100, Lin(120, 40))
          >>> g.R
          L:120/39

        """
        data = []
        ta, td = mean.dd
        for azi, dip in zip(180 * np.random.rand(N), sig * np.random.randn(N)):
            data.append(Lin(0, 90).rotate(Lin(azi, 0), dip))
        return cls(data, name=name).rotate(Lin(ta + 90, 0), 90 - td)

    @classmethod
    def randn_fol(cls, N=100, mean=Fol(0, 0), sig=20, name="Default"):
        """Method to create ``Group`` of normaly distributed random ``Fol`` objects.

        Keyword Args:
          N: number of objects to be generated
          mean: mean orientation given as ``Fol``. Default Fol(0, 0)
          sig: sigma of normal distribution. Default 20
          name: name of dataset. Default is 'Default'

        Example:
          >>> g = Group.randn_fol(100, Lin(240, 60))
          >>> g.R
          S:238/61

        """
        data = []
        ta, td = mean.dd
        for azi, dip in zip(180 * np.random.rand(N), sig * np.random.randn(N)):
            data.append(Fol(0, 0).rotate(Lin(azi, 0), dip))
        return cls(data, name=name).rotate(Lin(ta - 90, 0), td)

    @classmethod
    def uniform_lin(cls, N=500, name="Default"):
        """Method to create ``Group`` of uniformly distributed ``Lin`` objects.

        Keyword Args:
          N: approximate (maximum) number of objects to be generated
          name: name of dataset. Default is 'Default'

        Example:
          >>> g = Group.uniform_lin(300)
          >>> g.ortensor.eigenvals
          array([ 0.3354383 ,  0.33228085,  0.33228085])

        """
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
        return cls.from_array(azi, inc, typ=Lin, name=name)

    @classmethod
    def uniform_fol(cls, N=500, name="Default"):
        """Method to create ``Group`` of uniformly distributed ``Fol`` objects.

        Keyword Args:
          N: approximate (maximum) number of objects to be generated
          name: name of dataset. Default is 'Default'

        Example:
          >>> g = Group.uniform_fol(300)
          >>> g.ortensor.eigenvals
          array([ 0.3354383 ,  0.33228085,  0.33228085])

        """
        lins = cls.uniform_lin(N=N)
        azi, inc = lins.dd
        if settings["notation"] == "rhr":
            azi -= 90
        return cls.from_array(azi + 180, 90 - inc, typ=Fol, name=name)

    @classmethod
    def sfs_vec3(cls, N=1000, name="Default"):
        """Method to create ``Group`` of uniformly distributed ``Vec3`` objects.
        Spherical Fibonacci Spiral points on a sphere algorithm adopted from
        John Burkardt.

        http://people.sc.fsu.edu/~jburkardt/

        Keyword Args:
          N: number of objects to be generated. Default 1000
          name: name of dataset. Default is 'Default'

        Example:
          >>> v = Group.sfs_vec3(300)
          >>> v.ortensor.eigenvals
          array([ 0.33346453,  0.33333475,  0.33320072])
        """
        phi = (1 + np.sqrt(5)) / 2
        i2 = 2 * np.arange(N) - N + 1
        theta = 2 * np.pi * i2 / phi
        sp = i2 / N
        cp = np.sqrt((N + i2) * (N - i2)) / N
        dc = np.array([cp * np.sin(theta), cp * np.cos(theta), sp]).T
        return cls([Vec3(d) for d in dc], name=name)

    @classmethod
    def sfs_lin(cls, N=500, name="Default"):
        """Method to create ``Group`` of uniformly distributed ``Lin`` objects.
        Based on ``Group.sfs_vec3`` method, but only half of sphere is used.

        Args:
          N: number of objects to be generated. Default 500
          name: name of dataset. Default is 'Default'

        Example:
          >>> g = Group.sfs_lin(300)
          >>> g.ortensor.eigenvals
          array([ 0.33417707,  0.33333973,  0.33248319])
        """
        g = cls.sfs_vec3(N=2 * N)
        # no antipodal
        return cls([d.aslin for d in g if d[2] > 0], name=name)

    @classmethod
    def sfs_fol(cls, N=500, name="Default"):
        """Method to create ``Group`` of uniformly distributed ``Fol`` objects.
        Based on ``Group.sfs_vec3`` method, but only half of sphere is used.

        Args:
          N: number of objects to be generated. Default 500
          name: name of dataset. Default is 'Default'

        Example:
          >>> g = Group.sfs_fol(300)
          >>> g.ortensor.eigenvals
          array([ 0.33417707,  0.33333973,  0.33248319])
        """
        g = cls.sfs_vec3(N=2 * N)
        # no antipodal
        return cls([d.asfol for d in g if d[2] > 0], name=name)

    @classmethod
    def gss_vec3(cls, N=1000, name="Default"):
        """Method to create ``Group`` of uniformly distributed ``Vec3`` objects.
        Golden Section Spiral points on a sphere algorithm.

        http://www.softimageblog.com/archives/115

        Args:
          N: number of objects to be generated.  Default 1000
          name: name of dataset. Default is 'Default'

        Example:
          >>> v = Group.gss_vec3(300)
          >>> v.ortensor.eigenvals
          array([ 0.33335689,  0.33332315,  0.33331996])
        """
        inc = np.pi * (3 - np.sqrt(5))
        off = 2 / N
        k = np.arange(N)
        y = k * off - 1 + (off / 2)
        r = np.sqrt(1 - y * y)
        phi = k * inc
        dc = np.array([np.cos(phi) * r, y, np.sin(phi) * r]).T
        return cls([Vec3(d) for d in dc], name=name)

    @classmethod
    def gss_lin(cls, N=500, name="Default"):
        """Method to create ``Group`` of uniformly distributed ``Lin`` objects.
        Based on ``Group.gss_vec3`` method, but only half of sphere is used.

        Args:
          N: number of objects to be generated. Default 500
          name: name of dataset. Default is 'Default'

        Example:
          >>> g = Group.gss_lin(300)
          >>> g.ortensor.eigenvals
          array([ 0.33498373,  0.3333366 ,  0.33167967])
        """
        g = cls.gss_vec3(N=2 * N)
        # no antipodal
        return cls([d.aslin for d in g if d[2] > 0], name=name)

    @classmethod
    def gss_fol(cls, N=500, name="Default"):
        """Method to create ``Group`` of uniformly distributed ``Fol`` objects.
        Based on ``Group.gss_vec3`` method, but only half of sphere is used.

        Args:
          N: number of objects to be generated.  Default 500
          name: name of dataset. Default is 'Default'

        Example:
          >>> g = Group.gss_fol(300)
          >>> g.ortensor.eigenvals
          array([ 0.33498373,  0.3333366 ,  0.33167967])
        """
        g = cls.gss_vec3(N=2 * N)
        # no antipodal
        return cls([d.asfol for d in g if d[2] > 0], name=name)

    @classmethod
    def fisher_lin(cls, N=100, mean=Lin(0, 90), kappa=20, name="Default"):
        """Method to create ``Group`` of ``Lin`` objects distributed
        according to Fisher distribution.

        Args:
          N: number of objects to be generated
          kappa: precision parameter of the distribution. Default 20
          name: name of dataset. Default is 'Default'

        Example:
          >>> g = Group.fisher_lin(100, mean=Lin(120,10))
        """
        ta, td = mean.dd
        L = np.exp(-2 * kappa)
        a = np.random.random(N) * (1 - L) + L
        fac = np.sqrt(-np.log(a) / (2 * kappa))
        inc = 90 - 2 * asind(fac)
        azi = 360 * np.random.random(N)
        g = cls.from_array(azi, inc, typ=Lin, name=name)
        return g.rotate(Lin(ta + 90, 0), 90 - td)

    @classmethod
    def kent_lin(cls, p, kappa=20, beta=0, N=500, name="Default"):
        """Method to create ``Group`` of ``Lin`` objects distributed
        according to Kent distribution (Kent, 1982) - The 5-parameter
        Fisher–Bingham distribution.

        Args:
          p: Pair object defining orientation of data
          N: number of objects to be generated
          kappa: concentration parameter. Default 20
          beta: ellipticity 0 <= beta < kappa
          name: name of dataset. Default is 'Default'

        Example:
          >>> p = Pair(135, 30, 90, 22)
          >>> g = Group.kent_lin(p, 30, 5, 300)
        """
        assert issubclass(type(p), Pair), "Argument must be Pair object."
        k = KentDistribution(p.lvec, p.fvec.cross(p.lvec), p.fvec, kappa, beta)
        g = Group([Vec3(v).aslin for v in k.rvs(N)])
        return g

    def to_file(self, filename="group.dat"):
        """Save group to file.

        Keyword Args:
          filename (str): name of file to save. Default 'group.dat'

        """
        with open(filename, "wb") as file:
            pickle.dump(self, file)
        print("Group saved to file %s" % filename)

    @classmethod
    def from_file(cls, filename="group.dat"):
        """Load group to file.

        Keyword Args:
          filename (str): name of data file to load. Default 'group.dat'

        """
        with open(filename, "rb") as file:
            data = pickle.load(file)
        print("Group loaded from file %s" % filename)
        return cls(data, name=filename)

    def bootstrap(self, N=100, size=None):
        """Return iterator of bootstraped samples from ``Group``.

        Args:
          N: number of samples to be generated
          size: number of data in sample. Default is same as ``Group``.

        Example:
          >>> g = Group.randn_lin(100, mean=Lin(120,40))
          >>> sm = [gb.R for gb in g.bootstrap(100)]
          >>> g.fisher_stats
          {'csd': 18.985075817669784, 'a95': 3.4065695594364684, 'k': 18.203100466576508}
          >>> Group(sm).fisher_stats
          {'csd': 1.9142106832769188, 'a95': 0.33404753286607225, 'k': 1790.5669592301119}

        """
        if size is None:
            size = len(self)
        for ix in np.random.randint(0, len(self), (N, size)):
            yield self[ix]

    @classmethod
    def examples(cls, name=None):
        """Create ``Group`` from example datasets. Available names are returned
        when no name of example dataset is given as argument.

        Keyword Args:
          name: name of dataset

        Example:
          >>> g = Group.examples('B2')

        """
        azis = {}
        incs = {}
        typs = {}
        # Embleton (1970) - Measurements of magnetic remanence in specimens
        # of Palaeozoic red-beds from Argentina.
        azis["B2"] = [
            122.5,
            130.5,
            132.5,
            148.5,
            140.0,
            133.0,
            157.5,
            153.0,
            140.0,
            147.5,
            142.0,
            163.5,
            141.0,
            156.0,
            139.5,
            153.5,
            151.5,
            147.5,
            141.0,
            143.5,
            131.5,
            147.5,
            147.0,
            149.0,
            144.0,
            139.5,
        ]
        incs["B2"] = [
            55.5,
            58.0,
            44.0,
            56.0,
            63.0,
            64.5,
            53.0,
            44.5,
            61.5,
            54.5,
            51.0,
            56.0,
            59.5,
            56.5,
            54.0,
            47.5,
            61.0,
            58.5,
            57.0,
            67.5,
            62.5,
            63.5,
            55.5,
            62.0,
            53.5,
            58.0,
        ]
        typs["B2"] = Lin
        # Cohen (1983) - Facing directions of conically folded planes.
        azis["B4"] = [
            269,
            265,
            271,
            272,
            268,
            267,
            265,
            265,
            265,
            263,
            267,
            267,
            270,
            270,
            265,
            95,
            100,
            95,
            90,
            271,
            267,
            272,
            270,
            273,
            271,
            269,
            270,
            267,
            266,
            268,
            269,
            270,
            269,
            270,
            272,
            271,
            271,
            270,
            273,
            271,
            270,
            274,
            275,
            274,
            270,
            268,
            97,
            95,
            90,
            95,
            94,
            93,
            93,
            93,
            95,
            96,
            100,
            104,
            102,
            108,
            99,
            112,
            110,
            100,
            95,
            93,
            91,
            92,
            92,
            95,
            89,
            93,
            100,
            270,
            261,
            275,
            276,
            275,
            277,
            276,
            273,
            273,
            271,
            275,
            277,
            275,
            276,
            279,
            277,
            278,
            280,
            275,
            270,
            275,
            276,
            255,
            105,
            99,
            253,
            96,
            93,
            92,
            91,
            91,
            90,
            89,
            89,
            96,
            105,
            90,
            76,
            91,
            91,
            91,
            90,
            95,
            90,
            92,
            92,
            95,
            100,
            135,
            98,
            92,
            90,
            99,
            175,
            220,
            266,
            235,
            231,
            256,
            272,
            276,
            276,
            275,
            273,
            266,
            276,
            274,
            275,
            274,
            272,
            273,
            270,
            103,
            95,
            98,
            96,
            111,
            96,
            92,
            91,
            90,
            90,
        ]
        incs["B4"] = [
            48,
            57,
            61,
            59,
            58,
            60,
            59,
            58,
            60,
            59,
            59,
            53,
            50,
            48,
            61,
            40,
            56,
            67,
            52,
            49,
            60,
            47,
            50,
            48,
            50,
            53,
            52,
            58,
            60,
            60,
            62,
            61,
            62,
            60,
            58,
            59,
            56,
            53,
            49,
            49,
            53,
            50,
            46,
            45,
            60,
            68,
            75,
            47,
            48,
            50,
            48,
            49,
            45,
            41,
            42,
            40,
            51,
            70,
            74,
            71,
            51,
            75,
            73,
            60,
            49,
            44,
            41,
            51,
            45,
            50,
            41,
            44,
            68,
            67,
            73,
            50,
            40,
            60,
            47,
            47,
            54,
            60,
            62,
            57,
            43,
            53,
            40,
            40,
            42,
            44,
            60,
            65,
            76,
            63,
            71,
            80,
            77,
            72,
            80,
            60,
            48,
            49,
            48,
            46,
            44,
            43,
            40,
            58,
            65,
            39,
            48,
            38,
            43,
            42,
            49,
            39,
            43,
            44,
            48,
            61,
            68,
            80,
            60,
            49,
            45,
            62,
            79,
            79,
            72,
            76,
            77,
            71,
            60,
            42,
            50,
            41,
            60,
            73,
            43,
            50,
            46,
            51,
            56,
            50,
            40,
            73,
            57,
            60,
            54,
            71,
            54,
            50,
            48,
            48,
            51,
        ]
        typs["B4"] = Lin
        # Powell, Cole & Cudahy (1985) - Orientations of axial-plane cleavage
        # surfaces of F1 folds in Ordovician turbidites.
        azis["B11"] = [
            65,
            75,
            233,
            39,
            53,
            58,
            50,
            231,
            220,
            30,
            59,
            44,
            54,
            251,
            233,
            52,
            26,
            40,
            266,
            67,
            61,
            72,
            54,
            32,
            238,
            84,
            230,
            228,
            230,
            231,
            40,
            233,
            234,
            225,
            234,
            222,
            230,
            51,
            46,
            207,
            221,
            58,
            48,
            222,
            10,
            52,
            49,
            36,
            225,
            221,
            216,
            194,
            228,
            27,
            226,
            58,
            35,
            37,
            235,
            38,
            227,
            34,
            225,
            53,
            57,
            66,
            45,
            47,
            54,
            45,
            60,
            51,
            42,
            52,
            63,
        ]

        incs["B11"] = [
            50,
            53,
            85,
            82,
            82,
            66,
            75,
            85,
            87,
            85,
            82,
            88,
            86,
            82,
            83,
            86,
            80,
            78,
            85,
            89,
            85,
            85,
            86,
            67,
            87,
            86,
            81,
            85,
            79,
            86,
            88,
            84,
            87,
            88,
            83,
            82,
            89,
            82,
            82,
            67,
            85,
            87,
            82,
            82,
            82,
            75,
            68,
            89,
            81,
            87,
            63,
            86,
            81,
            81,
            89,
            62,
            81,
            88,
            70,
            80,
            77,
            85,
            74,
            90,
            90,
            90,
            90,
            90,
            90,
            90,
            90,
            90,
            90,
            90,
            90,
        ]
        typs["B11"] = Fol
        # Powell, Cole & Cudahy (1985) - Orientations of axial-plane cleavage
        # surfaces of F1 folds in Ordovician turbidites.
        azis["B12"] = [
            122,
            132,
            141,
            145,
            128,
            133,
            130,
            129,
            124,
            120,
            137,
            141,
            151,
            138,
            135,
            135,
            156,
            156,
            130,
            112,
            116,
            113,
            117,
            110,
            106,
            106,
            98,
            84,
            77,
            111,
            122,
            140,
            48,
            279,
            19,
            28,
            28,
            310,
            310,
            331,
            326,
            332,
            3,
            324,
            308,
            304,
            304,
            299,
            293,
            293,
            306,
            310,
            313,
            319,
            320,
            320,
            330,
            327,
            312,
            317,
            314,
            312,
            311,
            307,
            311,
            310,
            310,
            305,
            305,
            301,
            301,
            300,
        ]

        incs["B12"] = [
            80,
            72,
            63,
            51,
            62,
            53,
            53,
            52,
            48,
            45,
            44,
            44,
            34,
            37,
            38,
            40,
            25,
            15,
            22,
            63,
            35,
            28,
            28,
            22,
            33,
            37,
            32,
            27,
            24,
            8,
            6,
            8,
            11,
            8,
            6,
            6,
            8,
            20,
            21,
            18,
            25,
            28,
            32,
            32,
            32,
            34,
            38,
            37,
            44,
            45,
            48,
            42,
            47,
            45,
            43,
            45,
            50,
            70,
            59,
            66,
            65,
            70,
            66,
            67,
            83,
            66,
            69,
            69,
            72,
            67,
            69,
            82,
        ]
        typs["B12"] = Fol

        if name is None:
            print("Available sample datasets:")
            print(list(typs.keys()))
        else:
            return cls.from_array(azis[name], incs[name], typs[name], name=name)


class PairSet(list):
    """
    Represents a homogeneous group of ``Pair`` objects.
    """

    def __init__(self, data, name="Default"):
        assert issubclass(type(data), list), "Argument must be list of data."
        assert len(data) > 0, "Empty PairSet is not allowed."
        tp = type(data[0])
        assert issubclass(tp, Pair), "Data must be of Pair type."
        assert all(
            [isinstance(e, tp) for e in data]
        ), "All data in PairSet must be of same type."
        super(PairSet, self).__init__(data)
        self.type = tp
        self.name = name

    def __repr__(self):
        return "P:%g %s (%s)" % (len(self), self.type.__name__, self.name)

    def __add__(self, other):
        # merge sets
        assert self.type is other.type, "Only same type could be merged"
        return PairSet(list(self) + other, name=self.name)

    def __setitem__(self, key, value):
        assert isinstance(value, self.type), (
            "item is not of type %s" % self.type.__name__
        )
        super(FaultSet, self).__setitem__(key, value)

    def __getitem__(self, key):
        """PairSet fancy indexing"""
        if isinstance(key, slice):
            key = np.arange(*key.indices(len(self)))
        if isinstance(key, list) or isinstance(key, tuple):
            key = np.asarray(key)
        if isinstance(key, np.ndarray):
            if key.dtype == "bool":
                key = np.flatnonzero(key)
            return type(self)([self[i] for i in key])
        else:
            return super(type(self), self).__getitem__(key)

    def append(self, item):
        assert isinstance(item, self.type), (
            "item is not of type %s" % self.type.__name__
        )
        super(PairSet, self).append(item)

    def extend(self, items=()):
        for item in items:
            self.append(item)

    @property
    def data(self):
        return list(self)

    def rotate(self, axis, phi):
        """Rotate PairSet"""
        return type(self)([f.rotate(axis, phi) for f in self], name=self.name)

    @classmethod
    def from_csv(cls, fname, delimiter=",", facol=1, ficol=2, lacol=3, licol=4):
        """Read PairSet from csv file"""
        from os.path import basename

        dt = np.loadtxt(fname, dtype=float, delimiter=delimiter).T
        return cls.from_array(
            dt[facol - 1],
            dt[ficol - 1],
            dt[lacol - 1],
            dt[licol - 1],
            name=basename(fname),
        )

    def to_csv(self, fname, delimiter=",", rounded=False):
        if rounded:
            data = np.c_[
                np.round(self.fol.dd.T).astype(int), np.round(self.lin.dd.T).astype(int)
            ]
        else:
            data = np.c_[self.fol.dd.T, self.lin.dd.T]

        np.savetxt(fname, data, fmt="%g", delimiter=",", header=self.name)

    @classmethod
    def from_array(cls, fazis, fincs, lazis, lincs, name="Default"):
        """Create PairSet from arrays of dip directions and dips"""
        data = []
        for fazi, finc, lazi, linc in zip(fazis, fincs, lazis, lincs):
            data.append(Pair(fazi, finc, lazi, linc))
        return cls(data, name=name)

    @property
    def fol(self):
        """Return Fol part of PairSet as Group of Fol"""
        return Group([e.fol for e in self], name=self.name)

    @property
    def fvec(self):
        """Return vectors of Fol of PairSet as Group of Vec3"""
        return Group([e.fvec for e in self], name=self.name)

    @property
    def lin(self):
        """Return Lin part of PairSet as Group of Lin"""
        return Group([e.lin for e in self], name=self.name)

    @property
    def lvec(self):
        """Return vectors of Lin part of PairSet as Group of Vec3"""
        return Group([e.lvec for e in self], name=self.name)

    @property
    def misfit(self):
        """Return array of misfits"""
        return np.array([f.misfit for f in self])


class FaultSet(PairSet):
    """
    Represents a homogeneous group of ``Fault`` objects.

    """

    def __init__(self, data, name="Default"):
        assert issubclass(type(data), list), "Argument must be list of data."
        assert len(data) > 0, "Empty FaultSet is not allowed."
        tp = type(data[0])
        assert issubclass(tp, Fault), "Data must be of Fault type."
        assert all(
            [isinstance(e, tp) for e in data]
        ), "All data in FaultSet must be of same type."
        super(FaultSet, self).__init__(data)
        self.type = tp
        self.name = name

    def __repr__(self):
        return "F:%g %s (%s)" % (len(self), self.type.__name__, self.name)

    @classmethod
    def from_csv(cls, fname, delimiter=",", facol=1, ficol=2, lacol=3, licol=4, scol=5):
        """Read FaultSet from csv file"""
        from os.path import basename

        dt = np.loadtxt(fname, dtype=float, delimiter=delimiter).T
        return cls.from_array(
            dt[facol - 1],
            dt[ficol - 1],
            dt[lacol - 1],
            dt[licol - 1],
            dt[scol - 1],
            name=basename(fname),
        )

    def to_csv(self, fname, delimiter=",", rounded=False):
        if rounded:
            data = np.c_[
                np.round(self.fol.dd.T).astype(int),
                np.round(self.lin.dd.T).astype(int),
                self.sense.astype(int),
            ]
        else:
            data = np.c_[self.fol.dd.T, self.lin.dd.T, self.sense]

        np.savetxt(fname, data, fmt="%g", delimiter=",", header=self.name)

    @classmethod
    def from_array(cls, fazis, fincs, lazis, lincs, senses, name="Default"):
        """Create dataset from arrays of dip directions and dips"""
        data = []
        for fazi, finc, lazi, linc, sense in zip(fazis, fincs, lazis, lincs, senses):
            data.append(Fault(fazi, finc, lazi, linc, sense))
        return cls(data, name=name)

    @property
    def sense(self):
        """Return array of sense values"""
        return np.array([f.sense for f in self])

    @property
    def p(self):
        """Return p-axes of FaultSet as Group of Lin"""
        return Group([e.p for e in self], name=self.name + "-P")

    @property
    def pvec(self):
        """Return p-axes of FaultSet as Group of Vec3"""
        return Group([e.pvec for e in self], name=self.name)

    @property
    def tvec(self):
        """Return t-axes of FaultSet as Group of Vec3"""
        return Group([e.tvec for e in self], name=self.name)

    @property
    def t(self):
        """Return t-axes of FaultSet as Group of Lin"""
        return Group([e.t for e in self], name=self.name + "-T")

    @property
    def m(self):
        """Return m-planes of FaultSet as Group of Fol"""
        return Group([e.m for e in self], name=self.name + "-M")

    @property
    def d(self):
        """Return dihedra planes of FaultSet as Group of Fol"""
        return Group([e.d for e in self], name=self.name + "-D")

    def angmech(self, method="classic"):
        """Implementation of Angelier-Mechler dihedra method

        Args:
          method: 'probability' or 'classic'. Classic method assigns +/-1
          to individual positions, while 'probability' returns maximum
          likelihood estimate.
        """

        def angmech(dc, fs):
            val = 0
            for f in fs:
                val += 2 * float(np.sign(dc.dot(f.fvec)) == np.sign(dc.dot(f.lvec))) - 1
            return val

        def angmech2(dc, fs):
            val = 0
            d = Vec3(dc).aslin
            for f in fs:
                s = 2 * float(np.sign(dc.dot(f.fvec)) == np.sign(dc.dot(f.lvec))) - 1
                lprob = 1 - abs(45 - f.lin.angle(d)) / 45
                fprob = 1 - abs(45 - f.fol.angle(d)) / 45
                val += s * lprob * fprob
            return val

        d = StereoGrid()
        if method == "probability":
            d.apply_func(angmech2, self)
        else:
            d.apply_func(angmech, self)
        return d

    @classmethod
    def examples(cls, name=None):
        """Create ``FaultSet`` from example datasets. Available names are returned
        when no name of example dataset is given as argument.

        Keyword Args:
          name: name of dataset

        Example:
          >>> fs = FaultSet.examples('MELE')

        """
        fazis, fincs = {}, {}
        lazis, lincs = {}, {}
        senses = {}
        # Lexa (2008) - reactivated joints - Lipnice
        fazis["MELE"] = [
            95,
            66,
            42,
            14,
            126,
            12,
            14,
            150,
            35,
            26,
            138,
            140,
            132,
            50,
            52,
            70,
            152,
            70,
            184,
            194,
            330,
            150,
            72,
            80,
            188,
            186,
            72,
            138,
            72,
            184,
            308,
            128,
            60,
            130,
            105,
            130,
            124,
            135,
            292,
            30,
            36,
            282,
            95,
            88,
            134,
            120,
            26,
            2,
            8,
            6,
            140,
            60,
            60,
            98,
            88,
            94,
            110,
            114,
            8,
            100,
            16,
            20,
            120,
            10,
            120,
            10,
            124,
            30,
            22,
            204,
            4,
            254,
            296,
            244,
            210,
            22,
            250,
            210,
            130,
            206,
            210,
            4,
            258,
            260,
            272,
            96,
            105,
            120,
            214,
            96,
            22,
            88,
            26,
            110,
        ]
        fincs["MELE"] = [
            80,
            85,
            46,
            62,
            78,
            62,
            66,
            70,
            45,
            58,
            80,
            80,
            80,
            88,
            88,
            60,
            82,
            32,
            82,
            80,
            80,
            85,
            40,
            30,
            82,
            82,
            46,
            85,
            30,
            88,
            85,
            88,
            52,
            75,
            85,
            76,
            80,
            88,
            80,
            50,
            50,
            38,
            85,
            42,
            68,
            80,
            65,
            60,
            65,
            65,
            60,
            50,
            50,
            75,
            70,
            85,
            70,
            62,
            36,
            60,
            66,
            50,
            68,
            38,
            72,
            90,
            88,
            90,
            90,
            85,
            90,
            75,
            85,
            85,
            85,
            82,
            75,
            85,
            75,
            88,
            89,
            68,
            88,
            82,
            72,
            78,
            85,
            85,
            60,
            88,
            62,
            58,
            56,
            72,
        ]
        lazis["MELE"] = [
            119,
            154,
            110,
            296,
            41,
            295,
            291,
            232,
            106,
            105,
            49,
            227,
            45,
            139,
            142,
            149,
            241,
            89,
            98,
            110,
            55,
            60,
            91,
            105,
            98,
            96,
            103,
            226,
            104,
            95,
            37,
            217,
            112,
            48,
            16,
            46,
            39,
            46,
            15,
            108,
            100,
            4,
            8,
            102,
            51,
            207,
            299,
            283,
            290,
            287,
            62,
            333,
            7,
            185,
            359,
            5,
            21,
            31,
            90,
            14,
            290,
            102,
            49,
            93,
            35,
            280,
            213,
            120,
            292,
            114,
            274,
            320,
            19,
            332,
            299,
            295,
            332,
            297,
            49,
            296,
            300,
            276,
            176,
            275,
            253,
            103,
            184,
            30,
            134,
            6,
            108,
            49,
            112,
            27,
        ]
        lincs["MELE"] = [
            79,
            20,
            22,
            21,
            21,
            23,
            16,
            22,
            18,
            16,
            8,
            18,
            18,
            25,
            5,
            18,
            5,
            31,
            25,
            32,
            27,
            3,
            38,
            28,
            2,
            2,
            42,
            22,
            26,
            25,
            10,
            15,
            38,
            26,
            10,
            23,
            26,
            28,
            35,
            14,
            28,
            6,
            32,
            41,
            16,
            16,
            6,
            19,
            23,
            22,
            19,
            4,
            35,
            10,
            3,
            8,
            2,
            13,
            6,
            7,
            10,
            10,
            38,
            6,
            14,
            20,
            28,
            0,
            15,
            5,
            45,
            57,
            54,
            20,
            10,
            21,
            28,
            30,
            30,
            10,
            12,
            6,
            76,
            82,
            71,
            78,
            66,
            5,
            16,
            2,
            7,
            51,
            6,
            20,
        ]
        senses["MELE"] = [
            -1,
            -1,
            -1,
            1,
            -1,
            1,
            1,
            1,
            -1,
            -1,
            -1,
            1,
            -1,
            -1,
            -1,
            -1,
            -1,
            -1,
            1,
            1,
            1,
            -1,
            -1,
            -1,
            1,
            1,
            -1,
            1,
            -1,
            1,
            1,
            1,
            -1,
            -1,
            -1,
            -1,
            -1,
            -1,
            1,
            -1,
            -1,
            1,
            -1,
            -1,
            -1,
            1,
            1,
            1,
            1,
            1,
            -1,
            1,
            -1,
            1,
            -1,
            -1,
            -1,
            -1,
            -1,
            -1,
            1,
            -1,
            -1,
            -1,
            -1,
            1,
            1,
            -1,
            1,
            1,
            1,
            -1,
            -1,
            -1,
            -1,
            1,
            -1,
            -1,
            -1,
            -1,
            -1,
            1,
            -1,
            -1,
            -1,
            -1,
            1,
            -1,
            1,
            -1,
            -1,
            -1,
            -1,
            -1,
        ]

        if name is None:
            print("Available sample datasets:")
            print(list(senses.keys()))
        else:
            return cls.from_array(
                fazis[name],
                fincs[name],
                lazis[name],
                lincs[name],
                senses[name],
                name=name,
            )


class Ortensor(object):
    """
    Represents an orientation tensor, which characterize data distribution
    using eigenvalue method. See (Watson 1966, Scheidegger 1965).

    See following methods and properties for additional operations.

    Args:
        data (``Group``): grou of ``Vec3``, ``Fol`` or ``Lin`` objects

    Returns:
        ``Ortensor`` object

    Example:
        >>> g = Group.examples('B2')
        >>> ot = Ortensor(g)
        >>> ot
        Ortensor: B2 Kind: prolate
        (E1:0.9825,E2:0.01039,E3:0.007101)
        [[ 0.19780807 -0.13566589 -0.35878837]
        [-0.13566589  0.10492993  0.25970594]
        [-0.35878837  0.25970594  0.697262  ]]
      >>> ot.eigenlins.data
        [L:144/57, L:360/28, L:261/16]

    """

    def __init__(self, d, **kwargs):
        assert isinstance(d, Group), "Only group could be passed to Ortensor"
        self.cov = np.dot(np.array(d).T, np.array(d)) / len(d)
        self.name = d.name
        vc, vv = np.linalg.eig(self.cov)
        ix = np.argsort(vc)[::-1]
        self.eigenvals = vc[ix]
        self.vects = vv.T[ix]
        self.scaled = kwargs.get("scaled", False)

    def __repr__(self):
        return (
            "Ortensor: %s Kind: %s\n" % (self.name, self.kind)
            + "(E1:%.4g,E2:%.4g,E3:%.4g)\n" % tuple(self.eigenvals)
            + str(self.cov)
        )

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
        return Group(
            [
                e1 * Vec3(self.vects[0]),
                e2 * Vec3(self.vects[1]),
                e3 * Vec3(self.vects[2]),
            ]
        )

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
        return self.eigenvals[0] - self.eigenvals[1]

    @property
    def G(self):
        """Girdle index - Vollmer, 1990"""
        return 2 * (self.eigenvals[1] - self.eigenvals[2])

    @property
    def R(self):
        """Random index - Vollmer, 1990"""
        return 3 * self.eigenvals[2]

    @property
    def B(self):
        """Cylindricity index - Vollmer, 1990"""
        return self.P + self.G

    @property
    def I(self):
        """Intensity index - Lisle, 1985"""
        return 7.5 * np.sum((self.eigenvals - 1 / 3) ** 2)

    @property
    def kind(self):
        """Return descriptive type of ellipsoid"""
        return {False: "oblate", True: "prolate"}[self.shape > 1]

    @property
    def MADp(self):
        """Return approximate angular deviation from the major axis along E1"""
        return atand(np.sqrt((1 - self.E1) / self.E1))

    @property
    def MADo(self):
        """Return approximate deviation from the plane normal to E3"""
        return atand(np.sqrt(self.E3 / (1 - self.E3)))

    @property
    def MAD(self):
        """Return approximate deviation according to shape"""
        if self.shape > 1:
            return self.MADp
        else:
            return self.MADo


class StereoGrid(object):
    """
    The class to store regular grid of values to be contoured on ``StereoNet``.

    ``StereoGrid`` object could be calculated from ``Group`` object or by user-
    defined function, which accept unit vector as argument.

    Args:
      g: ``Group`` object of data to be used for desity calculation. If
      ommited, zero values grid is returned.

    Keyword Args:
      npoints: approximate number of grid points Default 1800
      grid: type of grid 'radial' or 'ortho'. Default 'radial'
      sigma: sigma for kernels. Default 1
      method: 'exp_kamb', 'linear_kamb', 'square_kamb', 'schmidt', 'kamb'.
        Default 'exp_kamb'
      trim: Set negative values to zero. Default False
      weighted: use euclidean norms as weights. Default False

    """

    def __init__(self, d=None, **kwargs):
        self.initgrid(**kwargs)
        if d:
            assert isinstance(d, Group), "StereoGrid need Group as argument"
            self.calculate_density(np.asarray(d), **kwargs)

    def __repr__(self):
        return (
            "StereoGrid with %d points.\n" % self.n
            + "Maximum: %.4g at %s\n" % (self.max, self.max_at)
            + "Minimum: %.4g at %s" % (self.min, self.min_at)
        )

    @property
    def min(self):
        return self.values.min()

    @property
    def max(self):
        return self.values.max()

    @property
    def min_at(self):
        return Vec3(self.dcgrid[self.values.argmin()]).aslin

    @property
    def max_at(self):
        return Vec3(self.dcgrid[self.values.argmax()]).aslin

    def initgrid(self, **kwargs):
        import matplotlib.tri as tri

        # parse options
        grid = kwargs.get("grid", "radial")
        if grid == "radial":
            ctn_points = int(
                np.round(np.sqrt(kwargs.get("npoints", 1800)) / 0.280269786)
            )
            # calc grid
            self.xg = 0
            self.yg = 0
            for rho in np.linspace(0, 1, np.round(ctn_points / 2 / np.pi)):
                theta = np.linspace(0, 360, np.round(ctn_points * rho + 1))[:-1]
                self.xg = np.hstack((self.xg, rho * sind(theta)))
                self.yg = np.hstack((self.yg, rho * cosd(theta)))
        elif grid == "ortho":
            n = int(np.round(np.sqrt(kwargs.get("npoints", 1800) - 4) / 0.8685725142))
            x, y = np.meshgrid(np.linspace(-1, 1, n), np.linspace(-1, 1, n))
            d2 = (x ** 2 + y ** 2) <= 1
            self.xg = np.hstack((0, 1, 0, -1, x[d2]))
            self.yg = np.hstack((1, 0, -1, 0, y[d2]))
        else:
            raise TypeError("Wrong grid type!")
        self.dcgrid = l2v(*getldd(self.xg, self.yg)).T
        self.n = self.dcgrid.shape[0]
        self.values = np.zeros(self.n, dtype=np.float)
        self.triang = tri.Triangulation(self.xg, self.yg)

    def calculate_density(self, dcdata, **kwargs):
        """Calculate density of elements from ``Group`` object.

        """
        # parse options
        sigma = kwargs.get("sigma", 1 / len(dcdata) ** (-1 / 7))
        weighted = kwargs.get("weighted", False)
        method = kwargs.get("method", "exp_kamb")
        trim = kwargs.get("trim", False)

        func = {
            "linear_kamb": _linear_inverse_kamb,
            "square_kamb": _square_inverse_kamb,
            "schmidt": _schmidt_count,
            "kamb": _kamb_count,
            "exp_kamb": _exponential_kamb,
        }[method]

        # weights are given by euclidean norms of data
        if weighted:
            weights = np.linalg.norm(dcdata, axis=1)
            weights /= weights.mean()
        else:
            weights = np.ones(len(dcdata))
        for i in range(self.n):
            dist = np.abs(np.dot(self.dcgrid[i], dcdata.T))
            count, scale = func(dist, sigma)
            count *= weights
            self.values[i] = (count.sum() - 0.5) / scale
        if trim:
            self.values[self.values < 0] = 0

    def apply_func(self, func, *args, **kwargs):
        """Calculate values using function passed as argument.
        Function must accept vector (3 elements array) as argument
        and return scalar value.

        """
        for i in range(self.n):
            self.values[i] = func(self.dcgrid[i], *args, **kwargs)

    def contourf(self, *args, **kwargs):
        """ Show filled contours of values."""
        fig, ax = plt.subplots()
        # Projection circle
        ax.text(0, 1.02, "N", ha="center", va="baseline", fontsize=16)
        ax.add_artist(plt.Circle((0, 0), 1, color="w", zorder=0))
        ax.add_artist(plt.Circle((0, 0), 1, color="None", ec="k", zorder=3))
        ax.set_aspect("equal")
        plt.tricontourf(self.triang, self.values, *args, **kwargs)
        plt.colorbar()
        plt.axis('off')
        plt.show()

    def contour(self, *args, **kwargs):
        """ Show contours of values."""
        fig, ax = plt.subplots()
        # Projection circle
        ax.text(0, 1.02, "N", ha="center", va="baseline", fontsize=16)
        ax.add_artist(plt.Circle((0, 0), 1, color="w", zorder=0))
        ax.add_artist(plt.Circle((0, 0), 1, color="None", ec="k", zorder=3))
        ax.set_aspect("equal")
        plt.tricontour(self.triang, self.values, *args, **kwargs)
        plt.colorbar()
        plt.axis('off')
        plt.show()

    def plotcountgrid(self):
        """ Show counting grid."""
        fig, ax = plt.subplots()
        # Projection circle
        ax.text(0, 1.02, "N", ha="center", va="baseline", fontsize=16)
        ax.add_artist(plt.Circle((0, 0), 1, color="w", zorder=0))
        ax.add_artist(plt.Circle((0, 0), 1, color="None", ec="k", zorder=3))
        ax.set_aspect("equal")
        plt.triplot(self.triang, "bo-")
        plt.axis('off')
        plt.show()


class Cluster(object):
    """
    Provides a hierarchical clustering using `scipy.cluster` routines.

    The distance matrix is calculated as an angle between features, where ``Fol`` and
    ``Lin`` use axial angles while ``Vec3`` uses direction angles.
    """

    def __init__(self, d, **kwargs):
        assert isinstance(d, Group), "Only group could be clustered"
        self.data = Group(d.copy())
        self.maxclust = kwargs.get("maxclust", 2)
        self.angle = kwargs.get("angle", None)
        self.method = kwargs.get("method", "average")
        self.pdist = self.data.angle()
        self.linkage()

    def __repr__(self):
        if hasattr(self, "groups"):
            info = "Already %d clusters created." % len(self.groups)
        else:
            info = "Not yet clustered. Use cluster() method."
        if self.angle is not None:
            crit = "Criterion: Angle\nSettings: angle=%.4g\n" % (self.angle)
        else:
            crit = "Criterion: Maxclust\nSettings: muxclust=%.4g\n" % (self.maxclust)
        return (
            "Clustering object\n"
            + "Number of data: %d\n" % len(self.data)
            + "Linkage method: %s\n" % self.method
            + crit
            + info
        )

    def cluster(self, **kwargs):
        """Do clustering on data

        Result is stored as tuple of Groups in ``groups`` property.

        Keyword Args:
          criterion: The criterion to use in forming flat clusters
          maxclust: number of clusters
          angle: maximum cophenetic distance(angle) in clusters
        """
        from scipy.cluster.hierarchy import fcluster

        self.maxclust = kwargs.get("maxclust", 2)
        self.angle = kwargs.get("angle", None)
        if self.angle is not None:
            self.idx = fcluster(self.Z, self.angle, criterion="distance")
        else:
            self.idx = fcluster(self.Z, self.maxclust, criterion="maxclust")
        self.groups = tuple(
            self.data[np.flatnonzero(self.idx == c)] for c in np.unique(self.idx)
        )

    def linkage(self, **kwargs):
        """Do linkage of distance matrix

        Keyword Args:
          method: The linkage algorithm to use
        """
        from scipy.cluster.hierarchy import linkage

        self.method = kwargs.get("method", "average")
        self.Z = linkage(self.pdist, method=self.method, metric=angle_metric)

    def dendrogram(self, **kwargs):
        """Show dendrogram

        See ``scipy.cluster.hierarchy.dendrogram`` for possible kwargs.
        """
        from scipy.cluster.hierarchy import dendrogram
        import matplotlib.pyplot as plt

        dendrogram(self.Z, **kwargs)
        plt.show()

    def elbow(self, no_plot=False, n=None):
        """Plot within groups variance vs. number of clusters.

        Elbow criterion could be used to determine number of clusters.
        """
        from scipy.cluster.hierarchy import fcluster
        import matplotlib.pyplot as plt

        if n is None:
            idx = fcluster(self.Z, len(self.data), criterion="maxclust")
            nclust = list(np.arange(1, np.sqrt(idx.max() / 2) + 1, dtype=int))
        else:
            nclust = list(np.arange(1, n + 1, dtype=int))
        within_grp_var = []
        mean_var = []
        for n in nclust:
            idx = fcluster(self.Z, n, criterion="maxclust")
            grp = [np.flatnonzero(idx == c) for c in np.unique(idx)]
            # between_grp_var = Group([self.data[ix].R.uv for ix in grp]).var
            var = [100 * self.data[ix].var for ix in grp]
            within_grp_var.append(var)
            mean_var.append(np.mean(var))
        if not no_plot:
            plt.boxplot(within_grp_var, positions=nclust)
            plt.plot(nclust, mean_var, "k")
            plt.xlabel("Number of clusters")
            plt.ylabel("Variance")
            plt.title("Within-groups variance vs. number of clusters")
            plt.show()
        else:
            return nclust, within_grp_var

    @property
    def R(self):
        """Return group of clusters resultants."""
        return Group([group.R for group in self.groups])


# HELPERS #


def G(s, typ=Lin, name="Default"):
    """
    Create a group from space separated string of dip directions and dips.
    """

    vals = np.fromstring(s, sep=" ")

    return Group.from_array(vals[::2], vals[1::2], typ=typ, name=name)
