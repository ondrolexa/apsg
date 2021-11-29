import warnings
import numpy as np
from scipy import linalg as spla

from apsg.config import apsg_conf
from apsg.helpers import sind, cosd, tand, asind, acosd, atand, atan2d
from apsg.helpers import is_like_vec3
from apsg.decorator import ensure_one_arg_matrix
from apsg.math import Vector3, Axial3


"""
from apsg_classes import *

u = Vector3(-2,1,1)
v = Vector3(1,5,3)
g = Group.from_list([u, v])
h = Group()
h.append(u)
h.append(v)

from scipy.spatial import transform as sct
f = Fol(120,30)
ang = math.pi/2 - 1e-6
sl = sct.Slerp([0, 1], sct.Rotation.from_rotvec([-ang*f, ang*f]))
[Lin(v) for v in sl(np.linspace(0, 1, 21)).apply(f.dipvec())]

or

[Lin(f.dipvec().rotate(f, a)) for a in np.linspace(-89.999999, 89.999999, 21)]

"""

class Lin(Axial):
    def __repr__(self):
        azi, inc = vec2geo_linear(self)
        return f"L:{azi:.0f}/{inc:.0f}"

    def cross(self, other):
        return Fol(super().cross(other))

    __pow__ = cross


class Fol(Axial):
    def __init__(self, *args):
        if len(args) == 1 and is_like_vec3(args[0]):
            coords = [float(v) for v in args[0]]
        elif len(args) == 2:
            coords = Geo2CoordsPlanar[apsg_conf["notation"]](*args)
        elif len(args) == 3:
            coords = [float(v) for v in args]
        else:
            raise TypeError("Not valid arguments for Fol")
        self._coords = tuple(coords)

    def __repr__(self):
        azi, inc = vec2geo_planar(self)
        return f"S:{azi:.0f}/{inc:.0f}"

    def cross(self, other):
        return Lin(super().cross(other))

    __pow__ = cross

    def dipvec(self):
        return Vector3(*vec2geo_planar(self))

    def rake(self, rake):
        return Vector3(self.dipvec().rotate(self, rake - 90))

    @ensure_one_arg_matrix
    def transform(self, F, **kwargs):
        """
        Return affine transformation of vector `u` by matrix `F`.

        Args:
            F: transformation matrix

        Keyword Args:
            norm: normalize transformed vectors. [True or False] Default False

        Returns:
            vector representation of affine transformation (dot product)
            of `self` by `F`

        Example:
            # Reflexion of `y` axis.
            >>> F = [[1, 0, 0], [0, -1, 0], [0, 0, 1]]
            >>> u = Vector3([1, 1, 1])
            >>> u.transform(F)
            V(1.000, -1.000, 1.000)

        """
        r = np.dot(self, np.linalg.inv(F))
        if kwargs.get("norm", False):
            r = r.normalized()
        return type(self)(r)


class Pair:
    """
    The class to store pair of planar and linear feature.

    When ``Pair`` object is created, both planar and linear feature are
    adjusted, so linear feature perfectly fit onto planar one. Warning
    is issued, when misfit angle is bigger than 20 degrees.

    Args:
        fazi (float): dip azimuth of planar feature in degrees
        finc (float): dip of planar feature in degrees
        lazi (float): plunge direction of linear feature in degrees
        linc (float): plunge of linear feature in degrees

    Example:
        >>> p = Pair(140, 30, 110, 26)

    """

    def __init__(self, *args):
        if len(args) == 0:
            fvec, lvec = Vector3(0, 0, 1), Vector3(1, 0, 0)
        elif len(args) == 1 and is_like_vec3(args[0]):
            f = Fol(args[0])
            fvec, lvec = Vector3(f), f.dipvec()
        elif len(args) == 2:
            fvec, lvec = Vector3(args[0]), Vector3(args[1])
        else:
            raise TypeError("Not valid arguments for Fol")

        misfit = 90 - fvec.angle(lvec)
        if misfit > 20:
            warnings.warn(f"Warning: Misfit angle is {misfit:.1f} degrees.")
        ax = fvec.cross(lvec)
        ang = (lvec.angle(fvec) - 90) / 2
        self.fvec = fvec.rotate(ax, ang)
        self.lvec = lvec.rotate(ax, -ang)
        self.misfit = misfit

    def __repr__(self):
        fazi, finc = vec2geo_planar(self.fvec)
        lazi, linc = vec2geo_linear(self.lvec)
        return f"P:{fazi:.0f}/{finc:.0f}-{lazi:.0f}/{linc:.0f}"

    def __eq__(self, other):
        """
        Return `True` if pairs are equal, otherwise `False`.
        """
        if isinstance(other, self.__class__):
            return False
        return (self.fol == other.fol) and (self.lin == other.lin)

    def __ne__(self, other):
        """
        Return `True` if pairs are not equal, otherwise `False`.

        """
        return not self == other

    @classmethod
    def random(cls):
        """
        Random Pair
        """

        lin, p = Vector3.random(), Vector3.random()
        fol = lin.cross(p)
        return cls(fol, lin)

    def rotate(self, axis, phi):
        """Rotates ``Pair`` by angle `phi` about `axis`.

        Args:
            axis (``Vector3``): axis of rotation
            phi (float): angle of rotation in degrees

        Example:
            >>> p = Pair(Fol(140, 30), Lin(110, 26))
            >>> p.rotate(Lin(40, 50), 120)
            P:210/83-287/60

        """
        return type(self)(self.fvec.rotate(axis, phi), self.lvec.rotate(axis, phi))

    @property
    def type(self):
        return type(self)

    @property
    def fol(self):
        """
        Return a planar feature of ``Pair`` as ``Fol``.
        """
        return Fol(self.fvec)

    @property
    def lin(self):
        """
        Return a linear feature of ``Pair`` as ``Lin``.
        """
        return Lin(self.lvec)

    @property
    def rax(self):
        """
        Return an oriented vector perpendicular to both ``Fol`` and ``Lin``.
        """
        return self.fvec.cross(self.lvec)

    def transform(self, F, **kwargs):
        """Return an affine transformation of ``Pair`` by matrix `F`.

        Args:
            F (``DefGrad`` or ``numpy.array``): transformation matrix

        Keyword Args:
            norm: normalize transformed vectors. True or False. Default False

        Returns:
            representation of affine transformation (dot product) of `self`
            by `F`

        Example:
          >>> F = [[1, 0, 0], [0, 1, 1], [0, 0, 1]]
          >>> p = Pair(90, 90, 0, 50)
          >>> p.transform(F)
          P:90/45-50/37

        """
        lvec = Vector3(np.dot(F, self.lvec))
        fvec = Vector3(np.dot(self.fvec, np.linalg.inv(F)))
        if kwargs.get("norm", False):
            lvec = lvec.normalized()
            fvec = fvec.normalized()
        return type(self)(fvec, lvec)

    def H(self, other):
        """
        Return ``DefGrad`` rotational matrix H which rotate ``Pair``
        to other ``Pair``.

        Args:
            other (``Pair``): other pair

        Returns:
            ``Defgrad`` rotational matrix

        Example:
            >>> p1 = Pair(58, 36, 81, 34)
            >>> p2 = Pair(217,42, 162, 27)
            >>> p1.transform(p1.H(p2)) == p2
            True

        """
        from apsg.tensors import DefGrad

        return DefGrad(DefGrad.from_pair(other) * DefGrad.from_pair(self).I)


### NOTATION TRANSORMATIONS ###


def fol2vec_dd(azi, inc):
    return -cosd(azi) * sind(inc), -sind(azi) * sind(inc), cosd(inc)


def fol2vec_rhr(strike, dip):
    return fol2vec_dd(strike + 90, dip)


def geo2vec_planar(*args):
    return {"dd": fol2vec_dd,
            "rhr": fol2vec_rhr
            }[apsg_conf['notation']](*args)

##############################

def lin2vec_dd(azi, inc):
    return cosd(azi) * cosd(inc), sind(azi) * cosd(inc), sind(inc)


def geo2vec_linear(*args):
    return {"dd": lin2vec_dd,
            "rhr": lin2vec_dd
            }[apsg_conf['notation']](*args)

##############################

def vec2fol_dd(v):
    n = v.uv()
    if n.z < 0:
        n = -n
    return (atan2d(n.y, n.x) + 180) % 360, 90 - asind(n.z)

def vec2fol_rhr(v):
    n = v.uv()
    if n.z < 0:
        n = -n
    return (atan2d(n.y, n.x) + 90) % 360, 90 - asind(n.z)


def vec2geo_planar(arg):
    return {"dd": vec2fol_dd,
            "rhr": vec2fol_rhr
            }[apsg_conf['notation']](arg)

##############################

def vec2lin_dd(v):
    n = v.uv()
    if n.z < 0:
        n = -n
    return atan2d(n.y, n.x) % 360, asind(n.z)


def vec2geo_linear(arg):
    return {"dd": vec2lin_dd,
            "rhr": vec2lin_dd
            }[apsg_conf['notation']](arg)

