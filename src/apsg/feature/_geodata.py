import warnings
import numpy as np

from apsg.config import apsg_conf
from apsg.helpers import sind, cosd, tand, asind, acosd, atand, atan2d
from apsg.helpers import geo2vec_linear, vec2geo_linear, geo2vec_planar, vec2geo_planar
from apsg.decorator import ensure_first_arg_same
from apsg.math import Vector3, Axial3


"""
to to
"""

class Lineation(Axial3):

    def __repr__(self):
        azi, inc = vec2geo_linear(self)
        return f"L:{azi:.0f}/{inc:.0f}"

    def cross(self, other):
        return Foliation(super().cross(other))

    __pow__ = cross


class Foliation(Axial3):

    def __init__(self, *args):
        if len(args) == 0:
            coords = (0, 0, 1)
        elif len(args) == 1 and np.asarray(args[0]).shape == Foliation.__shape__:
            coords = [float(v) for v in args[0]]
        elif len(args) == 2:
            coords = geo2vec_planar(*args)
        elif len(args) == 3:
            coords = [float(v) for v in args]
        else:
            raise TypeError("Not valid arguments for Foliation")
        self._coords = tuple(coords)

    def __repr__(self):
        azi, inc = vec2geo_planar(self)
        return f"S:{azi:.0f}/{inc:.0f}"

    def cross(self, other):
        return Lineation(super().cross(other))

    __pow__ = cross

    def dipvec(self):
        return Vector3(*vec2geo_planar(self))

    def rake(self, rake):
        return Vector3(self.dipvec().rotate(self, rake - 90))

    @ensure_first_arg_same
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
        elif len(args) == 1:
            f = Foliation(args[0])
            fvec, lvec = Vector3(f), f.dipvec()
        elif len(args) == 2:
            fvec, lvec = Vector3(args[0]), Vector3(args[1])
        else:
            raise TypeError("Not valid arguments for Foliation")

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
            >>> p = Pair(fol(140, 30), lin(110, 26))
            >>> p.rotate(lin(40, 50), 120)
            P:210/83-287/60

        """
        return type(self)(self.fvec.rotate(axis, phi), self.lvec.rotate(axis, phi))

    @property
    def type(self):
        return type(self)

    @property
    def fol(self):
        """
        Return a planar feature of ``Pair`` as ``Foliation``.
        """
        return Foliation(self.fvec)

    @property
    def lin(self):
        """
        Return a linear feature of ``Pair`` as ``Lineation``.
        """
        return Lineation(self.lvec)

    @property
    def rax(self):
        """
        Return an oriented vector perpendicular to both ``Foliation`` and ``Lineation``.
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
