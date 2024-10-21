import warnings
import numpy as np

from apsg.helpers._notation import (
    geo2vec_planar,
    vec2geo_planar,
    vec2geo_planar_signed,
    vec2geo_linear,
    vec2geo_linear_signed,
)
from apsg.decorator._decorator import ensure_first_arg_same, ensure_arguments
from apsg.math._vector import Vector3, Axial3


class Lineation(Axial3):
    """
    A class to represent axial (non-oriented) linear feature (lineation).

    There are different way to create ``Lineation`` object:

    - without arguments create default ``Lineation`` L:0/0
    - with single argument `l`, where:

        - `l` could be Vector3-like object
        - `l` could be string 'x', 'y' or 'z' - principal axes of coordinate system
        - `l` could be tuple of (x, y, z) - vector components
    - with 2 arguments plunge direction and plunge
    - with 3 numerical arguments defining vector components

    Args:
        azi (float): plunge direction of linear feature in degrees
        inc (float): plunge of linear feature in degrees

    Example:
        >>> lin()
        >>> lin('y')
        >>> lin(1,2,-1)
        >>> l = lin(110, 26)
    """

    def __repr__(self):
        azi, inc = self.geo
        return f"L:{azi:.0f}/{inc:.0f}"

    def cross(self, other):
        """Return Foliation defined by two linear features"""
        return Foliation(super().cross(other))

    __pow__ = cross

    @property
    def geo(self):
        """Return tuple of plunge direction and plunge"""
        return vec2geo_linear(self)

    def to_json(self):
        """Return as JSON dict"""
        azi, inc = vec2geo_linear_signed(self)
        return {"datatype": type(self).__name__, "args": (azi, inc)}


class Foliation(Axial3):
    """
    A class to represent non-oriented planar feature (foliation).

    There are different way to create ``Foliation`` object:

    - without arguments create default ``Foliation`` S:180/0
    - with single argument `f`, where:

        - `f` could be Vector3-like object
        - `f` could be string 'x', 'y' or 'z' - principal planes of coordinate system
        - `f` could be tuple of (x, y, z) - vector components
    - with 2 arguments follows active notation. See apsg_conf["notation"]
    - with 3 numerical arguments defining vector components of plane normal

    Args:
        azi (float): dip direction (or strike) of planar feature in degrees.
        inc (float): dip of planar feature in degrees

    Example:
        >>> fol()
        >>> fol('y')
        >>> fol(1,2,-1)
        >>> f = fol(250, 30)

    """

    def __init__(self, *args):
        if len(args) == 0:
            coords = (0, 0, 1)
        elif len(args) == 1:
            if np.asarray(args[0]).shape == Foliation.__shape__:
                coords = np.asarray(args[0])
            elif isinstance(args[0], str):
                if args[0].lower() == "x":
                    coords = (1, 0, 0)
                elif args[0].lower() == "y":
                    coords = (0, 1, 0)
                elif args[0].lower() == "z":
                    coords = (0, 0, 1)
                else:
                    raise TypeError(f"Not valid arguments for {type(self).__name__}")
            else:
                raise TypeError(f"Not valid arguments for {type(self).__name__}")
        elif len(args) == 2:
            coords = geo2vec_planar(*args)
        elif len(args) == 3:
            coords = [float(v) for v in args]
        else:
            raise TypeError("Not valid arguments for Foliation")
        self._coords = tuple(coords)

    def __repr__(self):
        azi, inc = self.geo
        return f"S:{azi:.0f}/{inc:.0f}"

    def cross(self, other):
        """Return Lineation defined by intersection of planar features"""
        return Lineation(super().cross(other))

    __pow__ = cross

    @property
    def geo(self):
        """Return tuple of dip direction and dip"""
        return vec2geo_planar(self)

    def to_json(self):
        """Return as JSON dict"""
        azi, inc = vec2geo_planar_signed(self)
        return {"datatype": type(self).__name__, "args": (azi, inc)}

    def dipvec(self):
        """Return dip vector"""
        return Vector3(*vec2geo_planar(self))

    def pole(self):
        """Return plane normal as vector"""
        return Vector3(self)

    def rake(self, rake):
        """Return rake vector"""
        return Vector3(self.dipvec().rotate(self, rake - 90))

    def transform(self, F, **kwargs):
        """
        Return affine transformation by matrix `F`.

        Args:
            F: transformation matrix

        Keyword Args:
            norm: normalize transformed ``Foliation``. [True or False] Default False

        Returns:
            vector representation of affine transformation (dot product)
            of `self` by `F`

        Example:
            >>> # Reflexion of `y` axis.
            >>> F = [[1, 0, 0], [0, -1, 0], [0, 0, 1]]
            >>> f = fol(45, 20)
            >>> f.transform(F)
            S:315/20
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

    There are different way to create ``Pair`` object:

    - without arguments create default Pair with fol(0,0) and lin(0,0)
    - with single argument `p`, where:

        - `p` could be Pair
        - `p` could be tuple of (fazi, finc, lazi, linc)
        - `p` could be tuple of (fx, fy ,fz, lx, ly, lz)
    - with 2 arguments f and l could be Vector3 like objects,
      e.g. Foliation and Lineation
    - with four numerical arguments defining `fol(fazi, finc)` and `lin(lazi, linc)`

    Args:
        fazi (float): dip azimuth of planar feature in degrees
        finc (float): dip of planar feature in degrees
        lazi (float): plunge direction of linear feature in degrees
        linc (float): plunge of linear feature in degrees

    Attributes:
        fvec (Vector3): corrected vector normal to plane
        lvec (Vector3): corrected vector of linear feature

    Example:
        >>> pair()
        >>> pair(p)
        >>> pair(f, l)
        >>> pair(fazi, finc, lazi, linc)
        >>> p = pair(140, 30, 110, 26)

    """

    __slots__ = ("fvec", "lvec", "misfit")
    __shape__ = (6,)

    def __init__(self, *args):
        if len(args) == 0:
            fvec, lvec = Vector3(0, 0, 1), Vector3(1, 0, 0)
        elif len(args) == 1 and issubclass(type(args[0]), Pair):
            fvec, lvec = args[0].fvec, args[0].lvec
        elif len(args) == 1 and np.asarray(args[0]).shape == (4,):
            fazi, finc, lazi, linc = (float(v) for v in args[0])
            fvec, lvec = Foliation(fazi, finc), Lineation(lazi, linc)
        elif len(args) == 1 and np.asarray(args[0]).shape == Pair.__shape__:
            fvec, lvec = Vector3(args[0][:3]), Vector3(args[0][-3:])
        elif len(args) == 2:
            if issubclass(type(args[0]), Vector3) and issubclass(
                type(args[1]), Vector3
            ):
                fvec, lvec = args
            else:
                raise TypeError("Not valid arguments for Pair")
        elif len(args) == 4:
            fvec = Foliation(args[0], args[1])
            lvec = Lineation(args[2], args[3])
        else:
            raise TypeError("Not valid arguments for Pair")

        fvec = Vector3(fvec)
        lvec = Vector3(lvec)
        misfit = abs(90 - fvec.angle(lvec))
        if misfit > 20:
            warnings.warn(f"Warning: Misfit angle is {misfit:.1f} degrees.")
        ax = fvec.cross(lvec)
        ang = (lvec.angle(fvec) - 90) / 2
        self.fvec = Vector3(fvec.rotate(ax, ang))
        self.lvec = Vector3(lvec.rotate(ax, -ang))
        self.misfit = misfit

    def __repr__(self):
        fazi, finc = self.fol.geo
        lazi, linc = self.lin.geo
        return f"P:{fazi:.0f}/{finc:.0f}-{lazi:.0f}/{linc:.0f}"

    @ensure_first_arg_same
    def __eq__(self, other):
        """
        Return `True` if pairs are equal, otherwise `False`.
        """
        return (self.fvec == other.fvec) and (self.lvec == other.lvec)

    def __ne__(self, other):
        """
        Return `True` if pairs are not equal, otherwise `False`.

        """
        return not self == other

    def __array__(self, dtype=None, copy=None):
        return np.hstack((self.fvec, self.lvec)).astype(dtype)

    def label(self):
        """Return label"""
        return str(self)

    def to_json(self):
        """Return as JSON dict"""
        fazi, finc = vec2geo_planar_signed(self.fvec)
        lazi, linc = vec2geo_linear_signed(self.lvec)
        return {"datatype": type(self).__name__, "args": (fazi, finc, lazi, linc)}

    @classmethod
    def random(cls):
        """
        Random Pair
        """

        lin, p = Vector3.random(), Vector3.random()
        fol = lin.cross(p)
        return cls(fol, lin)

    @ensure_arguments(Vector3)
    def rotate(self, axis, phi):
        """Rotates ``Pair`` by angle `phi` about `axis`.

        Args:
            axis (``Vector3``): axis of rotation
            phi (float): angle of rotation in degrees

        Example:
            >>> p = pair(fol(140, 30), lin(110, 26))
            >>> p.rotate(lin(40, 50), 120)
            P:210/83-287/60

        """
        return type(self)(self.fvec.rotate(axis, phi), self.lvec.rotate(axis, phi))

    @property
    def rax(self):
        return self.lvec.cross(self.fvec)

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

    def transform(self, F, **kwargs):
        """Return an affine transformation of ``Pair`` by matrix `F`.

        Args:
            F: transformation matrix

        Keyword Args:
            norm: normalize transformed vectors. True or False. Default False

        Returns:
            representation of affine transformation (dot product) of `self`
            by `F`

        Example:
          >>> F = defgrad.from_axisangle(lin(0,0), 60)
          >>> p = pair(90, 90, 0, 50)
          >>> p.transform(F)
          P:270/30-314/23

        """

        fvec = self.fol.transform(F)
        lvec = self.lin.transform(F)
        if kwargs.get("norm", False):
            fvec = fvec.normalized()
            lvec = lvec.normalized()
        return type(self)(fvec, lvec)


class Fault(Pair):
    """
    The class to store ``Pair`` with associated sense of movement.

    When ``Fault`` object is created, both planar and linear feature are
    adjusted, so linear feature perfectly fit onto planar one. Warning
    is issued, when misfit angle is bigger than 20 degrees.

    There are different way to create ``Fault`` object:

    - without arguments create default ``Fault`` with `fol(0,0)` and `lin(0,0)`
    - with single argument `p`:

        - `p` could be Fault
        - `p` could be tuple of (fazi, finc, lazi, linc, sense)
        - `p` could be tuple of (fx, fy ,fz, lx, ly, lz)
    - with 2 arguments p (Pair object) and sense
    - with 3 arguments f, l (Vector3 like objects), e.g. ``Foliation``
      and ``Lineation`` and sense
    - with 5 numerical arguments defining `fol(fazi, finc)`, `lin(lazi, linc)` and sense

    Args:
        fazi (float): dip azimuth of planar feature in degrees
        finc (float): dip of planar feature in degrees
        lazi (float): plunge direction of linear feature in degrees
        linc (float): plunge of linear feature in degrees
        sense (float or str): sense of movement +/11 hanging-wall down/up. When str,
            ,ust be one of 's', 'd', 'n', 'r'.


    Attributes:
        fvec (Vector3): corrected vector normal to plane
        lvec (Vector3): corrected vector of linear feature
        sense (int): sense of movement (+/-1)

    Example:
        >>> f = fault(140, 30, 110, 26, -1)
        >>> f = fault(140, 30, 110, 26, 'r')
        >>> p = pair(140, 30, 110, 26)
        >>> f = fault(p, 'n')
        >>> f = fault(fol(120, 80), lin(32, 10), 's')

    """

    __shape__ = (7,)

    def __init__(self, *args):
        if len(args) == 0:
            fvec, lvec = Vector3(0, 0, 1), Vector3(1, 0, 0)
        elif len(args) == 1 and np.asarray(args[0]).shape == (5,):
            fazi, finc, lazi, linc, sense = (float(v) for v in args[0])
            fvec, lvec = Foliation(fazi, finc), Lineation(lazi, linc)
            sense = self.calc_sense(fvec, lvec, sense)
            if sense < 0:
                lvec = -lvec
        elif len(args) == 1 and issubclass(type(args[0]), Pair):
            fvec, lvec = args[0].fvec, args[0].lvec
        elif len(args) == 2 and issubclass(type(args[0]), Pair):
            fvec, lvec = args[0].fvec, args[0].lvec
            sense = self.calc_sense(fvec, lvec, args[1])
            georax = lvec.lower().cross(fvec.lower())
            if args[0].rax == georax and sense < 0:
                lvec = -lvec
        elif len(args) == 2:
            if issubclass(type(args[0]), Vector3) and issubclass(
                type(args[1]), Vector3
            ):
                fvec, lvec = args[0], args[1]
        elif len(args) == 3:
            if issubclass(type(args[0]), Vector3) and issubclass(
                type(args[1]), Vector3
            ):
                fvec, lvec = args[0], args[1]
                sense = self.calc_sense(fvec, lvec, args[2])
                rax = lvec.cross(fvec)
                georax = lvec.lower().cross(fvec.lower())
                if rax == georax and sense < 0:
                    lvec = -lvec
        elif len(args) == 5:
            fvec = Foliation(args[0], args[1])
            lvec = Lineation(args[2], args[3])
            sense = self.calc_sense(fvec, lvec, args[4])
            if sense < 0:
                lvec = -lvec
        else:
            raise TypeError("Not valid arguments for Fault")
        super().__init__(fvec, lvec)

    @classmethod
    def calc_sense(cls, fvec, lvec, sense):
        if isinstance(sense, int) or isinstance(sense, float):
            return sense
        elif isinstance(sense, str):
            p = Pair(fvec, lvec)
            if sense.lower() == "s":
                if p.rax == p.rax.lower():
                    res = -1
                else:
                    res = 1
            elif sense.lower() == "d":
                if p.rax == p.rax.lower():
                    res = 1
                else:
                    res = -1
            elif sense.lower() == "n":
                res = 1
            elif sense.lower() == "r":
                res = -1
            return res

    def __repr__(self):
        fazi, finc = self.fol.geo
        lazi, linc = self.lin.geo
        schar = [" ", "+", "-"][self.sense]
        return f"F:{fazi:.0f}/{finc:.0f}-{lazi:.0f}/{linc:.0f} {schar}"

    @ensure_first_arg_same
    def __eq__(self, other):
        """
        Return `True` if pairs are equal, otherwise `False`.
        """
        return (
            (self.fvec == other.fvec)
            and (self.lvec == other.lvec)
            and (self.sense == other.sense)
        )

    def __ne__(self, other):
        """
        Return `True` if pairs are not equal, otherwise `False`.

        """
        return not self == other

    def __array__(self, dtype=None, copy=None):
        return np.hstack((self.fvec, self.lvec, self.sense)).astype(dtype)

    def to_json(self):
        """Return as JSON dict"""
        fazi, finc = vec2geo_planar_signed(self.fvec)
        lazi, linc = vec2geo_linear_signed(self.lvec)
        return {
            "datatype": type(self).__name__,
            "args": (fazi, finc, lazi, linc, self.sense),
        }

    @classmethod
    def random(cls):
        """
        Random Fault
        """
        import random

        lvec, p = Vector3.random(), Vector3.random()
        fvec = lvec.cross(p)
        return cls(fvec, lvec, random.choice([-1, 1]))

    @property
    def georax(self):
        return self.lvec.lower().cross(self.fvec.lower())

    @property
    def sense(self):
        if self.rax == self.georax:
            return 1
        else:
            return -1

    def p_vector(self, ptangle=90):
        """Return P axis as ``Vector3``"""
        return self.fvec.rotate(self.lvec.cross(self.fvec), -ptangle / 2)

    def t_vector(self, ptangle=90):
        """Return T-axis as ``Vector3``."""
        return self.fvec.rotate(self.lvec.cross(self.fvec), +ptangle / 2)

    @property
    def p(self):
        """Return P-axis as ``Lineation``"""
        return Lineation(self.p_vector())

    @property
    def t(self):
        """Return T-axis as ``Lineation``"""
        return Lineation(self.t_vector())

    @property
    def m(self):
        """Return kinematic M-plane as ``Foliation``"""
        return Foliation(self.lvec.cross(self.fvec))

    @property
    def d(self):
        """Return dihedra plane as ``Fol``"""
        return Foliation(self.lvec.cross(self.fvec).cross(self.fvec))


class Cone:
    """
    The class to store cone with given axis, secant line and revolution angle
    in degrees.

    There are different way to create ``Cone`` object according to number
    of arguments:

    - without args, you can create default``Cone`` with axis ``lin(0, 90)``,
      secant ``lin(0, 0)`` angle 360°
    - with single argument `c`, where `c` could be ``Cone``, 5-tuple of
      `(aazi, ainc, sazi, sinc, revangle)` or 7-tuple of
      `(ax, ay ,az, sx, sy, sz, revangle)`
    - with 3 arguments, where axis and secant line could be Vector3 like objects,
      e.g. Lineation and third argument is revolution angle
    - with 5 arguments defining axis `lin(aazi, ainc)`, secant line
      `lin(sazi, sinc)` and angle of revolution

    Attributes:
        axis (Vector3): axis of the cone
        secant (Vector3): secant line
        revangle (float): revolution angle

    Example:
        >>> cone()
        >>> cone(c)
        >>> cone(a, s, revangle)
        >>> cone(aazi, ainc, sazi, sinc, revangle)
        >>> c = cone(140, 30, 110, 26, 360)

    """

    __slots__ = ("axis", "secant", "revangle")
    __shape__ = (7,)

    def __init__(self, *args):
        if len(args) == 0:
            axis, secant, revangle = Vector3(0, 0, 1), Vector3(1, 0, 0), 360
        elif len(args) == 1 and issubclass(type(args[0]), Cone):
            axis, secant, revangle = args[0].axis, args[0].secant, args[0].revangle
        elif len(args) == 1 and np.asarray(args[0]).shape == (5,):
            aazi, ainc, sazi, sinc, revangle = (float(v) for v in args[0])
            axis, secant = Lineation(aazi, ainc), Lineation(sazi, sinc)
        elif len(args) == 1 and np.asarray(args[0]).shape == Cone.__shape__:
            axis, secant, revangle = (
                Vector3(args[0][:3]),
                Vector3(args[0][3:6]),
                args[0][-1],
            )
        elif len(args) == 2:
            if issubclass(type(args[0]), Vector3) and issubclass(
                type(args[1]), Vector3
            ):
                axis, secant = args
                revangle = 360
            elif issubclass(type(args[0]), Vector3) and np.isscalar(args[1]):
                axis = args[0]
                azi, inc = axis.geo
                secant = Vector3(azi, inc + args[1])
                revangle = 360
            else:
                raise TypeError("Not valid arguments for Cone")
        elif len(args) == 3:
            if issubclass(type(args[0]), Vector3) and issubclass(
                type(args[1]), Vector3
            ):
                axis, secant, revangle = args
            else:
                raise TypeError("Not valid arguments for Cone")
        elif len(args) == 4:
            axis = Lineation(args[0], args[1])
            secant = Lineation(args[2], args[3])
            revangle = 360
        elif len(args) == 5:
            axis = Lineation(args[0], args[1])
            secant = Lineation(args[2], args[3])
            revangle = args[4]
        else:
            raise TypeError("Not valid arguments for Cone")

        self.axis = Vector3(axis)
        self.secant = Vector3(secant)
        self.revangle = float(revangle)
        if self.axis.angle(self.secant) > 90:
            self.secant = -self.secant

    def __repr__(self):
        azi, inc = vec2geo_linear(self.axis)
        return f"C:{azi:.0f}/{inc:.0f} [{self.apical_angle():g}]"

    @ensure_first_arg_same
    def __eq__(self, other):
        return (
            (self.axis == other.axis)
            and (self.secant == other.secant)
            and (self.revangle == other.revangle)
        )

    def __ne__(self, other):
        return not self == other

    def __array__(self, dtype=None, copy=None):
        return np.hstack((self.axis, self.secant, self.revangle)).astype(dtype)

    def label(self):
        """Return label"""
        return str(self)

    def to_json(self):
        """Return as JSON dict"""
        aazi, ainc = vec2geo_linear_signed(self.axis)
        sazi, sinc = vec2geo_linear_signed(self.secant)
        return {
            "datatype": type(self).__name__,
            "args": (aazi, ainc, sazi, sinc, self.revangle),
        }

    @classmethod
    def random(cls):
        """
        Random Cone
        """

        axis, secant = Vector3.random(), Vector3.random()
        return cls(axis, secant, 360)

    @ensure_arguments(Vector3)
    def rotate(self, axis, phi):
        """Rotates ``Cone`` by angle `phi` about `axis`.

        Args:
            axis (``Vector3``): axis of rotation
            phi (float): angle of rotation in degrees

        Example:
            >>> c = cone(lin(140, 30), lin(110, 26), 360)
            >>> c.rotate(lin(40, 50), 120)
            C:210/83-287/60

        """
        return type(self)(
            self.axis.rotate(axis, phi), self.secant.rotate(axis, phi), self.revangle
        )

    def apical_angle(self):
        """Return apical angle"""
        return self.axis.angle(self.secant)

    @property
    def rotated_secant(self):
        """Return revangle rotated secant vector"""
        return self.secant.rotate(self.axis, self.revangle)
