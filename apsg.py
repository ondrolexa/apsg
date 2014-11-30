# -*- coding: utf-8 -*-
"""
Python module to manipulate, analyze and visualize structural geology data

"""

# import modulu
from __future__ import division, print_function
from copy import deepcopy
import sqlite3
import numpy as np
import matplotlib.pyplot as plt
try:
    import mplstereonet
except ImportError:
    pass

# lambda funkce
sind = lambda x: np.sin(np.deg2rad(x))
cosd = lambda x: np.cos(np.deg2rad(x))
asind = lambda x: np.rad2deg(np.arcsin(x))
acosd = lambda x: np.rad2deg(np.arccos(x))
atan2d = lambda x1, x2: np.rad2deg(np.arctan2(x1, x2))
getldd = lambda x, y: (atan2d(x, y) % 360, 90-2*asind(np.sqrt((x*x + y*y)/2)))
getfdd = lambda x, y: (atan2d(-x, -y) % 360, 2*asind(np.sqrt((x*x + y*y)/2)))
getldc = lambda u, v: (cosd(u)*cosd(v), sind(u)*cosd(v), sind(v))
getfdc = lambda u, v: (-cosd(u)*sind(v), -sind(u)*sind(v), cosd(v))


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
        return Vec3(getldc(azi, inc)).view(cls)

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

    @property
    def rhr(self):
        return self.dd


class Fol(Vec3):
    """Class for planar features
    """
    def __new__(cls, azi, inc):
        # casting to our class
        return Vec3(getfdc(azi, inc)).view(cls)

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


class Group(list):
    """Group class
    Group is homogeneous group of Vec3, Fol or Lin
    """
    def __init__(self, data,
                 name='Default',
                 color='blue',
                 fol={'lw': 1, 'ls': '-'},
                 lin={'marker': 'o', 's': 20},
                 vec={'marker': 'd', 's': 24, 'facecolors': None},
                 tmpl=None):
        if not issubclass(type(data), list):
            data = [data]
        assert len(data) > 0, 'Empty group is not allowed.'
        tp = type(data[0])
        assert issubclass(tp, Vec3), \
               'Data must be Fol, Lin or Vec3 type.'
        assert all([isinstance(e, tp) for e in data]), \
               'All data in group must be of same type.'
        super(Group, self).__init__(data)
        self.type = tp
        if tmpl is None:
            self.name = name
            self.color = color
            self.sym = {}
            self.sym['fol'] = fol
            self.sym['lin'] = lin
            self.sym['vec'] = vec
        else:
            self.name = tmpl.name
            self.color = tmpl.color
            self.sym = tmpl.sym

    def __repr__(self):
        return '%s: %g %s' % (self.name, len(self), self.type.__name__)

    def __add__(self, other):
        # merge Datasets
        assert isinstance(other, Group), 'Only groups could be merged'        
        assert self.type is other.type, 'Only same type groups could be merged'
        return Group(list(self) + other, tmpl=self)

    def __setitem__(self, key, value):
        assert isinstance(value, self.type), 'item is not of type %s' % self.type.__name__
        super(Group, self).__setitem__(key, value)

    def append(self, item):
        assert isinstance(item, self.type), 'item is not of type %s' % self.type.__name__
        super(Group, self).append(item)

    def extend(self, items = ()):
        for item in items:
            self.append(item)

    @classmethod
    def fromcsv(cls, fname, typ=Lin, acol=1, icol=2,
                name='Default', color='blue'):
        """Read dataset from csv"""
        import csv
        with open(fname, 'rb') as csvfile:
            sniffer = csv.Sniffer()
            dialect = sniffer.sniff(csvfile.read(1024))
            csvfile.seek(0)
            data = []
            reader = csv.reader(csvfile, dialect)
            if sniffer.has_header:
                reader.next()
            for row in reader:
                if len(row) > 1:
                    data.append(typ(float(row[acol-1]), float(row[icol-1])))
            return cls(data, name=name, color=color)

    @property
    def aslin(self):
        """Convert all data in Group to Lin"""
        return Group([e.aslin for e in self], tmpl=self)

    @property
    def asfol(self):
        """Convert all data in Group to Fol"""
        return Group([e.asfol for e in self], tmpl=self)

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
        return Group(res, tmpl=self)

    def rotate(self, axis, phi):
        """rotate Group"""
        return Group([e.rotate(axis, phi) for e in self], tmpl=self)

    def center(self):
        """rotate E3 direction of Group to vertical"""
        ot = self.ortensor
        azi, inc = ot.eigenlins[2][0].dd
        return self.rotate(Lin(azi - 90, 0), 90 - inc)

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
        return Group([e.transform(F) for e in self], tmpl=self)

    @property
    def dd(self):
        """array of dip directions and dips of Group"""
        return np.array([d.dd for d in self]).T

    @property
    def rhr(self):
        """array of strikes and dips of Group"""
        return np.array([d.rhr for d in self]).T

    def density(self, k=100, npoints=180):
        """calculate density of Group"""
        return Density(self, k, npoints)

    def plot(self):
        """Show Group on Schmidt net"""
        return SchmidtNet(self)

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


class Datasource(object):
    """PySDB database access class"""
    TESTSEL = "SELECT sites.id, sites.name, sites.x_coord, sites.y_coord, \
    sites.description, structdata.id, structdata.id_sites, \
    structdata.id_structype, structdata.azimuth, structdata.inclination, \
    structype.id, structype.structure, structype.description, \
    structype.structcode, structype.groupcode  \
    FROM sites \
    INNER JOIN structdata ON sites.id = structdata.id_sites \
    INNER JOIN structype ON structype.id = structdata.id_structype \
    LIMIT 1"
    STRUCTSEL = "SELECT structype.structure  \
    FROM sites  \
    INNER JOIN structdata ON sites.id = structdata.id_sites  \
    INNER JOIN structype ON structype.id = structdata.id_structype  \
    INNER JOIN units ON units.id = sites.id_units  \
    GROUP BY structype.structure  \
    ORDER BY structype.structure ASC"
    SELECT = "SELECT structdata.azimuth, structdata.inclination   \
    FROM sites   \
    INNER JOIN structdata ON sites.id = structdata.id_sites   \
    INNER JOIN structype ON structype.id = structdata.id_structype   \
    INNER JOIN units ON units.id = sites.id_units"

    def __new__(cls, db=None):
        try:
            cls.con = sqlite3.connect(db)
            cls.con.execute("pragma encoding='UTF-8'")
            cls.con.execute(Datasource.TESTSEL)
            print("Connected. PySDB version: %s" % cls.con.execute("SELECT value FROM meta WHERE name='version'").fetchall()[0][0])
            return super(Datasource, cls).__new__(cls)
        except sqlite3.Error as e:
            print("Error %s:" % e.args[0])
            raise sqlite3.Error

    def execsql(self, sql):
        return self.con.execute(sql).fetchall()

    @property
    def structures(self):
        return [el[0] for el in self.execsql(Datasource.STRUCTSEL)]

    def select(self, struct=None):
        tpsel = "SELECT planar FROM structype WHERE structure='%s'" % struct
        dtsel = Datasource.SELECT + " WHERE structype.structure='%s'" % struct
        tp = self.execsql(tpsel)[0][0]
        if tp:
            res = Group([Fol(el[0], el[1]) for el in self.execsql(dtsel)], name=struct)
        else:
            res = Group([Lin(el[0], el[1]) for el in self.execsql(dtsel)], name=struct)
        return res


class DefGrad(np.ndarray):
    """class to store deformation gradient tensor derived from numpy.ndarray
    """
    def __new__(cls, array):
        # casting to our class
        assert np.shape(array) == (3,3), 'DefGrad must be 3x3 2D array'
        obj = np.asarray(array).view(cls)
        return obj

    def __repr__(self):
        return 'DefGrad:\n' + str(self)

    def __mul__(self, other):
        assert np.shape(other) == (3,3), 'DefGrad could by multiplied with 3x3 2D array'
        return np.dot(self, other)

    def __pow__(self, n):
        # cross product or power of magnitude
        assert np.isscalar(n), 'Exponent must be integer.'
        return np.linalg.matrix_power(self, n)

    def __eq__(self, other):
        # equal
        return bool(np.sum(np.abs(self - other)) < 1e-14)

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
                [ x*xc+c,   xyc-zs,   zxc+ys ],
                [ xyc+zs,   y*yc+c,   yzc-xs ],
                [ zxc-ys,   yzc+xs,   z*zc+c ]])

    @classmethod
    def from_comp(cls,
                  xx=1, xy=0, xz=0,
                  yx=0, yy=1, yz=0,
                  zx=0, zy=0, zz=1):
        return cls([
                [ xx, xy, xz ],
                [ yx, yy, yz ],
                [ zx, zy, zz ]])

    @property
    def I(self):
        return np.linalg.inv(self)

    def rotate(self, vector, theta):
        R = DefGrad.from_axis(vector, theta)
        return R*self*R.T


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
        return e1 * Vec3(self.vects[0]),\
               e2 * Vec3(self.vects[1]),\
               e3 * Vec3(self.vects[2])

    @property
    def eigenlins(self):
        v1, v2, v3 = self.eigenvects
        d1 = Group(v1.aslin, name='E1', color='red')
        d2 = Group(v2.aslin, name='E2', color='magenta')
        d3 = Group(v3.aslin, name='E3', color='green')
        return d1, d2, d3

    @property
    def eigenfols(self):
        v1, v2, v3 = self.eigenvects
        d1 = Group(v1.asfol, name='E1', color='red')
        d2 = Group(v2.asfol, name='E2', color='magenta')
        d3 = Group(v3.asfol, name='E3', color='green')
        return d1, d2, d3


class StereoNet(object):
    """API to mplstereonet"""
    def __init__(self, *args, **kwargs):
        _, self._ax = mplstereonet.subplots(*args, **kwargs)
        self._grid_state = False
        self._cax = None
        self._lgd = None

    def draw(self):
        h,l = self._ax.get_legend_handles_labels()
        if h:
            self._lgd = self._ax.legend(h, l, bbox_to_anchor=(1.12, 1), loc=2, borderaxespad=0., numpoints=1, scatterpoints=1)
            plt.subplots_adjust(right=0.75)
        else:
            plt.subplots_adjust(right=0.9)
        plt.draw()

    def cla(self):
        self._ax.cla()
        self._ax.grid(self._grid_state)
        self._cax = None
        self._lgd = None
        self.draw()

    def grid(self, state=True):
        self._ax.grid(state)
        self._grid_state = state
        self.draw()

    def plane(self, obj, *args, **kwargs):
        assert obj.type is Fol, 'Only Fol instance could be plotted as plane.'
        strike, dip = obj.rhr
        self._ax.plane(strike, dip, *args, **kwargs)
        self.draw()

    def pole(self, obj, *args, **kwargs):
        assert obj.type is Fol, 'Only Fol instance could be plotted as pole.'
        strike, dip = obj.rhr
        self._ax.pole(strike, dip, *args, **kwargs)
        self.draw()

    def rake(self, obj, rake_angle, *args, **kwargs):
        assert obj.type is Fol, 'Only Fol instance could be used with rake.'
        strike, dip = obj.rhr
        self._ax.rake(strike, dip, rake_angle, *args, **kwargs)
        self.draw()

    def line(self, obj, *args, **kwargs):
        assert obj.type is Lin, 'Only Lin instance could be plotted as line.'
        bearing, plunge = obj.rhr
        self._ax.line(plunge, bearing, *args, **kwargs)
        self.draw()

    def cone(self, obj, angle, segments=100, bidirectional=True, **kwargs):
        assert obj.type is Lin, 'Only Lin instance could be used as cone axis.'
        bearing, plunge = obj.rhr
        self._ax.cone(plunge, bearing, angle, segments=segments,
                      bidirectional=bidirectional, **kwargs)
        self.draw()

    def density_contour(self, group, *args, **kwargs):
        assert type(group) is Group, 'Only group of data could be used for contouring.'
        if group.type is Lin:
            bearings, plunges = group.rhr
            kwargs['measurement'] = 'lines'
            self._cax = self._ax.density_contour(plunges, bearings, *args, **kwargs)
            plt.draw()
        elif group.type is Fol:
            strikes, dips = group.rhr
            kwargs['measurement'] = 'poles'
            self._cax = self._ax.density_contour(strikes, dips, *args, **kwargs)
            plt.draw()
        else:
            raise 'Only Fol or Lin group is allowed.'

    def density_contourf(self, group, *args, **kwargs):
        assert type(group) is Group, 'Only group of data could be used for contouring.'
        if group.type is Lin:
            bearings, plunges = group.rhr
            kwargs['measurement'] = 'lines'
            self._cax = self._ax.density_contourf(plunges, bearings, *args, **kwargs)
            plt.draw()
        elif group.type is Fol:
            strikes, dips = group.rhr
            kwargs['measurement'] = 'poles'
            self._cax = self._ax.density_contourf(strikes, dips, *args, **kwargs)
            plt.draw()
        else:
            raise 'Only Fol or Lin group is allowed.'

    def colorbar(self):
        if self._cax is not None:
            cbaxes = self._ax.figure.add_axes([0.015, 0.2, 0.02, 0.6]) 
            cb = plt.colorbar(self._cax, cax = cbaxes) 

    def savefig(self, filename='stereonet.pdf'):
        if self._lgd is None:
            self._ax.figure.savefig(filename)
        else:
            self._ax.figure.savefig(filename, bbox_extra_artists=(self._lgd,), bbox_inches='tight')

    def getlin(self):
        """get Lin by mouse click"""
        self.show()
        x, y = plt.ginput(1)[0]
        return Lin(*getldd(x, y))

    def getfol(self):
        """get Fol by mouse click"""
        self.show()
        x, y = plt.ginput(1)[0]
        return Fol(*getfdd(x, y))

    def getlins(self):
        """Collect Dataset of Lin by mouse clicks"""
        self.show()
        pts = plt.ginput(0, mouse_add=1, mouse_pop=2, mouse_stop=3)
        return Group([Lin(*getldd(x, y)) for x, y in pts])

    def getfols(self):
        """Collect Dataset of Fol by mouse clicks"""
        self.show()
        pts = plt.ginput(0, mouse_add=1, mouse_pop=2, mouse_stop=3)
        return Group([Fol(*getfdd(x, y)) for x, y in pts])


def fixpair(f, l):
    """Fix pair of planar and linear data, so Lin is within plane Fol::

        fok,lok = fixpair(f,l)
    """
    ax = f ** l
    ang = (Vec3(l).angle(f) - 90)/2
    return Vec3(f).rotate(ax, ang).asfol, Vec3(l).rotate(ax, -ang).aslin


def rose(a, bins=13, **kwargs):
    """Plot rose diagram"""
    if isinstance(a, Group):
        a, _ = a.dd
    fig = plt.figure()
    ax = fig.add_subplot(111, polar=True)
    ax.set_theta_direction(-1)
    ax.set_theta_zero_location('N')
    arad = a * np.pi / 180
    erad = np.linspace(0, 360, bins) * np.pi / 180
    plt.hist(arad, bins=erad, **kwargs)


if __name__ == "__main__":
    g = Group([Fol(0, 60),
               Fol(90, 60),
               Fol(180, 60),
               Fol(270, 60)], name='APSG')
    s = StereoNet()
    s.plane(g, 'k')
    s.line(g.cross(), 'go')
    plt.show()
