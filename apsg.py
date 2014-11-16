# -*- coding: utf-8 -*-
"""
Python module to manipulate, analyze and visualize structural geology data

Example::

    from apsg import *
    d=Dataset(name='lineace')
    d.append(Lin(120,40))
    d.append(Lin(153,18))
    d.append(Lin(140,35))
    s=SchmidtNet(d)
    s.add(d[0]**d[1])
    s.show()

"""

# import modulu
from __future__ import division, print_function
from copy import deepcopy
import sqlite3
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

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
        return abs(self - other) < 1e-15

    def __ne__(self, other):
        # not equal
        return not self == other

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

        >>> u.proj(v)
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

    @property
    def aspole(self):
        """Convert vector to Pole object.

        >>> u = Vec3([1,1,1])
        >>> u.aspole
        P:225/55
        """
        res = Pole(0, 0)
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
        return abs(self-other) < 1e-15 or abs(self+other) < 1e-15

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
        :type name: Vec3, Fol, Lin, Pole
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
    def xy(self):
        azi, inc = self.dd
        return (np.sqrt(2)*sind((90-inc)/2)*sind(azi),
                np.sqrt(2)*sind((90-inc)/2)*cosd(azi))


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
        return abs(self-other) < 1e-15 or abs(self+other) < 1e-15

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
        :type name: Vec3, Fol, Lin, Pole
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
        return np.dot(np.linalg.inv(F), self).view(type(self))

    @property
    def dd(self):
        n = self.uv
        if n[2] < 0:
            n = -n
        azi = (atan2d(n[1], n[0]) + 180) % 360
        inc = 90 - asind(n[2])
        return azi, inc

    @property
    def xy(self):
        azi, inc = self.dd
        return (-np.sqrt(2)*sind(inc/2)*sind(azi),
                -np.sqrt(2)*sind(inc/2)*cosd(azi))


class Pole(Fol):
    """Class for planar features represented as poles
    """
    def __new__(cls, azi, inc):
        # casting to our class
        return Vec3(getfdc(azi, inc)).view(cls)

    def __repr__(self):
        azi, inc = self.dd
        return 'P:%d/%d' % (round(azi), round(inc))


class Dataset(list):
    """Dataset class
    """
    def __init__(self, data=[],
                 name='Default',
                 color='blue',
                 fol={'lw': 1, 'ls': '-'},
                 lin={'marker': 'o', 's': 20},
                 pole={'marker': 'v', 's': 36, 'facecolors': None},
                 vec={'marker': 'd', 's': 24, 'facecolors': None},
                 tmpl=None):
        if not issubclass(type(data), list):
            data = [data]
        list.__init__(self, data)
        if tmpl is None:
            self.name = name
            self.color = color
            self.sym = {}
            self.sym['fol'] = fol
            self.sym['lin'] = lin
            self.sym['pole'] = pole
            self.sym['vec'] = vec
        else:
            self.name = tmpl.name
            self.color = tmpl.color
            self.sym = tmpl.sym


    @classmethod
    def fromcsv(cls, fname, typ=Lin, acol=1, icol=2,
                name='Default', color='blue'):
        """Read dataset from csv"""
        import csv
        with open(fname, 'rb') as csvfile:
            sniffer = csv.Sniffer()
            dialect = sniffer.sniff(csvfile.read(1024))
            csvfile.seek(0)
            d = cls(name=name, color=color)
            reader = csv.reader(csvfile, dialect)
            if sniffer.has_header:
                reader.next()
            for row in reader:
                if len(row) > 1:
                    d.append(typ(float(row[acol-1]), float(row[icol-1])))
            return d

    def __repr__(self):
        return self.name + ':' + repr(list(self))

    def __add__(self, d2):
        # merge Datasets
        return Dataset(list(self) + d2, tmpl=self)

    @property
    def lins(self):
        """return only Lin from Dataset"""
        return Dataset([e for e in self if type(e) == Lin], tmpl=self)

    @property
    def fols(self):
        """return only Fol from Dataset"""
        return Dataset([e for e in self if type(e) == Fol], tmpl=self)

    @property
    def poles(self):
        """return only Poles from Dataset"""
        return Dataset([e for e in self if type(e) == Pole], tmpl=self)

    @property
    def vecs(self):
        """return only Vec3 from Dataset"""
        return Dataset([e for e in self if type(e) == Vec3], tmpl=self)

    @property
    def numlins(self):
        """number of Lin in Dataset"""
        return len(self.lins)

    @property
    def numfols(self):
        """number of Fol in Dataset"""
        return len(self.fols)

    @property
    def numpoles(self):
        """number of Poles in Dataset"""
        return len(self.poles)

    @property
    def numvecs(self):
        """number of Vec3 in Dataset"""
        return len(self.vecs)

    @property
    def aslin(self):
        """Convert all data in Dataset to Lin"""
        return Dataset([e.aslin for e in self], tmpl=self)

    @property
    def asfol(self):
        """Convert all data in Dataset to Fol"""
        return Dataset([e.asfol for e in self], tmpl=self)

    @property
    def aspole(self):
        """Convert all data in Dataset to Pole"""
        return Dataset([e.aspole for e in self], tmpl=self)

    @property
    def resultant(self):
        """calculate resultant vector of Dataset"""
        r = deepcopy(self[0])
        for v in self[1:]:
            r += v
        return r

    @property
    def rdegree(self):
        """degree of preffered orientation od Dataset"""
        r = self.resultant
        n = len(self)
        return 100*(2*abs(r) - n)/n

    def cross(self, d=None):
        """return cross products of all pairs in Dataset"""
        res = Dataset(tmpl=self)
        res.name = 'Pairs'
        if d is None:
            for i in range(len(self)-1):
                for j in range(i+1, len(self)):
                    res.append(self[i]**self[j])
        else:
            for i in range(len(self)):
                for j in range(len(d)):
                    res.append(self[i]**d[j])
        return res

    def rotate(self, axis, phi):
        """rotate Dataset"""
        dr = Dataset(tmpl=self)
        for e in self:
            dr.append(e.rotate(axis, phi))
        return dr

    def center(self):
        """rotate E3 direction of Dataset to vertical"""
        ot = self.ortensor
        azi, inc = ot.eigenlins[2][0].dd
        return self.rotate(Lin(azi - 90, 0), 90 - inc)

    def angle(self, other):
        """list of angles between given feature and Dataset"""
        r = []
        for e in self:
            r.append(e.angle(other))
        return np.array(r)

    @property
    def ortensor(self):
        """return orientation tensor of Dataset"""
        return Ortensor(self)

    def transform(self, F):
        """Return affine transformation of dataset by matrix *F*"""
        dt = Dataset(tmpl=self)
        for e in self:
            dt.append(e.transform(F))
        return dt

    @property
    def dd(self):
        """array of dip directions and dips of Dataset"""
        return np.array([d.dd for d in self]).T

    def density(self, k=100, npoints=180):
        """calculate density of Dataset"""
        return Density(self, k, npoints)

    def plot(self):
        """Show Dataset on Schmidt net"""
        return SchmidtNet(self)

    @classmethod
    def randn_lin(self, N=100, main=Lin(0, 90), sig=20):
        d = []
        ta, td = main.dd
        for azi, dip in zip(180*np.random.rand(N), sig*np.random.randn(N)):
            d.append(Lin(0, 90).rotate(Lin(azi, 0), dip))
        return self(d).rotate(Lin(ta+90, 0), 90-td)

    @classmethod
    def randn_fol(self, N=100, main=Fol(0, 0), sig=20):
        d = []
        ta, td = main.dd
        for azi, dip in zip(180*np.random.rand(N), sig*np.random.randn(N)):
            d.append(Fol(0, 0).rotate(Lin(azi, 0), dip))
        return self(d).rotate(Lin(ta-90, 0), td)

    @classmethod
    def randn_pole(self, N=100, main=Pole(0, 0), sig=20):
        d = []
        ta, td = main.dd
        for azi, dip in zip(180*np.random.rand(N), sig*np.random.randn(N)):
            d.append(Pole(0, 0).rotate(Lin(azi, 0), dip))
        return self(d).rotate(Lin(ta-90, 0), td)


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
        return '(E1:%.4g,E2:%.4g,E3:%.4g)' % tuple(self.vals) + \
            '\n' + repr(self.M)

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
        d1 = Dataset(v1.aslin, name='E1', color='red')
        d2 = Dataset(v2.aslin, name='E2', color='magenta')
        d3 = Dataset(v3.aslin, name='E3', color='green')
        return d1, d2, d3

    @property
    def eigenfols(self):
        v1, v2, v3 = self.eigenvects
        d1 = Dataset(v1.asfol, name='E1', color='red')
        d2 = Dataset(v2.asfol, name='E2', color='magenta')
        d3 = Dataset(v3.asfol, name='E3', color='green')
        return d1, d2, d3


class Datasource(object):
    """PySDB database access class"""
    TESTSEL = "SELECT sites.id, sites.name, sites.x_coord, sites.y_coord, \
    sites.description, structdata.id, structdata.id_sites, \
    structdata.id_structype, structdata.azimuth, structdata.inclination, \
    structype.id, structype.structure, structype.description, \
    structype.structcode, structype.groupcode  \
    FROM sites \
    INNER JOIN structdata ON sites.id = structdata.id_sites \
    INNER JOIN structype ON structype.id = structdatnumlinsa.id_structype \
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
        fsel = Datasource.SELECT + " WHERE structype.planar=1"
        lsel = Datasource.SELECT + " WHERE structype.planar=0"
        if struct:
            fsel += " AND structype.structure='%s'" % struct
            lsel += " AND structype.structure='%s'" % struct
        fsel += " ORDER BY sites.name ASC"
        lsel += " ORDER BY sites.name ASC"

        fol = Dataset([Fol(el[0], el[1]) for el in self.execsql(fsel)])
        lin = Dataset([Lin(el[0], el[1]) for el in self.execsql(lsel)])
        return fol + lin


class Density(object):
    """Density grid class"""
    def __init__(self, d, k=100, npoints=180, nc=6, cmap=plt.cm.Greys):
        self.dcdata = np.asarray(d)
        self.k = k
        self.npoints = npoints
        self.nc = nc
        self.cm = cmap
        self.calculate()

    def __repr__(self):
        return ('Density grid from %d data with %d contours.\n' + \
                'Gridded on %d points.\n' + \
                'Values: k=%.4g E=%.4g s=%.4g\n' + \
                'Max. weight: %.4g') % (self.n, self.nc, self.npoints,
                                        self.k, self.E, self.s, self.weights.max())

    def calculate(self):
        import matplotlib.tri as tri
        self.xg = 0
        self.yg = 0
        for rho in np.linspace(0, 1, np.round(self.npoints/2/np.pi)):
            theta = np.linspace(0, 360, np.round(self.npoints*rho + 1))[:-1]
            self.xg = np.hstack((self.xg, rho*sind(theta)))
            self.yg = np.hstack((self.yg, rho*cosd(theta)))
        self.dcgrid = np.asarray(getldc(*getldd(self.xg, self.yg)))
        self.n = len(self.dcdata)
        self.E = self.n/self.k  # some points on periphery are equivalent
        self.s = np.sqrt((self.n*(0.5 - 1/self.k)/self.k))
        self.weights = np.zeros(len(self.xg))
        for i in range(self.n):
            self.weights += np.exp(self.k*(np.abs(np.dot(self.dcdata[i], self.dcgrid))-1))
        self.density = (self.weights - self.E)/self.s
        self.triang = tri.Triangulation(self.xg, self.yg)

    def plotcountgrid(self):
        plt.figure()
        plt.gca().set_aspect('equal')
        plt.triplot(self.triang, 'bo-')
        plt.show()


class SchmidtNet(object):
    """SchmidtNet class"""
    # store number of all used figures
    figlist = []

    def __init__(self, *data):
        # set figure number
        if SchmidtNet.figlist:
            self.fignum = max(SchmidtNet.figlist) + 1
        else:
            self.fignum = 1
        SchmidtNet.figlist.append(self.fignum)
        self.grid = True
        self.data = []
        self.density = None
        # calc grid
        grds = list(range(10, 100, 10)) + list(range(-80, 0, 10))
        self.xg = []
        self.yg = []
        a = Lin(0, 0)
        for dip in grds:
            l = Lin(0, dip)
            b = Fol(90, dip)
            t = Lin(90, dip)
            gc = map(l.rotate, 91*[a], np.linspace(-89.99, 89.99, 91))
            x, y = np.array([r.xy for r in gc]).T
            self.xg.extend(x)
            self.xg.append(np.nan)
            self.yg.extend(y)
            self.yg.append(np.nan)
            gc = map(t.rotate, 81*[b], np.linspace(-80, 80, 81))
            x, y = np.array([r.xy for r in gc]).T
            self.xg.extend(x)
            self.xg.append(np.nan)
            self.yg.extend(y)
            self.yg.append(np.nan)
        # add arguments
        for arg in data:
            self.add(arg)
        self.refresh()

    def clear(self):
        """remove all data from projection"""
        self.data = []
        self.density = None
        self.refresh()

    def add(self, *args):
        """Add data to projection"""
        if not issubclass(type(args), tuple):
            args = tuple(args)
        for arg in args:
            if type(arg) == Density:
                self.set_density(arg)
            elif type(arg) == Dataset:
                self.data.append(arg)
            elif type(arg) == Lin or type(arg) == Fol or type(arg) == Pole or type(arg) == Vec3:
                self.data.append(Dataset(arg))
            elif type(arg) == Ortensor:
                for v in arg.eigenlins:
                    self.data.append(v)
            else:
                raise Exception('Wrong argument! '+type(arg) +
                                ' cannot be plotted as linear feature.')
        self.refresh()

    def set_density(self, density):
        """Set density grid"""
        if type(density) == Density or density is None:
            self.density = density
            self.refresh()

    def refresh(self):
        """Draw figure"""
        # test if closed
        if not plt.fignum_exists(self.fignum):
            self.fig = plt.figure(num=self.fignum, facecolor='white')
            self.fig.canvas.set_window_title('Schmidt Net %d' % self.fignum)
            self.ax = self.fig.add_subplot(111)
        self.ax.cla()
        self.ax.set_aspect('equal')
        self.ax.set_autoscale_on(False)
        self.ax.axis([-1.05, 1.05, -1.05, 1.05])
        self.ax.set_axis_off()

        # Projection circle
        self.ax.text(0, 1.02, 'N', ha='center', fontsize=16)
        #self.ax.add_artist(plt.Circle((0, 0), 1, color='w', zorder=0))
        TH = np.linspace(0, 360, 361)
        self.ax.plot(sind(TH), cosd(TH), 'k')

        #density grid
        if self.density:
            cs = self.ax.tricontourf(self.density.triang, self.density.density,
                                     self.density.nc, cmap=self.density.cm, zorder=1)
            self.ax.tricontour(self.density.triang, self.density.density,
                               self.density.nc, colors='k', zorder=1)

        #grid
        if self.grid:
            self.ax.plot(self.xg, self.yg, 'k:')

        # init labels
        handles = []
        labels = []

        # plot data
        for arg in self.data:
            #fol great circle
            dd = arg.fols
            if dd:
                for d in dd:
                    l = Lin(*d.dd)
                    gc = map(l.rotate, 91*[d], np.linspace(-89.99, 89.99, 91))
                    x, y = np.array([r.xy for r in gc]).T
                    h = self.ax.plot(x, y, color=arg.color, zorder=2, **dd.sym['fol'])
                handles.append(h[0])
                labels.append('S ' + arg.name)
            #lin point
            dd = arg.lins
            if dd:
                x, y = np.array([e.xy for e in dd]).T
                h = self.ax.scatter(x, y, color=arg.color, zorder=4, **dd.sym['lin'])
                handles.append(h)
                labels.append('L ' + arg.name)
            #pole point
            dd = arg.poles
            if dd:
                x, y = np.array([e.xy for e in dd]).T
                h = self.ax.scatter(x, y, color=arg.color, zorder=3, **dd.sym['pole'])
                handles.append(h)
                labels.append('P ' + arg.name)
            #vector point
            dd = arg.vecs
            if dd:
                x, y = np.array([e.xy for e in dd]).T
                h = self.ax.scatter(x, y, color=arg.color, zorder=3, **dd.sym['vec'])
                handles.append(h)
                labels.append('V ' + arg.name)
        # legend
        if handles:
            self.ax.legend(handles, labels, bbox_to_anchor=(1.03, 1), loc=2,
                           borderaxespad=0., numpoints=1, scatterpoints=1)
        #density grid contours
        if self.density:
            divider = make_axes_locatable(self.ax)
            cax = divider.append_axes("left", size="5%", pad=0.5)
            cb = plt.colorbar(cs, cax=cax)
            # modify tick labels
            lbl = [item.get_text()+'S' for item in cb.ax.get_yticklabels()]
            lbl[lbl.index(next(l for l in lbl if l.startswith('0')))] = 'E'
            cb.set_ticklabels(lbl)
        #finish
        plt.subplots_adjust(left=0.02, bottom=0.05, right=0.75, top=0.95)
        #self.fig.canvas.draw()
        plt.draw()

    def show(self, *args, **kw):
        """Show figure"""
        if not plt.fignum_exists(self.fignum):
            self.refresh()
        plt.show(*args, **kw)

    def savefig(self, filename='schmidtnet.pdf'):
        if not plt.fignum_exists(self.fignum):
            self.refresh()
        plt.savefig(filename)

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

    def getpole(self):
        """get Pole by mouse click"""
        self.show()
        x, y = plt.ginput(1)[0]
        return Pole(*getfdd(x, y))

    def getlins(self):
        """Collect Dataset of Lin by mouse clicks"""
        self.show()
        pts = plt.ginput(0, mouse_add=1, mouse_pop=2, mouse_stop=3)
        l = Dataset()
        for x, y in pts:
            l.append(Lin(*getldd(x, y)))
        return l

    def getfols(self):
        """Collect Dataset of Fol by mouse clicks"""
        self.show()
        pts = plt.ginput(0, mouse_add=1, mouse_pop=2, mouse_stop=3)
        f = Dataset()
        for x, y in pts:
            f.append(Fol(*getfdd(x, y)))
        return f

    def getpoles(self):
        """Collect Dataset of Pole by mouse clicks"""
        self.show()
        pts = plt.ginput(0, mouse_add=1, mouse_pop=2, mouse_stop=3)
        f = Dataset()
        for x, y in pts:
            f.append(Pole(*getfdd(x, y)))
        return f


def fixpair(f, l):
    """Fix pair of planar and linear data, so Lin is within plane Fol::

        fok,lok = fixpair(f,l)
    """
    ax = f ** l
    ang = (Vec3(l).angle(f) - 90)/2
    return Vec3(f).rotate(ax, ang).asfol, Vec3(l).rotate(ax, -ang).aslin


def rose(a, bins=13, **kwargs):
    """Plot rose diagram"""
    if isinstance(a, Dataset):
        a, _ = a.dd
    fig = plt.figure()
    ax = fig.add_subplot(111, polar=True)
    ax.set_theta_direction(-1)
    ax.set_theta_zero_location('N')
    arad = a * np.pi / 180
    erad = np.linspace(0, 360, bins) * np.pi / 180
    plt.hist(arad, bins=erad, **kwargs)


if __name__ == "__main__":
    d = Dataset([Fol(0, 60),
                 Fol(90, 60),
                 Fol(180, 60),
                 Fol(270, 60)],
                name='apsg')
    c = Density(d)
    s = SchmidtNet(c, d)
    s.show()
