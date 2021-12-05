import sys
from itertools import combinations
import numpy as np

from apsg.config import apsg_conf
from apsg.math._vector import Vector3
from apsg.helpers._math import acosd
from apsg.feature._geodata import Lineation, Foliation, Pair
from apsg.feature._tensor import Ortensor3
from apsg.feature._statistics import KentDistribution, vonMisesFisher


class FeatureSet:
    __slots__ = ("data", "name")

    def __init__(self, data, name="Default"):
        dtype_cls = getattr(sys.modules[__name__], type(self).__feature_type__)
        assert all(
            [isinstance(obj, dtype_cls) for obj in data]
        ), f"Data must be instances of {type(self).__feature_type__}"
        self.data = tuple(data)
        self.name = name
        self._cache = {}

    def __copy__(self):
        return type(self).from_list(self.data, name=self.name)

    copy = __copy__

    def to_json(self):
        return {
            "datatype": type(self).__name__,
            "args": ({"collection": tuple(item.to_json() for item in self)},),
            "kwargs": {"name": self.name},
        }

    def label(self):
        return str(self)

    def __array__(self, dtype=None):
        return np.array(self.data, dtype=dtype)

    def __eq__(self, other):
        return NotImplemented

    def __ne__(self, other):
        return NotImplemented

    def __bool__(self):
        return len(self) != 0

    def __len__(self):
        return len(self.data)

    def __getitem__(self, key):
        return self.data[key]

    def __iter__(self):
        return iter(self.data)

    def __add__(self, other):
        if isinstance(other, type(self)):
            return type(self)(self.data + other.data, name=self.name)
        else:
            raise TypeError("Only {self.__name__} is allowed")

    def rotate(self, axis, phi):
        """Rotate ``FeatureSet`` object `phi` degress about `axis`."""
        return type(self)([e.rotate(axis, phi) for e in self], name=self.name)

    def bootstrap(self, n=100, size=None):
        """Return generator of bootstraped samples from ``FeatureSet``.

        Args:
          n: number of samples to be generated. Default 100.
          size: number of data in sample. Default is same as ``FeatureSet``.

        Example:
          >>> np.random.seed(6034782)
          >>> l = Vector3Set.random_fisher(n=100, position=lin(120,40))
          >>> sm = [lb.R() for lb in l.bootstrap()]
          >>> l.fisher_statistics()
          {'k': 19.912361106049794, 'a95': 3.2490273703993973, 'csd': 18.151964734256303}
          >>> Vector3Set(sm).fisher_statistics()
          {'k': 1735.3602067018592, 'a95': 0.33932243564473413, 'csd': 1.9444205467798013}
        """
        if size is None:
            size = len(self)
        for sample in np.random.randint(0, len(self), (n, size)):
            yield type(self)([self.data[ix] for ix in sample], name=self.name)


class Vector3Set(FeatureSet):
    __feature_type__ = "Vector3"

    def __repr__(self):
        return f"V({len(self)}) {self.name}"

    def __abs__(self):
        """Returns array of euclidean norms"""
        return np.asarray([abs(e) for e in self])

    def to_lin(self):
        """Return ``LineationSet`` object with all data converted to ``Lineation``."""
        return LineationSet([Lineation(e) for e in self], name=self.name)

    def to_fol(self):
        """Return ``FoliationSet`` object with all data converted to ``Foliation``."""
        return FoliationSet([Foliation(e) for e in self], name=self.name)

    def to_vec3(self):
        """Return ``Vector3Set`` object with all data converted to ``Vector3``."""
        return Vector3Set([Vector3(e) for e in self], name=self.name)

    def proj(self, vec):
        """Return projections of all features in ``FeatureSet`` onto vector.

        """
        return type(self)([e.project() for e in self], name=self.name)

    def dot(self, vec):
        """Return array of dot products of all features in ``FeatureSet`` with vector.

        """
        return np.array([e.dot(vec) for e in self])

    def cross(self, other=None):
        """Return cross products of all features in ``FeatureSet``

        Without arguments it returns cross product of all pairs in dataset.
        If argument is ``FeatureSet`` of same length or single data object
        element-wise cross-products are calculated.
        """
        res = []
        if other is None:
            res = [e.cross(f) for e, f in combinations(self.data, 2)]
        elif issubclass(type(other), FeatureSet):
            res = [e.cross(f) for e, f in zip(self, other)]
        elif issubclass(type(other), Vector3):
            res = [e.cross(other) for e in self]
        else:
            raise TypeError("Wrong argument type!")
        return G(res, name=self.name)

    __pow__ = cross

    def angle(self, other=None):
        """Return angles of all data in ``FeatureSet`` object

        Without arguments it returns angles of all pairs in dataset.
        If argument is ``FeatureSet`` of same length or single data object
        element-wise angles are calculated.
        """
        res = []
        if other is None:
            res = [e.angle(f) for e, f in combinations(self.data, 2)]
        elif issubclass(type(other), FeatureSet):
            res = [e.angle(f) for e, f in zip(self, other)]
        elif issubclass(type(other), Vector3):
            res = [e.angle(other) for e in self]
        else:
            raise TypeError("Wrong argument type!")
        return np.asarray(res)

    def normalized(self):
        """Return ``FeatureSet`` object with normalized (unit length) elements."""
        return type(self)([e.normalized() for e in self], name=self.name)

    uv = normalized

    def transform(self, F, **kwargs):
        """Return affine transformation of all features ``FeatureSet`` by matrix 'F'.

        Args:
          F: Transformation matrix. Should be array-like value e.g. ``DefGrad3``

        Keyword Args:
          norm: normalize transformed features. True or False. Default False

        """
        return type(self)([e.transform(F, **kwargs) for e in self], name=self.name)

    def is_upper(self):
        """
        Return boolean array of z-coordinate negative test
        """

        return np.asarray([e.is_upper() for e in self])

    def R(self, normalized=False):
        """Return resultant of data in ``FeatureSet`` object.

        Resultant is of same type as features in ``FeatureSet``. Note
        that ``Foliation`` and ``Lineation`` are axial in nature so
        resultant can give other result than expected. Anyway for axial
        data orientation tensor analysis will give you right answer.

        Args:
            normalized: if True returns mean resultant. Default False
        """
        R = sum(self)
        if normalized:
            R = R / len(self)
        return R

    def fisher_statistics(self):
        """Fisher's statistics

        fisher_statistics returns dictionary with keys:
            `k`    estimated precision parameter,
            `csd`  estimated angular standard deviation
            `a95`  confidence limit
        """
        stats = {"k": np.inf, "a95": 0, "csd": 0}
        N = len(self)
        R = abs(self.normalized().R())
        if N != R:
            stats["k"] = (N - 1) / (N - R)
            stats["csd"] = 81 / np.sqrt(stats["k"])
            stats["a95"] = acosd(1 - ((N - R) / R) * (20 ** (1 / (N - 1)) - 1))
        return stats

    def var(self):
        """Spherical variance based on resultant length (Mardia 1972).

        var = 1 - |R| / n
        """
        return 1 - abs(self.normalized().R(normalized=True))

    def delta(self):
        """Cone angle containing ~63% of the data in degrees.

        For enough large sample it approach angular standard deviation (csd)
        of Fisher statistics
        """
        return acosd(abs(self.R(normalized=True)))

    @property
    def rdegree(self):
        """Degree of preffered orientation of vectors in ``FeatureSet``.

        D = 100 * (2 * |R| - n) / n
        """
        N = len(self)
        return 100 * (2 * abs(self.normalized().R()) - N) / N

    def ortensor(self):
        """Return orientation tensor ``Ortensor`` of ``Group``."""

        return self._ortensor

    @property
    def _ortensor(self):
        if "ortensor" not in self._cache:
            self._cache["ortensor"] = Ortensor3.from_features(self)
        return self._cache["ortensor"]

    @property
    def _svd(self):
        if "svd" not in self._cache:
            self._cache["svd"] = np.linalg.svd(self._ortensor)
        return self._cache["svd"]

    def centered(self, max_vertical=False):
        """Rotate ``FeatureSet`` object to position that eigenvectors are parallel
        to axes of coordinate system: E1||X (north-south), E2||X(east-west),
        E3||X(vertical)

        Args:
            max_vertical: If True E1 is rotated to vertical. Default False

        """
        if max_vertical:
            return self.transform(self._svd[2]).rotate(Vector3(0, -1, 0), 90)
        else:
            return self.transform(self._svd[2])

    def halfspace(self):
        """Change orientation of vectors in ``FeatureSet``, so all have angle<=90 with
        resultant.

        """
        dtype_cls = getattr(sys.modules[__name__], type(self).__feature_type__)
        v = Vector3Set(self)
        v_data = list(v)
        alldone = np.all(v.angle(v.R()) <= 90)
        while not alldone:
            ang = v.angle(v.R())
            for ix, do in enumerate(ang > 90):
                if do:
                    v_data[ix] = -v_data[ix]
                v = Vector3Set(v_data)
                alldone = np.all(v.angle(v.R()) <= 90)
        return type(self)([dtype_cls(vec) for vec in v], name=self.name)

    def cluster(self):
        """Return hierarchical clustering ``Cluster`` of ``Group``."""
        return NotImplemented

    @classmethod
    def from_csv(cls, filename, acol=0, icol=1):
        """Create ``FeatureSet`` object from csv file

        Args:
          filename (str): name of CSV file to load

        Keyword Args:
          typ: Type of objects. Default ``Lin``
          acol (int or str): azimuth column (starts from 0). Default 0
          icol (int or str): inclination column (starts from 0). Default 1
            When acol and icol are strings they are used as column headers.

        Example:
          >>> gf = FoliationSet.from_csv('file1.csv')                 #doctest: +SKIP
          >>> gl = LineationSet.from_csv('file2.csv', acol=1, icol=2) #doctest: +SKIP

        """
        from os.path import basename
        import csv

        with open(filename) as csvfile:
            has_header = csv.Sniffer().has_header(csvfile.read(1024))
            csvfile.seek(0)
            dialect = csv.Sniffer().sniff(csvfile.read(1024))
            csvfile.seek(0)
            if isinstance(acol, int) and isinstance(icol, int):
                if has_header:
                    reader = csv.DictReader(csvfile, dialect=dialect)
                    aname, iname = reader.fieldnames[acol], reader.fieldnames[icol]
                    r = [(float(row[aname]), float(row[iname])) for row in reader]
                else:
                    reader = csv.reader(csvfile, dialect=dialect)
                    r = [(float(row[acol]), float(row[icol])) for row in reader]
            else:
                if has_header:
                    reader = csv.DictReader(csvfile, dialect=dialect)
                    r = [(float(row[acol]), float(row[icol])) for row in reader]
                else:
                    raise ValueError("No header line in CSV file...")

        azi, inc = zip(*r)
        return cls.from_array(azi, inc, name=basename(filename))

    def to_csv(self, filename, delimiter=","):
        """Save ``Group`` object to csv file

        Args:
          filename (str): name of CSV file to save.

        Keyword Args:
          delimiter (str): values delimiter. Default ','

        Note: Written values are rounded according to `ndigits` settings in apsg_conf

        """
        import csv

        n = apsg_conf["ndigits"]

        with open(filename, "w", newline="") as csvfile:
            fieldnames = ["azi", "inc"]
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            for dt in self:
                azi, inc = dt.geo
                writer.writerow({"azi": round(azi, n), "inc": round(inc, n)})

    @classmethod
    def from_array(cls, azis, incs, name="Default"):
        """Create ``Group`` object from arrays of azimuths and inclinations

        Args:
          azis: list or array of azimuths
          incs: list or array of inclinations

        Keyword Args:
          typ: type of data. ``Fol`` or ``Lin``
          name: name of ``Group`` object. Default is 'Default'

        Example:
          >>> f = FoliationSet.from_array([120,130,140], [10,20,30])
          >>> l = LineationSet.from_array([120,130,140], [10,20,30])
        """
        dtype_cls = getattr(sys.modules[__name__], cls.__feature_type__)
        return cls([dtype_cls(azi, inc) for azi, inc in zip(azis, incs)], name=name)

    @classmethod
    def random_normal(cls, n=100, position=Vector3(0, 0, 1), sigma=20, name="Default"):
        """Method to create ``FeatureSet`` of normaly distributed features.

        Keyword Args:
          n: number of objects to be generated
          position: mean orientation given as ``Vector3``. Default Vector3(0, 0, 1)
          sigma: sigma of normal distribution. Default 20
          name: name of dataset. Default is 'Default'

        Example:
          >>> np.random.seed(58463123)
          >>> g = Group.randn_lin(100, Lin(120, 40))
          >>> g.R
          L:118/42

        """
        data = []
        dtype_cls = getattr(sys.modules[__name__], cls.__feature_type__)
        orig = Vector3(0, 0, 1)
        ax = orig.cross(position)
        ang = orig.angle(position)
        for s, r in zip(
            180 * np.random.uniform(low=0, high=180, size=n),
            np.random.normal(loc=0, scale=sigma, size=n),
        ):
            v = orig.rotate(Vector3(s, 0), r).rotate(ax, ang)
            data.append(dtype_cls(v))
        return cls(data, name=name)

    @classmethod
    def random_fisher(cls, n=100, position=Vector3(0, 0, 1), kappa=20, name="Default"):
        """Return ``FeatureSet`` of random vectors sampled from von Mises Fisher distribution
        around center position with concentration kappa.

        Args:
          n: number of objects to be generated
          position: mean orientation given as ``Vector3``. Default Vector3(0, 0, 1)
          kappa: precision parameter of the distribution. Default 20
          name: name of dataset. Default is 'Default'

        Example:
          >>> l = LineationSet.random_fisher(position=lin(120,50))
        """
        dtype_cls = getattr(sys.modules[__name__], cls.__feature_type__)
        dc = vonMisesFisher(position, kappa, n)
        return cls([dtype_cls(d) for d in dc], name=name)

    @classmethod
    def random_fisher2(cls, n=100, position=Vector3(0, 0, 1), kappa=20, name="Default"):
        """Method to create ``FeatureSet`` of vectors distributed according to
        Fisher distribution.

        Note: For proper von Mises Fisher distrinbution implementation use ``random.fisher``
        method.

        Args:
          n: number of objects to be generated
          position: mean orientation given as ``Vector3``. Default Vector3(0, 0, 1)
          kappa: precision parameter of the distribution. Default 20
          name: name of dataset. Default is 'Default'

        Example:
          >>> l = LineationSet.random_fisher2(position=lin(120,50))
        """
        orig = Vector3(0, 0, 1)
        ax = orig.cross(position)
        ang = orig.angle(position)
        L = np.exp(-2 * kappa)
        a = np.random.random(n) * (1 - L) + L
        fac = np.sqrt(-np.log(a) / (2 * kappa))
        inc = 90 - 2 * np.degrees(np.arcsin(fac))
        azi = 360 * np.random.random(n)
        return cls.from_array(azi, inc, name=name).rotate(ax, ang)

    @classmethod
    def random_kent(cls, p, n=500, kappa=20, beta=None, name="Default"):
        """Return ``FeatureSet`` of random vectors sampled from Kent distribution
        (Kent, 1982) - The 5-parameter Fisher–Bingham distribution.

        Args:
          p: Pair object defining orientation of data
          N: number of objects to be generated
          kappa: concentration parameter. Default 20
          beta: ellipticity 0 <= beta < kappa
          name: name of dataset. Default is 'Default'

        Example:
          >>> p = pair(150, 40, 150, 40)
          >>> l = LineationSet.random_kent(p, n=300, kappa=30)
        """
        assert issubclass(type(p), Pair), "Argument must be Pair object."
        dtype_cls = getattr(sys.modules[__name__], cls.__feature_type__)
        if beta is None:
            beta = kappa / 2
        kd = KentDistribution(p.lvec, p.fvec.cross(p.lvec), p.fvec, kappa, beta)
        return cls([dtype_cls(d) for d in kd.rvs(n)], name=name)

    @classmethod
    def uniform_sfs(cls, n=100, name="Default"):
        """Method to create ``FeatureSet`` of uniformly distributed vectors.
        Spherical Fibonacci Spiral points on a sphere algorithm adopted from
        John Burkardt.

        http://people.sc.fsu.edu/~jburkardt/

        Keyword Args:
          n: number of objects to be generated. Default 1000
          name: name of dataset. Default is 'Default'

        Example:
          >>> v = Group.sfs_vec3(300)
          >>> v.ortensor.eigenvals
          (0.33346453471636356, 0.33333474915201167, 0.3332007161316248)
        """
        dtype_cls = getattr(sys.modules[__name__], cls.__feature_type__)
        phi = (1 + np.sqrt(5)) / 2
        i2 = 2 * np.arange(n) - n + 1
        theta = 2 * np.pi * i2 / phi
        sp = i2 / n
        cp = np.sqrt((n + i2) * (n - i2)) / n
        dc = np.array([cp * np.sin(theta), cp * np.cos(theta), sp]).T
        return cls([dtype_cls(d) for d in dc], name=name)

    @classmethod
    def uniform_gss(cls, n=100, name="Default"):
        """Method to create ``FeatureSet`` of uniformly distributed vectors.
        Golden Section Spiral points on a sphere algorithm.

        http://www.softimageblog.com/archives/115

        Args:
          n: number of objects to be generated.  Default 1000
          name: name of dataset. Default is 'Default'

        Example:
          >>> v = Group.gss_vec3(300)
          >>> v.ortensor.eigenvals
          (0.3333568856957158, 0.3333231511543691, 0.33331996314991513)
        """
        dtype_cls = getattr(sys.modules[__name__], cls.__feature_type__)
        inc = np.pi * (3 - np.sqrt(5))
        off = 2 / n
        k = np.arange(n)
        y = k * off - 1 + (off / 2)
        r = np.sqrt(1 - y * y)
        phi = k * inc
        dc = np.array([np.cos(phi) * r, y, np.sin(phi) * r]).T
        return cls([dtype_cls(d) for d in dc], name=name)


class LineationSet(Vector3Set):
    __feature_type__ = "Lineation"

    def __repr__(self):
        return f"L({len(self)}) {self.name}"


class FoliationSet(Vector3Set):
    __feature_type__ = "Foliation"

    def __repr__(self):
        return f"S({len(self)}) {self.name}"

    def dipvec(self):
        """Return ``FeatureSet`` object with plane dip vector."""
        return Vector3Set([e.dipvec() for e in self], name=self.name)


def G(lst, name="Default"):
    if hasattr(lst, "__len__"):
        dtype_cls = type(lst[0])
        assert all([isinstance(obj, dtype_cls) for obj in lst])
        if dtype_cls is Vector3:
            return Vector3Set(lst, name=name)
        elif dtype_cls is Lineation:
            return LineationSet(lst, name=name)
        elif dtype_cls is Foliation:
            return FoliationSet(lst, name=name)
        else:
            raise TypeError("Wrong datatype to create FeatureSet")


class PairSet(FeatureSet):
    __feature_type__ = "Pair"

    def __repr__(self):
        return f"P({len(self)}) {self.name}"


class FaultSet(FeatureSet):
    __feature_type__ = "Fault"

    def __repr__(self):
        return f"F({len(self)}) {self.name}"
