import sys
from itertools import combinations
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import vonmises
from scipy.cluster.hierarchy import linkage, fcluster, dendrogram

from apsg.config import apsg_conf
from apsg.math._vector import Vector2, Vector3
from apsg.helpers._math import acosd
from apsg.feature._geodata import Lineation, Foliation, Pair, Fault, Cone
from apsg.feature._tensor3 import OrientationTensor3, Ellipsoid, DeformationGradient3
from apsg.feature._tensor2 import OrientationTensor2, Ellipse
from apsg.feature._statistics import KentDistribution, vonMisesFisher


class FeatureSet:
    """
    Base class for containers
    """

    __slots__ = ("data", "name")

    def __init__(self, data, name="Default"):
        dtype_cls = getattr(sys.modules[__name__], type(self).__feature_type__)
        assert all(
            [isinstance(obj, dtype_cls) for obj in data]
        ), f"Data must be instances of {type(self).__feature_type__}"
        # cast to correct instances
        self.data = tuple([dtype_cls(d) for d in data])
        self.name = name
        self._cache = {}

    def __copy__(self):
        return type(self)(list(self.data), name=self.name)

    copy = __copy__

    def to_json(self):
        """Return as JSON dict"""
        return {
            "datatype": type(self).__name__,
            "args": ({"collection": tuple(item.to_json() for item in self)},),
            "kwargs": {"name": self.name},
        }

    def label(self):
        """Return label"""
        return self.name

    def __array__(self, dtype=None, copy=None):
        return np.array([np.array(p) for p in self.data], dtype=dtype)

    def __eq__(self, other):
        return NotImplemented

    def __ne__(self, other):
        return NotImplemented

    def __bool__(self):
        return len(self) != 0

    def __len__(self):
        return len(self.data)

    def __getitem__(self, key):
        if isinstance(key, slice):
            return type(self)(self.data[key])
        # elif isinstance(key, int):
        elif np.issubdtype(type(key), np.integer):
            return self.data[key]
        elif isinstance(key, np.ndarray):  # fancy indexing
            idxs = np.arange(len(self.data), dtype=int)[key]
            return type(self)([self.data[ix] for ix in idxs])
        else:
            raise TypeError(
                "Wrong index. Only slice, int and np.array are allowed for indexing."
            )

    def __iter__(self):
        return iter(self.data)

    def __add__(self, other):
        if isinstance(other, type(self)):
            return type(self)(self.data + other.data, name=self.name)
        else:
            raise TypeError(f"Only {self.__name__} is allowed")

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
          {'k': 19.91236110604979, 'a95': 3.249027370399397, 'csd': 18.15196473425630}
          >>> Vector3Set(sm).fisher_statistics()
          {'k': 1735.360206701859, 'a95': 0.3393224356447341, 'csd': 1.944420546779801}
        """
        if size is None:
            size = len(self)
        for i in range(n):
            yield self[np.random.choice(range(len(self)), size)]


class Vector2Set(FeatureSet):
    """
    Class to store set of ``Vector2`` features
    """

    __feature_type__ = "Vector2"

    def __repr__(self):
        return f"V2({len(self)}) {self.name}"

    def __abs__(self):
        """Returns array of euclidean norms"""
        return np.asarray([abs(e) for e in self])

    @property
    def x(self):
        """Return numpy array of x-components"""
        return np.array([e.x for e in self])

    @property
    def y(self):
        """Return numpy array of y-components"""
        return np.array([e.y for e in self])

    @property
    def direction(self):
        """Return array of direction angles"""
        return np.asarray([e.direction for e in self]).T

    def proj(self, vec):
        """Return projections of all features in ``Vector2Set`` onto vector."""
        return type(self)([e.project() for e in self], name=self.name)

    def dot(self, vec):
        """Return array of dot products of all features in ``Vector2Set`` with vector."""
        return np.array([e.dot(vec) for e in self])

    def cross(self, other=None):
        """Return cross products of all features in ``Vector2Set``

        Without arguments it returns cross product of all pairs in dataset.
        If argument is ``Vector2Set`` of same length or single data object
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
        return np.asarray(res)

    __pow__ = cross

    def angle(self, other=None):
        """Return angles of all data in ``Vector2Set`` object

        Without arguments it returns angles of all pairs in dataset.
        If argument is ``Vector2Set`` of same length or single data object
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
        """Return ``Vector2Set`` object with normalized (unit length) elements."""
        return type(self)([e.normalized() for e in self], name=self.name)

    uv = normalized

    def transform(self, F, **kwargs):
        """Return affine transformation of all features ``Vector2Set`` by matrix 'F'.

        Args:
          F: Transformation matrix. Array-like value e.g. ``DeformationGradient3``

        Keyword Args:
          norm: normalize transformed features. True or False. Default False

        """
        return type(self)([e.transform(F, **kwargs) for e in self], name=self.name)

    def R(self, mean=False):
        """Return resultant of data in ``Vector2Set`` object.

        Resultant is of same type as features in ``Vector2Set``. Note
        that ``Axial2`` is axial in nature so resultant can give
        other result than expected. Anyway for axial data orientation
        tensor analysis will give you right answer.

        Args:
            mean: if True returns mean resultant. Default False
        """
        R = sum(self)
        if mean:
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

        var = 1 - abs(R) / n
        """
        return 1 - abs(self.normalized().R(mean=True))

    def delta(self):
        """Cone angle containing ~63% of the data in degrees.

        For enough large sample it approach angular standard deviation (csd)
        of Fisher statistics
        """
        return acosd(abs(self.R(mean=True)))

    def rdegree(self):
        """Degree of preffered orientation of vectors in ``Vector2Set``.

        D = 100 * (2 * abs(R) - n) / n
        """
        N = len(self)
        return 100 * (2 * abs(self.normalized().R()) - N) / N

    def ortensor(self):
        """Return orientation tensor ``Ortensor`` of ``Group``."""

        return self._ortensor

    @property
    def _ortensor(self):
        if "ortensor" not in self._cache:
            self._cache["ortensor"] = OrientationTensor2.from_features(self)
        return self._cache["ortensor"]

    @property
    def _svd(self):
        if "svd" not in self._cache:
            self._cache["svd"] = np.linalg.svd(self._ortensor)
        return self._cache["svd"]

    def halfspace(self):
        """Change orientation of vectors in ``Vector2Set``, so all have angle<=90 with
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

    @classmethod
    def from_direction(cls, angles, name="Default"):
        """Create ``Vector2Set`` object from arrays of direction angles

        Args:
          angles: list or angles

        Keyword Args:
          name: name of ``Vector2Set`` object. Default is 'Default'

        Example:
          >>> f = vec2set.from_angles([120,130,140,125, 132. 131])
        """
        dtype_cls = getattr(sys.modules[__name__], cls.__feature_type__)
        return cls([dtype_cls(a) for a in angles], name=name)

    @classmethod
    def from_xy(cls, x, y, name="Default"):
        """Create ``Vector2Set`` object from arrays of x and y components

        Args:
          x: list or array of x components
          y: list or array of y components

        Keyword Args:
          name: name of ``Vector2Set`` object. Default is 'Default'

        Example:
          >>> v = vec2set.from_xy([-0.4330127, -0.4330127, -0.66793414],
                                  [0.75, 0.25, 0.60141061])
        """
        dtype_cls = getattr(sys.modules[__name__], cls.__feature_type__)
        return cls([dtype_cls(xx, yy) for xx, yy in zip(x, y)], name=name)

    @classmethod
    def random(cls, n=100, name="Default"):
        """Method to create ``Vector2Set`` of features with uniformly distributed
        random orientation.

        Keyword Args:
          n: number of objects to be generated
          name: name of dataset. Default is 'Default'

        Example:
          >>> np.random.seed(58463123)
          >>> l = vec2set.random(100)

        """
        dtype_cls = getattr(sys.modules[__name__], cls.__feature_type__)
        return cls([dtype_cls.random() for i in range(n)], name=name)

    @classmethod
    def random_vonmises(cls, n=100, position=0, kappa=5, name="Default"):
        """Return ``Vector2Set`` of random vectors sampled from von Mises distribution
        around center position with concentration kappa.

        Args:
          n: number of objects to be generated
          position: mean orientation given as angle. Default 0
          kappa: precision parameter of the distribution. Default 20
          name: name of dataset. Default is 'Default'

        Example:
          >>> l = linset.random_fisher(position=lin(120,50))
        """
        dtype_cls = getattr(sys.modules[__name__], cls.__feature_type__)
        angles = np.degrees(vonmises.rvs(kappa, loc=np.radians(position), size=n))
        return cls([dtype_cls(a) for a in angles], name=name)


class Vector3Set(FeatureSet):
    """
    Class to store set of ``Vector3`` features
    """

    __feature_type__ = "Vector3"

    def __repr__(self):
        return f"V3({len(self)}) {self.name}"

    def __abs__(self):
        """Returns array of euclidean norms"""
        return np.asarray([abs(e) for e in self])

    @property
    def x(self):
        """Return numpy array of x-components"""
        return np.array([e.x for e in self])

    @property
    def y(self):
        """Return numpy array of y-components"""
        return np.array([e.y for e in self])

    @property
    def z(self):
        """Return numpy array of z-components"""
        return np.array([e.z for e in self])

    @property
    def geo(self):
        """Return arrays of azi and inc according to apsg_conf['notation']"""
        azi, inc = np.asarray([e.geo for e in self]).T
        return azi, inc

    def to_lin(self):
        """Return ``LineationSet`` object with all data converted to ``Lineation``."""
        return LineationSet([Lineation(e) for e in self], name=self.name)

    def to_fol(self):
        """Return ``FoliationSet`` object with all data converted to ``Foliation``."""
        return FoliationSet([Foliation(e) for e in self], name=self.name)

    def to_vec(self):
        """Return ``Vector3Set`` object with all data converted to ``Vector3``."""
        return Vector3Set([Vector3(e) for e in self], name=self.name)

    def project(self, vec):
        """Return projections of all features in ``FeatureSet`` onto vector."""
        return type(self)([e.project(vec) for e in self], name=self.name)

    proj = project

    def reject(self, vec):
        """Return rejections of all features in ``FeatureSet`` onto vector."""
        return type(self)([e.reject(vec) for e in self], name=self.name)

    def dot(self, vec):
        """Return array of dot products of all features in ``FeatureSet`` with vector."""
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
          F: Transformation matrix. Array-like value e.g. ``DeformationGradient3``

        Keyword Args:
          norm: normalize transformed features. True or False. Default False

        """
        return type(self)([e.transform(F, **kwargs) for e in self], name=self.name)

    def is_upper(self):
        """
        Return boolean array of z-coordinate negative test
        """

        return np.asarray([e.is_upper() for e in self])

    def R(self, mean=False):
        """Return resultant of data in ``FeatureSet`` object.

        Resultant is of same type as features in ``FeatureSet``. Note
        that ``Foliation`` and ``Lineation`` are axial in nature so
        resultant can give other result than expected. Anyway for axial
        data orientation tensor analysis will give you right answer.

        Args:
            mean: if True returns mean resultant. Default False
        """
        R = sum(self)
        if mean:
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

    def fisher_cone_a95(self):
        """Confidence limit cone based on Fisher's statistics

        Cone axis is resultant and apical angle is a95 confidence limit
        """
        stats = self.fisher_statistics()
        return Cone(self.normalized().R(), stats["a95"])

    def fisher_cone_csd(self):
        """Angular standard deviation cone based on Fisher's statistics

        Cone axis is resultant and apical angle is angular standard deviation
        """
        stats = self.fisher_statistics()
        return Cone(self.normalized().R(), stats["csd"])

    def var(self):
        """Spherical variance based on resultant length (Mardia 1972).

        var = 1 - abs(R) / n
        """
        return 1 - abs(self.normalized().R(mean=True))

    def delta(self):
        """Cone angle containing ~63% of the data in degrees.

        For enough large sample it approach angular standard deviation (csd)
        of Fisher statistics
        """
        return acosd(abs(self.R(mean=True)))

    def rdegree(self):
        """Degree of preffered orientation of vectors in ``FeatureSet``.

        D = 100 * (2 * abs(R) - n) / n
        """
        N = len(self)
        return 100 * (2 * abs(self.normalized().R()) - N) / N

    def ortensor(self):
        """Return orientation tensor ``Ortensor`` of ``Group``."""
        return self._ortensor

    @property
    def _ortensor(self):
        if "ortensor" not in self._cache:
            self._cache["ortensor"] = OrientationTensor3.from_features(self)
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

    @classmethod
    def from_csv(cls, filename, acol=0, icol=1):
        """Create ``FeatureSet`` object from csv file of azimuths and inclinations

        Args:
          filename (str): name of CSV file to load

        Keyword Args:
          acol (int or str): azimuth column (starts from 0). Default 0
          icol (int or str): inclination column (starts from 0). Default 1
            When acol and icol are strings they are used as column headers.

        Example:
          >>> gf = folset.from_csv('file1.csv')                 #doctest: +SKIP
          >>> gl = linset.from_csv('file2.csv', acol=1, icol=2) #doctest: +SKIP

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
        """Save ``FeatureSet`` object to csv file of azimuths and inclinations

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
        """Create ``FeatureSet`` object from arrays of azimuths and inclinations

        Args:
          azis: list or array of azimuths
          incs: list or array of inclinations

        Keyword Args:
          name: name of ``FeatureSet`` object. Default is 'Default'

        Example:
          >>> f = folset.from_array([120,130,140], [10,20,30])
          >>> l = linset.from_array([120,130,140], [10,20,30])
        """
        dtype_cls = getattr(sys.modules[__name__], cls.__feature_type__)
        return cls([dtype_cls(azi, inc) for azi, inc in zip(azis, incs)], name=name)

    @classmethod
    def from_xyz(cls, x, y, z, name="Default"):
        """Create ``FeatureSet`` object from arrays of x, y and z components

        Args:
          x: list or array of x components
          y: list or array of y components
          z: list or array of z components

        Keyword Args:
          name: name of ``FeatureSet`` object. Default is 'Default'

        Example:
          >>> v = vecset.from_xyz([-0.4330127, -0.4330127, -0.66793414],
                                  [0.75, 0.25, 0.60141061],
                                  [0.5, 0.8660254, 0.43837115])
        """
        dtype_cls = getattr(sys.modules[__name__], cls.__feature_type__)
        return cls([dtype_cls(xx, yy, zz) for xx, yy, zz in zip(x, y, z)], name=name)

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
          >>> l = linset.random_normal(100, lin(120, 40))
          >>> l.R
          L:120/39

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
        """Return ``FeatureSet`` of random vectors sampled from von Mises Fisher
        distribution around center position with concentration kappa.

        Args:
          n: number of objects to be generated
          position: mean orientation given as ``Vector3``. Default Vector3(0, 0, 1)
          kappa: precision parameter of the distribution. Default 20
          name: name of dataset. Default is 'Default'

        Example:
          >>> l = linset.random_fisher(position=lin(120,50))
        """
        dtype_cls = getattr(sys.modules[__name__], cls.__feature_type__)
        dc = vonMisesFisher(position, kappa, n)
        return cls([dtype_cls(d) for d in dc], name=name)

    @classmethod
    def random_fisher2(cls, n=100, position=Vector3(0, 0, 1), kappa=20, name="Default"):
        """Method to create ``FeatureSet`` of vectors distributed according to
        Fisher distribution.

        Note: For proper von Mises Fisher distrinbution implementation use
        ``random.fisher`` method.

        Args:
          n: number of objects to be generated
          position: mean orientation given as ``Vector3``. Default Vector3(0, 0, 1)
          kappa: precision parameter of the distribution. Default 20
          name: name of dataset. Default is 'Default'

        Example:
          >>> l = linset.random_fisher2(position=lin(120,50))
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
    def random_kent(cls, p, n=100, kappa=20, beta=None, name="Default"):
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
          >>> l = linset.random_kent(p, n=300, kappa=30)
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
          >>> v = vecset.uniform_sfs(300)
          >>> v.ortensor().eigenvalues()
          (0.3334645347163635, 0.33333474915201167, 0.33320071613162483)
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
          >>> v = vecset.uniform_gss(300)
          >>> v.ortensor().eigenvalues()
          (0.33335688569571587, 0.33332315115436933, 0.33331996314991513)
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
    """
    Class to store set of ``Lineation`` features
    """

    __feature_type__ = "Lineation"

    def __repr__(self):
        return f"L({len(self)}) {self.name}"


class FoliationSet(Vector3Set):
    """
    Class to store set of ``Foliation`` features
    """

    __feature_type__ = "Foliation"

    def __repr__(self):
        return f"S({len(self)}) {self.name}"

    def dipvec(self):
        """Return ``FeatureSet`` object with plane dip vector."""
        return Vector3Set([e.dipvec() for e in self], name=self.name)


class PairSet(FeatureSet):
    """
    Class to store set of ``Pair`` features
    """

    __feature_type__ = "Pair"

    def __repr__(self):
        return f"P({len(self)}) {self.name}"

    def __array__(self, dtype=None, copy=None):
        return np.array([np.array(p) for p in self.data], dtype=dtype)

    @property
    def fol(self):
        """Return Foliations of pairs as FoliationSet"""
        return FoliationSet([e.fol for e in self], name=self.name)

    @property
    def fvec(self):
        """Return planar normal vectors of pairs as Vector3Set"""
        return Vector3Set([e.fvec for e in self], name=self.name)

    @property
    def lin(self):
        """Return Lineation of pairs as LineationSet"""
        return LineationSet([e.lin for e in self], name=self.name)

    @property
    def lvec(self):
        """Return lineation vectors of pairs as Vector3Set"""
        return Vector3Set([e.lvec for e in self], name=self.name)

    @property
    def misfit(self):
        """Return array of misfits"""
        return np.array([f.misfit for f in self])

    @property
    def rax(self):
        """
        Return vectors perpendicular to both planar and linear parts of
        pairs as Vector3Set
        """
        return Vector3Set([e.rax for e in self], name=self.name)

    def angle(self, other=None):
        """Return angles of all data in ``PairSet`` object

        Without arguments it returns angles of all pairs in dataset.
        If argument is ``PairSet`` of same length or single data object
        element-wise angles are calculated.
        """
        res = []
        if other is None:
            res = [
                abs(
                    DeformationGradient3.from_two_pairs(
                        e, f, symmetry=True
                    ).axisangle()[1]
                )
                for e, f in combinations(self.data, 2)
            ]
        elif issubclass(type(other), PairSet):
            res = [
                abs(
                    DeformationGradient3.from_two_pairs(
                        e, f, symmetry=True
                    ).axisangle()[1]
                )
                for e, f in zip(self, other)
            ]
        elif issubclass(type(other), Pair):
            res = [
                abs(
                    DeformationGradient3.from_two_pairs(
                        e, other, symmetry=True
                    ).axisangle()[1]
                )
                for e in self
            ]
        else:
            raise TypeError("Wrong argument type!")
        return np.asarray(res)

    @property
    def ortensor(self):
        """Return Lisle (1989) orientation tensor ``OrientationTensor3`` of orientations
        defined by pairs"""
        return OrientationTensor3.from_pairs(self)

    def label(self):
        return str(self)

    @classmethod
    def random(cls, n=25):
        """Create PairSet of random pairs"""
        return PairSet([Pair.random() for i in range(n)])

    @classmethod
    def from_csv(cls, filename, delimiter=",", facol=0, ficol=1, lacol=2, licol=3):
        """Read ``PairSet`` from csv file"""

        from os.path import basename
        import csv

        with open(filename) as csvfile:
            has_header = csv.Sniffer().has_header(csvfile.read(1024))
            csvfile.seek(0)
            dialect = csv.Sniffer().sniff(csvfile.read(1024))
            csvfile.seek(0)
            if (
                isinstance(facol, int)
                and isinstance(ficol, int)
                and isinstance(lacol, int)
                and isinstance(licol, int)
            ):
                if has_header:
                    reader = csv.DictReader(csvfile, dialect=dialect)
                    faname, finame = reader.fieldnames[facol], reader.fieldnames[ficol]
                    laname, liname = reader.fieldnames[lacol], reader.fieldnames[licol]
                    r = [
                        (
                            float(row[faname]),
                            float(row[finame]),
                            float(row[laname]),
                            float(row[liname]),
                        )
                        for row in reader
                    ]
                else:
                    reader = csv.reader(csvfile, dialect=dialect)
                    r = [
                        (
                            float(row[facol]),
                            float(row[ficol]),
                            float(row[lacol]),
                            float(row[licol]),
                        )
                        for row in reader
                    ]
            else:
                if has_header:
                    reader = csv.DictReader(csvfile, dialect=dialect)
                    r = [
                        (
                            float(row[facol]),
                            float(row[ficol]),
                            float(row[lacol]),
                            float(row[licol]),
                        )
                        for row in reader
                    ]
                else:
                    raise ValueError("No header line in CSV file...")

        fazi, finc, lazi, linc = zip(*r)
        return cls.from_array(fazi, finc, lazi, linc, name=basename(filename))

    def to_csv(self, filename, delimiter=","):
        """Save ``PairSet`` object to csv file

        Args:
          filename (str): name of CSV file to save.

        Keyword Args:
          delimiter (str): values delimiter. Default ','

        Note: Written values are rounded according to `ndigits` settings in apsg_conf

        """
        import csv

        n = apsg_conf["ndigits"]

        with open(filename, "w", newline="") as csvfile:
            fieldnames = ["azi", "inc", "lazi", "linc"]
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            for dt in self:
                fazi, finc = dt.fol.geo
                lazi, linc = dt.lin.geo
                writer.writerow(
                    {
                        "fazi": round(fazi, n),
                        "finc": round(finc, n),
                        "lazi": round(lazi, n),
                        "linc": round(linc, n),
                    }
                )

    @classmethod
    def from_array(cls, fazis, fincs, lazis, lincs, name="Default"):
        """Create ``PairSet`` from arrays of azimuths and inclinations

        Args:
          azis: list or array of azimuths
          incs: list or array of inclinations

        Keyword Args:
          name: name of ``PairSet`` object. Default is 'Default'
        """

        dtype_cls = getattr(sys.modules[__name__], cls.__feature_type__)
        return cls(
            [
                dtype_cls(fazi, finc, lazi, linc)
                for fazi, finc, lazi, linc in zip(fazis, fincs, lazis, lincs)
            ],
            name=name,
        )


class FaultSet(PairSet):
    """
    Class to store set of ``Fault`` features
    """

    __feature_type__ = "Fault"

    def __repr__(self):
        return f"F({len(self)}) {self.name}"

    @property
    def sense(self):
        """Return array of sense values"""
        return np.array([f.sense for f in self])

    @property
    def p_vector(self, ptangle=90):
        """Return p-axes of FaultSet as Vector3Set"""
        return Vector3Set([e.p_vector(ptangle) for e in self], name=self.name)

    @property
    def t_vector(self, ptangle=90):
        """Return t-axes of FaultSet as Vector3Set"""
        return Vector3Set([e.t_vector(ptangle) for e in self], name=self.name)

    @property
    def p(self):
        """Return p-axes of FaultSet as LineationSet"""
        return LineationSet([e.p for e in self], name=self.name + "-P")

    @property
    def t(self):
        """Return t-axes of FaultSet as LineationSet"""
        return LineationSet([e.t for e in self], name=self.name + "-T")

    @property
    def m(self):
        """Return m-planes of FaultSet as FoliationSet"""
        return FoliationSet([e.m for e in self], name=self.name + "-M")

    @property
    def d(self):
        """Return dihedra planes of FaultSet as FoliationSet"""
        return FoliationSet([e.d for e in self], name=self.name + "-D")

    def angle(self, other=None):
        """Return angles of all data in ``FaultSet`` object

        Without arguments it returns angles of all pairs in dataset.
        If argument is ``FaultSet`` of same length or single data object
        element-wise angles are calculated.
        """
        res = []
        if other is None:
            res = [
                abs(
                    DeformationGradient3.from_two_pairs(
                        e, f, symmetry=False
                    ).axisangle()[1]
                )
                for e, f in combinations(self.data, 2)
            ]
        elif issubclass(type(other), FaultSet):
            res = [
                abs(
                    DeformationGradient3.from_two_pairs(
                        e, f, symmetry=False
                    ).axisangle()[1]
                )
                for e, f in zip(self, other)
            ]
        elif issubclass(type(other), Fault):
            res = [
                abs(
                    DeformationGradient3.from_two_pairs(
                        e, other, symmetry=False
                    ).axisangle()[1]
                )
                for e in self
            ]
        else:
            raise TypeError("Wrong argument type!")
        return np.asarray(res)

    @classmethod
    def random(cls, n=25):
        """Create PairSet of random pairs"""
        return FaultSet([Fault.random() for i in range(n)])

    @classmethod
    def from_csv(
        cls, filename, delimiter=",", facol=0, ficol=1, lacol=2, licol=3, scol=4
    ):
        """Read ``FaultSet`` from csv file"""

        from os.path import basename
        import csv

        with open(filename) as csvfile:
            has_header = csv.Sniffer().has_header(csvfile.read(1024))
            csvfile.seek(0)
            dialect = csv.Sniffer().sniff(csvfile.read(1024))
            csvfile.seek(0)
            if (
                isinstance(facol, int)
                and isinstance(ficol, int)
                and isinstance(lacol, int)
                and isinstance(licol, int)
                and isinstance(scol, int)
            ):
                if has_header:
                    reader = csv.DictReader(csvfile, dialect=dialect)
                    faname, finame = reader.fieldnames[facol], reader.fieldnames[ficol]
                    laname, liname = reader.fieldnames[lacol], reader.fieldnames[licol]
                    sname = reader.fieldnames[scol]
                    r = [
                        (
                            float(row[faname]),
                            float(row[finame]),
                            float(row[laname]),
                            float(row[liname]),
                            int(row[sname]),
                        )
                        for row in reader
                    ]
                else:
                    reader = csv.reader(csvfile, dialect=dialect)
                    r = [
                        (
                            float(row[facol]),
                            float(row[ficol]),
                            float(row[lacol]),
                            float(row[licol]),
                            int(row[scol]),
                        )
                        for row in reader
                    ]
            else:
                if has_header:
                    reader = csv.DictReader(csvfile, dialect=dialect)
                    r = [
                        (
                            float(row[facol]),
                            float(row[ficol]),
                            float(row[lacol]),
                            float(row[licol]),
                            int(row[scol]),
                        )
                        for row in reader
                    ]
                else:
                    raise ValueError("No header line in CSV file...")

        fazi, finc, lazi, linc, sense = zip(*r)
        return cls.from_array(fazi, finc, lazi, linc, sense, name=basename(filename))

    def to_csv(self, filename, delimiter=","):
        """Save ``FaultSet`` object to csv file

        Args:
          filename (str): name of CSV file to save.

        Keyword Args:
          delimiter (str): values delimiter. Default ','

        Note: Written values are rounded according to `ndigits` settings in apsg_conf

        """
        import csv

        n = apsg_conf["ndigits"]

        with open(filename, "w", newline="") as csvfile:
            fieldnames = ["fazi", "finc", "lazi", "linc", "sense"]
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            for dt in self:
                fazi, finc = dt.fol.geo
                lazi, linc = dt.lin.geo
                writer.writerow(
                    {
                        "fazi": round(fazi, n),
                        "finc": round(finc, n),
                        "lazi": round(lazi, n),
                        "linc": round(linc, n),
                        "sense": dt.sense,
                    }
                )

    @classmethod
    def from_array(cls, fazis, fincs, lazis, lincs, senses, name="Default"):
        """Create ``PairSet`` from arrays of azimuths and inclinations

        Args:
          azis: list or array of azimuths
          incs: list or array of inclinations

        Keyword Args:
          name: name of ``PairSet`` object. Default is 'Default'
        """

        dtype_cls = getattr(sys.modules[__name__], cls.__feature_type__)
        return cls(
            [
                dtype_cls(fazi, finc, lazi, linc, sense)
                for fazi, finc, lazi, linc, sense in zip(
                    fazis, fincs, lazis, lincs, senses
                )
            ],
            name=name,
        )


class ConeSet(FeatureSet):
    """
    Class to store set of ``Cone`` features
    """

    __feature_type__ = "Cone"

    def __repr__(self):
        return f"C({len(self)}) {self.name}"


class EllipseSet(FeatureSet):
    """
    Class to store set of ``Ellipse`` features
    """

    __feature_type__ = "Ellipse"

    @property
    def S1(self) -> np.ndarray:
        """
        Return the array of maximum principal stretches.
        """
        return np.array([e.S1 for e in self])

    @property
    def S2(self) -> np.ndarray:
        """
        Return the array of minimum principal stretches.
        """
        return np.array([e.S2 for e in self])

    @property
    def e1(self) -> np.ndarray:
        """
        Return the maximum natural principal strains.
        """
        return np.array([e.e1 for e in self])

    @property
    def e2(self) -> np.ndarray:
        """
        Return the array of minimum natural principal strains.
        """
        return np.array([e.e2 for e in self])

    @property
    def ar(self) -> np.ndarray:
        """
        Return the array of axial ratios.
        """
        return np.array([e.ar for e in self])

    @property
    def orientation(self) -> np.ndarray:
        """
        Return the array of orientations of the maximum eigenvector.
        """
        return np.array([e.orientation for e in self])

    @property
    def e12(self) -> np.ndarray:
        """
        Return the array of differences between natural principal strains.
        """
        return np.array([e.e12 for e in self])


class OrientationTensor2Set(EllipseSet):
    """
    Class to store set of ``OrientationTensor2`` features
    """

    __feature_type__ = "OrientationTensor2"


class EllipsoidSet(FeatureSet):
    """
    Class to store set of ``Ellipsoid`` features
    """

    __feature_type__ = "Ellipsoid"

    @property
    def strength(self) -> np.ndarray:
        """
        Return the array of the Woodcock strength.
        """
        return np.array([e.strength for e in self])

    @property
    def shape(self) -> np.ndarray:
        """
        Return the array of the Woodcock shape.
        """
        return np.array([e.shape for e in self])

    @property
    def S1(self) -> np.ndarray:
        """
        Return the array of maximum principal stretches.
        """
        return np.array([e.S1 for e in self])

    @property
    def S2(self) -> np.ndarray:
        """
        Return the array of middle principal stretches.
        """
        return np.array([e.S2 for e in self])

    @property
    def S3(self) -> np.ndarray:
        """
        Return the array of minimum principal stretches.
        """
        return np.array([e.S3 for e in self])

    @property
    def e1(self) -> np.ndarray:
        """
        Return the array of the maximum natural principal strain.
        """
        return np.array([e.e1 for e in self])

    @property
    def e2(self) -> np.ndarray:
        """
        Return the array of the middle natural principal strain.
        """
        return np.array([e.e2 for e in self])

    @property
    def e3(self) -> np.ndarray:
        """
        Return the array of the minimum natural principal strain.
        """
        return np.array([e.e3 for e in self])

    @property
    def Rxy(self) -> np.ndarray:
        """
        Return the array of the Rxy ratios.
        """
        return np.array([e.Rxy for e in self])

    @property
    def Ryz(self) -> np.ndarray:
        """
        Return the array of the Ryz ratios.
        """
        return np.array([e.Ryz for e in self])

    @property
    def e12(self) -> np.ndarray:
        """
        Return the array of the e1 - e2 values.
        """
        return np.array([e.e12 for e in self])

    @property
    def e13(self) -> np.ndarray:
        """
        Return the array of the e1 - e3 values.
        """
        return np.array([e.e13 for e in self])

    @property
    def e23(self) -> np.ndarray:
        """
        Return the array of the e2 - e3 values.
        """
        return np.array([e.e23 for e in self])

    @property
    def k(self) -> np.ndarray:
        """
        Return the array of the strain symmetries.
        """
        return np.array([e.k for e in self])

    @property
    def d(self) -> np.ndarray:
        """
        Return the array of the strain intensities.
        """
        return np.array([e.d for e in self])

    @property
    def K(self) -> np.ndarray:
        """
        Return the array of the strain symmetries K (Ramsay, 1983).
        """
        return np.array([e.K for e in self])

    @property
    def D(self) -> np.ndarray:
        """
        Return the array of the strain intensities D (Ramsay, 1983)..
        """
        return np.array([e.D for e in self])

    @property
    def r(self) -> np.ndarray:
        """
        Return the array of the strain intensities (Watterson, 1968).
        """
        return np.array([e.r for e in self])

    @property
    def goct(self) -> np.ndarray:
        """
        Return the array of the natural octahedral unit shears (Nadai, 1963).
        """
        return np.array([e.goct for e in self])

    @property
    def eoct(self) -> np.ndarray:
        """
        Return the array of the natural octahedral unit strains (Nadai, 1963).
        """
        return np.array([e.eoct for e in self])

    @property
    def lode(self) -> np.ndarray:
        """
        Return the array of Lode parameters (Lode, 1926).
        """
        return np.array([e.lode for e in self])

    @property
    def P(self) -> np.ndarray:
        """
        Return the array of Point indexes (Vollmer, 1990).
        """
        return np.array([e.P for e in self])

    @property
    def G(self) -> np.ndarray:
        """
        Return the array of Girdle indexes (Vollmer, 1990).
        """
        return np.array([e.G for e in self])

    @property
    def R(self) -> np.ndarray:
        """
        Return the array of Random indexes (Vollmer, 1990).
        """
        return np.array([e.R for e in self])

    @property
    def B(self) -> np.ndarray:
        """
        Return the array of Cylindricity indexes (Vollmer, 1990).
        """
        return np.array([e.B for e in self])

    @property
    def Intensity(self) -> np.ndarray:
        """
        Return the array of Intensity indexes (Lisle, 1985).
        """
        return np.array([e.Intensity for e in self])

    @property
    def aMAD_l(self) -> np.ndarray:
        """
        Return approximate angular deviation from the major axis along E1.
        """
        return np.array([e.aMAD_l for e in self])

    @property
    def aMAD_p(self) -> np.ndarray:
        """
        Return approximate deviation from the plane normal to E3.
        """
        return np.array([e.aMAD_p for e in self])

    @property
    def aMAD(self) -> np.ndarray:
        """
        Return approximate deviation according to the shape
        """
        return np.array([e.aMAD for e in self])

    @property
    def MAD_l(self) -> np.ndarray:
        """
        Return maximum angular deviation (MAD) of linearly distributed vectors.
        Kirschvink 1980
        """
        return np.array([e.MAD_l for e in self])

    @property
    def MAD_p(self) -> np.ndarray:
        """
        Return maximum angular deviation (MAD) of planarly distributed vectors.
        Kirschvink 1980
        """
        return np.array([e.MAD_p for e in self])

    @property
    def MAD(self) -> np.ndarray:
        """
        Return approximate deviation according to shape
        """
        return np.array([e.MAD for e in self])


class OrientationTensor3Set(EllipsoidSet):
    """
    Class to store set of ``OrientationTensor3`` features
    """

    __feature_type__ = "OrientationTensor3"


class ClusterSet(object):
    """
    Provides a hierarchical clustering using `scipy.cluster` routines.
    The distance matrix is calculated as an angle between features, where ``Foliation``
    and ``Lineation`` use axial angles while ``Vector3`` uses direction angles.

    Args:
        azi (float): plunge direction of linear feature in degrees
        inc (float): plunge of linear feature in degrees

    Keyword Args:
        maxclust (int): Desired number of clusters. Default 2
        angle (float): Forms flat clusters so that the original observations in each
            cluster have no greater angle. Default is None to use maxclust criterion.
        method (str): Method for calculating the distance between the newly formed
            cluster and observations. Default is 'average' for UPGMA algorithm

    """

    def __init__(self, d, **kwargs):
        assert (
            isinstance(d, Vector2Set)
            or isinstance(d, Vector3Set)
            or isinstance(d, PairSet)
        ), "Only vec2set or vecset could be clustered"
        self.data = d.copy()
        self.maxclust = kwargs.get("maxclust", 2)
        self.angle = kwargs.get("angle", None)
        self.method = kwargs.get("method", "average")
        self.pdist = self.data.angle()
        self.linkage()
        self.cluster()

    def __repr__(self):
        info = f"Already {len(self.groups)} clusters created."
        if self.angle is not None:
            crit = f"Criterion: Angle\nSettings: distance={self.angle:.4g}\n"
        else:
            crit = f"Criterion: Maxclust\nSettings: maxclust={self.maxclust:.4g}\n"
        return (
            "ClusterSet\n"
            + f"Number of data: {len(self.data)}\n"
            + f"Linkage method: {self.method}\n"
            + crit
            + info
        )

    def cluster(self, **kwargs):
        """Do clustering on data

        Result is stored as tuple of Groups in ``groups`` property.

        Keyword Args:
          maxclust: number of clusters
          distance: maximum cophenetic distance in clusters
        """

        self.maxclust = kwargs.get("maxclust", self.maxclust)
        self.angle = kwargs.get("angle", self.angle)

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

        self.method = kwargs.get("method", self.method)
        self.Z = linkage(self.pdist, method=self.method, metric=angle_metric)

    def dendrogram(self, **kwargs):
        """Show dendrogram

        See ``scipy.cluster.hierarchy.dendrogram`` for possible kwargs.
        """

        fig, ax = plt.subplots(figsize=apsg_conf["figsize"])
        dendrogram(self.Z, ax=ax, **kwargs)
        plt.show()

    def elbow(self, no_plot=False, n=None):
        """Plot within groups variance vs. number of clusters.
        Elbow criterion could be used to determine number of clusters.
        """

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
            var = [100 * self.data[ix].var() for ix in grp]
            within_grp_var.append(var)
            mean_var.append(np.mean(var))
        if not no_plot:
            fig, ax = plt.subplots(figsize=apsg_conf["figsize"])
            ax.boxplot(within_grp_var, positions=nclust)
            ax.plot(nclust, mean_var, "k")
            ax.set_xlabel("Number of clusters")
            ax.set_ylabel("Variance")
            ax.set_title("Within-groups variance vs. number of clusters")
            plt.show()
        else:
            return nclust, within_grp_var

    @property
    def R(self):
        """Return group of clusters resultants."""
        return type(self.data)([group.R() for group in self.groups])


def G(lst, name="Default"):
    """
    Function to create appropriate container (FeatueSet) from list of features.

    Args:
        lst (list): Homogeneous list of objects of ``Vector2``, ``Vector3``,
            ``Lineation``, ``Foliation``, ``Pair``, ``Cone``, ``Ellipse``
            or ``OrientationTensor3``.

    Keyword Args:
        name (str): name of feature set. Default `Default`

    Example:
        >>> fols = [fol(120,30), fol(130, 40), fol(126, 37)]
        >>> f = G(fols)
    """
    if hasattr(lst, "__len__"):
        dtype_cls = type(lst[0])
        assert all([isinstance(obj, dtype_cls) for obj in lst])
        if dtype_cls is Vector3:
            return Vector3Set(lst, name=name)
        elif dtype_cls is Vector2:
            return Vector2Set(lst, name=name)
        elif dtype_cls is Lineation:
            return LineationSet(lst, name=name)
        elif dtype_cls is Foliation:
            return FoliationSet(lst, name=name)
        elif dtype_cls is Pair:
            return PairSet(lst, name=name)
        elif dtype_cls is Fault:
            return FaultSet(lst, name=name)
        elif dtype_cls is Cone:
            return ConeSet(lst, name=name)
        elif dtype_cls is Ellipsoid:
            return EllipsoidSet(lst, name=name)
        elif dtype_cls is OrientationTensor3:
            return OrientationTensor3Set(lst, name=name)
        elif dtype_cls is Ellipse:
            return EllipseSet(lst, name=name)
        elif dtype_cls is OrientationTensor2:
            return OrientationTensor2Set(lst, name=name)
        else:
            raise TypeError("Wrong datatype to create FeatureSet")


def angle_metric(u, v):
    return np.degrees(np.arccos(np.abs(np.dot(u, v))))
