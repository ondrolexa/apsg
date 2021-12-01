import sys
from itertools import combinations
import numpy as np

from apsg.math._vector import Vector3
from apsg.feature._geodata import Lineation, Foliation, Pair
from apsg.feature._tensor import Ortensor3


class FeatureSet:
    __slots__ = ("data", "name")

    def __init__(self, data, name="Default"):
        dtype_cls = getattr(sys.modules[__name__], type(self).__feature_type__)
        assert all([isinstance(obj, dtype_cls) for obj in data])
        self.data = tuple(data)
        self.name = name

    def __copy__(self):
        return type(self).from_list(self.data, name=self.name)

    def to_json(self):
        return {
            "datatype": type(self).__name__,
            "args": {"data": tuple(item.to_json() for item in self)},
            "kwargs": {"name": self.name},
        }

    copy = __copy__

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
        If argument is group of same length or single data object element-wise
        cross-product is calculated.
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

    def R(self):
        """Return resultant of data in ``FeatureSet`` object.

        Resultant is of same type as features in ``FeatureSet``. Note
        that ``Foliation`` and ``Lineation`` are axial in nature so
        resultant can give other result than expected. Anyway for axial
        data orientation tensor analysis will give you right answer.

        As axial summing is not commutative we use vectorial summing of
        centered data for Fol and Lin
        """
        return sum(self)

    def var(self):
        """Spherical variance based on resultant length (Mardia 1972).

        var = 1 - |R| / n
        """
        return 1 - abs(self.uv.R) / len(self)

    def ortensor(self):
        """Return orientation tensor ``Ortensor`` of ``Group``."""

        return Ortensor3.from_features(self)

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
        from os.path import basename
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
    def random_normal(cls,  N=100, mean=Vector3(0, 0, 1), sigma=20, name="Default"):
        """Method to create ``FeatureSet`` of normaly distributed features.

        Keyword Args:
          N: number of objects to be generated
          mean: mean orientation given as ``Vector3``. Default Vector3(0, 0, 1)
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
        ax = orig.cross(mean)
        ang = orig.angle(mean)
        for s, r in zip(180 * np.random.uniform(low=0, high=180, size=N), np.random.normal(loc=0, scale=sigma, size=N)):
            v = orig.rotate(Vector3(s, 0), r).rotate(ax, ang)
            data.append(dtype_cls(v))
        return cls(data, name=name)


class LineationSet(Vector3Set):
    __feature_type__ = "Lineation"

    def __repr__(self):
        return f"L({len(self)}) {self.name}"


class FoliationSet(Vector3Set):
    __feature_type__ = "Foliation"

    def __repr__(self):
        return f"S({len(self)}) {self.name}"


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