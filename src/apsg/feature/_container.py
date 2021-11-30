import sys
from itertools import combinations
import numpy as np

from apsg.math._vector import Vector3
from apsg.feature._geodata import Lineation, Foliation, Pair


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

    def to_lin(self):
        """Return ``LineationSet`` object with all data converted to ``Lineation``."""
        return LineationSet([Lineation(e) for e in self], name=self.name)

    def to_fol(self):
        """Return ``FoliationSet`` object with all data converted to ``Foliation``."""
        return FoliationSet([Foliation(e) for e in self], name=self.name)

    def to_vec3(self):
        """Return ``Vector3Set`` object with all data converted to ``Vector3``."""
        return Vector3Set([Vector3(e) for e in self], name=self.name)

    def cross(self, other=None):
        """Return cross products of all data in ``FeatureSet``

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

    def rotate(self, axis, phi):
        """Rotate ``FeatureSet`` object `phi` degress about `axis`."""
        return type(self)([e.rotate(axis, phi) for e in self], name=self.name)


class Vector3Set(FeatureSet):
    __feature_type__ = "Vector3"

    def __repr__(self):
        return f"V({len(self)}) {self.name}"


class LineationSet(FeatureSet):
    __feature_type__ = "Lineation"

    def __repr__(self):
        return f"L({len(self)}) {self.name}"


class FoliationSet(FeatureSet):
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
