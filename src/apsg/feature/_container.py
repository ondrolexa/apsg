import numpy as np 

from apsg.base_classes import Vec3, Axial, Matrix3
from apsg.geodata import Lin, Fol, Pair


class Group:
    __slots__ = ["data", "dtype", "name"]

    @staticmethod
    def from_list(lst, name="Default"):
        if hasattr(lst, "__len__"):
            g = Group(dtype=type(lst[0]), name=name)
            for obj in lst:
                g.append(obj)
            return g
        else:
            raise TypeError("Argument must be list-like value")

    def __init__(self, dtype=Vec3, name="Default"):
        self.data = []
        self.dtype = dtype
        self.name = name

    def __copy__(self):
        return Group.from_list(self.data, name=self.name)

    copy = __copy__

    def __repr__(self):
        return f"G:{len(self)} {self.dtype.__name__} '{self.name}'"

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

    def __setitem__(self, key, obj):
        if isinstance(obj, self.dtype):
            self.data[key] = obj

    def __iter__(self):
        return iter(self.data)

    def __add__(self, other):
        if isinstance(other, Group):
            if self.dtype == other.dtype:
                g = Group(dtype=self.dtype, name=self.name)
                g.data = self.data.copy()
                g.data.extend(other.data)
                return g
            else:
                raise TypeError("Only same dtype is allowed")
        else:
            raise TypeError("Only Group is allowed")

    def append(self, obj):
        if isinstance(obj, self.dtype):
            self.data.append(obj)

    def extend(self, other):
        if hasattr(other, "__len__"):
            for obj in other:
                self.append(obj)

    def to_lin(self):
        """Return ``Group`` object with all data converted to ``Lin``."""
        return Group.from_list([Lin(e) for e in self], name=self.name)

    def to_fol(self):
        """Return ``Group`` object with all data converted to ``Fol``."""
        return Group.from_list([Fol(e) for e in self], name=self.name)

    def to_vec3(self):
        """Return ``Group`` object with all data converted to ``Vec3``."""
        return Group.from_list([Vec3(e) for e in self], name=self.name)

    def cross(self, other=None):
        """Return cross products of all data in ``Group`` object

        Without arguments it returns cross product of all pairs in dataset.
        If argument is group or single data object all mutual cross products
        are returned.
        """
        res = []
        if other is None:
            res = [e.cross(f) for e, f in combinations(self.data, 2)]
        elif isinstance(other, Group):
            res = [e.cross(f) for e in self for f in other]
        elif issubclass(type(other), Vec3):
            res = [e.cross(other) for e, f in combinations(self.data, 2)]
        else:
            raise TypeError("Wrong argument type!")
        return Group.from_list(res, name=self.name)

    def rotate(self, axis, phi):
        """Rotate ``Group`` object `phi` degress about `axis`."""
        return Group([e.rotate(axis, phi) for e in self], name=self.name)


