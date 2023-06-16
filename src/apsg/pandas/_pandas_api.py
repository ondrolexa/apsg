import numpy as np
import pandas as pd
from pandas.core.dtypes.dtypes import PandasExtensionDtype
from pandas.api.extensions import ExtensionArray
from apsg.math import Vector3
from apsg.feature import Foliation, Lineation
from apsg.feature import FoliationSet, LineationSet, Vector3Set
from apsg.plotting import StereoNet

"""
import pandas as pd
from apsg import *
from apsg.pandas import *
df = pd.DataFrame({
    'name':['A', 'B', 'C'],
    'azi':[120, 140, 135],
    'inc':[30, 28, 42]
})

df.apsg.create_fols().fol.R()

df = pd.DataFrame({
    'name':['A', 'B', 'C'],
    'fazi':[120, 140, 135],
    'finc':[30, 28, 42],
    'lazi':[170, 120, 95],
    'linc':[20, 26, 22]
})

df.apsg.create_fols(columns=['fazi', 'finc']).apsg.create_lins(columns=['lazi', 'linc'])
"""


@pd.api.extensions.register_extension_dtype
class Vec3Dtype(PandasExtensionDtype):
    """
    Class to describe the custom Vector3 data type
    """

    type = Vector3  # Scalar type for data
    name = "vec"  # String identifying the data type name

    @classmethod
    def construct_array_type(cls):
        """
        Return array type associated with this dtype
        """
        return Vector3Array

    def __str__(self):
        return self.name

    def __hash__(self):
        return id(self)


@pd.api.extensions.register_extension_dtype
class FolDtype(Vec3Dtype):
    """
    Class to describe the custom Foliation data type
    """

    type = Foliation  # Scalar type for data
    name = "fol"  # String identifying the data type name

    @classmethod
    def construct_array_type(cls):
        """
        Return array type associated with this dtype
        """
        return FolArray


@pd.api.extensions.register_extension_dtype
class LinDtype(Vec3Dtype):
    """
    Class to describe the custom Lineation data type
    """

    type = Lineation  # Scalar type for data
    name = "lin"  # String identifying the data type name

    @classmethod
    def construct_array_type(cls):
        """
        Return array type associated with this dtype
        """
        return LinArray


class Vector3Array(ExtensionArray):
    """
    Custom Extension Array type for an array of Vector3
    """

    def __init__(self, vecs):
        """
        Initialise array of vecs

        Args:
            vecs (Vector3Set): set of vectors
        """
        self._obj = Vector3Set(vecs)

    @property
    def dtype(self):
        """
        Return Dtype instance (not class) associated with this Array
        """
        return Vec3Dtype()

    def __array__(self, dtype=str):
        return np.array([str(f) for f in self._obj], dtype=dtype)

    @classmethod
    def _from_sequence(cls, scalars):
        """
        Construct a new ExtensionArray from a sequence of scalars.
        Each element will be an instance of the scalar type for this array,
        or be converted into this type in this method.
        """
        return cls(scalars)

    @classmethod
    def _concat_same_type(cls, to_concat):
        """
        Concatenate multiple arrays of this dtype
        """
        print(to_concat)
        res = to_concat[0]._obj
        for other in to_concat[1:]:
            res += other._obj
        return cls(res)

    @property
    def nbytes(self):
        """
        The number of bytes needed to store this object in memory.
        """
        return np.array(self._obj).nbytes

    def __getitem__(self, item):
        """
        Retrieve single item or slice
        """
        if isinstance(item, int):
            # Get single vector
            return self._obj[item]

        else:
            # Get subset from slice  or boolean array
            return type(self)(self._obj[item])

    def __eq__(self, other):
        """
        Perform element-wise equality with a given vector value
        """
        if isinstance(other, (pd.Index, pd.Series, pd.DataFrame)):
            return NotImplemented

        return self._obj == other._obj

    def __len__(self):
        return len(self._obj)

    def isna(self):
        """
        Returns a 1-D array indicating if each value is missing
        """
        return np.isnan(self._obj).sum(axis=1) > 0

    def take(self, indices, *, allow_fill=False, fill_value=None):
        """
        Take element from array using positional indexing
        """
        from pandas.core.algorithms import take

        if allow_fill and fill_value is None:
            fill_value = self.dtype.na_value

        fols = take(self._obj, indices, fill_value=fill_value, allow_fill=allow_fill)
        return type(self)(fols)

    def copy(self):
        """
        Return copy of array
        """
        return type(self)(self._obj)


class FolArray(Vector3Array):
    """
    Custom Extension Array type for an array of fols
    """

    def __init__(self, fols):
        """
        Initialise array of fols

        Args:
            fols (FoliationSet): set of fols
        """
        self._obj = FoliationSet(fols)

    @property
    def dtype(self):
        """
        Return Dtype instance (not class) associated with this Array
        """
        return FolDtype()


class LinArray(Vector3Array):
    """
    Custom Extension Array type for an array of lins
    """

    def __init__(self, lins):
        """
        Initialise array of lins

        Args:
            lins (LineationSet): set of lins
        """
        self._obj = LineationSet(lins)

    @property
    def dtype(self):
        """
        Return Dtype instance (not class) associated with this Array
        """
        return LinDtype()


@pd.api.extensions.register_dataframe_accessor("apsg")
class APSGAccessor:
    def __init__(self, pandas_obj):
        self._obj = pandas_obj

    def create_vecs(self, columns=["x", "y", "z"], name="vecs"):
        res = self._obj.copy()
        seq = [Vector3(*row.values) for _, row in self._obj[columns].iterrows()]
        res[name] = Vector3Array(seq)
        return res

    def create_fols(self, columns=["azi", "inc"], name="fols"):
        res = self._obj.copy()
        seq = [Foliation(*row.values) for _, row in self._obj[columns].iterrows()]
        res[name] = FolArray(seq)
        return res

    def create_lins(self, columns=["azi", "inc"], name="lins"):
        res = self._obj.copy()
        seq = [Lineation(*row.values) for _, row in self._obj[columns].iterrows()]
        res[name] = LinArray(seq)
        return res


class FeatureSetAccessor:
    def __init__(self, pandas_obj):
        c = self._validate(pandas_obj)
        self._obj = pandas_obj
        self._col = c

    @property
    def getset(self):
        return self._obj[self._col].array._obj

    def R(self):
        # return resultant foliation of fols in DataFrame
        return self.getset.R()

    def ortensor(self):
        # return orientation tensor of fols in DataFrame
        return self.getset.ortensor()


@pd.api.extensions.register_dataframe_accessor("vec")
class Vec3Accessor(FeatureSetAccessor):
    @staticmethod
    def _validate(obj):
        # verify there is a vec type column
        ok = False
        for c, dtype in zip(obj.columns, obj.dtypes):
            if dtype == "vec":
                ok = True
                break
        if not ok:
            raise AttributeError("Must have column with 'vec' dtype.")
        else:
            return c

    def vector(self, snet=None, **kwargs):
        # plot vecs as vectors on StereoNet
        if snet is None:
            s = StereoNet()
            s.vector(self.getset, **kwargs)
            s.show()
        else:
            snet.vector(self.getset, **kwargs)


@pd.api.extensions.register_dataframe_accessor("fol")
class FolAccessor(FeatureSetAccessor):
    @staticmethod
    def _validate(obj):
        # verify there is a fol type column
        ok = False
        for c, dtype in zip(obj.columns, obj.dtypes):
            if dtype == "fol":
                ok = True
                break
        if not ok:
            raise AttributeError("Must have column with 'fol' dtype.")
        else:
            return c

    def great_circle(self, snet=None, **kwargs):
        # plot fols as great circles on StereoNet
        if snet is None:
            s = StereoNet()
            s.great_circle(self.getset, **kwargs)
            s.show()
        else:
            snet.great_circle(self.getset, **kwargs)

    def pole(self, snet=None, **kwargs):
        # plot fols as poles on StereoNet
        if snet is None:
            s = StereoNet()
            s.pole(self.getset, **kwargs)
            s.show()
        else:
            snet.pole(self.getset, **kwargs)


@pd.api.extensions.register_dataframe_accessor("lin")
class LinAccessor(FeatureSetAccessor):
    @staticmethod
    def _validate(obj):
        # verify there is a fol type column
        ok = False
        for c, dtype in zip(obj.columns, obj.dtypes):
            if dtype == "lin":
                ok = True
                break
        if not ok:
            raise AttributeError("Must have column with 'lin' dtype.")
        else:
            return c

    def line(self, snet=None, **kwargs):
        # plot lins as line on StereoNet
        if snet is None:
            s = StereoNet()
            s.line(self.getset, **kwargs)
            s.show()
        else:
            snet.line(self.getset, **kwargs)
