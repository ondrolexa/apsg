import warnings

import numpy as np
import pandas as pd
from pandas.api.extensions import ExtensionArray, ExtensionDtype

from apsg.feature import (
    Fault,
    FaultSet,
    Foliation,
    FoliationSet,
    Lineation,
    LineationSet,
    Vector3Set,
)
from apsg.math import Vector3


@pd.api.extensions.register_extension_dtype
class Vec3Dtype(ExtensionDtype):
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
        return Vec3Array

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


@pd.api.extensions.register_extension_dtype
class FaultDtype(Vec3Dtype):
    """
    Class to describe the custom Fault data type
    """

    type = Fault  # Scalar type for data
    name = "fault"  # String identifying the data type name

    @classmethod
    def construct_array_type(cls):
        """
        Return array type associated with this dtype
        """
        return FaultArray


class Vec3Array(ExtensionArray):
    """
    Custom Extension Array type for an array of Vector3
    """

    def __init__(self, vecs):
        """
        Initialise array of vecs

        Args:
            vecs (Vector3Setor list of Vector3): set of vectors
        """
        self._obj = Vector3Set(vecs)

    @property
    def dtype(self):
        """
        Return Dtype instance (not class) associated with this Array
        """
        return Vec3Dtype()

    def __array__(self, dtype=None, copy=None):
        if dtype in (None, object):
            out = np.empty(len(self._obj), dtype=object)
            for i, e in enumerate(self._obj):
                out[i] = e
            return out
        return np.array(list(self._obj), dtype=dtype)

    @classmethod
    def _from_sequence(cls, scalars, dtype=None, copy=False):
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
        data = [item for arr in to_concat for item in arr._obj]
        return cls(data)

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
        if isinstance(other, type(self)):
            return np.array([a == b for a, b in zip(self._obj, other._obj)])
        return np.array([a == other for a in self._obj])

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
        return type(self)(self._obj[indices])

    def copy(self):
        """
        Return copy of array
        """
        return type(self)(self._obj)


class FolArray(Vec3Array):
    """
    Custom Extension Array type for an array of fols
    """

    def __init__(self, fols):
        """
        Initialise array of fols

        Args:
            fols (FoliationSetor list of Foliation): set of fols
        """
        self._obj = FoliationSet(fols)

    @property
    def dtype(self):
        """
        Return Dtype instance (not class) associated with this Array
        """
        return FolDtype()


class LinArray(Vec3Array):
    """
    Custom Extension Array type for an array of lins
    """

    def __init__(self, lins):
        """
        Initialise array of lins

        Args:
            lins (LineationSetor list of Lineation): set of lins
        """
        self._obj = LineationSet(lins)

    @property
    def dtype(self):
        """
        Return Dtype instance (not class) associated with this Array
        """
        return LinDtype()


class FaultArray(Vec3Array):
    """
    Custom Extension Array type for an array of faults
    """

    def __init__(self, faults):
        """
        Initialise array of faults

        Args:
            faults (FaultSet or list of Fault): set of faults
        """
        self._obj = FaultSet(faults)

    @property
    def dtype(self):
        """
        Return Dtype instance (not class) associated with this Array
        """
        return FaultDtype()


@pd.api.extensions.register_series_accessor("G")
class GAccessor:
    """Series accessor returning APSG feature set via ``series.G()``."""

    def __init__(self, pandas_obj):
        if not isinstance(
            pandas_obj.array, (Vec3Array, LinArray, FolArray, FaultArray)
        ):
            raise AttributeError(
                "Series must contain an APSG feature array "
                "(Vec3Array, LinArray, FolArray, or FaultArray)."
            )
        self._series = pandas_obj

    def __call__(self):
        res = self._series.array._obj
        res.name = self._series.name
        return res

    def plot(self, **kwargs):
        from apsg.plotting import quicknet

        quicknet(self(), **kwargs)


@pd.api.extensions.register_dataframe_accessor("apsg")
class APSGAccessor:
    """
    `apsg` DataFrame accessor to create aspg columns from data
    """

    def __init__(self, pandas_obj):
        self._obj = pandas_obj

    @staticmethod
    def _warn_on_invalid_name(name):
        if not name.isidentifier():
            warnings.warn(
                f"Column name '{name}' is not a valid Python identifier. "
                f"Use df['{name}'].G() instead of df.{name}.G()."
            )

    def create_vecs(self, columns=["x", "y", "z"], name="vecs"):
        """Create column with `Vector3` features

        Keyword Args:
            columns (list): Columns containing either x, y and z components
              or azi and inc. Default ["x", "y", "z"]
            name (str): Name of created column. Default 'vecs'
        """
        res = self._obj.copy()
        seq = [Vector3(*row) for row in self._obj[columns].itertuples(index=False)]
        res[name] = Vec3Array(seq)
        self._warn_on_invalid_name(name)
        return res

    def add_vecs(self, features, name="vecs"):
        """Add column with `Vector3` features

        Keyword Args:
            name (str): Name of created column. Default 'vecs'
        """
        if not isinstance(features, Vector3Set):
            raise TypeError("Argument must be Vector3Set")
        res = self._obj.copy()
        res[name] = Vec3Array(features.data)
        self._warn_on_invalid_name(name)
        return res

    def create_fols(self, columns=["azi", "inc"], name="fols"):
        """Create column with `Foliation` features

        Keyword Args:
            columns (list): Columns containing azi and inc.
              Default ["azi", "inc"]
            name (str): Name of created column. Default 'fols'
        """
        res = self._obj.copy()
        seq = [Foliation(*row) for row in self._obj[columns].itertuples(index=False)]
        res[name] = FolArray(seq)
        self._warn_on_invalid_name(name)
        return res

    def add_fols(self, features, name="fols"):
        """Add column with `Foliation` features

        Keyword Args:
            name (str): Name of created column. Default 'fols'
        """
        if not isinstance(features, FoliationSet):
            raise TypeError("Argument must be FoliationSet")
        res = self._obj.copy()
        res[name] = FolArray(features.data)
        self._warn_on_invalid_name(name)
        return res

    def create_lins(self, columns=["azi", "inc"], name="lins"):
        """Create column with `Lineation` features

        Keyword Args:
            columns (list): Columns containing azi and inc.
              Default ["azi", "inc"]
            name (str): Name of created column. Default 'lins'
        """
        res = self._obj.copy()
        seq = [Lineation(*row) for row in self._obj[columns].itertuples(index=False)]
        res[name] = LinArray(seq)
        self._warn_on_invalid_name(name)
        return res

    def add_lins(self, features, name="lins"):
        """Add column with `Lineation` features

        Keyword Args:
            name (str): Name of created column. Default 'lins'
        """
        if not isinstance(features, LineationSet):
            raise TypeError("Argument must be LineationSet")
        res = self._obj.copy()
        res[name] = LinArray(features.data)
        self._warn_on_invalid_name(name)
        return res

    def create_faults(
        self, columns=["fazi", "finc", "lazi", "linc", "sense"], name="faults"
    ):
        """Create column with `Fault` features

        Keyword Args:
            columns (list): Columns containing fault plane (fazi, finc),
              lineation (lazi, linc) and sense.
              Default ['fazi', 'finc', 'lazi', 'linc', 'sense']
            name (str): Name of created column. Default 'faults'
        """
        res = self._obj.copy()
        seq = [Fault(*row) for row in self._obj[columns].itertuples(index=False)]
        res[name] = FaultArray(seq)
        self._warn_on_invalid_name(name)
        return res

    def add_faults(self, features, name="faults"):
        """Add column with `Fault` features

        Keyword Args:
            name (str): Name of created column. Default 'faults'
        """
        if not isinstance(features, FaultSet):
            raise TypeError("Argument must be FaultSet")
        res = self._obj.copy()
        res[name] = FaultArray(features.data)
        self._warn_on_invalid_name(name)
        return res
