import numpy as np
import pandas as pd
from pandas.core.dtypes.dtypes import PandasExtensionDtype
from pandas.api.extensions import ExtensionArray
from apsg.math import Vector3
from apsg.feature import Foliation, Lineation, Fault
from apsg.feature import FoliationSet, LineationSet, Vector3Set, FaultSet
from apsg.plotting import StereoNet


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


class Vector3Array(ExtensionArray):
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

    def __array__(self, dtype=str, copy=None):
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
            fols (FoliationSetor list of Foliation): set of fols
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
            lins (LineationSetor list of Lineation): set of lins
        """
        self._obj = LineationSet(lins)

    @property
    def dtype(self):
        """
        Return Dtype instance (not class) associated with this Array
        """
        return LinDtype()


class FaultArray(Vector3Array):
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


@pd.api.extensions.register_dataframe_accessor("apsg")
class APSGAccessor:
    """
    `apsg` DataFrame accessor to create aspg columns from data
    """

    def __init__(self, pandas_obj):
        self._obj = pandas_obj

    def create_vecs(self, columns=["x", "y", "z"], name="vecs"):
        """Create column with `Vector3` features

        Keyword Args:
            columns (list): Columns containing either x, y and z components
              or azi and inc. Default ["x", "y", "z"]
            name (str): Name of created column. Default 'vecs'
        """
        res = self._obj.copy()
        seq = [Vector3(*row.values) for _, row in self._obj[columns].iterrows()]
        res[name] = Vector3Array(seq)
        return res

    def create_fols(self, columns=["azi", "inc"], name="fols"):
        """Create column with `Foliation` features

        Keyword Args:
            columns (list): Columns containing azi and inc.
              Default ["azi", "inc"]
            name (str): Name of created column. Default 'fols'
        """
        res = self._obj.copy()
        seq = [Foliation(*row.values) for _, row in self._obj[columns].iterrows()]
        res[name] = FolArray(seq)
        return res

    def create_lins(self, columns=["azi", "inc"], name="lins"):
        """Create column with `Lineation` features

        Keyword Args:
            columns (list): Columns containing azi and inc.
              Default ["azi", "inc"]
            name (str): Name of created column. Default 'lins'
        """
        res = self._obj.copy()
        seq = [Lineation(*row.values) for _, row in self._obj[columns].iterrows()]
        res[name] = LinArray(seq)
        return res

    def create_faults(
        self, columns=["fazi", "finc", "lazi", "linc", "sense"], name="faults"
    ):
        """Create column with `Fault` features

        Keyword Args:
            columns (list): Columns containing azi and inc.
              Default ['fazi', 'finc', 'lazi', 'linc', 'sense']
            name (str): Name of created column. Default 'lins'
        """
        res = self._obj.copy()
        seq = [Fault(*row.values) for _, row in self._obj[columns].iterrows()]
        res[name] = FaultArray(seq)
        return res


class VectorSetBaseAccessor:
    """
    Base class of DataFrame accessors provides methods for FeatureSet
    """

    def __init__(self, pandas_obj):
        c = self._validate(pandas_obj)
        self._obj = pandas_obj
        self._col = c

    @property
    def getset(self):
        """Get ``FeatureSet``"""
        res = self._obj[self._col].array._obj
        res.name = self._col
        return res

    def R(self):
        """Return resultant of data in ``FeatureSet``."""
        return self.getset.R()

    def fisher_k(self):
        """Precision parameter based on Fisher's statistics"""
        stats = self.getset.fisher_statistics()
        return stats["k"]

    def fisher_csd(self):
        """Angular standard deviation based on Fisher's statistics"""
        stats = self.getset.fisher_statistics()
        return stats["csd"]

    def fisher_a95(self):
        """95% confidence limit based on Fisher's statistics"""
        stats = self.getset.fisher_statistics()
        return stats["a95"]

    def var(self):
        """Spherical variance based on resultant length (Mardia 1972).

        var = 1 - abs(R) / n
        """
        return self.getset.var()

    def delta(self):
        """Cone angle containing ~63% of the data in degrees.

        For enough large sample it approach angular standard deviation (csd)
        of Fisher statistics
        """
        return self.getset.delta()

    def rdegree(self):
        """Degree of preffered orientation of vectors in ``FeatureSet``.

        D = 100 * (2 * abs(R) - n) / n
        """
        return self.getset.rdegree()

    def ortensor(self):
        """Return orientation tensor ``Ortensor`` of vectors in ``FeatureSet``."""
        return self.getset.ortensor()

    def contour(self, snet=None, **kwargs):
        """Plot data contours on StereoNet"""
        if snet is None:
            s = StereoNet()
            s.contour(self.getset, **kwargs)
            s.show()
        else:
            snet.contour(self.getset, **kwargs)


@pd.api.extensions.register_dataframe_accessor("vec")
class Vec3Accessor(VectorSetBaseAccessor):
    """
    `vec` DataFrame accessor provides methods for Vector3Set
    """

    @staticmethod
    def _validate(obj):
        ok = False
        for c, dtype in zip(obj.columns, obj.dtypes):
            if dtype == "vec":
                ok = True
                break
        if not ok:
            raise AttributeError("Must have column with 'vec' dtype.")
        else:
            return c

    def plot(self, snet=None, **kwargs):
        """Plot vecs as vectors on StereoNet"""
        if snet is None:
            s = StereoNet()
            s.vector(self.getset, **kwargs)
            s.show()
        else:
            snet.vector(self.getset, **kwargs)


@pd.api.extensions.register_dataframe_accessor("fol")
class FolAccessor(VectorSetBaseAccessor):
    """
    `fol` DataFrame accessor provides methods for FoliationSet
    """

    @staticmethod
    def _validate(obj):
        ok = False
        for c, dtype in zip(obj.columns, obj.dtypes):
            if dtype == "fol":
                ok = True
                break
        if not ok:
            raise AttributeError("Must have column with 'fol' dtype.")
        else:
            return c

    def plot(self, snet=None, aspole=False, **kwargs):
        """Plot fols as great circles on StereoNet"""
        if snet is None:
            s = StereoNet()
            if aspole:
                s.pole(self.getset, **kwargs)
            else:
                s.great_circle(self.getset, **kwargs)
            s.show()
        else:
            if aspole:
                snet.pole(self.getset, **kwargs)
            else:
                snet.great_circle(self.getset, **kwargs)


@pd.api.extensions.register_dataframe_accessor("lin")
class LinAccessor(VectorSetBaseAccessor):
    """
    `lin` DataFrame accessor provides methods for LineationSet
    """

    @staticmethod
    def _validate(obj):
        ok = False
        for c, dtype in zip(obj.columns, obj.dtypes):
            if dtype == "lin":
                ok = True
                break
        if not ok:
            raise AttributeError("Must have column with 'lin' dtype.")
        else:
            return c

    def plot(self, snet=None, **kwargs):
        """Plot lins as line on StereoNet"""
        if snet is None:
            s = StereoNet()
            s.line(self.getset, **kwargs)
            s.show()
        else:
            snet.line(self.getset, **kwargs)


@pd.api.extensions.register_dataframe_accessor("fault")
class FaultAccessor:
    """
    `fault` DataFrame accessor provides methods for FaultSet
    """

    def __init__(self, pandas_obj):
        c = self._validate(pandas_obj)
        self._obj = pandas_obj
        self._col = c

    @property
    def getset(self):
        """Get ``FeatureSet``"""
        res = self._obj[self._col].array._obj
        res.name = self._col
        return res

    @staticmethod
    def _validate(obj):
        ok = False
        for c, dtype in zip(obj.columns, obj.dtypes):
            if dtype == "fault":
                ok = True
                break
        if not ok:
            raise AttributeError("Must have column with 'fault' dtype.")
        else:
            return c

    def plot(self, snet=None, **kwargs):
        """Plot vecs as vectors on StereoNet"""
        if snet is None:
            s = StereoNet()
            s.fault(self.getset, **kwargs)
            s.show()
        else:
            snet.fault(self.getset, **kwargs)
