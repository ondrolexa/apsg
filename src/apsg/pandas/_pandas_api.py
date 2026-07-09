import numpy as np
import pandas as pd
from pandas.api.extensions import ExtensionArray, ExtensionDtype
from pandas.core.groupby import SeriesGroupBy

from apsg.feature import (
    Direction,
    Direction2Set,
    Fault,
    FaultSet,
    FeatureSet,
    Foliation,
    FoliationSet,
    Lineation,
    LineationSet,
    Vector2Set,
    Vector3Set,
)
from apsg.math import Vector2, Vector3


@pd.api.extensions.register_extension_dtype
class Vec3Dtype(ExtensionDtype):
    """
    Class to describe the custom Vector3 data type.
    """

    type = Vector3  # Scalar type for data
    name = "vec"  # String identifying the data type name

    @classmethod
    def construct_array_type(cls):
        """
        Return array type associated with this dtype.
        """
        return Vec3Array

    def __str__(self):
        return self.name

    def __hash__(self):
        return id(self)


@pd.api.extensions.register_extension_dtype
class Vec2Dtype(ExtensionDtype):
    """
    Class to describe the custom Vector2 data type.
    """

    type = Vector2
    name = "vec2"

    @classmethod
    def construct_array_type(cls):
        """
        Return array type associated with this dtype.
        """
        return Vec2Array

    def __str__(self):
        return self.name

    def __hash__(self):
        return id(self)


@pd.api.extensions.register_extension_dtype
class DirDtype(Vec2Dtype):
    """
    Class to describe the custom Direction data type.
    """

    type = Direction
    name = "dir"

    @classmethod
    def construct_array_type(cls):
        """
        Return array type associated with this dtype.
        """
        return DirArray


@pd.api.extensions.register_extension_dtype
class FolDtype(Vec3Dtype):
    """
    Class to describe the custom Foliation data type.
    """

    type = Foliation  # Scalar type for data
    name = "fol"  # String identifying the data type name

    @classmethod
    def construct_array_type(cls):
        """
        Return array type associated with this dtype.
        """
        return FolArray


@pd.api.extensions.register_extension_dtype
class LinDtype(Vec3Dtype):
    """
    Class to describe the custom Lineation data type.
    """

    type = Lineation  # Scalar type for data
    name = "lin"  # String identifying the data type name

    @classmethod
    def construct_array_type(cls):
        """
        Return array type associated with this dtype.
        """
        return LinArray


@pd.api.extensions.register_extension_dtype
class FaultDtype(Vec3Dtype):
    """
    Class to describe the custom Fault data type.
    """

    type = Fault  # Scalar type for data
    name = "fault"  # String identifying the data type name

    @classmethod
    def construct_array_type(cls):
        """
        Return array type associated with this dtype.
        """
        return FaultArray


class _MaskedFeatureArray(ExtensionArray):
    """Base class for APSG ExtensionArrays with NA support.

    Stores data in ``_data`` (object ndarray, ``None`` for NA),
    ``_mask`` (bool ndarray), and ``_obj`` (FeatureSet of only valid entries).

    Subclasses must set ``_FEATURE_SET_CLASS``.
    """

    _FEATURE_SET_CLASS = None

    def __init__(self, data, mask=None):
        data = list(data)
        if mask is None:
            mask = np.zeros(len(data), dtype=bool)
        self._mask = np.asarray(mask, dtype=bool)
        self._data = np.empty(len(data), dtype=object)
        for i, v in enumerate(data):
            self._data[i] = v
        valid = [self._data[i] for i in range(len(self._data)) if not self._mask[i]]
        if valid:
            self._obj = self._FEATURE_SET_CLASS(valid)
        else:
            self._obj = self._FEATURE_SET_CLASS([])

    def __len__(self):
        return len(self._data)

    def __getitem__(self, item):
        if isinstance(item, int):
            if self._mask[item]:
                return pd.NA
            return self._data[item]
        elif isinstance(item, slice):
            return type(self)(self._data[item], mask=self._mask[item])
        else:
            idx = np.asarray(item)
            if idx.dtype == bool:
                return type(self)(self._data[idx], mask=self._mask[idx])
            return type(self)(
                [self._data[i] for i in idx],
                mask=np.array([self._mask[i] for i in idx]),
            )

    def __eq__(self, other):
        if isinstance(other, (pd.Index, pd.Series, pd.DataFrame)):
            return NotImplemented
        if isinstance(other, type(self)):
            return np.array(
                [
                    False if self._mask[i] else self._data[i] == other._data[i]
                    for i in range(len(self))
                ]
            )
        return np.array(
            [
                False if self._mask[i] else self._data[i] == other
                for i in range(len(self))
            ]
        )

    @classmethod
    def _from_sequence(cls, scalars, dtype=None, copy=False):
        data = []
        mask = []
        for v in scalars:
            if v is pd.NA or v is None:
                data.append(None)
                mask.append(True)
            elif isinstance(v, float) and np.isnan(v):
                data.append(None)
                mask.append(True)
            else:
                data.append(v)
                mask.append(False)
        return cls(data, mask=np.array(mask))

    @classmethod
    def _concat_same_type(cls, to_concat):
        data = []
        mask = []
        for arr in to_concat:
            data.extend(arr._data)
            mask.extend(arr._mask)
        return cls(data, mask=np.array(mask))

    @property
    def nbytes(self):
        return self._data.nbytes + self._mask.nbytes

    def isna(self):
        return self._mask.copy()

    def take(self, indices, *, allow_fill=False, fill_value=None):
        if allow_fill:
            data = []
            mask = []
            for i in indices:
                if i == -1:
                    data.append(fill_value)
                    mask.append(False)
                else:
                    data.append(self._data[i])
                    mask.append(self._mask[i])
            return type(self)(data, mask=np.array(mask))
        return type(self)(
            [self._data[i] for i in indices],
            mask=np.array([self._mask[i] for i in indices]),
        )

    def copy(self):
        return type(self)(list(self._data), mask=self._mask.copy())

    def __array__(self, dtype=None, copy=None):
        if dtype in (None, object):
            out = np.empty(len(self), dtype=object)
            for i in range(len(self)):
                out[i] = pd.NA if self._mask[i] else self._data[i]
            return out
        valid = [self._data[i] for i in range(len(self)) if not self._mask[i]]
        return np.array(valid, dtype=dtype)


class Vec3Array(_MaskedFeatureArray):
    """
    Custom Extension Array type for an array of Vector3.
    """

    _FEATURE_SET_CLASS = Vector3Set

    @property
    def dtype(self):
        """
        Return Dtype instance (not class) associated with this Array.
        """
        return Vec3Dtype()


class Vec2Array(_MaskedFeatureArray):
    """
    Custom Extension Array type for an array of Vector2.
    """

    _FEATURE_SET_CLASS = Vector2Set

    @property
    def dtype(self):
        """
        Return Dtype instance (not class) associated with this Array.
        """
        return Vec2Dtype()


class DirArray(_MaskedFeatureArray):
    """
    Custom Extension Array type for an array of directions.
    """

    _FEATURE_SET_CLASS = Direction2Set

    @property
    def dtype(self):
        """
        Return Dtype instance (not class) associated with this Array.
        """
        return DirDtype()


class FolArray(_MaskedFeatureArray):
    """
    Custom Extension Array type for an array of fols.
    """

    _FEATURE_SET_CLASS = FoliationSet

    @property
    def dtype(self):
        """
        Return Dtype instance (not class) associated with this Array.
        """
        return FolDtype()


class LinArray(_MaskedFeatureArray):
    """
    Custom Extension Array type for an array of lins.
    """

    _FEATURE_SET_CLASS = LineationSet

    @property
    def dtype(self):
        """
        Return Dtype instance (not class) associated with this Array.
        """
        return LinDtype()


class FaultArray(_MaskedFeatureArray):
    """
    Custom Extension Array type for an array of faults.
    """

    _FEATURE_SET_CLASS = FaultSet

    @property
    def dtype(self):
        """
        Return Dtype instance (not class) associated with this Array.
        """
        return FaultDtype()


_PLOT_FN_MAP = {
    Vector3: dict(plot="StereoNet", kinds=["line", "vector"]),
    Lineation: dict(plot="StereoNet", kinds=["line"]),
    Foliation: dict(plot="StereoNet", kinds=["gc", "pole"]),
    Fault: dict(plot="StereoNet", kinds=["fault", "pair", "hoeppner"]),
    Vector2: dict(plot="RosePlot", kinds=["bar", "pdf"]),
    Direction: dict(plot="RosePlot", kinds=["bar", "pdf"]),
}


@pd.api.extensions.register_series_accessor("G")
class GAccessor:
    """Series accessor returning APSG feature set via ``series.G()``."""

    def __init__(self, pandas_obj):
        if not isinstance(pandas_obj.array, _MaskedFeatureArray):
            raise AttributeError(
                "Series must contain an APSG feature array "
                "(Vec3Array, LinArray, FolArray, FaultArray, "
                "Vec2Array, or DirArray)."
            )
        self._series = pandas_obj

    def __call__(self):
        res = self._series.array._obj
        res.name = self._series.name
        return res

    def plot(self, **kwargs):
        from apsg.plotting import RosePlot, StereoNet

        kind = kwargs.pop("kind", None)
        plot_kws = kwargs.pop("plot_kws", {})
        show = kwargs.pop("show", True)

        scalar_type = self._series.array.dtype.type
        plot_info = _PLOT_FN_MAP.get(scalar_type)
        if plot_info is None:
            raise ValueError(f"No plot function registered for {scalar_type.__name__}")

        if kind is None:
            kind = plot_info["kinds"][0]

        plot_cls = {"StereoNet": StereoNet, "RosePlot": RosePlot}[plot_info["plot"]]
        plot_obj = plot_cls(**plot_kws)
        getattr(plot_obj, kind)(self(), **kwargs)

        if show:
            plot_obj.show()
            return None
        return plot_obj


class GAccessorGroupBy:
    """SeriesGroupBy accessor returning APSG feature array apply with dtype recast.

    ``df.groupby(...)["col"].G.apply(func)`` applies *func* to each group
    and returns a ``Series`` whose ``ExtensionDtype`` matches the scalar type
    returned by *func* (``Lineation`` → ``lin``, ``Foliation`` → ``fol``,
    ``Fault`` → ``fault``, ``Vector3`` → ``vec``). Non-APSG return values
    pass through unchanged.
    """

    _INPUT_DTYPE_MAP = {
        Vec3Dtype: (Vector3, Vec3Array, Vec3Dtype),
        LinDtype: (Lineation, LinArray, LinDtype),
        FolDtype: (Foliation, FolArray, FolDtype),
        FaultDtype: (Fault, FaultArray, FaultDtype),
        Vec2Dtype: (Vector2, Vec2Array, Vec2Dtype),
        DirDtype: (Direction, DirArray, DirDtype),
    }

    def __init__(self, pandas_obj):
        self._groupby = pandas_obj

    def apply(self, func, *args, **kwargs):
        """Apply func to each group and recast the result to the matching extension dtype."""
        raw = self._groupby.apply(func, *args, **kwargs)
        return self._recast(raw)

    def transform(self, func, *args, **kwargs):
        """Transform each group and recast the result to the matching extension dtype."""

        def wrapper(s):
            res = func(s, *args, **kwargs)
            if isinstance(res, FeatureSet):
                return list(res)
            return res

        raw = self._groupby.transform(wrapper)
        return self._recast(raw)

    def aggregate(self, func, *args, **kwargs):
        """Aggregate each group and recast the result to the matching extension dtype."""
        raw = self._groupby.aggregate(func, *args, **kwargs)
        return self._recast(raw)

    agg = aggregate

    def _recast(self, result):
        if not isinstance(result, pd.Series):
            return result
        scalars = list(result)
        if not scalars:
            return result
        # Case 1: scalars are already APSG objects (e.g. a function that
        # returned a DataFrame or a function that bypassed unpacking)
        info = self._SCALAR_TYPE_MAP.get(type(scalars[0]))
        if info is not None:
            array_cls, dtype_cls = info
            if all(isinstance(v, type(scalars[0])) for v in scalars):
                return self._build_result(result, scalars, array_cls)
            return result

        # Case 2: pandas unpacked APSG scalars into coordinate tuples
        # (Vector3.__getitem__ makes them look sequence-like).
        # Try to reconstruct from the input Series' dtype.
        first = scalars[0]
        if not isinstance(first, (tuple, list)) or len(first) not in (2, 3, 6, 7):
            return result
        input_dtype = self._groupby.obj.dtype
        input_info = self._INPUT_DTYPE_MAP.get(type(input_dtype))
        if input_info is None:
            return result
        scalar_cls, array_cls, _ = input_info
        try:
            reconstructed = [scalar_cls(*v) for v in scalars]
            return self._build_result(result, reconstructed, array_cls)
        except Exception:
            return result

    @staticmethod
    def _build_result(result, scalars, array_cls):
        new_arr = array_cls._from_sequence(scalars)
        return pd.Series(new_arr, index=result.index, name=result.name)

    _SCALAR_TYPE_MAP = {
        Vector3: (Vec3Array, Vec3Dtype),
        Lineation: (LinArray, LinDtype),
        Foliation: (FolArray, FolDtype),
        Fault: (FaultArray, FaultDtype),
        Vector2: (Vec2Array, Vec2Dtype),
        Direction: (DirArray, DirDtype),
    }


# Enable .G() on SeriesGroupBy (in addition to the existing accessor on pd.Series)
SeriesGroupBy.G = property(lambda self: GAccessorGroupBy(self))


@pd.api.extensions.register_dataframe_accessor("apsg")
class APSGAccessor:
    """
    `apsg` DataFrame accessor to create apsg columns from data.

    Note:
        Column name for created or added APSG features should be valid
        Python identifier to enable attribute-style access (e.g. ``df.my_col.G()``).
        For non-identifier names, use bracket indexing: ``df['my col'].G()``.
    """

    def __init__(self, pandas_obj):
        self._obj = pandas_obj

    def create_vecs(self, columns=["x", "y", "z"], name="vecs"):
        """Create column with `Vector3` features.

        Keyword Args:
            columns (list): Columns containing either x, y and z components
              or azi and inc. Default ["x", "y", "z"]
            name (str): Name of created column. Default 'vecs'
        """
        res = self._obj.copy()
        data = []
        mask = []
        for row in self._obj[columns].itertuples(index=False):
            if any(pd.isna(v) for v in row):
                data.append(None)
                mask.append(True)
            else:
                data.append(Vector3(*row))
                mask.append(False)
        res[name] = Vec3Array(data, mask=np.array(mask))
        return res

    def add_vecs(self, features, name="vecs"):
        """Add column with `Vector3` features.

        Keyword Args:
            name (str): Name of created column. Default 'vecs'
        """
        if not isinstance(features, Vector3Set):
            raise TypeError("Argument must be Vector3Set")
        res = self._obj.copy()
        res[name] = Vec3Array(features.data)
        return res

    def create_vecs2(self, columns=["x", "y"], name="vecs2"):
        """Create column with `Vector2` features.

        Keyword Args:
            columns (list): Columns containing x and y components. Default ["x", "y"]
            name (str): Name of created column. Default 'vecs2'
        """
        res = self._obj.copy()
        data = []
        mask = []
        for row in self._obj[columns].itertuples(index=False):
            if any(pd.isna(v) for v in row):
                data.append(None)
                mask.append(True)
            else:
                data.append(Vector2(*row))
                mask.append(False)
        res[name] = Vec2Array(data, mask=np.array(mask))
        return res

    def add_vecs2(self, features, name="vecs2"):
        """Add column with `Vector2` features.

        Keyword Args:
            name (str): Name of created column. Default 'vecs2'
        """
        if not isinstance(features, Vector2Set):
            raise TypeError("Argument must be Vector2Set")
        res = self._obj.copy()
        res[name] = Vec2Array(features.data)
        return res

    def create_dirs(self, columns=["angle"], name="dirs"):
        """Create column with `Direction` features.

        Keyword Args:
            columns (list): Column containing angle in degrees. Default ["angle"]
            name (str): Name of created column. Default 'dirs'
        """
        res = self._obj.copy()
        data = []
        mask = []
        for row in self._obj[columns].itertuples(index=False):
            if any(pd.isna(v) for v in row):
                data.append(None)
                mask.append(True)
            else:
                data.append(Direction(*row))
                mask.append(False)
        res[name] = DirArray(data, mask=np.array(mask))
        return res

    def add_dirs(self, features, name="dirs"):
        """Add column with `Direction` features.

        Keyword Args:
            name (str): Name of created column. Default 'dirs'
        """
        if not isinstance(features, Direction2Set):
            raise TypeError("Argument must be Direction2Set")
        res = self._obj.copy()
        res[name] = DirArray(features.data)
        return res

    def create_fols(self, columns=["azi", "inc"], name="fols"):
        """Create column with `Foliation` features.

        Keyword Args:
            columns (list): Columns containing azi and inc.
              Default ["azi", "inc"]
            name (str): Name of created column. Default 'fols'
        """
        res = self._obj.copy()
        data = []
        mask = []
        for row in self._obj[columns].itertuples(index=False):
            if any(pd.isna(v) for v in row):
                data.append(None)
                mask.append(True)
            else:
                data.append(Foliation(*row))
                mask.append(False)
        res[name] = FolArray(data, mask=np.array(mask))
        return res

    def add_fols(self, features, name="fols"):
        """Add column with `Foliation` features.

        Keyword Args:
            name (str): Name of created column. Default 'fols'
        """
        if not isinstance(features, FoliationSet):
            raise TypeError("Argument must be FoliationSet")
        res = self._obj.copy()
        res[name] = FolArray(features.data)
        return res

    def create_lins(self, columns=["azi", "inc"], name="lins"):
        """Create column with `Lineation` features.

        Keyword Args:
            columns (list): Columns containing azi and inc.
              Default ["azi", "inc"]
            name (str): Name of created column. Default 'lins'
        """
        res = self._obj.copy()
        data = []
        mask = []
        for row in self._obj[columns].itertuples(index=False):
            if any(pd.isna(v) for v in row):
                data.append(None)
                mask.append(True)
            else:
                data.append(Lineation(*row))
                mask.append(False)
        res[name] = LinArray(data, mask=np.array(mask))
        return res

    def add_lins(self, features, name="lins"):
        """Add column with `Lineation` features.

        Keyword Args:
            name (str): Name of created column. Default 'lins'
        """
        if not isinstance(features, LineationSet):
            raise TypeError("Argument must be LineationSet")
        res = self._obj.copy()
        res[name] = LinArray(features.data)
        return res

    def create_faults(
        self, columns=["fazi", "finc", "lazi", "linc", "sense"], name="faults"
    ):
        """Create column with `Fault` features.

        Keyword Args:
            columns (list): Columns containing fault plane (fazi, finc),
              lineation (lazi, linc) and sense.
              Default ['fazi', 'finc', 'lazi', 'linc', 'sense']
            name (str): Name of created column. Default 'faults'
        """
        res = self._obj.copy()
        data = []
        mask = []
        for row in self._obj[columns].itertuples(index=False):
            if any(pd.isna(v) for v in row):
                data.append(None)
                mask.append(True)
            else:
                data.append(Fault(*row))
                mask.append(False)
        res[name] = FaultArray(data, mask=np.array(mask))
        return res

    def add_faults(self, features, name="faults"):
        """Add column with `Fault` features.

        Keyword Args:
            name (str): Name of created column. Default 'faults'
        """
        if not isinstance(features, FaultSet):
            raise TypeError("Argument must be FaultSet")
        res = self._obj.copy()
        res[name] = FaultArray(features.data)
        return res
