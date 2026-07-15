import numpy as np
import pandas as pd
from pandas.api.extensions import ExtensionArray, ExtensionDtype

from apsg.feature import (
    Direction,
    Direction2Set,
    Fault,
    FaultSet,
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
