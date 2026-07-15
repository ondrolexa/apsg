import pandas as pd

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
from apsg.pandas._pandas_api import (
    DirArray,
    FaultArray,
    FolArray,
    LinArray,
    Vec2Array,
    Vec3Array,
)

_APSG_SCALAR_TYPES = (Vector3, Vector2, Direction, Foliation, Lineation, Fault)

_SCALAR_TYPE_MAP = {
    Vector3: Vec3Array,
    Vector2: Vec2Array,
    Direction: DirArray,
    Foliation: FolArray,
    Lineation: LinArray,
    Fault: FaultArray,
}


def _recast_column(index, values, name=None):
    """Wrap a list of per-row/per-group results in a proper APSG ExtensionDtype
    Series if every non-null value is the same APSG scalar type, else a plain
    ``pd.Series``.
    """
    non_null = [v for v in values if v is not None]
    if non_null:
        first_type = type(non_null[0])
        array_cls = _SCALAR_TYPE_MAP.get(first_type)
        # exact-type check (not isinstance) - Lineation/Foliation subclass
        # Vector3 and Direction subclasses Vector2, so isinstance would blur them
        if array_cls is not None and all(type(v) is first_type for v in non_null):
            return pd.Series(array_cls._from_sequence(values), index=index, name=name)
    return pd.Series(values, index=index, name=name)


class _FeatureAccessor:
    """Base class for callable, live DataFrame accessors of APSG feature types.

    A concrete subclass (e.g. ``LinAccessor``) is registered on a fixed short
    name (e.g. ``"lin"``) via ``pd.api.extensions.register_dataframe_accessor``.
    ``df.lin()`` builds the ``FeatureSet`` live from whichever columns are
    currently configured (``df.lin.set_columns(...)``), skipping rows with NaN
    in those columns - no intermediate column is ever created.
    """

    _SCALAR_CLS = None
    _FEATURE_SET_CLS = None
    _ARRAY_CLS = None
    _DEFAULT_COLUMNS = {}  # {param_name: default_column_name}, order == ctor arg order
    _NAME = None
    _PLOT_CLASS = None  # "StereoNet" or "RosePlot"
    _PLOT_KINDS = ()  # first entry is the default `kind`

    def __init__(self, pandas_obj):
        self._df = pandas_obj

    @property
    def _attrs_key(self):
        return f"_apsg_{self._NAME}_columns"

    @property
    def _columns(self):
        # pandas' DataFrame accessor descriptor does not cache the accessor
        # instance (a fresh one is built on every `df.lin` access), so the
        # column mapping is stored on `df.attrs` instead - the one place that
        # actually persists across separate accesses to the same DataFrame,
        # and is independently copied (not shared) on `df.copy()`.
        return self._df.attrs.setdefault(self._attrs_key, dict(self._DEFAULT_COLUMNS))

    def set_columns(self, **kwargs):
        """Reconfigure which existing DataFrame columns feed this accessor.

        Args:
            **kwargs: column overrides keyed by the scalar type's constructor
                parameter names (e.g. ``azi=``, ``inc=`` for ``lin``/``fol``).

        Returns:
            _FeatureAccessor: self, for chaining.
        """
        invalid = set(kwargs) - set(self._DEFAULT_COLUMNS)
        if invalid:
            raise TypeError(
                f"Unknown column parameter(s) {sorted(invalid)} for "
                f"{type(self).__name__}; expected one of {sorted(self._DEFAULT_COLUMNS)}"
            )
        self._columns.update(kwargs)
        return self

    def _rows(self, df):
        """Yield ``(index_label, scalar_or_None)`` for every row of `df`,
        skipping (``None``) rows with NaN in any configured column.
        """
        cols = [self._columns[p] for p in self._DEFAULT_COLUMNS]
        sub = df[cols]
        for idx, row in zip(sub.index, sub.itertuples(index=False)):
            if any(pd.isna(v) for v in row):
                yield idx, None
            else:
                yield idx, self._SCALAR_CLS(*row)

    def __call__(self, name=None):
        """Build the ``FeatureSet`` live from the currently configured columns."""
        scalars = [s for _, s in self._rows(self._df) if s is not None]
        return self._FEATURE_SET_CLS(
            scalars, name=name if name is not None else self._NAME
        )

    def plot(self, **kwargs):
        """Quickplot the live ``FeatureSet``. Same kwargs as before: ``kind=``,
        ``plot_kws=``, ``show=``, plus passthrough kwargs to the underlying
        ``StereoNet``/``RosePlot`` method.
        """
        from apsg.plotting import RosePlot, StereoNet

        kind = kwargs.pop("kind", None) or self._PLOT_KINDS[0]
        plot_kws = kwargs.pop("plot_kws", {})
        show = kwargs.pop("show", True)

        plot_cls = {"StereoNet": StereoNet, "RosePlot": RosePlot}[self._PLOT_CLASS]
        plot_obj = plot_cls(**plot_kws)
        getattr(plot_obj, kind)(self(), **kwargs)

        if show:
            plot_obj.show()
            return None
        return plot_obj

    def groupby(self, by):
        """Group the whole DataFrame by `by`, building this accessor's
        ``FeatureSet`` per group on demand. Returns a `_FeatureGroupBy`.
        """
        return _FeatureGroupBy(self, by)


class _FeatureGroupBy:
    """GroupBy wrapper for a `_FeatureAccessor`.

    There is no materialized column to group - the whole frame is grouped
    with plain ``df.groupby(by)``, and each group's ``FeatureSet`` is built on
    demand from the parent accessor's currently configured columns.
    """

    def __init__(self, accessor, by):
        self._accessor = accessor
        self._df = accessor._df
        self._by = by
        self._grouped = self._df.groupby(by)

    def _group_scalars(self, subframe):
        rows = list(self._accessor._rows(subframe))
        valid = [(idx, s) for idx, s in rows if s is not None]
        return rows, [idx for idx, _ in valid], [s for _, s in valid]

    def _group_index(self, keys):
        if isinstance(self._by, (list, tuple)):
            return pd.MultiIndex.from_tuples(keys, names=self._by)
        return pd.Index(keys, name=self._by)

    def apply(self, func, *args, **kwargs):
        """Apply `func` to each group's FeatureSet; recast to a proper dtype."""
        keys, values = [], []
        for key, subframe in self._grouped:
            _, _, scalars = self._group_scalars(subframe)
            fs = self._accessor._FEATURE_SET_CLS(scalars, name=str(key))
            keys.append(key)
            values.append(func(fs, *args, **kwargs))
        return _recast_column(self._group_index(keys), values)

    def transform(self, func, *args, **kwargs):
        """Transform each group's FeatureSet; result is aligned to the
        original DataFrame's row order/index, recast to a proper dtype.

        `func` must return either a scalar (broadcast across the group's
        valid, non-NaN rows) or a sequence sized to the group's FeatureSet
        length (not the raw group size - rows with NaN in the configured
        columns are excluded from the FeatureSet the callback sees).
        """
        out = {}
        for key, subframe in self._grouped:
            rows, valid_idx, scalars = self._group_scalars(subframe)
            for idx, _ in rows:
                out.setdefault(idx, None)
            if not scalars:
                continue
            fs = self._accessor._FEATURE_SET_CLS(scalars, name=str(key))
            res = func(fs, *args, **kwargs)
            if isinstance(res, _APSG_SCALAR_TYPES) or not hasattr(res, "__len__"):
                per_row = [res] * len(valid_idx)
            else:
                seq = list(res)
                if len(seq) != len(valid_idx):
                    raise ValueError(
                        f"transform function returned {len(seq)} values for group "
                        f"{key!r}, which has {len(valid_idx)} valid (non-NaN) rows"
                    )
                per_row = seq
            for idx, v in zip(valid_idx, per_row):
                out[idx] = v
        values = [out[idx] for idx in self._df.index]
        return _recast_column(self._df.index, values)

    def aggregate(self, func, *args, **kwargs):
        """Aggregate each group's FeatureSet with a single function, or with a
        list of functions / ``(name, func)`` tuples producing a multi-column
        ``pd.DataFrame`` (each column independently recast).
        """
        if isinstance(func, list):
            cols = {}
            for item in func:
                if isinstance(item, tuple):
                    col_name, fn = item
                else:
                    fn = item
                    col_name = (
                        fn if isinstance(fn, str) else getattr(fn, "__name__", str(fn))
                    )
                cols[col_name] = self._aggregate_one(fn, *args, **kwargs)
            return pd.DataFrame(cols)
        return self._aggregate_one(func, *args, **kwargs)

    agg = aggregate

    def _aggregate_one(self, func, *args, **kwargs):
        fn = len if func in ("count", "len") else func
        keys, values = [], []
        for key, subframe in self._grouped:
            _, _, scalars = self._group_scalars(subframe)
            fs = self._accessor._FEATURE_SET_CLS(scalars, name=str(key))
            keys.append(key)
            values.append(fn(fs, *args, **kwargs))
        return _recast_column(self._group_index(keys), values)


@pd.api.extensions.register_dataframe_accessor("vec")
class VecAccessor(_FeatureAccessor):
    """Callable DataFrame accessor for `Vector3` data. Default columns x, y, z."""

    _SCALAR_CLS = Vector3
    _FEATURE_SET_CLS = Vector3Set
    _ARRAY_CLS = Vec3Array
    _DEFAULT_COLUMNS = {"x": "x", "y": "y", "z": "z"}
    _NAME = "vec"
    _PLOT_CLASS = "StereoNet"
    _PLOT_KINDS = ("line", "vector")


@pd.api.extensions.register_dataframe_accessor("vec2")
class Vec2Accessor(_FeatureAccessor):
    """Callable DataFrame accessor for `Vector2` data. Default columns x, y."""

    _SCALAR_CLS = Vector2
    _FEATURE_SET_CLS = Vector2Set
    _ARRAY_CLS = Vec2Array
    _DEFAULT_COLUMNS = {"x": "x", "y": "y"}
    _NAME = "vec2"
    _PLOT_CLASS = "RosePlot"
    _PLOT_KINDS = ("bar", "pdf")


@pd.api.extensions.register_dataframe_accessor("dir")
class DirAccessor(_FeatureAccessor):
    """Callable DataFrame accessor for `Direction` data. Default column angle."""

    _SCALAR_CLS = Direction
    _FEATURE_SET_CLS = Direction2Set
    _ARRAY_CLS = DirArray
    _DEFAULT_COLUMNS = {"angle": "angle"}
    _NAME = "dir"
    _PLOT_CLASS = "RosePlot"
    _PLOT_KINDS = ("bar", "pdf")


@pd.api.extensions.register_dataframe_accessor("fol")
class FolAccessor(_FeatureAccessor):
    """Callable DataFrame accessor for `Foliation` data. Default columns azi, inc."""

    _SCALAR_CLS = Foliation
    _FEATURE_SET_CLS = FoliationSet
    _ARRAY_CLS = FolArray
    _DEFAULT_COLUMNS = {"azi": "azi", "inc": "inc"}
    _NAME = "fol"
    _PLOT_CLASS = "StereoNet"
    _PLOT_KINDS = ("gc", "pole")


@pd.api.extensions.register_dataframe_accessor("lin")
class LinAccessor(_FeatureAccessor):
    """Callable DataFrame accessor for `Lineation` data. Default columns azi, inc."""

    _SCALAR_CLS = Lineation
    _FEATURE_SET_CLS = LineationSet
    _ARRAY_CLS = LinArray
    _DEFAULT_COLUMNS = {"azi": "azi", "inc": "inc"}
    _NAME = "lin"
    _PLOT_CLASS = "StereoNet"
    _PLOT_KINDS = ("line",)


@pd.api.extensions.register_dataframe_accessor("fault")
class FaultAccessor(_FeatureAccessor):
    """Callable DataFrame accessor for `Fault` data.

    Default columns fazi, finc, lazi, linc, sense.
    """

    _SCALAR_CLS = Fault
    _FEATURE_SET_CLS = FaultSet
    _ARRAY_CLS = FaultArray
    _DEFAULT_COLUMNS = {
        "fazi": "fazi",
        "finc": "finc",
        "lazi": "lazi",
        "linc": "linc",
        "sense": "sense",
    }
    _NAME = "fault"
    _PLOT_CLASS = "StereoNet"
    _PLOT_KINDS = ("fault", "pair", "hoeppner")
