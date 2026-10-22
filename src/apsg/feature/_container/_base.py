"""Base class of all feature sets, shared helpers and the ``G()`` factory."""

import csv

import numpy as np

from apsg.feature._geodata import (
    Cone,
    Direction,
    Fault,
    Foliation,
    Lineation,
    Pair,
)
from apsg.feature._tensor2 import Ellipse, OrientationTensor2, Stress2
from apsg.feature._tensor3 import (
    Ellipsoid,
    OrientationTensor3,
    Stress3,
)
from apsg.math._vector import Vector2, Vector3


def _per_item(name, doc):
    """Return a read-only property collecting attribute `name` of every item
    of the set into an array."""

    def fget(self) -> np.ndarray:
        return np.array([getattr(item, name) for item in self])

    return property(fget, doc=doc)


def _is_number(text):
    try:
        float(text)
    except ValueError:
        return False
    return True


def _finite_float(text):
    """Return `text` as a finite float, raising ``ValueError`` otherwise."""
    value = float(text)
    if not np.isfinite(value):
        raise ValueError(f"{text!r} is not a finite number")
    return value


def _read_csv_columns(filename, cols, delimiter=None, convert=_finite_float):
    """Return the selected columns of a CSV file, as tuples of values.

    Columns are given by index (from 0) or, when the file has a header line,
    by column name. The CSV dialect is sniffed unless ``delimiter`` is given.
    Empty lines are skipped. Each value is converted by `convert`, a callable
    or one callable per column. Malformed files and values raise
    ``ValueError`` naming the file and line.
    """
    converters = convert if isinstance(convert, tuple) else (convert,) * len(cols)
    with open(filename, newline="") as csvfile:
        sample = csvfile.read(1024)
        csvfile.seek(0)
        if not sample.strip():
            raise ValueError(f"CSV file {filename} is empty")
        try:
            # only common delimiters, so binary junk isn't read as a header line
            dialect = (
                csv.Sniffer().sniff(sample, delimiters=",;\t |")
                if delimiter is None
                else None
            )
        except csv.Error:
            raise ValueError(
                f"Can't detect the CSV dialect of {filename}, give the delimiter"
            ) from None
        try:
            has_header = csv.Sniffer().has_header(sample)
        except csv.Error:
            # e.g. a single column: a header is a first line that is not numbers
            first = next(csv.reader([sample.splitlines()[0]], delimiter=delimiter))
            has_header = not all(_is_number(field) for field in first)
        if dialect is not None:
            reader = csv.reader(csvfile, dialect=dialect)
        else:
            reader = csv.reader(csvfile, delimiter=delimiter)
        try:
            rows = [(reader.line_num, row) for row in reader if row]
        except csv.Error as err:
            raise ValueError(f"Can't read CSV file {filename}: {err}") from None
    if has_header:
        header = [name.strip() for name in rows[0][1]]
        rows = rows[1:]
        idx = []
        for col in cols:
            if isinstance(col, int):
                idx.append(col)
            elif col in header:
                idx.append(header.index(col))
            else:
                raise ValueError(f"No column {col!r} in CSV file header {header}.")
        if max(idx) >= len(header):
            raise ValueError(
                f"The header of {filename} has {len(header)} columns, column "
                f"{max(idx)} (from 0) is needed"
            )
    elif all(isinstance(col, int) for col in cols):
        idx = list(cols)
    else:
        raise ValueError("No header line in CSV file, columns must be indexes.")
    values = []
    for line, row in rows:
        if max(idx) >= len(row):
            raise ValueError(
                f"Line {line} of {filename} has {len(row)} columns, column "
                f"{max(idx)} (from 0) is needed"
            )
        try:
            values.append(
                tuple(f(row[i].strip()) for f, i in zip(converters, idx, strict=True))
            )
        except ValueError as err:
            raise ValueError(f"Line {line} of {filename}: {err}") from None
    return [tuple(column) for column in zip(*values, strict=True)] or [()] * len(cols)


class FeatureSet:
    """
    Base class for containers.
    """

    __slots__ = ("_cache", "data", "name")

    def __init__(self, data, name="Default"):
        self.data = tuple(data)
        self.name = name
        self._cache = {}

    def __copy__(self):
        return type(self)(list(self.data), name=self.name)

    copy = __copy__

    def to_json(self):
        """Return as JSON dict."""

        return {
            "datatype": self.__class__.__name__,
            "args": ({"collection": tuple(item.to_json() for item in self)},),
            "kwargs": {"name": self.name},
        }

    def _require(self, n_min, what):
        """Raise unless the set has at least `n_min` items, needed by `what`."""
        if len(self) >= n_min:
            return
        if len(self) == 0:
            raise ValueError(f"{what} of an empty {type(self).__name__}")
        raise ValueError(
            f"{what} needs at least {n_min} items, {type(self).__name__} has "
            f"{len(self)}"
        )

    def _check_same_length(self, other, what):
        """Raise unless `other` has as many items as this set, for an
        item-by-item operation `what`."""
        if len(other) != len(self):
            raise ValueError(
                f"{what} needs sets of the same length, got {len(self)} and "
                f"{len(other)}"
            )

    def attrs(self):
        import pandas as pd

        return pd.DataFrame([item._attrs for item in self.data])

    def label(self):
        """Return label."""

        return self.name

    def __eq__(self, other):
        return NotImplemented

    def __ne__(self, other):
        return not self.__eq__(other)

    def __bool__(self):
        return len(self) != 0

    def __len__(self):
        return len(self.data)

    def __getitem__(self, key):
        if isinstance(key, slice):
            return self.__class__(self.data[key], name=self.name)
        # elif isinstance(key, int):
        elif np.issubdtype(type(key), np.integer):
            return self.data[key]
        else:  # fancy indexing
            try:
                idxs = np.arange(len(self.data), dtype=int)[np.asarray(key)]
                return self.__class__([self.data[ix] for ix in idxs], name=self.name)
            except (TypeError, IndexError, ValueError):
                raise TypeError(
                    "Wrong index. Only slice, int and array like are allowed for indexing."
                )

    def __iter__(self):
        return iter(self.data)

    def __add__(self, other):
        if isinstance(other, type(self)):
            return type(self)(self.data + other.data, name=self.name)
        else:
            raise TypeError(f"Only {self.__class__.__name__} is allowed")

    def filter(self, **kwargs):
        """Filter ``FeatureSet`` objects attrs."""

        sel = list(self.data)
        for key, val in kwargs.items():
            sel = [item for item in sel if item._attrs.get(key, None) == val]
        return type(self)(sel)

    def rotate(self, axis, phi):
        """Rotate ``FeatureSet`` object `phi` degress about `axis`."""

        return type(self)([e.rotate(axis, phi) for e in self], name=self.name)

    def bootstrap(self, n=100, size=None, replace=True):
        """Return generator of bootstraped samples from ``FeatureSet``.

        Args:
            n (int): number of samples to be generated. Default 100.
            size (int): number of data in sample. Default is same as ``FeatureSet``.
            replace (bool): Whether the sample is with or without replacement. Default is True.

        Examples:
            >>> np.random.seed(6034782)
            >>> l = Vector3Set.random_fisher(n=100, position=lin(120,40))
            >>> sm = [lb.R() for lb in l.bootstrap()]
            >>> l.fisher_statistics()
            {'mu': V:120/40, 'k': 19.91236110604979, 'alpha': 3.249027370399397}
            >>> Vector3Set(sm).fisher_statistics()
            {'mu': V:120/40, 'k': 1735.360206701859, 'alpha': 0.3393224356447341}
        Returns:
            generator: generator of bootstraped samples from ``FeatureSet``.
        """
        if size is None:
            size = len(self)
        for i in range(n):
            yield self[np.random.choice(range(len(self)), size=size, replace=replace)]

    def fit(self, model, **kwargs):
        """Fit a probability distribution to the data.

        Args:
            model: distribution class (e.g. ``Kent``) or its name, either the
                class name or the short name (e.g. ``"kent"``, ``"watson"``,
                ``"vonmises"``, ``"lisle"`` for pairs), matched to the kind of
                data of the set.

        Keyword Args:
            kwargs: passed to the ``fit`` method of the distribution, e.g.
                ``method``.

        Examples:
            >>> l = linset.random_fisher(position=lin(120, 40))
            >>> l.fit("watson")
            Watson(mean=L:120/40, kappa=...)

        Returns:
            OrientationDistribution: the fitted distribution.
        """
        from apsg.feature._distributions import get_distribution

        if issubclass(self.__feature_class__, Vector3):
            dim = 3
        elif issubclass(self.__feature_class__, Vector2):
            dim = 2
        elif issubclass(self.__feature_class__, Pair):
            dim = "pair"
        else:
            raise TypeError(f"{type(self).__name__} can't be fitted by a distribution")
        dist_cls = get_distribution(model, dim=dim)
        if dist_cls.dim != dim:
            raise ValueError(f"{dist_cls.__name__} is not a {dim}D distribution")
        return dist_cls.fit(self, **kwargs)


_CONTAINER_REGISTRY: dict = {}


def _build_registry():
    # imported here, on the first G() call, as the set modules import this one
    from apsg.feature._container._othersets import ConeSet
    from apsg.feature._container._pairsets import FaultSet, PairSet
    from apsg.feature._container._tensorsets import (
        EllipseSet,
        EllipsoidSet,
        OrientationTensor2Set,
        OrientationTensor3Set,
        Stress2Set,
        Stress3Set,
    )
    from apsg.feature._container._vectorsets import (
        Direction2Set,
        FoliationSet,
        LineationSet,
        Vector2Set,
        Vector3Set,
    )

    _CONTAINER_REGISTRY.update(
        {
            Vector3: Vector3Set,
            Vector2: Vector2Set,
            Direction: Direction2Set,
            Lineation: LineationSet,
            Foliation: FoliationSet,
            Pair: PairSet,
            Fault: FaultSet,
            Cone: ConeSet,
            Ellipsoid: EllipsoidSet,
            OrientationTensor3: OrientationTensor3Set,
            Ellipse: EllipseSet,
            OrientationTensor2: OrientationTensor2Set,
            Stress3: Stress3Set,
            Stress2: Stress2Set,
        }
    )


def G(lst, name="Default", feature=None):
    """
    Function to create appropriate container (FeatueSet) from list of features.

    Args:
        lst (list): Homogeneous list of objects of ``Vector2``, ``Vector3``,
            ``Direction``, ``Lineation``, ``Foliation``, ``Pair``, ``Cone``, ``Ellipse``
            or ``OrientationTensor3``.

    Keyword Args:
        name (str): name of feature set. Default `Default`.
        feature (type): feature class of the items, e.g. ``Lineation``. Needed for
            an empty `lst`, otherwise taken from its first item. Default None.

    Examples:
        >>> fols = [fol(120,30), fol(130, 40), fol(126, 37)]
        >>> f = G(fols)
    Returns:
        FeatureSet: The created container.
    """
    if not hasattr(lst, "__len__"):
        raise TypeError("Wrong datatype to create FeatureSet")
    if not _CONTAINER_REGISTRY:
        _build_registry()
    if feature is not None:
        dtype_cls = feature
    elif len(lst) > 0:
        dtype_cls = type(lst[0])
    else:
        raise ValueError("Give the feature class to create a FeatureSet from no items")
    if not all(isinstance(obj, dtype_cls) for obj in lst):
        raise TypeError("All elements must be of the same type")
    container_cls = _CONTAINER_REGISTRY.get(dtype_cls)
    if container_cls is None:
        raise TypeError("Wrong datatype to create FeatureSet")
    return container_cls(lst, name=name)
