"""
GroupBy helper functions for APSG feature columns.

Each function takes a ``pd.Series`` sub-group (from ``df.groupby(...)["col"]``),
calls ``.G()`` to obtain a ``FeatureSet``, then chains APSG methods on it.

Import via::

    from apsg.pandas import gbf

The functions are designed as **examples of the pattern** so users can easily
write their own:

::

    def my_apply_func(series):
        \"""Compute … for each group.

        Args:
            series (pd.Series): Sub-group Series containing an APSG feature
                array (Vec3Array, LinArray, FolArray, or FaultArray).

        Returns:
            Lineation: Single scalar per group.

        Examples:
            >>> df.groupby("group")["col"].G.apply(my_apply_func)
        \"""
        return series.G().some_method()


    def my_transform_func(series):
        \"""Compute … per element within each group.

        Args:
            series (pd.Series): Sub-group Series containing an APSG feature
                array.

        Returns:
            numpy.ndarray: Same-length array of per-element results.

        Examples:
            >>> df.groupby("group")["col"].G.transform(my_transform_func)
        \"""
        return series.G().some_elementwise_method()


    # Functions with extra parameters pass through *args / **kwargs.
    def my_param_func(series, alpha):
        return series.G().some_method(alpha=alpha)

    >>> df.groupby("group")["col"].G.apply(my_param_func, alpha=95)
"""

# ---------------------------------------------------------------------------
# Imports
# ---------------------------------------------------------------------------

from apsg.feature import (
    Direction,
    Direction2Set,
    FaultSet,
    FoliationSet,
    LineationSet,
    Vector2Set,
    Vector3Set,
)

# ---------------------------------------------------------------------------
# Apply helpers — return a single scalar per group
# ---------------------------------------------------------------------------


def resultant(series):
    """Resultant vector of a group.

    Args:
        series (pd.Series): Sub-group Series containing an APSG feature
            array (Vec3Array, LinArray, FolArray, or FaultArray).

    Returns:
        Vector3 | Lineation | Foliation | Fault: Resultant of the group.
        The return type matches the input feature type
        (e.g. ``LineationSet.R()`` returns ``Lineation``).

    Examples:
        >>> df.groupby("structure")["lins"].G.apply(gbf.resultant)
    """
    return series.G().R()


def resultant_magnitude(series):
    """Length (magnitude) of the resultant vector of a group.

    Args:
        series (pd.Series): Sub-group Series containing an APSG feature array.

    Returns:
        float: Magnitude of the resultant.

    Examples:
        >>> df.groupby("structure")["lins"].G.apply(gbf.resultant_magnitude)
    """
    return abs(series.G().R())


def mean(series):
    """Mean orientation of a group.

    The mean is computed differently based on the feature type:

    - ``Vector2`` / ``Vector3``: arithmetic mean via ``.R(mean=True)``
    - ``Direction``: orientation tensor eigenvector azimuth via
      ``.ortensor().orientation``
    - ``Lineation``: major eigenvector of the orientation tensor
      via ``.ortensor().eigenlins(which=0)``
    - ``Foliation``: major eigenvector (pole) of the orientation tensor
      via ``.ortensor().eigenfols(which=0)``
    - ``Fault``: raises ``TypeError``

    Args:
        series (pd.Series): Sub-group Series containing an APSG feature
            array.

    Returns:
        Vector2 | Vector3 | Lineation | Foliation | float:
        The mean orientation. The exact type depends on the input
        feature type (see description above).

    Raises:
        TypeError: If the series contains Fault or unsupported feature data.

    Examples:
        >>> df.groupby("structure")["vecs"].G.apply(gbf.mean)
        >>> df.groupby("structure")["dirs"].G.apply(gbf.mean)
        >>> df.groupby("structure")["lins"].G.apply(gbf.mean)
        >>> df.groupby("structure")["fols"].G.apply(gbf.mean)
    """
    data = series.G()
    if isinstance(data, Direction2Set):
        return data.ortensor().orientation
    if isinstance(data, LineationSet):
        return data.ortensor().eigenlins(which=0)
    if isinstance(data, FoliationSet):
        return data.ortensor().eigenfols(which=0)
    if isinstance(data, (Vector2Set, Vector3Set)):
        return data.R(mean=True)
    if isinstance(data, FaultSet):
        raise TypeError("mean is not defined for Fault data")
    raise TypeError(f"mean is not defined for {type(data).__name__}")


# ---------------------------------------------------------------------------
# Transform helpers — return same-length arrays per group
# ---------------------------------------------------------------------------


def angle_to_mean(series):
    """Per-element angular distance (in degrees) to the group mean.

    The mean is computed via :func:`mean`, which dispatches based on
    the feature type in the series.

    Args:
        series (pd.Series): Sub-group Series containing an APSG feature
            array.

    Returns:
        numpy.ndarray: Array of angles in degrees, one per element in the
        group.

    Raises:
        TypeError: If the series contains Fault or unsupported data.

    Examples:
        >>> df.groupby("structure")["lins"].G.transform(gbf.angle_to_mean)
        >>> df.groupby("structure")["vecs"].G.transform(gbf.angle_to_mean)
        >>> df.groupby("structure")["dirs"].G.transform(gbf.angle_to_mean)
    """
    data = series.G()
    m = mean(series)
    if isinstance(data, Direction2Set):
        m = Direction(m)
    return data.angle(m)


# ---------------------------------------------------------------------------
# Custom-parameter example — shows how to add parameters
# ---------------------------------------------------------------------------


def eigenlin(series, which=0):
    """N-th eigenvector of the orientation tensor as a ``Lineation``.

    This is a generalised version of :func:`mean` for Lineation data that accepts
    an extra ``which`` parameter, demonstrating how to write custom
    groupby functions with additional arguments.

    Args:
        series (pd.Series): Sub-group Series containing an APSG feature array.
        which (int, optional): Eigenvector index:
            0 = major, 1 = intermediate, 2 = minor. Defaults to 0.

    Returns:
        Lineation: The requested eigenvector.

    Examples:
        >>> df.groupby("structure")["lins"].G.apply(gbf.eigenlin)
        >>> df.groupby("structure")["lins"].G.apply(gbf.eigenlin, which=1)

        Aggregate with multiple ``which`` values via lambdas:

        >>> df.groupby("group")["lins"].G.aggregate([
        ...     ("major", lambda s: gbf.eigenlin(s, which=0)),
        ...     ("minor", lambda s: gbf.eigenlin(s, which=2)),
        ... ])
    """
    return series.G().ortensor().eigenlins(which=which)
