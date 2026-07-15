"""
GroupBy helper functions for APSG feature accessors.

Each function takes a ``FeatureSet`` directly (the argument
``df.<accessor>.groupby(by).apply/transform/aggregate(func)`` passes to
``func`` for each group), and chains APSG methods on it.

Import via::

    from apsg.pandas import gbf

The functions are designed as **examples of the pattern** so users can easily
write their own:

::

    def my_apply_func(fs):
        \"""Compute … for each group.

        Args:
            fs (FeatureSet): Vector3Set/Vector2Set/LineationSet/FoliationSet/
                FaultSet group.

        Returns:
            Lineation: Single scalar per group.

        Examples:
            >>> df.lin.groupby("group").apply(my_apply_func)
        \"""
        return fs.some_method()


    def my_transform_func(fs):
        \"""Compute … per element within each group.

        Args:
            fs (FeatureSet): the group's FeatureSet.

        Returns:
            numpy.ndarray: Same-length array of per-element results.

        Examples:
            >>> df.lin.groupby("group").transform(my_transform_func)
        \"""
        return fs.some_elementwise_method()


    # Functions with extra parameters pass through *args / **kwargs.
    def my_param_func(fs, alpha):
        return fs.some_method(alpha=alpha)

    >>> df.lin.groupby("group").apply(my_param_func, alpha=95)
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


def resultant(fs):
    """Resultant vector of a group.

    Args:
        fs (FeatureSet): Vector3Set/Vector2Set/LineationSet/FoliationSet/
            FaultSet group.

    Returns:
        Vector3 | Vector2 | Lineation | Foliation | Fault: Resultant of the
        group. The return type matches the input feature type
        (e.g. ``LineationSet.R()`` returns ``Lineation``).

    Examples:
        >>> df.lin.groupby("structure").apply(gbf.resultant)
    """
    return fs.R()


def resultant_magnitude(fs):
    """Length (magnitude) of the resultant vector of a group.

    Args:
        fs (FeatureSet): the group's FeatureSet.

    Returns:
        float: Magnitude of the resultant.

    Examples:
        >>> df.lin.groupby("structure").apply(gbf.resultant_magnitude)
    """
    return abs(fs.R())


def mean(fs):
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
        fs (FeatureSet): the group's FeatureSet.

    Returns:
        Vector2 | Vector3 | Lineation | Foliation | float:
        The mean orientation. The exact type depends on the input
        feature type (see description above).

    Raises:
        TypeError: If `fs` is a FaultSet or an unsupported feature type.

    Examples:
        >>> df.vec.groupby("structure").apply(gbf.mean)
        >>> df.dir.groupby("structure").apply(gbf.mean)
        >>> df.lin.groupby("structure").apply(gbf.mean)
        >>> df.fol.groupby("structure").apply(gbf.mean)
    """
    if isinstance(fs, Direction2Set):
        return fs.ortensor().orientation
    if isinstance(fs, LineationSet):
        return fs.ortensor().eigenlins(which=0)
    if isinstance(fs, FoliationSet):
        return fs.ortensor().eigenfols(which=0)
    if isinstance(fs, (Vector2Set, Vector3Set)):
        return fs.R(mean=True)
    if isinstance(fs, FaultSet):
        raise TypeError("mean is not defined for Fault data")
    raise TypeError(f"mean is not defined for {type(fs).__name__}")


# ---------------------------------------------------------------------------
# Transform helpers — return same-length arrays per group
# ---------------------------------------------------------------------------


def angle_to_mean(fs):
    """Per-element angular distance (in degrees) to the group mean.

    The mean is computed via :func:`mean`, which dispatches based on
    the feature type of `fs`.

    Args:
        fs (FeatureSet): the group's FeatureSet.

    Returns:
        numpy.ndarray: Array of angles in degrees, one per element in the
        group.

    Raises:
        TypeError: If `fs` is a FaultSet or unsupported feature type.

    Examples:
        >>> df.lin.groupby("structure").transform(gbf.angle_to_mean)
        >>> df.vec.groupby("structure").transform(gbf.angle_to_mean)
        >>> df.dir.groupby("structure").transform(gbf.angle_to_mean)
    """
    m = mean(fs)
    if isinstance(fs, Direction2Set):
        m = Direction(m)
    return fs.angle(m)


# ---------------------------------------------------------------------------
# Custom-parameter example — shows how to add parameters
# ---------------------------------------------------------------------------


def eigenlin(fs, which=0):
    """N-th eigenvector of the orientation tensor as a ``Lineation``.

    This is a generalised version of :func:`mean` for Lineation data that accepts
    an extra ``which`` parameter, demonstrating how to write custom
    groupby functions with additional arguments.

    Args:
        fs (FeatureSet): the group's FeatureSet.
        which (int, optional): Eigenvector index:
            0 = major, 1 = intermediate, 2 = minor. Defaults to 0.

    Returns:
        Lineation: The requested eigenvector.

    Examples:
        >>> df.lin.groupby("structure").apply(gbf.eigenlin)
        >>> df.lin.groupby("structure").apply(gbf.eigenlin, which=1)

        Aggregate with multiple ``which`` values via lambdas:

        >>> df.lin.groupby("group").aggregate([
        ...     ("major", lambda s: gbf.eigenlin(s, which=0)),
        ...     ("minor", lambda s: gbf.eigenlin(s, which=2)),
        ... ])
    """
    return fs.ortensor().eigenlins(which=which)
