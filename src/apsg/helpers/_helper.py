# some utils
import json
import numbers


def eformat(f, prec):
    """Format float in scientific notation with given precision."""
    s = f"{f:e}"
    m, e = s.split("e")
    return "{:.{:d}f}E{:0d}".format(float(m), prec, int(e))


def is_jsonable(x):
    """Check if value is JSON-serializable."""
    try:
        json.dumps(x)
        return True
    except (TypeError, OverflowError):
        return False


def check_count(n, name="n", minimum=0):
    """Return `n` as int, raising unless it is an integer of at least `minimum`."""
    if isinstance(n, bool) or not isinstance(n, numbers.Integral):
        raise TypeError(f"{name} must be an integer, got {n!r}")
    if n < minimum:
        raise ValueError(f"{name} must be at least {minimum}, got {n}")
    return int(n)


def checked_attrs(kwargs):
    """Return feature attributes ``kwargs``, raising if not JSON-serializable."""
    if not is_jsonable(kwargs):
        raise TypeError("Provided attributes are not serializable.")
    return kwargs


_AXES = {"x": (1, 0, 0), "y": (0, 1, 0), "z": (0, 0, 1)}


def axis_coords(name, dim=3):
    """Return components of the coordinate axis named `name` ('x', 'y' or 'z').

    Returns ``None`` when `name` is not an axis name of a `dim`-dimensional
    coordinate system.
    """
    coords = _AXES.get(name.lower())
    if coords is None or not any(coords[:dim]):
        return None
    return coords[:dim]
