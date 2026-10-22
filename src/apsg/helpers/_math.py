import math

import numpy as np

sqrt2 = math.sqrt(2.0)


def _short_repr(value, limit=40):
    text = repr(value)
    return text if len(text) <= limit else text[: limit - 3] + "..."


def real(value, what="value"):
    """Return `value` as a finite float.

    Raises ``TypeError`` unless `value` is a real number (strings, booleans,
    complex numbers and containers are refused) and ``ValueError`` when it is
    not finite or too large for a float.
    """
    if isinstance(value, (str, bytes, bool, np.bool_, complex, np.complexfloating)):
        raise TypeError(f"{what} must be a real number, got {_short_repr(value)}")
    try:
        x = float(value)
    except OverflowError:
        raise ValueError(f"{what} is too large, got {_short_repr(value)}") from None
    except (TypeError, ValueError):
        raise TypeError(
            f"{what} must be a real number, got {_short_repr(value)}"
        ) from None
    if not math.isfinite(x):
        raise ValueError(f"{what} must be finite, got {_short_repr(value)}")
    return x


# DEGREES TRIGINOMETRY


def sind(x):
    """Calculate sine of angle in degrees."""
    return math.sin(math.radians(x))


def cosd(x):
    """Calculate cosine of angle in degrees."""
    return math.cos(math.radians(x))


def tand(x):
    """Calculate tangent of angle in degrees."""
    return math.tan(math.radians(x))


def asind(x):
    """Calculate arc sine in degrees."""
    return math.degrees(math.asin(max(min(x, 1), -1)))


def acosd(x):
    """Calculate arc cosine in degrees."""
    return math.degrees(math.acos(max(min(x, 1), -1)))


def atand(x):
    """Calculate arc tangent in degrees."""
    return math.degrees(math.atan(x))


def atan2d(y, x):
    """
    Calculate arc tangent in degrees in the range from pi to -pi.

    Args:
        y (float): y coordinate
        x (float): x coordinate
    """
    return math.degrees(math.atan2(y, x))
