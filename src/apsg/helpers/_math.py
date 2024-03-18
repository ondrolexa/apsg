import math

sqrt2 = math.sqrt(2.0)

# DEGREES TRIGINOMETRY


def sind(x):
    """Calculate sine of angle in degrees"""
    return math.sin(math.radians(x))


def cosd(x):
    """Calculate cosine of angle in degrees"""
    return math.cos(math.radians(x))


def tand(x):
    """Calculate tangent of angle in degrees"""
    return math.tan(math.radians(x))


def asind(x):
    """Calculate arc sine in degrees"""
    return math.degrees(math.asin(max(min(x, 1), -1)))


def acosd(x):
    """Calculate arc cosine in degrees"""
    return math.degrees(math.acos(max(min(x, 1), -1)))


def atand(x):
    """Calculate arc tangent in degrees"""
    return math.degrees(math.atan(x))


def atan2d(y, x):
    """
    Calculate arc tangent in degrees in the range from pi to -pi.

    Args:
        y (float): y coordinate
        x (float): x coordinate
    """
    return math.degrees(math.atan2(y, x))
