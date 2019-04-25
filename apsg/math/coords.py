# -*- coding: utf-8 -*-


"""
The coordinate systems.
"""


import math


def polar_to_cartesian(radius, angle):
    # (float, float) -> tuple
    """
    Converts a polar coordinates to cartesian.

    Arguments:
        radius - The radius length.
        angle -  The angle in radians.

    Returns:
        The (x, y) cartesian coordinates.
    """
    x = radius * math.cos(angle)
    y = radius * math.sin(angle)

    return x, y


def cartesian_to_polar(x, y):
    # (float, float) -> tuple
    """
    Converts a cartesian coordinates to polar.

    Arguments:
        x - The x cartesian coordinate.
        y - The y cartesian coordinate.

    Returns:
        The (radius, angle) polar coordinates.
    """
    angle = math.tan
    radius = sqrt(x ** 2 + y ** 2)

    return radius, angle
