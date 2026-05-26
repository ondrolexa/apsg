==============
helpers module
==============

The :mod:`apsg.helpers` module provides utility functions used throughout APSG for
geological notation conversion and geometric calculations. It includes trigonometric
functions in degrees, converters between geological and Cartesian coordinates, and
formatting helpers.

Usage
-----

Geological coordinate conversions::

    >>> from apsg.helpers import geo2vec_planar, geo2vec_linear, vec2geo_planar, vec2geo_linear
    >>> v = geo2vec_planar(120, 30)       # dip direction=120, dip=30
    >>> v = geo2vec_linear(210, 45)       # trend=210, plunge=45
    >>> strike, dip = vec2geo_planar(v)
    >>> trend, plunge = vec2geo_linear(v)

Trigonometric functions in degrees::

    >>> from apsg.helpers import sind, cosd, tand, asind, acosd, atand, atan2d
    >>> sind(90)
    1.0
    >>> cosd(0)
    1.0
    >>> asind(1)
    90.0

.. automodule:: apsg.helpers
    :members:
    :show-inheritance:
