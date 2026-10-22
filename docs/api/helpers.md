# apsg.helpers

The `apsg.helpers` module provides utility functions used throughout APSG for
geological notation conversion and geometric calculations. It includes trigonometric
functions in degrees, converters between geological and Cartesian coordinates, and
formatting helpers.

## Usage

Geological coordinate conversions:

```pycon
>>> from apsg.helpers import geo2vec_planar, geo2vec_linear, vec2geo_planar, vec2geo_linear
>>> v = geo2vec_planar(120, 30)       # dip direction=120, dip=30
>>> v = geo2vec_linear(210, 45)       # trend=210, plunge=45
>>> strike, dip = vec2geo_planar(v)
>>> trend, plunge = vec2geo_linear(v)
```

Trigonometric functions in degrees:

```pycon
>>> from apsg.helpers import sind, cosd, tand, asind, acosd, atand, atan2d
>>> sind(90)
1.0
>>> cosd(0)
1.0
>>> asind(1)
90.0
```

## Reference

### Trigonometry in degrees

::: apsg.helpers.sind

::: apsg.helpers.cosd

::: apsg.helpers.tand

::: apsg.helpers.asind

::: apsg.helpers.acosd

::: apsg.helpers.atand

::: apsg.helpers.atan2d

::: apsg.helpers.sqrt2

### Notation conversion

::: apsg.helpers.geo2vec_planar

::: apsg.helpers.geo2vec_linear

::: apsg.helpers.vec2geo_planar

::: apsg.helpers.vec2geo_linear

::: apsg.helpers.parse_quadrant_planar

::: apsg.helpers.parse_quadrant_linear

::: apsg.helpers.NOTATIONS

### Formatting and checks

::: apsg.helpers.format_planar

::: apsg.helpers.format_linear

::: apsg.helpers.eformat

::: apsg.helpers.is_jsonable
