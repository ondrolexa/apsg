# apsg.math

The `apsg.math` module provides basic linear algebra classes for structural geology.
It includes 2D and 3D vectors (`Vector2`, `Vector3`), axial vectors (`Axial2`, `Axial3`),
and matrices (`Matrix2`, `Matrix3`) with operations commonly used in orientation analysis.

The main APSG namespace provides lowercase aliases for commonly used classes (e.g. `vec2` for
`Vector2`, `vec` for `Vector3`, `matrix` for `Matrix3`).
See the [alias table](index.md#aliases) for the full list.

## Usage

3D vectors:

```pycon
>>> from apsg import vec
>>> v = vec(1, 2, 3)
>>> v.magnitude()
>>> v.normalized()
>>> v.geo
>>> v.angle(vec(0, 0, 1))
```

Vector arithmetic:

```pycon
>>> u = vec(45, 30)   # from trend/plunge
>>> v = vec(1, 0, 0)
>>> u + v
>>> u.cross(v)
>>> u.dot(v)
```

2D vectors:

```pycon
>>> from apsg import vec2
>>> v2 = vec2(1, 0)
```

Matrices:

```pycon
>>> from apsg import matrix
>>> m = matrix([[1, 0, 0], [0, 2, 0], [0, 0, 3]])
>>> m.eigenvalues()
>>> m.eigenvectors()
```

Rotations:

```pycon
>>> from apsg import vec
>>> v = vec(1, 0, 0)
>>> v.rotate(vec(0, 0, 1), 90)
```

## Reference

### Vectors

::: apsg.math.Vector2

::: apsg.math.Axial2

::: apsg.math.Vector3

::: apsg.math.Axial3

### Matrices

::: apsg.math.Matrix2

::: apsg.math.Matrix3
