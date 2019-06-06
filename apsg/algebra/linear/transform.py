# -*- coding: utf-8 -*-


"""
Contains transformations as rotation, translation etc. represented as matrices.

# Transformation

## Rotation

### Rotation 2D

R = [[cos(pfi), -sin(phi)], [sin(phi), cos(phi)]]

### Rotation 3D

#### Elemental Rotation

Rx = [ [1, 0, 0], [0, cos(phi), -sin(phi)], [0, sin(phi), cos(phi)] ]

Ry = [ [cos(phi), 0, sin(phi)], [0, 1, 0], [-sin(phi), 0, cos(phi)] ]

Rz = [ [cos(phi), -sin(phi), 0], [sin(phi), cos(phi), 0], [0, 0, 1] ]

#### Yaw, Pitch, Roll

R = Rx(alpha) @ Ry(beta) @ Rz(gamma)

## Reflection

...

## Orthogonal projection

...

# Shear 2D

[[1, m], [0, 1]]

## Resources

- https://en.wikipedia.org/wiki/Rotation_matrix

"""
