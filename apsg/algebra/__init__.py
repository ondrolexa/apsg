# -*- coding: utf-8 -*-


"""
Matrix, Vector and Tensor algebra in low dimension.


# Linear Algebra

--------------------------------------------------------------------------------

## Matrix

### Fact, Terms and Definitions

> A *matrix* is rectangular array of items arranged in rows and columns.

Our implementation is focused on real matrices, that is, matrix elements are
real (complex) numbers represented `by float` type.

- Matrix Dimension: The number of rows and columns that a matrix has is
  called its dimension or its order. By convention, rows are listed first;
  and columns, second.

- Identity Matrix: ...

- A *square matrix* is a matrix with the same number of rows and columns (n × n).

- A *square diagonal matrix* is a square matrix 

- A *scalar matrix* is a square diagonal matrix with equal main diagonal entries.

- Row Vector: Matrix of dimension 1 × n.

- Column Vector: Matrix of dimension n × 1

- Idempotent Matrix: ...

- Rectangular Diagonal Matrix

- Symmetric Diagonal Matrix

- Size - The *size* of a matrix is defined by the number of rows and columns that it contains.
- Dimensions - A matrix with m rows and n columns is called an m × n matrix or m-by-n matrix, while m and n are called its dimensions.
- Row Vector - Matrices with a single row are called row vectors
- Empty Matrix - In some contexts, such as computer algebra programs, it is useful to consider a matrix with no rows or no columns, called an empty matrix.

-  block matrix


# Transformation

--------------------------------------------------------------------------------

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


# Geometric Algebra

--------------------------------------------------------------------------------

...
"""


__all__ = (
	"Tensor",
	"Matrix2", "Matrix3", "Matrix4", "MatrixError",
	"Vector2", "Vector3", "Vector4", "VectorError",
	)


from apsg.algebra.tensor import Tensor
from apsg.algebra.matrix import Matrix2, Matrix3, Matrix4, MatrixError
from apsg.algebra.vector import Vector2, Vector3, Vector4, VectorError
from apsg.algebra.helper import sind, cosd, tand, asind, acosd # TODO Re-import other functions?
