# -*- coding: utf-8 -*-


"""
A matrix, vector and tensor algebra in low dimension.
"""


__all__ = (
	"Tensor",
	"Matrix2", "Matrix3", "Matrix4", "MatrixError",
	"Vector2", "Vector3", "Vector4", "VectorError",
	)


from apsg.math.algebra.tensor import Tensor
from apsg.math.algebra.matrix import Matrix2, Matrix3, Matrix4, MatrixError
from apsg.math.algebra.vector import Vector2, Vector3, Vector4, VectorError
