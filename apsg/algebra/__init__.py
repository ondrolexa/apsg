# -*- coding: utf-8 -*-


"""
Matrix, Vector and Tensor algebra in low dimension.
"""


__all__ = (
	"Tensor",
	"Matrix2", "Matrix3", "Matrix4", "MatrixError",
	"Vector2", "Vector3", "Vector4", "VectorError",
	)


from apsg.algebra.linear.tensor import Tensor
from apsg.algebra.linear.matrix import Matrix2, Matrix3, Matrix4, MatrixError
from apsg.algebra.linear.vector import Vector2, Vector3, Vector4, VectorError
from apsg.algebra.linear.helper import sind, cosd, tand, asind, acosd # TODO Re-import other functions?
