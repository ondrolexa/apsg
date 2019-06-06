# -*- coding: utf-8 -*-


"""
A vector algebra types and functions.
"""


import math

import numpy as np

from apsg.algebra.linear.helper import acosd
from apsg.algebra.linear.scalar import Scalar
from apsg.algebra.linear.matrix import Matrix, MajorOrder
# TODO: Use `MajorOrder` in matrix and reuse in vector.


__all__ = ("Vector2", "Vector3", "Vector4", "VectorError")


# #############################################################################
# Low Level API for developers.
# #############################################################################


class VectorError(Exception):
    """
    Raises when there is any problem with vector.
    """


class NonConformableVectors(VectorError): # Derive from matrix exception?
    """
    Raises when vectors are not conformable for a certain operation.
    """


class Vector(Matrix):
    """
    A vector base class represented as M x 1 (row) or 1 x N (column) vector.
    """

    def __init__(self, *elements, order):
        """
        Create a new vector.

        Arguments:
            elements (float) - The vector coordinate values.
            type (enum) - The row or column orientation.
        """
        super(Vector, self).__init__(*elements)
        self._order = MajorOrder(order)

    def __getitem__(self, index):
        return self._elements[index]

    def __abs__(self):
        # type: () -> Scalar
        return math.sqrt(sum(self._elements))

    def dot(self, other):
        # type: (Vector) -> Scalar
        # todo: Don't call protected attribute -- use operator `[]`?
        # todo: Use @ operator?
        return sum([i * j for (i, j) in zip(self._elements, other._elements)])

    @property
    def order(self):
        return self._order

    @property
    def unit(self): # normalized
        """
        Normalize the vector to unit length.

        Returns:
          The unit vector of `self`.

        Example:
          >>> v = Vector(1, 1, 1)
          >>> v.unit
          V(0.577, 0.577, 0.577)

        """
        # return self / abs(self)
        return self.__class__(*[x / abs(self) for x in self._elements])

    def angle(self, other):
        # (Vector) -> float # AngleInDegrees
        """
        Calculate the angle between two vectors in degrees.

        Args:
            other: other ``Vector`` vector

        Returns:
            The angle between `self` and `other` in degrees.

        Example:
            >>> v = Vector(1, 0, 0)
            >>> u = Vector(0, 0, 1)
            >>> v.angle(u)
            90.0
        """
        # if isinstance(other, Group):
        #     return other.angle(self)
        # else:
        return acosd(np.clip(np.dot(self.unit, other.unit), -1, 1))


# #############################################################################
# High Level API for power users.
# #############################################################################


class Vector2(Vector):
    """
    Represents a two-dimensional row or column vector.

    Examples:
        >>> u = Vector2(1, 0)
        >>> v = Vector2(0, 1)
        >>> u + v
        Vector2([(1.0,), (1.0,)])

        # FIXME We want this ``Vector2(1, 1)``!
    """

    __shape__ = (2, 1)

    def __init__(self, *elements, order="column"):
        expected_number_of_elements = self.__shape__[0] * self.__shape__[1]

        if len(elements) > expected_number_of_elements:
            raise Exception( # FIXME Use non generic exception!
                "Wrong number of elements, expected {0}, got {1}".format(
                    expected_number_of_elements, len(elements)))

        super(Vector2, self).__init__(*elements, order=order)

    @property
    def x(self):
        return self[0]

    @property
    def y(self):
        return self[1]

    @classmethod
    def unit_x(cls):
        return cls(1, 0)

    @classmethod
    def unit_y(cls):
        return cls(0, 1)


class Vector3(Vector):
    """
    Represents a three-dimensional row or column vector.
    """

    __shape__ = (3, 1)

    def __init__(self, *elements, order="column"):
        expected_number_of_elements = self.__shape__[0] * self.__shape__[1]

        if len(elements) > expected_number_of_elements:
            raise WrongNumberOfElements(
                "Wrong number of elements, expected {0}, got {1}".format(
                    expected_number_of_elements, len(elements)))

        super(Vector3, self).__init__(*elements, order=order)

    def __pow__(self, other):  # (Vector3) -> Vector3
        """
        Calculate the vector product between ``self`` and ``other`` vector.

        Returns:
            The vector product of ``self`` and ``other`` vector.
        """
        return self.__class__(
            self.y * other.z - self.z * other.y,
            self.z * other.x - self.x * other.z,
            self.x * other.y - self.y * other.x,
        )

    @classmethod
    def unit_x(cls):  # e1
        return cls(1, 0, 0)

    @classmethod
    def unit_y(cls):  # e2
        return cls(0, 1, 0)

    @classmethod
    def unit_z(cls):  # e3
        return cls(0, 0, 1)

    @property
    def x(self):
        return self[0]

    @property
    def y(self):
        return self[1]

    @property
    def z(self):
        return self[2]

    def cross(self, other):  # (Vector3) -> Vector3
        """
        Calculate the vector product between ``self`` and ``other`` vector.

        Returns:
            The vector product of ``self`` and ``other`` vector.
        """
        return self ** other

    @property
    def is_upper(self):
        # FIXME This is a coordinate system dependent. How to keep it independent?
        # () -> bool
        """
        Return `True` if z-coordinate is negative, otherwise `False`.
        """
        return np.sign(self.z) < 0

    @property
    def flip(self): # flipped
        """
        Return a new vector with inverted `z` coordinate.
        """
        return self.__class__(self.x, self.y, -self.z)


class Vector4(Vector):
    """
    Represents a four-dimensional row or column vector.
    """

    __shape__ = (4, 1)

    def __init__(self, *elements, order="column"):
        expected_number_of_elements = self.__shape__[0] * self.__shape__[1]

        if len(elements) > expected_number_of_elements:
            raise WrongNumberOfElements(
                "Wrong number of elements, expected {0}, got {1}".format(
                    expected_number_of_elements, len(elements)))

        super(Vector3, self).__init__(*elements, order=order)

    @classmethod
    def unit_x(cls):
        return cls(1, 0, 0, 0)

    @classmethod
    def unit_y(cls):
        return cls(0, 1, 0, 0)

    @classmethod
    def unit_z(cls):
        return cls(0, 0, 1, 0)

    @classmethod
    def unit_w(cls):
        return cls(0, 0, 0, 1)

    @property
    def x(self):
        return self[0]

    @property
    def y(self):
        return self[1]

    @property
    def z(self):
        return self[2]

    @property
    def w(self):
        return self[3]
