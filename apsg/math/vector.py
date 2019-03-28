# -*- coding: utf-8 -*-


"""
A vector algebra types and functions.

== Contains

- ``Vector``

- ``Vector2``

- ``Vector3``

"""


from apsg.math.scalar import Scalar
from apsg.math.matrix import Matrix


class Vector(Matrix):
    """
    Vector is represented as N x 1 matrix.s
    """

    __size__ = (0, 1)

    def __init__(self, *elements):
        super(Vector, self).__init__(*elements)

    def __getitem__(self, index):
        return self._elements[index]


class Vector2(Vector):
    """
    Represents a two-dimensional vector.

    Examples:
        >>> u = Vector2(1, 0)
        >>> v = Vector2(0, 1)
        >>> u + v
        Vector2([(1,), (1,)]) # FIXME We want this ``Vector2(1, 1)``!

    """

    __shape__ = (2, 1)

    def __init__(self, *elements):
        if len(elements) > 2:
            raise Exception("Wrong number of elements")

        super(Vector2, self).__init__(*elements)

    @property
    def x(self):
        return self[0]

    @property
    def y(self):
        return self[1]


class Vector3(Vector):

    __shape__ = (2, 1)

    def __init__(self, *elements):
        if len(elements) > 3:
            raise Exception("Wrong number of elements")

        super(Vector3, self).__init__(*elements)

    def __pow__(self, other): # (Vector3) -> Vector3
        """
        Calculate the vector product.

        Returns:
            The vector product of ``self`` and ``other`` vector.
        """
        return self.__class__(
            self.y * other.z - self.z * other.y,
            self.z * other.x - self.x * other.z,
            self.x * other.y - self.y * other.x
        )

    def cross(self, other): # (Vector3) -> Vector3
        """
         Calculate the vector product.

        Returns:
            The vector product of ``self`` and ``other`` vector.
        """
        return self ** other

    @property
    def x(self):
        return self[0]

    @property
    def y(self):
        return self[1]

    @property
    def z(self):
        return self[1]


if __name__ == '__main__':
    v1 = Vector3(1, 0, 0)
    v2 = Vector3(0, 1, 0)

    print(v1 % v2)

    v1 = Vector3(1, 2, 3)
