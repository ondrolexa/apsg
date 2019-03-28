# -*- coding: utf-8 -*-


import operator as op
import itertools as it


"""
A matrix algebra types and functions.

== Overview

- ``Matrix``

- ``Matrix2``

- ``Matrix3``

"""


__all__ = ("Matrix2", "Matrix3")


class NonConformableMatrix(Exception):
    """
    Raises when matices are not conformable for a certain operation, see
    `https://en.wikipedia.org/wiki/Conformable_matrix`.
    """


class Matrix(object):

    __shape__ = (0, 0) # (uint, uint)

    def __init__(self, *elements):
        """
        Take sequence of elements.
        """
        self._elements = elements

    def __getitem__(self, indexes):
        """
        Gets the element with a given indexes.

        Examples:

            >>> Matrix.__shape__ = (2, 2)

            >>> m = Matrix(11, 12, 21, 22)
            >>> m[0, 0], m[0, 1], m[1, 0], m[1, 1]
            (11, 12, 21, 22)

        """
        if len(indexes) != len(self.__shape__):
            raise Exception("Number of indexes must match the shape.")

        i, j = indexes
        return self._elements[i * self.__shape__[1] + j]

    # Factory methods

    @classmethod
    def from_rows(cls, row):
        """
        [[1, 0], [0, 1]]
          row1    row2
        """
        ...

    @classmethod
    def from_columns(cls, columns):
        ...

    # Properties

    @property
    def rows(self):
        # Use the ``itertools.grouper``?
        group = lambda t, n: zip(*[t[i::n] for i in range(n)])
        return list( group(self._elements, self.__shape__[1]) )

    @property
    def row_count(self): # () -> int
        return self.__class__.__shape__[0]

    @property
    def column_count(self): # () -> int
        return self.__class__.__shape__[1]

    def __array__(self):
        return NotImplemented

    def __repr__(self): # () -> str
        return self.__class__.__name__ + "(" + str(self.rows) + ")"

    def __str__(self): # () -> str
        return repr(self)

    def __eq__(self, other): # (Matrix) -> bool
        return self._elements == other # FIXME

    def __ne__(self, other): # (Matrix) -> bool
        return not (self == other)

    def __hash__(self): # () -> int
        return hash((self._elements), self.__class__.__name___)

    def __matmul__(self, other): # (Matrix) -> Matrix
        row_count = self.row_count
        column_count = self.column_count
        return NotImplemented


    def __add__(self, other): # (Matrix) -> Matrix
        """
        Raises:
            An exception if ``other`` matrix is not conformable.
        """
        if self.row_count != other.row_count or self.column_count != other.column_count:
            raise NonConformableMatrix()
        return self.__class__( *[a + b for a, b in zip(self._elements, other._elements)] )

    def __mul__(self, scalar): # (float) -> Matrix
        """
        Calculate left scalar-matrix multiplication.
        """
        return self.__class__(*map(lambda x: scalar * x, self._elements))

    def __rmul__(self, scalar):
        """Calculate right scalar-matrix multiplication."""
        return self * scalar


class Matrix2(Matrix):
    """
    Represents a square matrix 2×2 of float values.
    The matrix elements has indexes `i` for row and `j` for column writen as `m_{ij}`,
    e.g `m_{12}` represents the element at first row and second column.

    m_{11} | m_{12}
    m_{22} | m_{22}

    Examples:
        >>> m = Matrix2(11, 12, 21, 22)
        >>> m[0, 0], m[0, 1], m[1, 0], m[1, 1]
        (11, 12, 21, 22)

        >>> m1 = Matrix2(1, 2, 3, 4)
        >>> m2 = Matrix2(1, 2, 3, 4)

        Operators

        >>> m1 = Matrix2(1, 2, 3, 4)
        >>> m1.rows
        [(1, 2), (3, 4)]

        >>> m2 = Matrix2(4, 3, 2, 1)
        >>> m2.rows
        [(4, 3), (2, 1)]


        ``+`` Matrix addition

        >>> m1 + m2
        Matrix2([(5, 5), (5, 5)])

        >>> m2 + m1
        Matrix2([(5, 5), (5, 5)])


        ``*`` Matrix, scalar multiplication

        >>> m1 * 2
        Matrix2([(2, 4), (6, 8)])

        >>> 2 * m1
        Matrix2([(2, 4), (6, 8)])


        ``@`` Matrix, matrix multiplication
        # >>> m1 @ m2
        # Matrix2([(7, 10), (15, 14)])

    """
    __shape__ = (2, 2)

    def __init__(self, *elements):
        super(Matrix2, self).__init__(*elements)


class Matrix3(Matrix):
    """
    Represents a square matrix 3×3 of float values.
    The matrix elements has indexes `i` for row and `j` for column writen as `m_{ij}`,
    e.g `m_{12}` represents the element at first row and second column.

    m_{11} | m_{12} | m_{13}
    m_{22} | m_{22} | m_{23}
    m_{31} | m_{32} | m_{33}

    """

    __shape__ = (3, 3)

    def __init__(self, *elements):
        super(Matrix3, self).__init__(*elements)


if __name__ == '__main__':
    pass
    # v = Vector2(0, 1)
    # print(
    #     m.row_count,
    #     m.column_count
    # )

