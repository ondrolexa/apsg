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
    Raises when matrices are not conformable for a certain operation.

    For more information see `https://en.wikipedia.org/wiki/Conformable_matrix`.
    """


class Matrix(object):
    """
    Represents a square matrix M×N of float values.
    The matrix elements has indexes `i` for row and `j` for column writen as `m_{ij}`,
    e.g `m_{12}` represents the element at first row and second column.

    How to derive from this class.

    >>> class Vector(Matrix):  __shape__ = (2, 1)

    >>> v = Vector(1, 2)
    >>> len(v)
    2

    See the ``vector`` module for more details.

    """

    __shape__ = (0, 0) # (uint, uint)

    __slots__ = ("_elements") # Don't forget define this again in each subclass!

    def __new__(cls, *args, **kwargs):
        """
        Create a new instance.

        Check that ``__shape__`` contains the same values e.g (2, 2).

        Raises:
            The ``AssertionError`` when some component of the ``__shape__``
            class attribute has zero value.
        """
        if (0 == cls.__shape__[0]) or (0 == cls.__shape__[1]):
            raise AssertionError("Please define non zero `__shape__` values e.g (2, 1).")

        return super(Matrix, cls).__new__(cls)

    def __init__(self, *elements):  # (floats) -> Matrix
        """
        Take sequence of elements.
        """
        if len(elements) != self.row_count * self.column_count:
            raise AssertionError(
                "The number of elements must be equal to ``{class_name}`` dimension, which is {dimension}".format(
                    class_name=self.__class__.__name__, dimension=self.__shape__[0] * self.__shape__[1]))
        self._elements = elements

    def __getitem__(self, indexes): # (tuple) -> float
        """
        Get the element with a given indexes.

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

    def __len__(self): # () -> int
        """
        Get the number of rows.

        Note:
            For the ``Vector`` class it returns the number of items.
            It is consistent with the idea that the column vector is represented as M × 1 matrix.

        Returns:
            The number of rows.

        Examples:
            >>> Matrix.__shape__ = (2, 2)
            >>> m = Matrix(1, 2, 3, 4)
            >>> len(m)
            2
        """
        return self.__class__.__shape__[0]

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

    # Magic methods

    def __array__(self):
        """
        Get the instance as ``numpy.array``.
        """
        return NotImplemented

    def __repr__(self):
        # type: () -> str
        return self.__class__.__name__ + "(" + str(self.rows) + ")"

    def __str__(self):
        # type: () -> str
        return repr(self)

    def __eq__(self, other):
        # type: (Matrix) -> bool
        return self._elements == other # FIXME

    def __ne__(self, other):
        # type: (Matrix) -> bool
        return not (self == other)

    def __hash__(self):
        # type: () -> int
        return hash((self._elements), self.__class__.__name___)

    # Operators

    def __matmul__(self, other):
        # type: (Matrix) -> Matrix
        """
        Calculate matrix multiplication.

        Raises:
            An exception if ``other`` matrix is not conformable.
        """

        row_count = self.row_count
        column_count = self.column_count
        return NotImplemented

    def __add__(self, other):
        # type: (Matrix) -> Matrix
        """
        Calculate matrix addition.

        Raises:
            An exception if ``other`` matrix is not conformable.
        """
        if self.row_count != other.row_count or self.column_count != other.column_count:
            raise NonConformableMatrix()
        return self.__class__( *[a + b for a, b in zip(self._elements, other._elements)] )

    def __mul__(self, scalar):
        # type: (float) -> Matrix
        """
        Calculate left scalar-matrix multiplication.
        """
        return self.__class__(*map(lambda x: scalar * x, self._elements))

    def __rmul__(self, scalar):
        # type: (float) -> Matrix
        """Calculate right scalar-matrix multiplication."""
        return self * scalar

    # Properties

    @property
    def rows(self):
        # Use the ``itertools.grouper``?
        group = lambda t, n: zip(*[t[i::n] for i in range(n)])
        return list( group(self._elements, self.__shape__[1]) )

    @property
    def row_count(self): # () -> int
        return len(self)

    @property
    def column_count(self): # () -> int
        return self.__class__.__shape__[1]

    def dimension(self):
        return self.row_count * self.column_count


class SquareMatrix(Matrix):
    """
    Represents a square matrix M × N of float values.
    """

    __slots__ = ("_elements") # Don't forget define this again in each subclass!

    def __new__(cls, *args, **kwargs):
        """
        Create a new instance.

        Check that ``__shape__`` contains the same values e.g (2, 2).
        """
        if (cls.__shape__[0] != cls.__shape__[1]):
            raise AssertionError("The ``__shape__`` must contain the same values e.g (2, 2).")

        return super(SquareMatrix, cls).__new__(cls, *args, **kwargs)


class Matrix2(SquareMatrix):
    """
    Represents a square matrix 2 × 2 of float values.
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

    __slots__ = ("_elements") # Don't forget define this again in each subclass!

    def __init__(self, *elements):
        super(Matrix2, self).__init__(*elements)


class Matrix3(SquareMatrix):
    """
    Represents a square matrix 3 × 3 of float values.
    The matrix elements has indexes `i` for row and `j` for column writen as `m_{ij}`,
    e.g `m_{12}` represents the element at first row and second column.

    m_{11} | m_{12} | m_{13}
    m_{22} | m_{22} | m_{23}
    m_{31} | m_{32} | m_{33}

    """

    __shape__ = (3, 3)

    __slots__ = ("_elements") # Don't forget define this again in each subclass!

    def __init__(self, *elements):
        super(Matrix3, self).__init__(*elements)


if __name__ == '__main__':
    m = Matrix2(1, 2, 3, 4)
    m = Matrix3(1, 2, 3, 4, 5, 6, 7, 8, 9)
    # Try wrong number of elements.

    class DiagonalMatrix(SquareMatrix):
        __shape__ = (2, 2)
        # Try change to (1, 2) or (0, 1) or (0, 0).
    m = DiagonalMatrix(1, 2, 3, 4)

