# -*- coding: utf-8 -*-


import operator as op


__all__ = ("Matrix2", "Matrix3")


class NonConformableMatrix(Exception):
    """
    Raises when matices are not conformable for a certain operation, see
    `https://en.wikipedia.org/wiki/Conformable_matrix`.
    """


class Matrix(object):

    __size__ = (0, 0) # (uint, uint)

    def __init__(self, *elements):
        """
        Take sequence of elements.
        """
        self._elements = elements

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

    @property
    def rows(self):
        return [self._elements]

    @property
    def row_count(self): # () -> int
        return self.__class__.__size__[0]

    @property
    def column_count(self): # () -> int
        return self.__class__.__size__[1]

    def __array__(self):
        return NotImplemented

    def __repr__(self): # () -> str
        return self.__class__.__name__ + "(" + str(self._elements) + ")"

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
        print(other)

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

    """
    __size__ = (2, 2)

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

    __size__ = (3, 3)

    def __init__(self, *elements):
        super(Matrix3, self).__init__(*elements)


if __name__ == '__main__':

    m1 = Matrix2(1, 0, 0, 1)
    m2 = Matrix2(0, 1, 1, 0)

    print(" __add__")
    print(m1 + m2)
    print(m2 + m1)

    print("__mul__")
    print(m1 * 2)
    print(2 * m1)

    print("__matmul__")
    print(m1 @ m2)
    print(m1 % m2)

    # v = Vector2(0, 1)
    # print(
    #     m.row_count,
    #     m.column_count
    # )

