# -*- coding: utf-8 -*-


import typing
import operator
import itertools

from collections.abc import Iterable

from apsg.math.scalar import Scalar


"""
A matrix algebra types and functions.
"""


__all__ = ("Matrix2", "Matrix3", "Matrix4")


class NonConformableMatrix(Exception):
    """
    Raises when matrices are not conformable for a certain operation.

    For more information see `https://en.wikipedia.org/wiki/Conformable_matrix`.
    """


class Matrix(object):
    """
    Represents a matrix of dimension M × M.

    This type is immutable -- it's elements can't be changed
    after an initialization.

    This type also has a structural equality -- the two instances are equal if and
    only if their elements are equal.

    The matrix elements has indexes `i` for row and `j` for column written as `m_{ij}`,
    e.g `m_{12}` represents the element at first row and second column.

    Tests:

        Matrix 2 × 2

        >>> m = Matrix2(1, 0, 0, 1)
        >>> m.dimension
        4

        Calculate a determinant.
        >>> m.determinant()
        1

        Create 3 × 3 Matrix.
        >>> m = Matrix3(1, 0, 0, 0, 1, 0, 0, 0, 1)

        Check th
        >>> m.dimension
        9

        Calculate a determinant.
        >>> m.determinant()
        1

        >>> m = Matrix2(1, -2, 3, -4)
        >>> -m
        Matrix2([(-1, 2), (-3, 4)])

        >>> m = Matrix2(1, 2, 3, 4)
        >>> m.transpose()
        Matrix2([(1, 3), (2, 4)])

    Examples:

        >>> from apsg.math.matrix import Matrix

        How to properly derive from this class? At least you have to define
        the ``__shape__`` class attribute with non zero values e.g:

        >>> class Mat(Matrix): __shape__ = (2, 2)

        Also don't forget to define ``__slots__`` class attribute in each
        subclass! For more details see other classes in this or
        ``apsg.math.vector`` module.
    """

    __shape__ = (0, 0)  # (number of rows: uint, number of columns: uint)
    __slots__ = ("_elements",)

    # #########################################################################
    # Magic methods
    # #########################################################################

    def __new__(cls, *args, **kwargs):
        """
        Create a new instance.

        Check that ``__shape__`` contains the same values e.g (2, 2).

        Raises:
            The ``AssertionError`` when some component of the ``__shape__``
            class attribute has zero value.
        """
        if (0 == cls.__shape__[0]) or (0 == cls.__shape__[1]):
            raise AssertionError(
                "Please define non zero `__shape__` values e.g (2, 1)."
            )

        return super(Matrix, cls).__new__(cls)

    def __init__(self, *elements):
        # type: (Tuple[float]) -> Matrix
        """
        Creates an matrix with given elements.

        Args:
            elements - An elements in row order, e.g an identity matrix
            I := [[1, 0] [0, 1]] is created as Matrix(1, 0, 0, 1).
        Raises:
            Exception if dimension of matrix does not match the number of items.
        """

        # NOTE: The elements are stored in one-dimensional tuple.
        # An access of an element in `i`-th row and `j`-th column is implemented with
        # formula:  `m[i, j] = number_of_columns * i + j`, see the ``__getitem__`` method.

        # if 1 == len(elements) and isinstance(elements, Iterable):
        #     elements = *elements

        if len(elements) != self.__shape__[0] * self.__shape__[1]:
            raise AssertionError(
                "The number of elements must be equal to ``{class_name}`` dimension, which is {dimension}".format(
                    class_name=self.__class__.__name__,
                    dimension=self.__shape__[0] * self.__shape__[1],
                )
            )

        self._elements = elements

    def __repr__(self):
        # type: () -> str
        return self.__class__.__name__ + "(" + str(self.rows) + ")"

    __str__ = __repr__

    def __eq__(self, other):
        # type: (Matrix) -> bool
        """
        Returns:
            bool: `True` when the components are equal otherwise `False`.
        """
        # isinstance(other, self.__class__) # ???
        return self._elements == other  # FIXME

    def __ne__(self, other):
        # type: (Matrix) -> bool
        """
        Returns:
            bool: `True` when the components are not equal otherwise `False`.

        Note:
            This have to be implemented for Python 2 compatibility.
        """
        return not (self == other)

    def __hash__(self):
        # type: () -> int
        return hash((self._elements, self.__class__.__name___))

    def __len__(self):
        # () -> int
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
        return self.__shape__[
            0
        ]  # * self.__shape__[1] # FIXME: Return dimension not number of rows

    def __iter__(self):
        """
        Return the iterator.
        """
        return iter(self._elements)

    def __getitem__(self, indexes):
        # type: (Tuple[int]) -> float
        """
        Get the element with a given indexes.

        Examples:
            >>> Matrix.__shape__ = (2, 2)

            >>> m = Matrix(11, 12, 21, 22)
            >>> m[0, 0], m[0, 1], m[1, 0], m[1, 1]
            (11, 12, 21, 22)
        """
        # FIXME How to implement this for matrix and vector?
        if len(indexes) != 2:
            raise Exception("Number of indexes must be 2.")

        i, j = indexes
        return self._elements[i * self.__shape__[1] + j]

    def __array__(self, dtype=None):
        """
        Get the instance as `numpy.array`.
        """
        return np.array(self.rows, dtype=dtype)

    # #########################################################################
    # Operators
    # #########################################################################

    def __neg__(self):
        # type: () -> Matrix
        """
        Change the sign of each element.
        """
        return self.__class__(*map(operator.neg, self._elements))

    def __mul__(self, scalar):
        # type: (Scalar) -> Matrix FIXME
        """
        Calculate the scalar-matrix multiplication.

        Arguments:
            scalar - A scalar value.

        Returns:
            Matrix: A new matrix.
        """
        return self.__class__(*map(lambda x: scalar * x, self._elements))

    def __rmul__(self, scalar):
        # type: (Scalar) -> Matrix
        """
        Calculate the matrix-scalar multiplication.

        Arguments:
            scalar -- A scalar value.

        Returns:
            Matrix: A new matrix.
        """
        return self * scalar

    def __add__(self, other):
        # type: (Matrix) -> Matrix
        """
        Calculate matrix addition.

        Arguments:
            Matrix: An other matrix.

        Returns:
            Matrix: A new matrix.

        Raises:
            NonConformableMatrix: Raises if other matrix is not conformable for addition.
        """
        if self.row_count != other.row_count or self.column_count != other.column_count:
            raise NonConformableMatrix()

        return self.__class__(*[a + b for a, b in zip(self._elements, other._elements)])

    def __matmul__(self, other):
        # type: (Matrix) -> Matrix
        """
        Calculate the matrix-matrix multiplication.

        Arguments:
            Matrix: An other matrix.

        Returns:
            Matrix: A new matrix.

        Raises:
            NonConformableMatrix: Raises if other matrix is not conformable for multiplication.
        """

        row_count = self.row_count
        column_count = self.column_count
        return NotImplemented

    # #########################################################################
    # Factories
    # #########################################################################

    @classmethod
    def uniform(cls, value):
        # type: (Scalar) -> Matrix
        """
        Create a new instance filled with the same value.
        """
        return cls(value for _ in range(0, cls.__shape__[0] * cls.__shape__[1]))

    @classmethod
    def ones(cls):
        # type: (Scalar) -> Matrix
        """
        Create a new instance filled with values 1.0.
        """
        return cls.uniform(1.0)

    @classmethod
    def zeros(cls):
        # type: (Scalar) -> Matrix
        """
        Create a new instance filled with values 0.0.
        """
        return cls.uniform(0.0)

    # @classmethod
    # def from_rows(cls, rows):
    #     ...

    # @classmethod
    # def from_columns(cls, columns):
    #     ...

    # #########################################################################
    # Properties
    # #########################################################################

    @property
    def min(self):
        # type: () -> float
        """
        Get the element with minimal value.
        """
        return min(self._elements)

    @property
    def max(self):
        # type: () -> float
        """
        Get the element with maximal value.
        """
        return max(self._elements)

    @property
    def rows(self):
        # type: () -> List[List[float]]
        """
        Get the matrix rows.

        Returns: The matrix rows.
        """
        group = lambda t, n: zip(*[t[i::n] for i in range(n)])
        # Use the ``itertools.grouper`` instead of custom function?
        return list(group(self._elements, self.__shape__[1]))

    @property
    def columns(self):
        # type: () -> List[List[float]]
        """
        Get the matrix columns.

        Returns: The matrix columns.
        """
        return self.transpose().rows

    @property
    def row_count(self):
        # type: () -> int
        """
        Get the number of rows.
        """
        return len(self.rows)

    @property
    def column_count(self):
        # type: () -> int
        """
        Get the number of columns.
        """
        return len(self.columns)

    @property
    def dimension(self):
        # type: () -> int
        """
        Get the dimension of matrix.

        This is calculated as number of rows × number of columns.
        """
        return self.row_count * self.column_count

    # #########################################################################
    # Methods
    # #########################################################################

    def row(self, index):
        # type: (int) -> List[float]
        return self.rows[index]

    def column(self, index):
        # type: (int) -> List[float]
        return self.columns[index]

    def transpose(self):
        # type: () -> Matrix
        """
        Transpose the matrix.
        """
        return self.__class__(*list(itertools.chain.from_iterable(zip(*self.rows))))

    def is_square(self):
        # type: () -> bool
        """
        Check if ``self`` is a square matrix.
        """
        return self.row_count == self.column_count


class SquareMatrix(Matrix):
    """
    Represents a square matrix M × N of float values.
    """

    __slots__ = ("_elements",)  # Don't forget define this again in each subclass!

    def __new__(cls, *args, **kwargs):
        """
        Create a new instance.

        Check that ``__shape__`` contains the same values e.g (2, 2).
        """
        if cls.__shape__[0] != cls.__shape__[1]:
            raise AssertionError(
                "The ``__shape__`` must contain the same values e.g (2, 2)."
            )

        return super(SquareMatrix, cls).__new__(cls, *args, **kwargs)

    @classmethod
    def diagonal(cls, value):
        """
        Create a diagonal matrix.
        """

    @classmethod
    def identity(cls):
        """
        Create a identity (or unit) matrix.
        """
        return cls.diagonal(1.0)

    # upper_triangular(cls)

    # lower_triangular(cls)

    @classmethod
    def symmetric(cls, *values):
        """
        Create a symmetric matrix.

        |1 | 2 | 3 |
        |x | 4 | 5 |
        |x | x | 6 |
        """
        return NotImplemented

    def determinant(self):
        """
        Calculate a determinant of matrix.
        """
        if 1 == self.dimension:
            return self[0]
            # This is techniccaly an vector with 1 valeu => scalar.
            # Should we allow it?

        if 4 == self.dimension:
            # 2 × 2 matrix
            return self[0, 0] * self[1, 1] - self[0, 1] * self[1, 0]

        if 9 == self.dimension:
            # 3 × 3 matrix
            return (
                (self[0, 0] * self[1, 1] * self[2, 2])
                + (self[1, 0] * self[2, 1] * self[0, 2])
                + (self[2, 0] * self[0, 1] * self[1, 2])
                - (self[0, 2] * self[1, 1] * self[2, 0])
                - (self[1, 2] * self[2, 1] * self[0, 0])
                - (self[2, 2] * self[0, 1] * self[1, 0])
            )

        # FIXME: There should be some general method e.g  Leibniz eq., but it is very slow.
        return NotImplemented

    # def inverted() # type: () -> SquareMatrix


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
    __slots__ = ("_elements",)  # Don't forget define this again in each subclass!

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
    __slots__ = ("_elements",)  # Don't forget define this again in each subclass!

    def __init__(self, *elements):
        super(Matrix3, self).__init__(*elements)


class Matrix4(SquareMatrix):
    """
    Represents a square matrix 4 × 4 of float values.

    The matrix elements has indexes `i` for row and `j` for column writen as `m_{ij}`,
    e.g `m_{12}` represents the element at first row and second column.

    m_{11} | m_{12} | m_{13} | m_{14}
    m_{22} | m_{22} | m_{23} | m_{24}
    m_{31} | m_{32} | m_{33} | m_{34}
    m_{41} | m_{42} | m_{43} | m_{44}

    """

    __shape__ = (4, 4)
    __slots__ = ("_elements",)  # Don't forget define this again in each subclass!

    def __init__(self, *elements):
        super(Matrix3, self).__init__(*elements)
