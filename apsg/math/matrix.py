# -*- coding: utf-8 -*-


import typing
import operator
import itertools
import functools

from collections.abc import Iterable

from apsg.math.scalar import Scalar


"""
A matrix algebra types and functions.

== Glossary:

- Matrix Dimension: The number of rows and columns that a matrix has is
  called its dimension or its order. By convention, rows are listed first;
  and columns, second.

"""


__all__ = ("Matrix2", "Matrix3", "Matrix4")



class NonConformableMatrix(Exception):
    """
    Raises when matrices are not conformable for a certain operation.

    In mathematics, a matrix is conformable if its dimensions are suitable
    for defining some operation, otherwise it is non conformable.
    """


class NonConformableMatrixForAddition(NonConformableMatrix):
     """
     Raises when matrices are not conformable for addition.
     """


class NonConformableMatrixForMultiplication(NonConformableMatrix):
     """
     Raises when matrices are not conformable for multiplication.
     """


class Slots(type):
    """Append predefined attribute to subclass `__slots__`."""

    def __new__(mcs, name, bases, dic):
        dic['__slots__'] += ('_elements',)
        return type.__new__(mcs, name, bases, dic)


class Matrix(object):
    """
    Represents a (real) matrix of dimension M × N

    "unit":     1.0
    "zero":     0.0
    "inverse": -1.0

    The matrix elements has row index `i` and column index `j` (`m_{ij}`).

    In mathematics one writes `m_{11}` to access the element at first row and
    first column, but implementation is **zero-based**, i.e. the upper
    left-hand corner of a matrix is element (0,0), not element (1,1).


    `Matrix` is immutable -- it's elements can't be changed after the initialization.
    This means that in-place operators as `+=, -=` etc. are not implemented.

    `Matrix` has a structural equality -- the two instances are equal if and
    only if their elements are equal.

    `Matrix` has no memory-saving optimization for sparse matrices.

    Todo:
        - Matrix can be created with any 2D sequence as `numpy.array` or neted lists eg
          Matrix2( [1, 2], [3, 4] ), Matrix2([ [1, 2], [3, 4] ]), Matrix2( ( (1, 2), (3, 4) )

    Examples:

        This only for faster development -- proper unit tests will be added soon.

        Matrix 2 × 2
        ------------

        Create the matrix.
        >>> A = Matrix2(1, 0, 0, 1)       # Implicit conversion to float!
        >>> B = Matrix2(0.0, 1.0, 1.0, 0.0)


        Add two matrices.
        >>> A + B
        Matrix2([(1.0, 1.0), (1.0, 1.0)])

        >>> (A + B) == (B + A)
        True


        Subtract two matrices.
        >>> A - B
        Matrix2([(1.0, -1.0), (-1.0, 1.0)])

        >>> (A - B) != (B - A)
        True


        Multiply matrix by scalar
        >>> 3 * A
        Matrix2([(3.0, 0.0), (0.0, 3.0)])
        >>> A * 3
        Matrix2([(3.0, 0.0), (0.0, 3.0)])

        Multiply matrix by matrix:
        >>> A @ B
        None


        Negate matrix
        >>> -A
        Matrix2([(-1.0, 0.0), (0.0, -1.0)])


        Check hashes.
        >>> C1 = Matrix2.ones()
        >>> C2 = Matrix2.ones()
        >>> (C1 == C2) and (hash(C1) == hash(C2))
        True


        Get a rows.
        >>> A.rows
        [(1.0, 0.0), (0.0, 1.0)]


        Get number of rows
        >>> A.row_count
        2


        Get a columns.
        >>> A.columns
        [(1.0, 0.0), (0.0, 1.0)]


        Get number of columns
        >>> A.column_count
        2


        Get element at [i,j] position.
        >>> A[0,0], A[0,1], A[1,0], A[1,1]
        (1.0, 0.0, 0.0, 1.0)

        __iter__ works!
        >>> [i for i in A]
        [1.0, 0.0, 0.0, 1.0]

        __array__ works!
        >>> import numpy
        >>> numpy.array(A)
        array([[1., 0.],
               [0., 1.]])

        Note: NumPy shows floats `0.0` as `0.`!


        Get a dimension.
        >>> A.dimension
        4


        Get minimum value.
        >>> A.min
        0.0

        Get maximum value.
        >>> A.max
        1.0


        Calculate a determinant.
        >>> A.determinant()
        1.0


        Check if it is a square matrix.
        >>> A.is_square
        True


        # Matrix 2 × 3
        # ------------
        # >>> Matrix.__shape__ = (2, 3)
        # >>> R = Matrix(1,1,1, 2,2,2)
        # >>> R.rows
        # [(1.0, 1.0, 1.0), (2.0, 2.0, 2.0)]
        # >>> R.columns
        # [(1.0, 2.0), (1.0, 2.0), (1.0, 2.0)]

        # >>> R.is_square FIXME
        # False


        # Create diagonal matrix.
        # >>> m = Matrix2.diagonal()
        # >>> m.rows
        # [(1, 0), (0, 1)]


        Matrix 3 × 3
        ------------

        >>> m = Matrix3(1, 0, 0, 0, 1, 0, 0, 0, 1)

        Check th
        >>> m.dimension
        9

        Calculate a determinant.
        >>> m.determinant()
        1.0

        >>> m = Matrix2(1, -2, 3, -4)
        >>> -m
        Matrix2([(-1.0, 2.0), (-3.0, 4.0)])

        >>> m = Matrix2(1, 2, 3, 4)
        >>> m.transpose()
        Matrix2([(1.0, 3.0), (2.0, 4.0)])

        >>> from apsg.math.matrix import Matrix

        How to properly derive from this class? At least you have to define
        the ``__shape__`` class attribute with non zero values e.g:

        >>> class Mat(Matrix): __shape__ = (2, 2)

        Also don't forget to define ``__slots__`` class attribute in each
        subclass! For more details see other classes in this or
        ``apsg.math.vector`` module.
    """

    __shape__ = (0, 0)
    # (number of rows: uint, number of columns: uint)

    __metaclass__ = Slots
    # For Python 3 use `Matrix(metaclass=MatrixSlot)`

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
        # type: (Tuple[Scalar]) -> Matrix
        """
        Creates an matrix with given elements.

        Args:
            elements - An elements in row order, e.g an identity matrix
            I := [[1, 0] [0, 1]] is created as Matrix(1, 0, 0, 1).

            dtype - The type of elements.
        Raises:
            Exception if dimension of matrix does not match the number of items.
        """

        # NOTE: The elements are stored in one-dimensional tuple.
        # An access of an element in `i`-th row and `j`-th column is implemented with
        # formula:  `m[i, j] = number_of_columns * i + j`, see the ``__getitem__`` method.

        # if 1 == len(elements) and isinstance(elements, Iterable):
        #     elements = *elements

        number_of_expected_elements = self.__shape__[0] * self.__shape__[1]
        if len(elements) != number_of_expected_elements:
            raise AssertionError(
                "The number of elements must be equal to ``{class_name}`` dimension, expected {expected}, got {got}".format(
                    class_name=self.__class__.__name__,
                    expected=number_of_expected_elements,
                    got=len(elements)
                )
            )

        self._elements = tuple([
            # Convert -0.0 to 0.0, see https://en.wikipedia.org/wiki/Signed_zero
            # This is especially important for `__neg__` method!
            float(max(0, e)) if (e == 0) else \
                float(e) for e in elements]) # Should we implicitly convert to float?

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
        return hash((self._elements, self.__class__.__name__))

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
        """
        # FIXME How to implement this for matrix and vector together?
        if len(indexes) != 2:
            raise Exception("Number of indexes must be 2.")

        i, j = indexes
        return self._elements[i * self.__shape__[1] + j]

    def __array__(self, dtype=None):
        """
        Get the instance as `numpy.array`.
        """
        import numpy
        return numpy.array(self.rows, dtype=dtype)

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

    def __sub__(self, other):
        # type: (Matrix) -> Matrix
        """
        Calculate matrix subtratcion.

        Arguments:
            Matrix: An other matrix.

        Returns:
            Matrix: A new matrix.

        Raises:
            NonConformableMatrix: Raises if other matrix is not conformable for addition.
        """
        return self + (-other)

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
        if self.column_count != other.row_count:
            raise NonConformableMatrix("FIXME: Reasonable message here!")

        return self.__class__(*(self.rows @ other.rows))

    # #########################################################################
    # Factories
    # #########################################################################

    @classmethod
    def uniform(cls, value):
        # type: (Scalar) -> Matrix
        """
        Create a new instance filled with the same value.
        """
        return cls(*[value for _ in range(0, cls.__shape__[0] * cls.__shape__[1])])

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
        return cls.uniform(0.0) # NOTE Always use float?

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
    def shape(self):
        # type: () -> Tuple[int, int]
        return tuple(self.__shape__)

    @property
    def dimension(self):
        # type: () -> int
        """
        Get the dimension of matrix.

        This is calculated as number of rows × number of columns.
        """
        return self.row_count * self.column_count

    @property
    def is_square(self):
        # type: () -> bool
        """
        Check if ``self`` is a square matrix.
        """
        return self.row_count == self.column_count

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
        # Flatten the zipped rows before passing to constructor.
        # xs = [y for x in zip(self.rows) for y in x]
        return self.__class__(
            *list(itertools.chain.from_iterable(zip(*self.rows))))


class SquareMatrix(Matrix):
    """
    Represents a square matrix M × N.
    """

    def __new__(cls, *args, **kwargs):
        """
        Create a new instance.

        Raises:
            AssertionError - If the ``__shape__`` doesn't contain the same values e.g (1, 2).
        """
        if cls.__shape__[0] != cls.__shape__[1]:
            raise AssertionError(
                "The ``__shape__`` must contain the same values e.g (2, 2)."
            )

        return super(SquareMatrix, cls).__new__(cls, *args, **kwargs)

    # #########################################################################
    # Factories
    # #########################################################################

    @classmethod
    def diagonal(cls, *values):
        """
        Create a diagonal matrix.
        """
        # TODO Check the length of values.
        # Put the value on each diagonal element otherwise 0.0.
        # [print(i, cls.__shape__ - 1) for i in range(functools.reduce((lambda x, y: x * y), cls.__shape__))])
        # print([0.0 if (i % cls.__shape__[1]) else values[0] for i in range(functools.reduce((lambda x, y: x * y), cls.__shape__))])

        # return cls(*[0 if (i % skip_index) else values[i] for i in \
        #     range(functools.reduce((lambda x, y: x * y), cls.__shape__))])

    @classmethod
    def identity(cls):
        """
        Create a identity (or unit) matrix.

        | 1 | 0 | 0 |
        | 0 | 1 | 0 |
        | 0 | 0 | 1 |
        """
        return cls.diagonal(*[1.0 for _ in \
            range(functools.reduce((lambda x, y: x * y), cls.__shape__))])

    @classmethod
    def upper_triangular(cls):
        # type: () -> SquareMatrix
        return NotImplemented

    @classmethod
    def lower_triangular(cls):
        # type: () -> SquareMatrix
        return NotImplemented

    @classmethod
    def symmetric(cls, *values):
        """
        Create a symmetric matrix.

        | 1 | 2 | 3 |
        | x | 4 | 5 |
        | x | x | 6 |
        """
        return NotImplemented

    # #########################################################################
    # Methods
    # #########################################################################

    def trace(self):
        """
        Calculate the matrix trace.
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
        return NotImplementedError

    def inverted():
        # type: () -> SquareMatrix
        return NotImplemented


class Matrix2(SquareMatrix):
    """
    Represents a square matrix of dimension 2 × 2.

    | m_{11} | m_{12} |
    | m_{22} | m_{22} |
    """

    __shape__ = (2, 2)


class Matrix3(SquareMatrix):
    """
    Represents a square matrix of dimension 3 × 3.

    | m_{11} | m_{12} | m_{13} |
    | m_{22} | m_{22} | m_{23} |
    | m_{31} | m_{32} | m_{33} |
    """

    __shape__ = (3, 3)


class Matrix4(SquareMatrix):
    """
    Represents a square matrix of dimension 4 × 4.

    | m_{11} | m_{12} | m_{13} | m_{14} |
    | m_{22} | m_{22} | m_{23} | m_{24} |
    | m_{31} | m_{32} | m_{33} | m_{34} |
    | m_{41} | m_{42} | m_{43} | m_{44} |
    """

    __shape__ = (4, 4)
