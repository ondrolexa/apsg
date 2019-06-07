# -*- coding: utf-8 -*-


"""
Matrix algebra types and functions.
"""


from __future__ import division

import enum
import typing
import operator
import itertools
import functools


from collections.abc import Iterable

from apsg.math.algebra.scalar import Scalar


__all__ = (
    # Low level API
    " MajorOrder", "Matrix", "SquareMatrix", "MatrixError",
    # High Level API
    "Matrix2", "Matrix3", "Matrix4",
    )


# #############################################################################
# Low Level API for developers.
# #############################################################################


class _SlotsInjectorMeta(type):
    """Inject predefined attribute to subclass `__slots__`."""

    def __new__(mcs, name, bases, dic):
        dic['__slots__'] += ('_elements',)
        return type.__new__(mcs, name, bases, dic)


class MatrixError(Exception):
    """
    Raises when there is any problem with matrix.
    """


class NonConformableMatrix(MatrixError):
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


@enum.unique
class MajorOrder(enum.Enum): #
    """
    The vector type; distinguish between row and column vector.

    As an example see the numpy:
    - `array( [ [1, 2, 3] ] )` -- shape (1, 3) => column vector
    - `array( [ [1], [2], [3] ] )` -- shape (3, 1) => row vector.
    """
    ROW = "row"
    COLUMN = "column"


class Matrix(object):
    """
    A rectangular matrix with `M` rows and `N` columns.

    The matrix elements has row index `i` and column index `j` (`m_{ij}`).

    In mathematics one writes `m_{11}` to access the element at first row and
    first column, but implementation is **zero-based**, i.e. the upper
    left-hand corner of a matrix is element (0,0), not element (1,1).

    `Matrix` is immutable -- it's elements can't be changed after the initialization.
    This means that in-place operators as `+=, -=` etc. are not implemented.

    `Matrix` has a structural equality -- the two instances are equal if and
    only if their dimensions and elements are equal.

    For real matrix following `float` values represents:

    - unit element: 1.0
    - zero element: 0.0
    - inverse element: -1.0

    `Matrix` has no memory-saving optimization for sparse matrices.

    Todo:
        - Matrix can be created with any 2D sequence as `numpy.array` or neted lists eg
          Matrix2( [1, 2], [3, 4] ), Matrix2([ [1, 2], [3, 4] ]), Matrix2( ( (1, 2), (3, 4) )

    Also don't forget to define ``__slots__`` class attribute in each
    subclass! For more details see other classes in this or
    ``apsg.math.vector`` module.

    Examples:

        See the tutorial.
    """

    __shape__ = (0, 0)
    # (number of rows: uint, number of columns: uint)

    __metaclass__ = _SlotsInjectorMeta
    # For Python 3 use `Matrix(metaclass=__slots_injector)`

    # #########################################################################
    # Public Methods & Properties
    # #########################################################################

    # =========================================================================
    # Magic Methods
    # =========================================================================

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

    def __init__(self, *elements, **kwargs):
        # type: (Tuple[Scalar]) -> Matrix
        """
        Creates a matrix with given elements.

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
            4
        """
        return len(self._elements)

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

    # -------------------------------------------------------------------------
    # Arithmetic Operators: unary (-); binary (+), (-), (*), (/), (@)
    # -------------------------------------------------------------------------

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
        # Note: The scalar value is always converted to float!
        return self.__class__(*map(lambda x: float(scalar) * x, self._elements))

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

    def __truediv__(self, scalar):
        # type: (Scalar) -> Matrix
        """
        Calculate the matrix-scalar division.

        This is simply a multiplication with inverted scalar value (1/scalar).
        """
        # Note: The scalar value is always converted to float!
        return (1.0 / float(scalar)) * self

    def __rtruediv__(self, scalar):
        # type: (Scalar) -> Matrix
        """
        Calculate the matrix-scalar division.

        This is simply a multiplication with inverted scalar value (1/scalar).
        """
        return self / scalar

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

        import numpy
        rows = (numpy.array(self.rows)  @ numpy.array(other.rows)).flatten()

        result = self.__class__(*rows)

        # If the result is matrix 1x1 then convert the result to scalar.
        if result.shape[0] == self.shape[1] == 1:
            return float(result.rows[0][0])

        # If the result is matrix 1xN or Mx1 then convert to vector.
        if result.shape[0] == 1 or result.shape[1]:
            return result # TODO Vector

        return result

    # __rmatmul__(self, other):
    # If dimensions are (X, Y) and (X, Y) then swap one of the operand.
    # This prevents the explicit conversion from row to column vector.

    # =========================================================================
    # Static & Class Methods
    # =========================================================================

    # -------------------------------------------------------------------------
    # Factories
    # -------------------------------------------------------------------------

    @classmethod
    def uniform(cls, value):
        # type: (Scalar) -> Matrix
        """
        Create a new instance filled with the same values.
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

    # =========================================================================
    # Properties
    # =========================================================================

    @property
    def min(self):
        # type: () -> float
        """
        Get the element with min value.
        """
        return min(self._elements)

    @property
    def max(self):
        # type: () -> float
        """
        Get the element with max value.
        """
        return max(self._elements)

    @property
    def rows(self):
        # type: () -> List[Vector]
        """
        Get the matrix rows.

        Returns: The matrix rows.
        """
        # Choose the vector class with equal number of rows.

        # We keep the number of these "tricks" and circular module
        # dependencies at the minimum but this is quite useful!

        # This is some kind of *Composite Pattern* where superclass
        # returns its subclass.

        # Custom `group` lambda function.
        # Use the `itertools.grouper` instead?
        group = lambda t, n: zip(*[t[i::n] for i in range(n)])

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
        return self.__shape__

    @property
    def dimension(self):
        # type: () -> int
        """
        Get the dimension of matrix.
        """
        return self.__shape__

    @property
    def is_square(self):
        # type: () -> bool
        """
        Check if ``self`` is a square matrix.
        """
        return self.row_count == self.column_count

    # is_

    # #########################################################################
    # Methods
    # #########################################################################

    def row(self, index):
        # type: (int) -> Vector
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

        # Create a new instance.
        instance = self.__class__(
            *list(itertools.chain.from_iterable(zip(*self.rows))))

        # Swap the shape rows and columns.
        self.__shape__ = tuple(reversed(self.__shape__))

        return instance


class SquareMatrix(Matrix):
    """
    A square matrix of real values.
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
    def diagonal(cls, *values): # FIXME
        """
        Create a diagonal matrix.
        """
        # TODO Check the length of values.
        # Put the value on each diagonal element otherwise 0.0.
        # [print(i, cls.__shape__ - 1) for i in range(functools.reduce((lambda x, y: x * y), cls.__shape__))])
        # print([0.0 if (i % cls.__shape__[1]) else values[0] for i in range(functools.reduce((lambda x, y: x * y), cls.__shape__))])

        return cls(*[0 if (i % skip_index) else values[i] for i in \
            range(functools.reduce((lambda x, y: x * y), cls.__shape__))])

    @classmethod
    def identity(cls): # FIXME
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
        if 1 == self.row_count:
            # FIXME What a row vector?
            # NumPy distinguishes row and column vector.
            return self[0]
            # This is techniccaly an vector with 1 value => scalar.
            # Should we allow it?

        if 2 == self.row_count:
            # 2 × 2 matrix
            return self[0, 0] * self[1, 1] - self[0, 1] * self[1, 0]

        if 3 == self.row_count:
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

    # def is_regular(self): ...

    # def is_symmetric(self): ...

    # eigenvalues()

    # eigenvectors(


# #############################################################################
# High Level API for power users.
# #############################################################################


class Matrix2(SquareMatrix):
    """
    Represents a square matrix of dimension 2 × 2.
    """

    __shape__ = (2, 2)


class Matrix3(SquareMatrix):
    """
    Represents a square matrix of dimension 3 × 3.
    """

    __shape__ = (3, 3)


class Matrix4(SquareMatrix):
    """
    Represents a square matrix of dimension 4 × 4.

    Useful if you want to combine translations and rotations
    or apply perspective projection.
    """

    __shape__ = (4, 4)
