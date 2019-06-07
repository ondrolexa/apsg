# -*- coding: utf-8 -*-


"""

The tests for vector types:

- `Vector2`
- `Vector3`
- `Vector4`

"""

import math


import pytest
import numpy as np

from apsg.algebra import Vector3


# ==============================================================================
# Operators and Magic Methods
# ==============================================================================

# `==` operator

def test_that_equality_operator_is_reflexive():
    u = Vector3(1, 2, 3)
    assert u == u


def test_that_equality_operator_is_symetric():
    u = Vector3(1, 2, 3)
    v = Vector3(1, 2, 3)

    assert u == v and v == u


def test_that_equality_operator_is_transitive():
    u = Vector3(1, 2, 3)
    v = Vector3(1, 2, 3)
    w = Vector3(1, 2, 3)

    assert u == v and v == w and u == w


def test_that_equality_operator_returns_false_for_none():
    assert not (Vector3(1, 0, 0) == None)


# `!=` operator

def test_inequality_operator():
    lhs = Vector3(1, 2, 3)
    rhs = Vector3(3, 2, 1)

    assert lhs != rhs


# `hash` method


def test_that_vector_hashing_works(is_hashable):
    assert is_hashable(Vector3(1, 2, 3))


def test_that_hash_is_same_for_identical_vectors():
    lhs = Vector3(1, 2, 3)
    rhs = Vector3(1, 2, 3)

    assert hash(lhs) == hash(rhs)


def test_that_hash_is_not_same_for_different_vectors():
    lhs = Vector3(1, 2, 3)
    rhs = Vector3(3, 2, 1)

    assert not hash(lhs) == hash(rhs)


# `+` operator


def test_add_operator():
    lhs = Vector3(1, 1, 1)
    rhs = Vector3(1, 1, 1)

    current = lhs + rhs
    expects = Vector3(2, 2, 2)

    assert current == expects


def test_sub_operator():
    lhs = Vector3(1, 2, 3)
    rhs = Vector3(3, 1, 2)

    current = lhs - rhs
    expects = Vector3(-2, 1, 1)

    assert current == expects


# `*` operator (scalar product)

@pytest.mark.skip() # "Use @ operator"
def test_mull_operator():
    import numpy as np

    lhs = Vector3(1, 1, 1)
    rhs = Vector3(1, 1, 1)

    current = lhs * rhs
    expects = lhs.dot(rhs)

    assert np.allclose(current, expects)


# `**` operator (vector product)


def test_pow_operator_with_vector():
    lhs = Vector3(1, 0, 0)
    rhs = Vector3(0, 1, 0)

    current = lhs ** rhs
    expects = lhs.cross(rhs)

    assert current == expects


def test_absolute_value():
    current = abs(Vector3(1, 1, 1))
    expects = math.sqrt(3)

    assert current == expects


def test_length_method():
    w = Vector3(1, 2, 3)

    assert len(w) == 3


def test_getitem_operator():
    v = Vector3(1, 2, 3)

    assert all((v[0] == 1, v[1] == 2, v[2] == 3))

# ==============================================================================
# Properties
# ==============================================================================

# `is_upper` property


def test_that_vector_is_upper_when_z_coordinate_is_negative():
    vec = Vector3(0, 0, -1)

    assert vec.is_upper


def test_that_vector_is_not_upper_z_coordinate_is_positive():
    vec = Vector3(0, 0, 1)

    assert not vec.is_upper


# `flip` property


def test_that_vector_is_flipped():
    current = Vector3(0, 0, 1).flip # flipped
    expects = Vector3(0, 0, -1)

    assert current == expects


# `unit` property


def test_that_vector_is_normalized():
    current = Vector3(1, 0, 0).unit # normalized or as_unit or to_unit
    # expects = Vector3(0.26726124191242442, 0.5345224838248488, 0.8017837257372732)
    expects = Vector3(1, 0, 0)

    assert current == expects


# ==============================================================================
# Methods
# ==============================================================================


# `angle` method


# def test_that_angle_between_vectors_is_0_degrees_when_they_are_collinear():
#     lhs = Vector3(1, 0, 0)
#     rhs = Vector3(2, 0, 0)

#     current = lhs.angle(rhs)
#     expects = 0

#     assert current == expects


# def test_that_angle_between_vectors_is_90_degrees_when_they_are_perpendicular():
#     lhs = Vector3(1, 0, 0)
#     rhs = Vector3(0, 1, 1)

#     current = lhs.angle(rhs)
#     expects = 90  # degrees

#     assert current == expects


# def test_that_angle_between_vectors_is_180_degrees_when_they_are_opposite():
#     lhs = Vector3(1, 0, 0)
#     rhs = Vector3(-1, 0, 0)

#     current = lhs.angle(rhs)
#     expects = 180  # degrees

#     assert current == expects


# ``dot`` method


# def test_scalar_product_of_same_vectors():
#     i = Vector3(1, 2, 3)

#     assert np.allclose(i.dot(i), abs(i) ** 2)


def test_scalar_product_of_orthonornal_vectors():
    u = Vector3(1, 0, 0)
    v = Vector3(0, 1, 0)

    assert u.dot(v) == 0
