# -*- coding: utf-8 -*-


from apsg.math import Vector3


def test_that_vector_hashing_works(is_hashable):
    assert is_hashable(Vector3(1, 2, 3))


# ------------------------------------------------------------------------------
# Operators
# ------------------------------------------------------------------------------

# ``==`` operator

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
