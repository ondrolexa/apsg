# -*- coding: utf-8 -*-


"""
How to create a helper test function?
- https://stackoverflow.com/a/42156088
- https://stackoverflow.com/a/51389067
"""


import pytest


@pytest.fixture
def is_hashable():
    def func(obj):
        try:
            hash(obj)
            return True
        except TypeError:
            return False
    return func


@pytest.fixture
def has_same_hash_when_value_objects_are_equals(lhs, rhs):
    def  func(lhs, rhs):
        if lhs != rhs:
            raise Exception("Objects have to be equal!")
        return hash(lhs == rhs)

    return func


@pytest.fixture
def has_not_same_hash_when_value_objects_are_not_equals(lhs, rhs):
    def  func(lhs, rhs):
        if lhs == rhs:
            raise Exception("Objects have not to be equal!")
        return hash(lhs != rhs)

    return func
