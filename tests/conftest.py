"""
Several test helper functions.
"""

import pytest


class Helpers:
    @staticmethod
    def help_me():
        return "no"

    @staticmethod
    def is_hashable(obj):
        try:
            hash(obj)
            return True
        except TypeError:
            return False

    @staticmethod
    def has_same_hash_when_value_objects_are_equals(lhs, rhs):
        if lhs != rhs:
            raise Exception("Objects have to be equal!")
        return hash(lhs == rhs)

    @staticmethod
    def has_not_same_hash_when_value_objects_are_not_equals(lhs, rhs):
        if lhs == rhs:
            raise Exception("Objects have not to be equal!")
        return hash(lhs != rhs)


@pytest.fixture
def helpers():
    return Helpers
