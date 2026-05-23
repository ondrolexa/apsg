import pytest

from apsg.feature._geodata import Foliation, Lineation, Pair
from apsg.math._vector import Vector2, Vector3


@pytest.fixture
def v2():
    return Vector2(3, 4)


@pytest.fixture
def v2_unit():
    return Vector2(1, 0)


@pytest.fixture
def v3():
    return Vector3(1, 2, 3)


@pytest.fixture
def fol():
    return Foliation(250, 30)


@pytest.fixture
def lin():
    return Lineation(110, 26)


@pytest.fixture
def pair():
    return Pair(140, 30, 110, 26)
