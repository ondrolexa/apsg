import matplotlib.pyplot as plt
import pytest

from apsg.feature._geodata import Foliation, Lineation, Pair
from apsg.math._vector import Vector2, Vector3


@pytest.fixture(autouse=True)
def _close_figures():
    """Close every matplotlib figure after each test.

    Tests that call a plot object's ``init_figure()``/``_render()``
    directly (rather than ``show()``/``savefig()``, which already close
    figure 0 first) leave figures open, which otherwise both accumulates
    past matplotlib's max-open-figure warning threshold and makes
    RosePlot/FabricPlot's hardcoded figure 0 collide with the next test
    that reuses it.
    """
    yield
    plt.close("all")


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
