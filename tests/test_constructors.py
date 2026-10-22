"""Every documented constructor call form, checked against an equivalent one."""

import warnings

import numpy as np
import pytest

from apsg.feature import Arc, Cone, Direction, Fault, Foliation, Lineation, Pair
from apsg.math import Vector2, Vector3

L = Lineation(120, 30)
F = Foliation(200, 40)
P = Pair(140, 30, 110, 26)


def _close(a, b):
    return type(a) is type(b) and np.allclose(np.asarray(a), np.asarray(b))


@pytest.mark.parametrize(
    ("made", "expected"),
    [
        (Vector2(), Vector2(1, 0)),
        (Vector2("x"), Vector2(1, 0)),
        (Vector2("Y"), Vector2(0, 1)),
        (Vector2(90), Vector2(0, 1)),
        (Vector2((1, 2)), Vector2(1, 2)),
        (Vector2(np.array([1.0, 2.0])), Vector2(1, 2)),
        (Direction("y"), Direction(0, 1)),
        (Vector3(), Vector3(1, 0, 0)),
        (Vector3("y"), Vector3(0, 1, 0)),
        (Vector3("Z"), Vector3(0, 0, 1)),
        (Vector3((1, 2, 3)), Vector3(1, 2, 3)),
        (Vector3(np.array([1.0, 2.0, 3.0])), Vector3(1, 2, 3)),
        (Vector3(0, 90), Vector3(0, 0, 1)),
        (Lineation(), Lineation(1, 0, 0)),
        (Lineation("y"), Lineation(0, 1, 0)),
        (Lineation("N45E,30"), Lineation(45, 30)),
        (Foliation(), Foliation(0, 0, 1)),
        (Foliation("x"), Foliation(1, 0, 0)),
        (Foliation("z"), Foliation(0, 0, 1)),
        (Foliation((0, 0, 1)), Foliation(0, 0, 1)),
        (Foliation("N30E,40NW"), Foliation(300, 40)),
        (Foliation(F), F),
    ],
)
def test_vector_forms(made, expected):
    assert _close(made, expected)


@pytest.mark.parametrize(
    ("made", "expected"),
    [
        (Pair(), Pair(Vector3(0, 0, 1), Vector3(1, 0, 0))),
        (Pair(P), P),
        (Pair((140, 30, 110, 26)), P),
        (Pair(np.asarray(P)), P),
        (Pair(Foliation(140, 30), Lineation(110, 26)), P),
        (Fault(), Fault(Vector3(0, 0, 1), Vector3(1, 0, 0))),
        (Fault((140, 30, 110, 26, -1)), Fault(140, 30, 110, 26, -1)),
        (Fault(np.array([140, 30, 110, 26, 1])), Fault(140, 30, 110, 26, 1)),
        (Fault(P), Fault(P.fvec, P.lvec)),
        (Fault(P, -1), Fault(140, 30, 110, 26, -1)),
        (Fault(P, "n"), Fault(140, 30, 110, 26, 1)),
        (Fault(P, "r"), Fault(140, 30, 110, 26, "r")),
        (Fault(Foliation(140, 30), Lineation(110, 26), -1), Fault(P, -1)),
        (Fault(140, 30, -90), Fault(Foliation(140, 30), Foliation(140, 30).rake(-90))),
        (Cone(), Cone(Vector3(0, 0, 1), Vector3(1, 0, 0))),
        (Cone(Cone(L, F)), Cone(L, F)),
        (Cone((120, 30, 200, 10, 180)), Cone(120, 30, 200, 10, 180)),
        (Cone(L, Lineation(200, 10)), Cone(120, 30, 200, 10)),
        (Cone(L, 20), Cone(L, Vector3(120, 50))),
        (Cone(L, Lineation(200, 10), 180), Cone(120, 30, 200, 10, 180)),
        (
            Cone(np.r_[np.asarray(L), np.asarray(F), 90]),
            Cone(Vector3(L), Vector3(F), 90),
        ),
        (Arc(), Arc(Lineation(0, 0), Lineation(90, 0))),
        (Arc(Arc(L, F)), Arc(L, F)),
        (Arc((1, 0, 0, 0, 1, 0)), Arc(Vector3(1, 0, 0), Vector3(0, 1, 0))),
        (Arc(120, 30, 200, 10), Arc(L, Lineation(200, 10))),
    ],
)
def test_feature_forms(made, expected):
    assert _close(made, expected)


def test_fault_senses():
    assert Fault(140, 30, 110, 26, -1).sense == -1
    assert Fault(140, 30, 110, 26, 1).sense == 1
    assert Fault(140, 30, 110, 26, "r").sense == -1
    assert Fault(140, 30, 110, 26, "n").sense == 1
    assert Fault((140, 30, 110, 26, -1)).sense == -1


def test_arc_copy_keeps_options():
    a = Arc(Arc(L, F, curvature=0.5, positive=False, short=False))
    assert (a.curvature, a.positive, a.short) == (0.5, False, False)
    a = Arc(Arc(L, F, curvature=0.5), curvature=0.2)
    assert a.curvature == 0.2


@pytest.mark.parametrize(
    "make",
    [
        lambda: Vector2("z"),
        lambda: Vector2(1, 2, 3),
        lambda: Vector3("q"),
        lambda: Vector3(object()),
        lambda: Vector3(1, 2, 3, 4),
        lambda: Lineation("zz"),
        lambda: Foliation("bad"),
        lambda: Foliation(1, 2, 3, 4),
        lambda: Pair(1, 2),
        lambda: Pair(1, 2, 3),
        lambda: Fault(120, 30),
        lambda: Fault(1, 2, 3, 4),
        lambda: Cone(L, "x"),
        lambda: Cone(1, 2, 3, 4, 5, 6),
        lambda: Arc(1, 2, 3),
    ],
)
def test_invalid_forms_raise_type_error(make):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        with pytest.raises(TypeError):
            make()


@pytest.mark.parametrize(
    "make",
    [
        lambda **kw: Vector2(**kw),
        lambda **kw: Vector3(**kw),
        lambda **kw: Foliation(**kw),
        lambda **kw: Pair(**kw),
        lambda **kw: Fault(**kw),
        lambda **kw: Cone(**kw),
        lambda **kw: Arc(**kw),
    ],
)
def test_attrs(make):
    assert make(name="a")._attrs == {"name": "a"}
    assert make()._attrs == {}
    with pytest.raises(TypeError, match="serializable"):
        make(bad=object())
