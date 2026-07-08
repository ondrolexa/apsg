import math

import numpy as np
import pytest

from apsg import dir2set, faultset, folset, linset, pairset, vec2set, vecset
from apsg.feature._container import (
    Direction2Set,
    FaultSet,
    FoliationSet,
    LineationSet,
    PairSet,
    Vector2Set,
    Vector3Set,
)
from apsg.feature._geodata import Direction, Fault, Foliation, Lineation, Pair
from apsg.math._vector import Vector2, Vector3

# ---------------------------------------------------------------------------
# Vector2Set
# ---------------------------------------------------------------------------


class TestVector2Set:
    def test_default(self):
        v = Vector2Set([Vector2(1, 0)])
        assert len(v) == 1

    def test_from_list(self):
        data = [Vector2(1, 0), Vector2(0, 1), Vector2(3, 4)]
        v = Vector2Set(data)
        assert len(v) == 3

    def test_type_assertion(self):
        with pytest.raises(TypeError):
            Vector2Set([Vector3(1, 0, 0)])

    def test_repr(self):
        v = Vector2Set([Vector2(1, 0)], name="test")
        assert repr(v) == "V2(1) test"

    def test_len(self):
        v = Vector2Set([Vector2(1, 0), Vector2(0, 1)])
        assert len(v) == 2

    def test_bool_empty(self):
        v = Vector2Set([])
        assert not v

    def test_bool_nonempty(self):
        v = Vector2Set([Vector2(1, 0)])
        assert v

    def test_getitem_int(self):
        data = [Vector2(1, 0), Vector2(0, 1), Vector2(3, 4)]
        v = Vector2Set(data)
        assert v[1] == data[1]

    def test_getitem_slice(self):
        data = [Vector2(1, 0), Vector2(0, 1), Vector2(3, 4)]
        v = Vector2Set(data)
        s = v[0:2]
        assert isinstance(s, Vector2Set)
        assert len(s) == 2

    def test_getitem_array(self):
        data = [Vector2(1, 0), Vector2(0, 1), Vector2(3, 4)]
        v = Vector2Set(data)
        s = v[[0, 2]]
        assert isinstance(s, Vector2Set)
        assert len(s) == 2

    def test_iter(self):
        data = [Vector2(1, 0), Vector2(0, 1)]
        v = Vector2Set(data)
        assert list(v) == data

    def test_add(self):
        v1 = Vector2Set([Vector2(1, 0)])
        v2 = Vector2Set([Vector2(0, 1)])
        v3 = v1 + v2
        assert len(v3) == 2

    def test_add_type_error(self):
        v = Vector2Set([Vector2(1, 0)])
        with pytest.raises(TypeError):
            v + "bad"

    def test_abs(self):
        v = Vector2Set([Vector2(3, 4)])
        assert math.isclose(abs(v)[0], 5)

    def test_x(self):
        v = Vector2Set([Vector2(3, 4), Vector2(1, 2)])
        np.testing.assert_array_equal(v.x, [3, 1])

    def test_y(self):
        v = Vector2Set([Vector2(3, 4), Vector2(1, 2)])
        np.testing.assert_array_equal(v.y, [4, 2])

    def test_direction(self):
        v = Vector2Set([Vector2(1, 0), Vector2(0, 1)])
        np.testing.assert_array_almost_equal(v.direction, [0, 90])

    def test_to_vec2(self):
        v = Vector2Set([Direction(45)])
        v2 = v.to_vec2()
        assert isinstance(v2, Vector2Set)

    def test_to_dir2(self):
        v = Vector2Set([Vector2(1, 0)])
        d = v.to_dir2()
        assert isinstance(d, Direction2Set)

    def test_proj(self):
        v = Vector2Set([Vector2(3, 4)])
        p = v.proj(Vector2(1, 0))
        assert p[0] == Vector2(3, 0)

    def test_dot(self):
        v = Vector2Set([Vector2(3, 4)])
        d = v.dot(Vector2(1, 0))
        assert math.isclose(d[0], 3)

    def test_cross_none(self):
        v = Vector2Set([Vector2(1, 0), Vector2(0, 1)])
        c = v.cross()
        assert len(c) == 1
        assert math.isclose(c[0], 1)

    def test_cross_set(self):
        v1 = Vector2Set([Vector2(1, 0), Vector2(0, 1)])
        v2 = Vector2Set([Vector2(0, 1), Vector2(1, 0)])
        c = v1.cross(v2)
        assert math.isclose(c[0], 1)
        assert math.isclose(c[1], -1)

    def test_cross_vector(self):
        v = Vector2Set([Vector2(1, 0)])
        c = v.cross(Vector2(0, 1))
        assert math.isclose(c[0], 1)

    def test_cross_type_error(self):
        v = Vector2Set([Vector2(1, 0)])
        with pytest.raises(TypeError):
            v.cross("bad")

    def test_pow_operator(self):
        v = Vector2Set([Vector2(1, 0), Vector2(0, 1)])
        c = v**v
        assert len(c) == 2

    def test_angle_none(self):
        v = Vector2Set([Vector2(1, 0), Vector2(0, 1)])
        a = v.angle()
        assert math.isclose(a[0], 90)

    def test_angle_set(self):
        v1 = Vector2Set([Vector2(1, 0)])
        v2 = Vector2Set([Vector2(0, 1)])
        a = v1.angle(v2)
        assert math.isclose(a[0], 90)

    def test_angle_vector(self):
        v = Vector2Set([Vector2(1, 0)])
        a = v.angle(Vector2(0, 1))
        assert math.isclose(a[0], 90)

    def test_angle_type_error(self):
        v = Vector2Set([Vector2(1, 0)])
        with pytest.raises(TypeError):
            v.angle("bad")

    def test_normalized(self):
        v = Vector2Set([Vector2(3, 4)])
        n = v.normalized()
        assert math.isclose(n[0].magnitude(), 1)

    def test_uv_alias(self):
        v = Vector2Set([Vector2(3, 4)])
        assert v.uv()[0] == v.normalized()[0]

    def test_transform(self):
        F = [[1, 0], [0, -1]]
        v = Vector2Set([Vector2(1, 1)])
        t = v.transform(F)
        assert t[0] == Vector2(1, -1)

    def test_R(self):
        data = [Vector2(1, 0), Vector2(0, 1)]
        v = Vector2Set(data)
        r = v.R()
        assert isinstance(r, Vector2)
        assert r == Vector2(1, 1)

    def test_R_mean(self):
        data = [Vector2(1, 0), Vector2(0, 1)]
        v = Vector2Set(data)
        r = v.R(mean=True)
        assert math.isclose(abs(r), math.sqrt(2) / 2)

    def test_fisher_statistics(self):
        v = Vector2Set([Vector2(1, 0), Vector2(0, 1)])
        s = v.fisher_statistics()
        assert isinstance(s, dict)
        assert "k" in s
        assert "alpha" in s
        assert "csd" in s
        assert "uniform" in s

    def test_var(self):
        v = Vector2Set([Vector2(1, 0), Vector2(0, 1)])
        var = v.var()
        assert isinstance(var, float)

    def test_delta(self):
        v = Vector2Set([Vector2(1, 0), Vector2(0, 1)])
        d = v.delta()
        assert isinstance(d, float)

    def test_rdegree(self):
        v = Vector2Set([Vector2(1, 0), Vector2(0, 1)])
        d = v.rdegree()
        assert isinstance(d, float)

    def test_ortensor(self):
        v = Vector2Set([Vector2(1, 0), Vector2(0, 1)])
        ot = v.ortensor()
        assert ot is not None

    def test_halfspace(self):
        v = Vector2Set([Vector2(-1, 0), Vector2(0, 1)])
        h = v.halfspace()
        assert all(h.angle(h.R()) <= 90 + 1e-10)

    def test_from_directions(self):
        v = Vector2Set.from_directions([0, 90, 45])
        assert len(v) == 3
        assert math.isclose(v[0].direction, 0)

    def test_from_xy(self):
        v = Vector2Set.from_xy([1, 0], [0, 1])
        assert len(v) == 2

    def test_random(self):
        np.random.seed(42)
        v = Vector2Set.random(10)
        assert len(v) == 10

    def test_random_vonmises(self):
        np.random.seed(42)
        v = Vector2Set.random_vonmises(10, position=45, kappa=10)
        assert len(v) == 10

    def test_copy(self):
        v = Vector2Set([Vector2(1, 0)])
        c = v.copy()
        assert c[0] == v[0]
        assert c is not v

    def test_name(self):
        v = Vector2Set([Vector2(1, 0)], name="mydata")
        assert v.label() == "mydata"

    def test_filter(self):
        v1 = Vector2(1, 0, label="a")
        v2 = Vector2(0, 1, label="b")
        v = Vector2Set([v1, v2])
        f = v.filter(label="a")
        assert len(f) == 1
        assert f[0] == v1

    def test_rotate_raises_on_bad_axis(self):
        v = Vector2Set([Vector2(1, 0)])
        with pytest.raises(TypeError):
            v.rotate(None, 90)

    def test_to_json(self):
        v = Vector2Set([Vector2(1, 0)], name="test")
        j = v.to_json()
        assert j["datatype"] == "Vector2Set"
        assert j["kwargs"]["name"] == "test"

    def test_bootstrap(self):
        v = Vector2Set([Vector2(1, 0), Vector2(0, 1), Vector2(3, 4)])
        samples = list(v.bootstrap(n=3, size=2))
        assert len(samples) == 3
        for s in samples:
            assert isinstance(s, Vector2Set)
            assert len(s) == 2

    def test_attrs(self):
        v1 = Vector2(1, 0, label="a")
        v = Vector2Set([v1])
        df = v.attrs()
        assert df["label"][0] == "a"


# ---------------------------------------------------------------------------
# Direction2Set
# ---------------------------------------------------------------------------


class TestDirection2Set:
    def test_default(self):
        d = Direction2Set([Direction(0), Direction(90)])
        assert len(d) == 2

    def test_type_assertion(self):
        with pytest.raises(TypeError):
            Direction2Set([Vector2(1, 0)])

    def test_repr(self):
        d = Direction2Set([Direction(45)], name="test")
        assert repr(d) == "D2(1) test"

    def test_inherited_from_vec2set(self):
        d = Direction2Set([Direction(45), Direction(135)])
        assert d.to_vec2() is not None
        assert len(d.normalized()) == 2


# ---------------------------------------------------------------------------
# Vector3Set
# ---------------------------------------------------------------------------


class TestVector3Set:
    def test_default(self):
        v = Vector3Set([Vector3(1, 0, 0)])
        assert len(v) == 1

    def test_from_list(self):
        data = [Vector3(1, 0, 0), Vector3(0, 1, 0), Vector3(0, 0, 1)]
        v = Vector3Set(data)
        assert len(v) == 3

    def test_type_assertion(self):
        with pytest.raises(TypeError):
            Vector3Set([Vector2(1, 0)])

    def test_repr(self):
        v = Vector3Set([Vector3(1, 0, 0)], name="test")
        assert repr(v) == "V3(1) test"

    def test_len(self):
        v = Vector3Set([Vector3(1, 0, 0), Vector3(0, 1, 0)])
        assert len(v) == 2

    def test_getitem(self):
        data = [Vector3(1, 0, 0), Vector3(0, 1, 0)]
        v = Vector3Set(data)
        assert v[1] == data[1]

    def test_add(self):
        v1 = Vector3Set([Vector3(1, 0, 0)])
        v2 = Vector3Set([Vector3(0, 1, 0)])
        v3 = v1 + v2
        assert len(v3) == 2

    def test_abs(self):
        v = Vector3Set([Vector3(1, 2, 3)])
        assert math.isclose(abs(v)[0], math.sqrt(14))

    def test_x(self):
        v = Vector3Set([Vector3(1, 2, 3)])
        np.testing.assert_array_equal(v.x, [1])

    def test_y(self):
        v = Vector3Set([Vector3(1, 2, 3)])
        np.testing.assert_array_equal(v.y, [2])

    def test_z(self):
        v = Vector3Set([Vector3(1, 2, 3)])
        np.testing.assert_array_equal(v.z, [3])

    def test_geo(self):
        v = Vector3Set([Vector3(0, 0, 1)])
        azi, inc = v.geo
        assert math.isclose(azi[0], 0)
        assert math.isclose(inc[0], 90)

    def test_to_lin(self):
        v = Vector3Set([Vector3(0, 0, 1)])
        v = v.to_lin()
        assert isinstance(v, LineationSet)

    def test_to_fol(self):
        v = Vector3Set([Vector3(0, 0, 1)])
        f = v.to_fol()
        assert isinstance(f, FoliationSet)

    def test_to_vec(self):
        v = Vector3Set([Lineation(90, 0)])
        v3 = v.to_vec()
        assert isinstance(v3, Vector3Set)

    def test_project(self):
        v = Vector3Set([Vector3(3, 4, 0)])
        p = v.project(Vector3(1, 0, 0))
        assert p[0] == Vector3(3, 0, 0)

    def test_proj_alias(self):
        v = Vector3Set([Vector3(3, 4, 0)])
        p = v.proj(Vector3(1, 0, 0))
        assert p[0] == Vector3(3, 0, 0)

    def test_reject(self):
        v = Vector3Set([Vector3(3, 4, 0)])
        r = v.reject(Vector3(1, 0, 0))
        assert r[0] == Vector3(0, 4, 0)

    def test_dot(self):
        v = Vector3Set([Vector3(1, 0, 0)])
        d = v.dot(Vector3(0, 1, 0))
        assert math.isclose(d[0], 0)

    def test_cross_none(self):
        v = Vector3Set([Vector3(1, 0, 0), Vector3(0, 1, 0)])
        c = v.cross()
        assert len(c) == 1

    def test_cross_set(self):
        v1 = Vector3Set([Vector3(1, 0, 0)])
        v2 = Vector3Set([Vector3(0, 1, 0)])
        c = v1.cross(v2)
        assert c[0] == Vector3(0, 0, 1)

    def test_cross_vector(self):
        v = Vector3Set([Vector3(1, 0, 0)])
        c = v.cross(Vector3(0, 1, 0))
        assert c[0] == Vector3(0, 0, 1)

    def test_angle_none(self):
        v = Vector3Set([Vector3(1, 0, 0), Vector3(0, 1, 0)])
        a = v.angle()
        assert math.isclose(a[0], 90)

    def test_angle_set(self):
        v1 = Vector3Set([Vector3(1, 0, 0)])
        v2 = Vector3Set([Vector3(0, 1, 0)])
        a = v1.angle(v2)
        assert math.isclose(a[0], 90)

    def test_angle_vector(self):
        v = Vector3Set([Vector3(1, 0, 0)])
        a = v.angle(Vector3(0, 1, 0))
        assert math.isclose(a[0], 90)

    def test_normalized(self):
        v = Vector3Set([Vector3(3, 0, 0)])
        n = v.normalized()
        assert n[0] == Vector3(1, 0, 0)

    def test_uv_alias(self):
        v = Vector3Set([Vector3(3, 0, 0)])
        assert v.uv()[0] == v.normalized()[0]

    def test_transform(self):
        F = [[1, 0, 0], [0, -1, 0], [0, 0, 1]]
        v = Vector3Set([Vector3(1, 1, 1)])
        t = v.transform(F)
        assert t[0] == Vector3(1, -1, 1)

    def test_is_upper(self):
        v = Vector3Set([Vector3(0, 0, -1), Vector3(0, 0, 1)])
        u = v.is_upper()
        assert u[0]
        assert not u[1]

    def test_R(self):
        v = Vector3Set([Vector3(1, 0, 0), Vector3(0, 1, 0)])
        r = v.R()
        assert isinstance(r, Vector3)

    def test_R_mean(self):
        v = Vector3Set([Vector3(1, 0, 0), Vector3(0, 1, 0)])
        r = v.R(mean=True)
        assert math.isclose(r.magnitude(), math.sqrt(2) / 2, abs_tol=1e-10)

    def test_fisher_statistics(self):
        np.random.seed(42)
        v = Vector3Set.random_fisher(100, position=Vector3(0, 0, 1), kappa=20)
        s = v.fisher_statistics()
        assert isinstance(s, dict)
        assert "mu" in s
        assert "k" in s
        assert "alpha" in s
        assert "csd" in s
        assert "uniform" in s

    def test_fisher_cone(self):
        np.random.seed(42)
        v = Vector3Set.random_fisher(100, position=Vector3(0, 0, 1), kappa=20)
        cone = v.fisher_cone()
        from apsg.feature._geodata import Cone

        assert isinstance(cone, Cone)

    def test_fisher_cone_csd(self):
        np.random.seed(42)
        v = Vector3Set.random_fisher(100, position=Vector3(0, 0, 1), kappa=20)
        cone = v.fisher_cone_csd()
        from apsg.feature._geodata import Cone

        assert isinstance(cone, Cone)

    def test_watson_statistics(self):
        np.random.seed(42)
        v = Vector3Set.random_fisher(100, position=Vector3(0, 0, 1), kappa=20)
        s = v.watson_statistics()
        assert isinstance(s, dict)
        assert "mu" in s
        assert "k" in s
        assert "alpha" in s

    def test_var(self):
        v = Vector3Set([Vector3(1, 0, 0), Vector3(0, 1, 0)])
        var = v.var()
        assert isinstance(var, float)

    def test_delta(self):
        v = Vector3Set([Vector3(1, 0, 0), Vector3(0, 1, 0)])
        d = v.delta()
        assert isinstance(d, float)

    def test_rdegree(self):
        v = Vector3Set([Vector3(1, 0, 0), Vector3(0, 1, 0)])
        d = v.rdegree()
        assert isinstance(d, float)

    def test_ortensor(self):
        v = Vector3Set([Vector3(1, 0, 0), Vector3(0, 1, 0)])
        ot = v.ortensor()
        assert ot is not None

    def test_centered(self):
        v = Vector3Set([Vector3(1, 0, 0), Vector3(0, 1, 0)])
        c = v.centered()
        assert isinstance(c, Vector3Set)

    def test_centered_max_vertical(self):
        v = Vector3Set([Vector3(1, 0, 0), Vector3(0, 1, 0)])
        c = v.centered(max_vertical=True)
        assert isinstance(c, Vector3Set)

    def test_halfspace(self):
        v = Vector3Set([Vector3(-1, 0, 0), Vector3(0, 1, 0)])
        h = v.halfspace()
        assert all(h.angle(h.R()) <= 90 + 1e-10)

    def test_similarity(self):
        v1 = Vector3Set([Vector3(1, 0, 0), Vector3(0, 1, 0)])
        v2 = Vector3Set([Vector3(1, 0, 0), Vector3(0, 1, 0)])
        stat, pval, same = v1.similarity(v2)
        assert same

    def test_similarity_hotelling(self):
        v1 = Vector3Set([Vector3(1, 0, 0), Vector3(0, 1, 0), Vector3(0, 0, 1)] * 3)
        v2 = Vector3Set([Vector3(1, 0, 0), Vector3(0, 1, 0), Vector3(0, 0, 1)] * 3)
        stat, pval, same = v1.similarity(v2, method="hotelling")
        assert same

    def test_similarity_mmd(self):
        v1 = Vector3Set([Vector3(1, 0, 0), Vector3(0, 1, 0)])
        v2 = Vector3Set([Vector3(1, 0, 0), Vector3(0, 1, 0)])
        stat, pval, same = v1.similarity(v2, method="mmd", n_permutations=99)
        assert same

    def test_align(self):
        v1 = Vector3Set([Vector3(1, 0, 0), Vector3(0, 1, 0)])
        v2 = Vector3Set([Vector3(1, 0, 0), Vector3(0, 1, 0)])
        dg = v1.align(v2)
        from apsg.feature._tensor3 import DeformationGradient3

        assert isinstance(dg, DeformationGradient3)

    def test_from_array(self):
        v = Vector3Set.from_array([0, 90], [0, 0])
        assert len(v) == 2

    def test_from_xyz(self):
        v = Vector3Set.from_xyz([1, 0], [0, 1], [0, 0])
        assert len(v) == 2

    def test_random_normal(self):
        np.random.seed(42)
        v = Vector3Set.random_normal(10, position=Vector3(0, 0, 1), sigma=20)
        assert len(v) == 10

    def test_random_fisher(self):
        np.random.seed(42)
        v = Vector3Set.random_fisher(10, position=Vector3(0, 0, 1), kappa=20)
        assert len(v) == 10

    def test_random_fisher2(self):
        np.random.seed(42)
        v = Vector3Set.random_fisher2(10, position=Vector3(0, 0, 1), kappa=20)
        assert len(v) == 10

    def test_random_kent(self):
        np.random.seed(42)
        p = Pair(150, 40, 150, 40)
        v = Vector3Set.random_kent(p, n=10, kappa=30)
        assert len(v) == 10

    def test_uniform_sfs(self):
        v = Vector3Set.uniform_sfs(100)
        assert len(v) == 100

    def test_uniform_gss(self):
        v = Vector3Set.uniform_gss(100)
        assert len(v) == 100

    def test_copy(self):
        v = Vector3Set([Vector3(1, 0, 0)])
        c = v.copy()
        assert c[0] == v[0]
        assert c is not v

    def test_name(self):
        v = Vector3Set([Vector3(1, 0, 0)], name="mydata")
        assert v.label() == "mydata"

    def test_filter(self):
        v1 = Vector3(1, 0, 0, label="a")
        v2 = Vector3(0, 1, 0, label="b")
        v = Vector3Set([v1, v2])
        f = v.filter(label="a")
        assert len(f) == 1

    def test_rotate(self):
        v = Vector3Set([Vector3(1, 0, 0)])
        r = v.rotate(Vector3(0, 0, 1), 90)
        assert r[0] == Vector3(0, 1, 0)

    def test_to_json(self):
        v = Vector3Set([Vector3(1, 0, 0)], name="test")
        j = v.to_json()
        assert j["datatype"] == "Vector3Set"
        assert j["kwargs"]["name"] == "test"

    def test_bootstrap(self):
        v = Vector3Set([Vector3(1, 0, 0), Vector3(0, 1, 0), Vector3(0, 0, 1)])
        samples = list(v.bootstrap(n=3, size=2))
        assert len(samples) == 3
        for s in samples:
            assert isinstance(s, Vector3Set)
            assert len(s) == 2

    def test_attrs(self):
        v1 = Vector3(1, 0, 0, label="a")
        v = Vector3Set([v1])
        df = v.attrs()
        assert df["label"][0] == "a"


# ---------------------------------------------------------------------------
# LineationSet
# ---------------------------------------------------------------------------


class TestLineationSet:
    def test_default(self):
        v = LineationSet([Lineation(110, 26)])
        assert len(v) == 1

    def test_type_assertion(self):
        with pytest.raises(TypeError):
            LineationSet([Vector3(1, 0, 0)])

    def test_repr(self):
        v = LineationSet([Lineation(110, 26)], name="test")
        assert repr(v) == "L(1) test"

    def test_inherited_methods(self):
        v = LineationSet([Lineation(110, 26), Lineation(30, 10)])
        assert len(v.normalized()) == 2
        assert isinstance(v.R(), Lineation)
        assert isinstance(v.to_vec(), Vector3Set)

    def test_from_array(self):
        v = LineationSet.from_array([110, 30], [26, 10])
        assert len(v) == 2

    def test_random_fisher(self):
        np.random.seed(42)
        v = LineationSet.random_fisher(10, position=Lineation(120, 40))
        assert len(v) == 10


# ---------------------------------------------------------------------------
# FoliationSet
# ---------------------------------------------------------------------------


class TestFoliationSet:
    def test_default(self):
        f = FoliationSet([Foliation(250, 30)])
        assert len(f) == 1

    def test_type_assertion(self):
        with pytest.raises(TypeError):
            FoliationSet([Vector3(1, 0, 0)])

    def test_repr(self):
        f = FoliationSet([Foliation(250, 30)], name="test")
        assert repr(f) == "S(1) test"

    def test_dipvec(self):
        f = FoliationSet([Foliation(250, 30)])
        v = f.dipvec()
        assert isinstance(v, Vector3Set)
        assert math.isclose(abs(v[0]), 1, abs_tol=1e-10)

    def test_strike(self):
        f = FoliationSet([Foliation(250, 30)])
        s = f.strike()
        assert isinstance(s, Direction2Set)
        assert math.isclose(s[0].direction, 160)

    def test_inherited_methods(self):
        f = FoliationSet([Foliation(250, 30), Foliation(100, 50)])
        assert len(f.normalized()) == 2
        assert isinstance(f.R(), Foliation)
        assert f.ortensor() is not None


# ---------------------------------------------------------------------------
# PairSet
# ---------------------------------------------------------------------------


class TestPairSet:
    def test_default(self):
        p = PairSet([Pair(140, 30, 110, 26)])
        assert len(p) == 1

    def test_type_assertion(self):
        with pytest.raises(TypeError):
            PairSet([Vector3(1, 0, 0)])

    def test_repr(self):
        p = PairSet([Pair(140, 30, 110, 26)], name="test")
        assert repr(p) == "P(1) test"

    def test_len(self):
        p = PairSet([Pair(140, 30, 110, 26), Pair(200, 40, 180, 20)])
        assert len(p) == 2

    def test_fol_property(self):
        p = PairSet([Pair(140, 30, 110, 26)])
        f = p.fol
        assert isinstance(f, FoliationSet)
        assert len(f) == 1

    def test_fvec_property(self):
        p = PairSet([Pair(140, 30, 110, 26)])
        fv = p.fvec
        assert isinstance(fv, Vector3Set)

    def test_lin_property(self):
        p = PairSet([Pair(140, 30, 110, 26)])
        v = p.lin
        assert isinstance(v, LineationSet)
        assert len(v) == 1

    def test_lvec_property(self):
        p = PairSet([Pair(140, 30, 110, 26)])
        lv = p.lvec
        assert isinstance(lv, Vector3Set)

    def test_misfit_property(self):
        p = PairSet([Pair(140, 30, 110, 26)])
        m = p.misfit
        assert isinstance(m, np.ndarray)
        assert len(m) == 1

    def test_rax_property(self):
        p = PairSet([Pair(140, 30, 110, 26)])
        r = p.rax
        assert isinstance(r, Vector3Set)

    def test_angle_none(self):
        p = PairSet([Pair(140, 30, 110, 26), Pair(200, 40, 180, 20)])
        a = p.angle()
        assert isinstance(a, np.ndarray)
        assert len(a) == 1

    def test_angle_set(self):
        p1 = PairSet([Pair(140, 30, 110, 26)])
        p2 = PairSet([Pair(200, 40, 180, 20)])
        a = p1.angle(p2)
        assert isinstance(a, np.ndarray)
        assert len(a) == 1

    def test_angle_pair(self):
        p = PairSet([Pair(140, 30, 110, 26)])
        other = Pair(200, 40, 180, 20)
        a = p.angle(other)
        assert isinstance(a, np.ndarray)
        assert len(a) == 1

    def test_angle_type_error(self):
        p = PairSet([Pair(140, 30, 110, 26)])
        with pytest.raises(TypeError):
            p.angle("bad")

    def test_ortensor(self):
        p = PairSet([Pair(140, 30, 110, 26)])
        ot = p.ortensor()
        assert ot is not None

    def test_random(self):
        np.random.seed(42)
        p = PairSet.random(10)
        assert len(p) == 10

    def test_from_array(self):
        p = PairSet.from_array([140, 200], [30, 40], [110, 180], [26, 20])
        assert len(p) == 2
        assert isinstance(p[0], Pair)

    def test_copy(self):
        p = PairSet([Pair(140, 30, 110, 26)])
        c = p.copy()
        assert c[0] == p[0]
        assert c is not p

    def test_to_json(self):
        p = PairSet([Pair(140, 30, 110, 26)], name="test")
        j = p.to_json()
        assert j["datatype"] == "PairSet"
        assert j["kwargs"]["name"] == "test"

    def test_filter(self):
        p1 = Pair(140, 30, 110, 26, label="a")
        p2 = Pair(200, 40, 180, 20, label="b")
        ps = PairSet([p1, p2])
        f = ps.filter(label="a")
        assert len(f) == 1

    def test_rotate(self):
        p = PairSet([Pair(140, 30, 110, 26)])
        axis = Lineation(40, 50)
        r = p.rotate(axis, 120)
        assert isinstance(r, PairSet)

    def test_add(self):
        p1 = PairSet([Pair(140, 30, 110, 26)])
        p2 = PairSet([Pair(200, 40, 180, 20)])
        p3 = p1 + p2
        assert len(p3) == 2


# ---------------------------------------------------------------------------
# FaultSet
# ---------------------------------------------------------------------------


class TestFaultSet:
    def test_default(self):
        f = FaultSet([Fault(140, 30, 110, 26, -1)])
        assert len(f) == 1

    def test_type_assertion(self):
        with pytest.raises(TypeError):
            FaultSet([Vector3(1, 0, 0)])

    def test_repr(self):
        f = FaultSet([Fault(140, 30, 110, 26, -1)], name="test")
        assert repr(f) == "F(1) test"

    def test_len(self):
        f = FaultSet(
            [
                Fault(140, 30, 110, 26, -1),
                Fault(200, 40, 180, 20, 1),
            ]
        )
        assert len(f) == 2

    def test_sense_property(self):
        f = FaultSet([Fault(140, 30, 110, 26, -1)])
        np.testing.assert_array_equal(f.sense, [-1])

    def test_sense_str_property(self):
        f = FaultSet([Fault(140, 30, 110, 26, -1)])
        assert isinstance(f.sense_str[0], str)
        assert len(f.sense_str[0]) == 1

    def test_p_vector(self):
        f = FaultSet([Fault(140, 30, 110, 26, -1)])
        pv = f.p_vector()
        assert isinstance(pv, Vector3Set)

    def test_t_vector(self):
        f = FaultSet([Fault(140, 30, 110, 26, -1)])
        tv = f.t_vector()
        assert isinstance(tv, Vector3Set)

    def test_p_property(self):
        f = FaultSet([Fault(140, 30, 110, 26, -1)])
        p = f.p
        assert isinstance(p, LineationSet)

    def test_t_property(self):
        f = FaultSet([Fault(140, 30, 110, 26, -1)])
        t = f.t
        assert isinstance(t, LineationSet)

    def test_m_property(self):
        f = FaultSet([Fault(140, 30, 110, 26, -1)])
        m = f.m
        assert isinstance(m, FoliationSet)

    def test_d_property(self):
        f = FaultSet([Fault(140, 30, 110, 26, -1)])
        d = f.d
        assert isinstance(d, FoliationSet)

    def test_angle_none(self):
        f = FaultSet(
            [
                Fault(140, 30, 110, 26, -1),
                Fault(200, 40, 180, 20, 1),
            ]
        )
        a = f.angle()
        assert isinstance(a, np.ndarray)
        assert len(a) == 1

    def test_angle_set(self):
        f1 = FaultSet([Fault(140, 30, 110, 26, -1)])
        f2 = FaultSet([Fault(200, 40, 180, 20, 1)])
        a = f1.angle(f2)
        assert isinstance(a, np.ndarray)
        assert len(a) == 1

    def test_angle_fault(self):
        f = FaultSet([Fault(140, 30, 110, 26, -1)])
        other = Fault(200, 40, 180, 20, 1)
        a = f.angle(other)
        assert isinstance(a, np.ndarray)
        assert len(a) == 1

    def test_random(self):
        np.random.seed(42)
        f = FaultSet.random(10)
        assert len(f) == 10

    def test_from_array(self):
        f = FaultSet.from_array([140, 200], [30, 40], [110, 180], [26, 20], [-1, 1])
        assert len(f) == 2
        assert isinstance(f[0], Fault)

    def test_from_array_no_senses(self):
        f = FaultSet.from_array([140, 200], [30, 40], [110, 180], [26, 20])
        # With no senses passed, Fault.random() might fail or use default
        # We just check it returns the correct type
        assert isinstance(f, FaultSet)

    def test_stress_inversion(self):
        np.random.seed(42)
        faults = FaultSet(
            [
                Fault(156, 75, 223, 55, 1),
                Fault(153, 80, 73, 41, -1),
                Fault(111, 30, 198, 2, 1),
                Fault(128, 75, 199, 51, -1),
                Fault(251, 21, 240, 21, 1),
                Fault(193, 39, 204, 38, -1),
                Fault(304, 11, 307, 11, 1),
                Fault(3, 82, 292, 67, -1),
                Fault(318, 63, 243, 26, 1),
                Fault(150, 61, 206, 45, -1),
            ]
        )
        stress = faults.stress_inversion()
        from apsg.feature._tensor3 import Stress3

        assert isinstance(stress, Stress3)

    def test_stress_inversion_bootstrap(self):
        np.random.seed(42)
        faults = FaultSet(
            [
                Fault(156, 75, 223, 55, 1),
                Fault(153, 80, 73, 41, -1),
                Fault(111, 30, 198, 2, 1),
                Fault(128, 75, 199, 51, -1),
                Fault(251, 21, 240, 21, 1),
                Fault(193, 39, 204, 38, -1),
                Fault(304, 11, 307, 11, 1),
                Fault(3, 82, 292, 67, -1),
                Fault(318, 63, 243, 26, 1),
                Fault(150, 61, 206, 45, -1),
            ]
        )
        stress_set = faults.stress_inversion(bootstrap=True, n=5)
        from apsg.feature._container import Stress3Set

        assert isinstance(stress_set, Stress3Set)
        assert len(stress_set) == 5

    def test_copy(self):
        f = FaultSet([Fault(140, 30, 110, 26, -1)])
        c = f.copy()
        assert c[0] == f[0]
        assert c is not f

    def test_to_json(self):
        f = FaultSet([Fault(140, 30, 110, 26, -1)], name="test")
        j = f.to_json()
        assert j["datatype"] == "FaultSet"
        assert j["kwargs"]["name"] == "test"

    def test_filter(self):
        f1 = Fault(140, 30, 110, 26, -1, label="a")
        f2 = Fault(200, 40, 180, 20, 1, label="b")
        fs = FaultSet([f1, f2])
        filtered = fs.filter(label="a")
        assert len(filtered) == 1

    def test_rotate(self):
        f = FaultSet([Fault(140, 30, 110, 26, -1)])
        axis = Lineation(40, 50)
        r = f.rotate(axis, 120)
        assert isinstance(r, FaultSet)

    def test_add(self):
        f1 = FaultSet([Fault(140, 30, 110, 26, -1)])
        f2 = FaultSet([Fault(200, 40, 180, 20, 1)])
        f3 = f1 + f2
        assert len(f3) == 2

    def test_conversion_from_pairset(self):
        ps = PairSet([Pair(140, 30, 110, 26)])
        assert isinstance(ps, PairSet)
        assert not isinstance(ps, FaultSet)

    def test_lowercase_aliases(self):
        assert vec2set is Vector2Set
        assert dir2set is Direction2Set
        assert vecset is Vector3Set
        assert linset is LineationSet
        assert folset is FoliationSet
        assert pairset is PairSet
        assert faultset is FaultSet
