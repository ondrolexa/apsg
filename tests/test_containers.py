import math

import numpy as np
import pytest

from apsg import (
    G,
    arcset,
    coneset,
    dir2set,
    ellipsoidset,
    faultset,
    folset,
    linset,
    pairset,
    stress2set,
    stressset,
    vec2set,
    vecset,
)
from apsg.feature._container import (
    ArcSet,
    ConeSet,
    Direction2Set,
    EllipsoidSet,
    FaultSet,
    FoliationSet,
    LineationSet,
    OrientationTensor3Set,
    PairSet,
    Stress2Set,
    Stress3Set,
    Vector2Set,
    Vector3Set,
)
from apsg.feature._geodata import (
    Arc,
    Cone,
    Direction,
    Fault,
    Foliation,
    Lineation,
    Pair,
)
from apsg.feature._statistics import jelinek_statistics
from apsg.feature._tensor2 import Stress2
from apsg.feature._tensor3 import Ellipsoid, OrientationTensor3, Stress3
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

    def test_csd(self):
        np.random.seed(42)
        v = Vector2Set.random_vonmises(100, position=0, kappa=20)
        assert isinstance(v.csd(), float)

    def test_uniformity_test_uniform(self):
        np.random.seed(42)
        v = Vector2Set.random(200)
        s = v.uniformity_test()
        assert isinstance(s, dict)
        assert "R" in s
        assert "statistic" in s
        assert "p_value" in s
        assert s["p_value"] > 0.05
        assert s["uniform"] is True

    def test_uniformity_test_clustered(self):
        np.random.seed(42)
        v = Vector2Set.random_vonmises(100, position=0, kappa=20)
        s = v.uniformity_test()
        assert s["p_value"] < 0.05
        assert s["uniform"] is False

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

    def test_R_order_independent(self):
        np.random.seed(1)
        angles = np.random.uniform(0, 180, 20)
        d = Direction2Set([Direction(a) for a in angles])
        shuffled = Direction2Set([Direction(a) for a in angles[::-1]])
        assert np.allclose(np.array(d.R()), np.array(shuffled.R()))

    def test_uniformity_test_uniform(self):
        np.random.seed(42)
        d = Direction2Set([Direction(a) for a in np.random.uniform(0, 180, 200)])
        s = d.uniformity_test()
        assert isinstance(s, dict)
        assert s["p_value"] > 0.05
        assert s["uniform"] is True

    def test_uniformity_test_clustered_antipodal_mix(self):
        # A single tight axial cluster around the 10/190 axis, but recorded with
        # both antipodal representations mixed together (as real field
        # measurements of an axial feature often are). A naive Rayleigh test on
        # the raw (undoubled) vectors would report this as uniform (R ~ 0), since
        # the two antipodal halves cancel in the vector sum.
        np.random.seed(42)
        angles = np.concatenate(
            [np.random.normal(10, 3, 50), np.random.normal(190, 3, 50)]
        )
        d = Direction2Set([Direction(a) for a in angles])
        s = d.uniformity_test()
        assert s["p_value"] < 0.05
        assert s["uniform"] is False


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

    def test_csd(self):
        np.random.seed(42)
        v = Vector3Set.random_fisher(100, position=Vector3(0, 0, 1), kappa=20)
        assert isinstance(v.csd(), float)

    def test_uniformity_test_uniform(self):
        np.random.seed(42)
        v = Vector3Set.gss(200)
        s = v.uniformity_test()
        assert isinstance(s, dict)
        assert "R" in s
        assert "statistic" in s
        assert "p_value" in s
        assert s["p_value"] > 0.05
        assert s["uniform"] is True

    def test_uniformity_test_clustered(self):
        np.random.seed(42)
        v = Vector3Set.random_fisher(100, position=Vector3(0, 0, 1), kappa=20)
        s = v.uniformity_test()
        assert s["p_value"] < 0.05
        assert s["uniform"] is False

    def test_watson_statistics(self):
        np.random.seed(42)
        v = Vector3Set.random_fisher(100, position=Vector3(0, 0, 1), kappa=20)
        s = v.watson_statistics()
        assert isinstance(s, dict)
        assert "mu" in s
        assert "k" in s
        assert "alpha" in s

    def test_bingham_statistics(self):
        np.random.seed(42)
        v = Vector3Set.random_fisher(100, position=Vector3(0, 0, 1), kappa=20)
        s = v.bingham_statistics()
        assert isinstance(s, dict)
        assert "mu" in s
        assert "axes" in s
        assert "gamma" in s
        assert s["n"] == 100
        assert s["level"] == 0.95
        assert s["which"] == 0
        assert all(0 <= g <= 90 for g in s["gamma"])

    def test_bingham_statistics_shrinks_with_n(self):
        np.random.seed(42)
        small = Vector3Set.random_fisher(20, position=Vector3(0, 0, 1), kappa=20)
        large = Vector3Set.random_fisher(400, position=Vector3(0, 0, 1), kappa=20)
        assert max(large.bingham_statistics()["gamma"]) < max(
            small.bingham_statistics()["gamma"]
        )

    def test_bingham_statistics_which(self):
        np.random.seed(42)
        v = Vector3Set.random_fisher(100, position=Vector3(0, 0, 1), kappa=20)
        ot = v.ortensor()
        s = v.bingham_statistics(which=2)
        assert s["mu"] == ot.eigenvectors(2)

    def test_bingham_statistics_invalid_which(self):
        v = Vector3Set.random_fisher(50, position=Vector3(0, 0, 1), kappa=20)
        with pytest.raises(ValueError):
            v.bingham_statistics(which=3)

    def test_bingham_statistics_degenerate(self):
        # perfectly symmetric data around one axis -> tau2 == tau3
        v = Vector3Set(
            [
                Vector3(1, 0, 0),
                Vector3(-1, 0, 0),
                Vector3(0, 1, 0),
                Vector3(0, -1, 0),
                Vector3(0, 0, 1),
                Vector3(0, 0, -1),
            ]
        )
        s = v.bingham_statistics()
        assert 90.0 in [round(g, 6) for g in s["gamma"]]

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
        _stat, _pval, same = v1.similarity(v2)
        assert same

    def test_similarity_hotelling(self):
        v1 = Vector3Set([Vector3(1, 0, 0), Vector3(0, 1, 0), Vector3(0, 0, 1)] * 3)
        v2 = Vector3Set([Vector3(1, 0, 0), Vector3(0, 1, 0), Vector3(0, 0, 1)] * 3)
        _stat, _pval, same = v1.similarity(v2, method="hotelling")
        assert same

    def test_similarity_mmd(self):
        v1 = Vector3Set([Vector3(1, 0, 0), Vector3(0, 1, 0)])
        v2 = Vector3Set([Vector3(1, 0, 0), Vector3(0, 1, 0)])
        _stat, _pval, same = v1.similarity(v2, method="mmd", n_permutations=99)
        assert same

    def test_align(self):
        v1 = Vector3Set([Vector3(1, 0, 0), Vector3(0, 1, 0)])
        v2 = Vector3Set([Vector3(1, 0, 0), Vector3(0, 1, 0)])
        R = v1.align(v2)
        from apsg.feature._tensor3 import Rotation3

        assert isinstance(R, Rotation3)

    def test_from_array(self):
        v = Vector3Set.from_array([0, 90], [0, 0])
        assert len(v) == 2

    def test_from_xyz(self):
        v = Vector3Set.from_xyz([1, 0], [0, 1], [0, 0])
        assert len(v) == 2

    def test_random(self):
        np.random.seed(42)
        v = Vector3Set.random(2000)
        assert len(v) == 2000
        eigenvalues = v.ortensor().eigenvalues()
        for ev in eigenvalues:
            assert math.isclose(ev, 1 / 3, abs_tol=0.02)

    def test_random_normal(self):
        np.random.seed(42)
        v = Vector3Set.random_normal(10, position=Vector3(0, 0, 1), sigma=20)
        assert len(v) == 10

    def test_random_fisher(self):
        np.random.seed(42)
        v = Vector3Set.random_fisher(10, position=Vector3(0, 0, 1), kappa=20)
        assert len(v) == 10

    def test_random_kent(self):
        np.random.seed(42)
        p = Pair(150, 40, 150, 40)
        v = Vector3Set.random_kent(p, n=10, kappa=30)
        assert len(v) == 10

    def test_sfs(self):
        v = Vector3Set.sfs(100)
        assert len(v) == 100

    def test_gss(self):
        v = Vector3Set.gss(100)
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

    def test_R_order_independent(self):
        np.random.seed(42)
        v = LineationSet.random_fisher(20, position=Lineation(120, 40))
        shuffled = LineationSet(list(reversed(v)))
        assert np.allclose(np.array(v.R()), np.array(shuffled.R()))

    def test_from_array(self):
        v = LineationSet.from_array([110, 30], [26, 10])
        assert len(v) == 2

    def test_from_csv_name_defaults_to_filename(self, tmp_path):
        v = LineationSet.from_array([110, 30], [26, 10])
        path = tmp_path / "mele.csv"
        v.to_csv(str(path))
        loaded = LineationSet.from_csv(str(path))
        assert loaded.name == "mele.csv"

    def test_from_csv_name_kwarg(self, tmp_path):
        v = LineationSet.from_array([110, 30], [26, 10])
        path = tmp_path / "mele.csv"
        v.to_csv(str(path))
        loaded = LineationSet.from_csv(str(path), name="Mele")
        assert loaded.name == "Mele"
        assert len(loaded) == 2

    def test_random_fisher(self):
        np.random.seed(42)
        v = LineationSet.random_fisher(10, position=Lineation(120, 40))
        assert len(v) == 10

    def test_uniformity_test_uniform(self):
        np.random.seed(42)
        v = LineationSet.gss(200)
        s = v.uniformity_test()
        assert isinstance(s, dict)
        assert "statistic" in s
        assert "critical_value" in s
        assert s["uniform"] is True

    def test_uniformity_test_clustered_antipodal_mix(self):
        # A tight axial cluster around lin(120, 40), with half the vectors
        # flipped to their antipode (a no-op for axial equality, but exercises
        # that mixing both antipodal representations doesn't cancel the signal
        # the way it would for a plain Rayleigh test on the raw resultant).
        np.random.seed(42)
        v = LineationSet.random_fisher(100, position=Lineation(120, 40), kappa=50)
        flipped = LineationSet([-e if i % 2 else e for i, e in enumerate(v)])
        s = flipped.uniformity_test()
        assert s["uniform"] is False


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

    def test_R_order_independent(self):
        np.random.seed(1)
        data = [
            Foliation(a, i)
            for a, i in zip(np.random.uniform(0, 360, 20), np.random.uniform(0, 90, 20))
        ]
        f = FoliationSet(data)
        shuffled = FoliationSet(list(reversed(data)))
        assert np.allclose(np.array(f.R()), np.array(shuffled.R()))

    def test_uniformity_test_uniform(self):
        np.random.seed(42)
        f = FoliationSet.gss(200)
        s = f.uniformity_test()
        assert isinstance(s, dict)
        assert "statistic" in s
        assert "critical_value" in s
        assert s["uniform"] is True

    def test_uniformity_test_clustered_antipodal_mix(self):
        np.random.seed(42)
        f = FoliationSet.random_fisher(100, position=Foliation(250, 30), kappa=50)
        flipped = FoliationSet([-e if i % 2 else e for i, e in enumerate(f)])
        s = flipped.uniformity_test()
        assert s["uniform"] is False


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

    def test_from_csv_name_kwarg(self, tmp_path):
        p = PairSet([Pair(140, 30, 110, 26), Pair(200, 40, 180, 20)])
        path = tmp_path / "mele.csv"
        p.to_csv(str(path))
        loaded = PairSet.from_csv(str(path), name="Mele")
        assert loaded.name == "Mele"
        assert len(loaded) == 2
        loaded_default = PairSet.from_csv(str(path))
        assert loaded_default.name == "mele.csv"

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

    def test_rake_property(self):
        p = PairSet([Pair(140, 30, 110, 26)])
        r = p.rake
        assert isinstance(r, np.ndarray)
        assert len(r) == 1

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

    def test_from_csv_name_kwarg(self, tmp_path):
        f = FaultSet([Fault(140, 30, 110, 26, -1), Fault(200, 40, 180, 20, 1)])
        path = tmp_path / "mele.csv"
        f.to_csv(str(path))
        loaded = FaultSet.from_csv(str(path), name="Mele")
        assert loaded.name == "Mele"
        assert len(loaded) == 2
        loaded_default = FaultSet.from_csv(str(path))
        assert loaded_default.name == "mele.csv"

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


# ---------------------------------------------------------------------------
# ConeSet
# ---------------------------------------------------------------------------


class TestConeSet:
    def test_default(self):
        c = ConeSet([Cone(140, 30, 110, 26, 360)])
        assert len(c) == 1

    def test_type_assertion(self):
        with pytest.raises(TypeError):
            ConeSet([Pair(140, 30, 110, 26)])

    def test_revangle_property(self):
        c = ConeSet([Cone(140, 30, 110, 26, 360), Cone(90, 70, 45, 30, 115)])
        r = c.revangle
        assert isinstance(r, np.ndarray)
        assert len(r) == 2

    def test_apical_angle_property(self):
        c = ConeSet([Cone(140, 30, 110, 26, 360), Cone(90, 70, 45, 30, 115)])
        a = c.apical_angle
        assert isinstance(a, np.ndarray)
        assert len(a) == 2

    def test_lowercase_alias(self):
        assert coneset is ConeSet


# ---------------------------------------------------------------------------
# ArcSet
# ---------------------------------------------------------------------------


class TestArcSet:
    def test_default(self):
        a = ArcSet([Arc(Lineation(0, 0), Lineation(90, 0))])
        assert len(a) == 1

    def test_type_assertion(self):
        with pytest.raises(TypeError):
            ArcSet([Pair(140, 30, 110, 26)])

    def test_curvature_property(self):
        a = ArcSet(
            [
                Arc(Lineation(0, 0), Lineation(90, 0), curvature=0.2),
                Arc(Lineation(10, 0), Lineation(80, 0), curvature=0.8),
            ]
        )
        c = a.curvature
        assert isinstance(c, np.ndarray)
        assert len(c) == 2

    def test_positive_short_properties(self):
        a = ArcSet(
            [
                Arc(Lineation(0, 0), Lineation(90, 0), positive=True, short=True),
                Arc(Lineation(10, 0), Lineation(80, 0), positive=False, short=False),
            ]
        )
        assert isinstance(a.positive, np.ndarray)
        assert isinstance(a.short, np.ndarray)
        assert list(a.positive) == [True, False]
        assert list(a.short) == [True, False]

    def test_lowercase_alias(self):
        assert arcset is ArcSet

    def test_from_vectors_multiple_args(self):
        a = ArcSet.from_vectors(Lineation(0, 0), Lineation(45, 20), Lineation(90, 0))
        assert isinstance(a, ArcSet)
        assert len(a) == 2
        assert a[0].p1 == Lineation(0, 0)
        assert a[0].p2 == Lineation(45, 20)
        assert a[1].p1 == Lineation(45, 20)
        assert a[1].p2 == Lineation(90, 0)

    def test_from_vectors_single_set(self):
        v = linset([Lineation(0, 0), Lineation(45, 20), Lineation(90, 0)])
        a = ArcSet.from_vectors(v)
        assert len(a) == 2
        assert a[0].p1 == Lineation(0, 0)
        assert a[1].p2 == Lineation(90, 0)

    def test_from_vectors_forwards_arc_kwargs(self):
        a = ArcSet.from_vectors(
            Lineation(0, 0),
            Lineation(90, 0),
            curvature=0.4,
            positive=False,
            short=False,
        )
        assert len(a) == 1
        assert a[0].curvature == pytest.approx(0.4)
        assert a[0].positive is False
        assert a[0].short is False

    def test_from_vectors_too_few_points_raises(self):
        with pytest.raises(TypeError):
            ArcSet.from_vectors(Lineation(0, 0))
        with pytest.raises(TypeError):
            ArcSet.from_vectors(linset([Lineation(0, 0)]))
        with pytest.raises(TypeError):
            ArcSet.from_vectors()


# ---------------------------------------------------------------------------
# EllipsoidSet
# ---------------------------------------------------------------------------


class TestEllipsoidSet:
    def test_default(self):
        e = EllipsoidSet([Ellipsoid([[8, 0, 0], [0, 2, 0], [0, 0, 1]])])
        assert len(e) == 1

    def test_type_assertion(self):
        with pytest.raises(TypeError):
            EllipsoidSet([Pair(140, 30, 110, 26)])

    def test_kind_property(self):
        e = EllipsoidSet([Ellipsoid([[8, 0, 0], [0, 2, 0], [0, 0, 1]])])
        k = e.kind
        assert isinstance(k, np.ndarray)
        assert len(k) == 1

    def test_P_j_property(self):
        e = EllipsoidSet([Ellipsoid([[8, 0, 0], [0, 2, 0], [0, 0, 1]])])
        p = e.P_j
        assert isinstance(p, np.ndarray)
        assert len(p) == 1

    def test_T_property(self):
        e = EllipsoidSet([Ellipsoid([[8, 0, 0], [0, 2, 0], [0, 0, 1]])])
        t = e.T
        assert isinstance(t, np.ndarray)
        assert len(t) == 1

    def test_lowercase_alias(self):
        assert ellipsoidset is EllipsoidSet


# ---------------------------------------------------------------------------
# Stress3Set
# ---------------------------------------------------------------------------


class TestStress3Set:
    def test_default(self):
        s = Stress3Set([Stress3([[10, 2, -3], [2, 5, 1], [-3, 1, -2]])])
        assert len(s) == 1

    def test_type_assertion(self):
        with pytest.raises(TypeError):
            Stress3Set([Pair(140, 30, 110, 26)])

    def test_mean_stress_property(self):
        s = Stress3Set(
            [
                Stress3([[10, 2, -3], [2, 5, 1], [-3, 1, -2]]),
                Stress3([[8, 0, 0], [0, 5, 0], [0, 0, 1]]),
            ]
        )
        m = s.mean_stress
        assert isinstance(m, np.ndarray)
        assert len(m) == 2

    def test_I1_I2_I3_properties(self):
        s = Stress3Set(
            [
                Stress3([[10, 2, -3], [2, 5, 1], [-3, 1, -2]]),
                Stress3([[8, 0, 0], [0, 5, 0], [0, 0, 1]]),
            ]
        )
        assert isinstance(s.I1, np.ndarray)
        assert isinstance(s.I2, np.ndarray)
        assert isinstance(s.I3, np.ndarray)
        assert len(s.I1) == len(s.I2) == len(s.I3) == 2

    def test_shape_ratio_property(self):
        s = Stress3Set(
            [
                Stress3([[10, 2, -3], [2, 5, 1], [-3, 1, -2]]),
                Stress3([[8, 0, 0], [0, 5, 0], [0, 0, 1]]),
            ]
        )
        r = s.shape_ratio
        assert isinstance(r, np.ndarray)
        assert len(r) == 2

    def test_lowercase_alias(self):
        assert stressset is Stress3Set


# ---------------------------------------------------------------------------
# mean_tensor (Jelinek 1978) of EllipsoidSet and Stress3Set
# ---------------------------------------------------------------------------

# 8-specimen AMS example (k11, k22, k33, k12, k23, k13) from the jelinekstat package
_AMS = np.array(
    [
        [1.02327, 1.02946, 0.94727, -0.01495, -0.03599, -0.05574],
        [1.02315, 1.01803, 0.95882, -0.00924, -0.02058, -0.03151],
        [1.02801, 1.03572, 0.93627, -0.03029, -0.03491, -0.06088],
        [1.02775, 1.00633, 0.96591, -0.01635, -0.04148, -0.02006],
        [1.02143, 1.01775, 0.96082, -0.02798, -0.04727, -0.02384],
        [1.01823, 1.01203, 0.96975, -0.01126, -0.02833, -0.03649],
        [1.01486, 1.02067, 0.96446, -0.01046, -0.01913, -0.03864],
        [1.04596, 1.01133, 0.94271, -0.01660, -0.04711, -0.03636],
    ]
)


def _ams_matrices():
    a, b, c, d, e, f = _AMS.T
    return np.array([[a, d, f], [d, b, e], [f, e, c]]).transpose(2, 0, 1)


def _noisy_matrices(rng, n, sigma=0.05, true=(1.3, 1.0, 0.7)):
    noise = rng.normal(scale=sigma, size=(n, 3, 3))
    return np.diag(true) + (noise + noise.transpose(0, 2, 1)) / 2


class TestMeanTensor:
    def test_ellipsoidset(self):
        es = EllipsoidSet([Ellipsoid(m) for m in _ams_matrices()])
        r = es.mean_tensor()
        assert set(r) == {
            "mean",
            "eigenvalues",
            "ellipses",
            "n",
            "level",
            "normalize",
            "anisoft",
        }
        assert isinstance(r["mean"], Ellipsoid)
        assert r["n"] == 8
        assert r["level"] == 0.95
        assert r["normalize"] is False
        assert r["anisoft"] is False
        assert len(r["ellipses"]) == 3
        for which, ell in enumerate(r["ellipses"]):
            assert ell["which"] == which
            assert isinstance(ell["mu"], Vector3)
            assert all(isinstance(a, Vector3) for a in ell["axes"])
            assert len(ell["gamma"]) == 2
            u, v = ell["axes"]
            assert abs(u.dot(ell["mu"])) < 1e-9
            assert abs(v.dot(ell["mu"])) < 1e-9
            assert abs(u.dot(v)) < 1e-9

    def test_mean_is_arithmetic_mean(self):
        mats = _ams_matrices()
        r = EllipsoidSet([Ellipsoid(m) for m in mats]).mean_tensor()
        np.testing.assert_allclose(np.asarray(r["mean"]), mats.mean(axis=0))
        np.testing.assert_allclose(
            r["eigenvalues"], np.linalg.eigvalsh(mats.mean(axis=0))[::-1]
        )

    def test_stress3set(self):
        ss = Stress3Set([Stress3(m) for m in _ams_matrices()])
        r = ss.mean_tensor()
        assert isinstance(r["mean"], Stress3)
        assert r["mean"] == Stress3(_ams_matrices().mean(axis=0))

    def test_orientationtensor3set_inherits(self):
        os = OrientationTensor3Set([OrientationTensor3(m) for m in _ams_matrices()])
        assert isinstance(os.mean_tensor()["mean"], OrientationTensor3)

    def test_too_few_tensors(self):
        es = EllipsoidSet([Ellipsoid(m) for m in _ams_matrices()[:2]])
        with pytest.raises(ValueError):
            es.mean_tensor()

    def test_normalize(self):
        mats = _ams_matrices() * np.arange(1, 9)[:, None, None]
        es = EllipsoidSet([Ellipsoid(m) for m in mats])
        r = es.mean_tensor(normalize=True)
        assert r["normalize"] is True
        assert np.trace(np.asarray(r["mean"])) == pytest.approx(3)
        # scale of individual tensors does not matter once normalized
        r0 = EllipsoidSet([Ellipsoid(m) for m in _ams_matrices()]).mean_tensor(
            normalize=True
        )
        np.testing.assert_allclose(np.asarray(r["mean"]), np.asarray(r0["mean"]))
        np.testing.assert_allclose(
            [e["gamma"] for e in r["ellipses"]], [e["gamma"] for e in r0["ellipses"]]
        )

    def test_normalize_zero_trace(self):
        ss = Stress3Set(
            [Stress3(np.diag(d)) for d in ([2, 0, -2], [3, -1, -2], [1, 1, -2])]
        )
        with pytest.raises(ValueError):
            ss.mean_tensor(normalize=True)
        assert isinstance(ss.mean_tensor()["mean"], Stress3)

    def test_level_monotonic(self):
        es = EllipsoidSet([Ellipsoid(m) for m in _ams_matrices()])
        lo = es.mean_tensor(level=0.8)["ellipses"]
        hi = es.mean_tensor(level=0.99)["ellipses"]
        for a, b in zip(lo, hi):
            assert all(x < y for x, y in zip(a["gamma"], b["gamma"]))

    def test_more_specimens_narrower(self):
        mats = _noisy_matrices(np.random.default_rng(5), 200)
        few = EllipsoidSet([Ellipsoid(m) for m in mats[:10]]).mean_tensor()
        many = EllipsoidSet([Ellipsoid(m) for m in mats]).mean_tensor()
        assert many["ellipses"][2]["gamma"][0] < few["ellipses"][2]["gamma"][0]

    def test_isotropic_is_degenerate(self):
        es = EllipsoidSet([Ellipsoid(np.eye(3)) for _ in range(3)])
        for ell in es.mean_tensor()["ellipses"]:
            assert ell["gamma"] == (90.0, 90.0)

    def test_identical_tensors_have_zero_gamma(self):
        es = EllipsoidSet([Ellipsoid(np.diag([3.0, 2.0, 1.0])) for _ in range(4)])
        for ell in es.mean_tensor()["ellipses"]:
            assert ell["gamma"] == (0.0, 0.0)

    def test_rotation_covariance(self):
        from scipy.spatial.transform import Rotation

        R = Rotation.random(random_state=11).as_matrix()
        mats = _ams_matrices()
        rotated = R @ mats @ R.T
        r0 = EllipsoidSet([Ellipsoid(m) for m in mats]).mean_tensor()
        r1 = EllipsoidSet([Ellipsoid(m) for m in rotated]).mean_tensor()
        np.testing.assert_allclose(r0["eigenvalues"], r1["eigenvalues"])
        for e0, e1 in zip(r0["ellipses"], r1["ellipses"]):
            np.testing.assert_allclose(e0["gamma"], e1["gamma"])
            # eigenvector sign is arbitrary
            assert abs(np.dot(np.asarray(e1["mu"]), R @ np.asarray(e0["mu"]))) == (
                pytest.approx(1)
            )

    def test_ams_regression(self):
        es = EllipsoidSet([Ellipsoid(m) for m in _ams_matrices()])
        r = es.mean_tensor(normalize=True)
        # normalized mean tensor and its eigenvalues as published for jelinekstat
        np.testing.assert_allclose(
            r["eigenvalues"], [1.042394, 1.033976, 0.923631], atol=1e-6
        )
        np.testing.assert_allclose(
            [e["gamma"] for e in r["ellipses"]],
            [(42.094, 6.509), (42.129, 6.290), (8.984, 3.038)],
            atol=1e-3,
        )

    def test_anisoft_matches_jelinekstat(self):
        # jelinekstat additionally scales the covariance by ((n - 1) / n)**2;
        # with anisoft=True its published values are reproduced
        es = EllipsoidSet([Ellipsoid(m) for m in _ams_matrices()])
        r = es.mean_tensor(normalize=True, anisoft=True)
        assert r["anisoft"] is True
        np.testing.assert_allclose(
            np.radians([e["gamma"] for e in r["ellipses"]]),
            [
                (0.66888885, 0.09950548),
                (0.66949335, 0.09615434),
                (0.13745895, 0.04640122),
            ],
            atol=1e-6,
        )

    def test_anisoft_scales_tangent_of_semi_angles(self):
        mats = _noisy_matrices(np.random.default_rng(8), 10)
        for cls, kls in ((EllipsoidSet, Ellipsoid), (Stress3Set, Stress3)):
            ts = cls([kls(m) for m in mats])
            plain, aniso = ts.mean_tensor(), ts.mean_tensor(anisoft=True)
            assert plain["anisoft"] is False
            assert aniso["anisoft"] is True
            for p, a in zip(plain["ellipses"], aniso["ellipses"]):
                np.testing.assert_allclose(
                    np.tan(np.radians(a["gamma"])),
                    9 / 10 * np.tan(np.radians(p["gamma"])),
                )
        # the mean tensor and the axes do not depend on it
        np.testing.assert_allclose(np.asarray(plain["mean"]), np.asarray(aniso["mean"]))

    def test_anisoft_keeps_degenerate_ellipse(self):
        es = EllipsoidSet([Ellipsoid(np.eye(3)) for _ in range(3)])
        for ell in es.mean_tensor(anisoft=True)["ellipses"]:
            assert ell["gamma"] == (90.0, 90.0)

    @pytest.mark.parametrize("which", [0, 1, 2])
    def test_confidence_coverage(self, which):
        # the ellipse of the true principal axis has to cover it ~95 % of the time
        rng = np.random.default_rng(1234)
        trials, hits = 1000, 0
        x = np.eye(3)[which]
        for _ in range(trials):
            ell = jelinek_statistics(_noisy_matrices(rng, 8))["ellipses"][which]
            mu = ell["mu"]
            d = (x if x @ mu > 0 else -x) / abs(x @ mu) - mu
            t0, t1 = np.tan(np.radians(ell["gamma"]))
            hits += (d @ ell["axes"][0] / t0) ** 2 + (d @ ell["axes"][1] / t1) ** 2 <= 1
        assert 0.92 <= hits / trials <= 0.98


# ---------------------------------------------------------------------------
# Stress2Set
# ---------------------------------------------------------------------------


class TestStress2Set:
    def test_default(self):
        s = Stress2Set([Stress2([[8, 0], [0, 1]])])
        assert len(s) == 1

    def test_type_assertion(self):
        with pytest.raises(TypeError):
            Stress2Set([Pair(140, 30, 110, 26)])

    def test_repr(self):
        s = Stress2Set([Stress2([[8, 0], [0, 1]])], name="test")
        assert repr(s) == "Sig2(1) test"

    def test_mean_stress_property(self):
        s = Stress2Set([Stress2([[8, 0], [0, 1]]), Stress2([[5, 0], [0, 2]])])
        m = s.mean_stress
        assert isinstance(m, np.ndarray)
        assert len(m) == 2

    def test_sigma1_sigma2_properties(self):
        s = Stress2Set([Stress2([[8, 0], [0, 1]]), Stress2([[5, 0], [0, 2]])])
        assert isinstance(s.sigma1, np.ndarray)
        assert isinstance(s.sigma2, np.ndarray)
        assert len(s.sigma1) == len(s.sigma2) == 2

    def test_I1_I2_I3_properties(self):
        s = Stress2Set([Stress2([[8, 0], [0, 1]]), Stress2([[5, 0], [0, 2]])])
        assert isinstance(s.I1, np.ndarray)
        assert isinstance(s.I2, np.ndarray)
        assert isinstance(s.I3, np.ndarray)
        assert len(s.I1) == len(s.I2) == len(s.I3) == 2

    def test_sigma1dir_sigma2dir_properties(self):
        s = Stress2Set([Stress2([[8, 0], [0, 1]]), Stress2([[5, 0], [0, 2]])])
        assert isinstance(s.sigma1dir, Vector2Set)
        assert isinstance(s.sigma2dir, Vector2Set)
        assert len(s.sigma1dir) == len(s.sigma2dir) == 2

    def test_lowercase_alias(self):
        assert stress2set is Stress2Set


# ---------------------------------------------------------------------------
# G() factory registry
# ---------------------------------------------------------------------------


class TestGFactoryStress:
    def test_stress3_registry(self):
        g = G([Stress3([[10, 2, -3], [2, 5, 1], [-3, 1, -2]])] * 2)
        assert isinstance(g, Stress3Set)
        assert len(g) == 2

    def test_stress2_registry(self):
        g = G([Stress2([[8, 0], [0, 1]])] * 2)
        assert isinstance(g, Stress2Set)
        assert len(g) == 2
