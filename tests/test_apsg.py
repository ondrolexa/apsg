import math

import pytest

from apsg import cone, dir2, fault, fol, lin, pair
from apsg.config import apsg_conf_context
from apsg.feature._geodata import Cone, Direction, Fault, Foliation, Lineation, Pair
from apsg.helpers._notation import (
    azi2bearing,
    bearing2azi,
    format_quadrant_linear,
    format_quadrant_planar,
    parse_quadrant_linear,
    parse_quadrant_planar,
)
from apsg.math._vector import Vector2, Vector3

# ---------------------------------------------------------------------------
# Quadrant notation helpers
# ---------------------------------------------------------------------------


class TestQuadrantNotation:
    @pytest.mark.parametrize(
        "azi,bearing",
        [
            (0, "N0E"),
            (10, "N10E"),
            (45, "N45E"),
            (90, "N90E"),
            (91, "S89E"),
            (135, "S45E"),
            (180, "S0E"),
            (181, "S1W"),
            (225, "S45W"),
            (270, "S90W"),
            (271, "N89W"),
            (315, "N45W"),
            (360, "N0E"),
        ],
    )
    def test_azi2bearing(self, azi, bearing):
        assert azi2bearing(azi) == bearing

    @pytest.mark.parametrize(
        "bearing,azi",
        [
            ("N0E", 0),
            ("N45E", 45),
            ("N90E", 90),
            ("S45E", 135),
            ("S0E", 180),
            ("S45W", 225),
            ("S90W", 270),
            ("N45W", 315),
        ],
    )
    def test_bearing2azi(self, bearing, azi):
        assert math.isclose(bearing2azi(bearing), azi)

    def test_bearing2azi_bad_string(self):
        with pytest.raises(ValueError):
            bearing2azi("X45E")

    def test_bearing2azi_bad_angle(self):
        with pytest.raises(ValueError):
            bearing2azi("N91E")

    @pytest.mark.parametrize(
        "s,strike,dip",
        [
            ("N30E,40NW", 210, 40),
            ("N80W,75SW", 100, 75),
        ],
    )
    def test_parse_quadrant_planar(self, s, strike, dip):
        got_strike, got_dip = parse_quadrant_planar(s)
        assert math.isclose(got_strike, strike)
        assert math.isclose(got_dip, dip)

    @pytest.mark.parametrize("s", ["N30E,40NW", "N80W,75SW", "N45W,20NE"])
    def test_format_quadrant_planar_roundtrip(self, s):
        # Round-trips exactly for strike bearings already given in the canonical
        # N-quadrant form; a strike is an axial line, so an S-quadrant input
        # bearing describing the same plane is canonicalized to its N-quadrant
        # equivalent on formatting (see test_format_quadrant_planar_canonicalizes_strike).
        assert format_quadrant_planar(*parse_quadrant_planar(s)) == s

    def test_format_quadrant_planar_canonicalizes_strike(self):
        # 'S45E,20NE' and 'N45W,20NE' describe the identical plane (strike is a
        # bidirectional line); formatting always prefers the N-quadrant bearing.
        assert (
            format_quadrant_planar(*parse_quadrant_planar("S45E,20NE")) == "N45W,20NE"
        )

    def test_parse_quadrant_planar_bad_string(self):
        with pytest.raises(ValueError):
            parse_quadrant_planar("not a measurement")

    @pytest.mark.parametrize("s", ["N45E,30", "S10W,10"])
    def test_format_quadrant_linear_roundtrip(self, s):
        assert format_quadrant_linear(*parse_quadrant_linear(s)) == s

    def test_parse_quadrant_linear_bad_string(self):
        with pytest.raises(ValueError):
            parse_quadrant_linear("not a measurement")


# ---------------------------------------------------------------------------
# Vector2
# ---------------------------------------------------------------------------


class TestVector2:
    def test_default(self):
        v = Vector2()
        assert v.x == 1
        assert v.y == 0

    def test_from_string(self):
        assert Vector2("x").x == 1
        assert Vector2("x").y == 0
        assert Vector2("y").x == 0
        assert Vector2("y").y == 1
        with pytest.raises(TypeError):
            Vector2("z")

    def test_from_angle(self):
        v = Vector2(90)
        assert math.isclose(v.x, 0, abs_tol=1e-10)
        assert math.isclose(v.y, 1, abs_tol=1e-10)

    def test_from_two_args(self):
        v = Vector2(3, 4)
        assert v.x == 3
        assert v.y == 4

    def test_from_tuple(self):
        v = Vector2((3, 4))
        assert v.x == 3
        assert v.y == 4

    def test_from_vector2(self):
        v = Vector2(Vector2(3, 4))
        assert v.x == 3
        assert v.y == 4

    def test_angle_zero(self):
        assert Vector2(0) == Vector2(1, 0)

    def test_repr(self):
        assert repr(Vector2(3, 4)).startswith("Vector2(")

    def test_magnitude(self, v2):
        assert math.isclose(v2.magnitude(), 5)

    def test_abs(self, v2):
        assert math.isclose(abs(v2), 5)

    def test_normalized(self, v2):
        n = v2.normalized()
        assert math.isclose(n.magnitude(), 1)

    def test_uv_alias(self, v2):
        assert v2.uv() == v2.normalized()

    def test_dot(self):
        assert math.isclose(Vector2(1, 0).dot(Vector2(0, 1)), 0)
        assert math.isclose(Vector2(1, 0).dot(Vector2(1, 0)), 1)
        assert math.isclose(Vector2(3, 4).dot(Vector2(2, 3)), 18)

    def test_cross(self):
        assert math.isclose(Vector2(1, 0).cross(Vector2(0, 1)), 1)
        assert math.isclose(Vector2(0, 1).cross(Vector2(1, 0)), -1)
        assert math.isclose(Vector2(1, 1).cross(Vector2(2, 2)), 0)

    def test_angle_between(self):
        a = Vector2(1, 0)
        b = Vector2(0, 1)
        assert math.isclose(a.angle(b), 90)

    def test_add(self):
        assert Vector2(1, 2) + Vector2(3, 4) == Vector2(4, 6)

    def test_sub(self):
        assert Vector2(3, 5) - Vector2(1, 2) == Vector2(2, 3)

    def test_mul(self):
        assert Vector2(1, 2) * 3 == Vector2(3, 6)

    def test_rmul(self):
        assert 3 * Vector2(1, 2) == Vector2(3, 6)

    def test_truediv(self):
        v = Vector2(4, 8) / 2
        assert v == Vector2(2, 4)

    def test_neg(self):
        assert -Vector2(1, 2) == Vector2(-1, -2)

    def test_pos(self):
        assert +Vector2(1, 2) == Vector2(1, 2)

    def test_eq(self):
        assert Vector2(1, 2) == Vector2(1, 2)
        assert Vector2(1, 2) != Vector2(1, 3)

    def test_matmul(self):
        r = Vector2(1, 2) @ Vector2(3, 4)
        assert math.isclose(r, 11)

    def test_is_unit(self):
        assert Vector2(1, 0).is_unit()
        assert not Vector2(3, 4).is_unit()

    def test_project(self):
        v = Vector2(3, 4).project(Vector2(1, 0))
        assert v == Vector2(3, 0)

    def test_reject(self):
        v = Vector2(3, 4).reject(Vector2(1, 0))
        assert v == Vector2(0, 4)

    def test_direction(self):
        assert math.isclose(Vector2(1, 0).direction, 0)
        assert math.isclose(Vector2(0, 1).direction, 90)

    def test_transform(self):
        F = [[1, 0], [0, -1]]
        u = Vector2(1, 1)
        assert u.transform(F) == Vector2(1, -1)

    def test_transform_norm(self):
        F = [[2, 0], [0, 2]]
        u = Vector2(1, 0)
        r = u.transform(F, norm=True)
        assert math.isclose(r.magnitude(), 1)

    def test_to_json(self):
        v = Vector2(3, 4)
        j = v.to_json()
        assert j["datatype"] == "Vector2"
        assert j["args"] == ((3, 4),)

    def test_to_json_with_attrs(self):
        v = Vector2(1, 2, label="test")
        j = v.to_json()
        assert j["kwargs"] == {"label": "test"}

    def test_random(self):
        v = Vector2.random()
        assert isinstance(v, Vector2)
        assert math.isclose(abs(v), 1, abs_tol=1e-10)

    def test_unit_x(self):
        assert Vector2.unit_x() == Vector2(1, 0)

    def test_unit_y(self):
        assert Vector2.unit_y() == Vector2(0, 1)

    def test_copy(self):
        v = Vector2(3, 4)
        c = v.copy()
        assert c == v
        assert c is not v

    def test_hash(self):
        assert hash(Vector2(1, 0)) == hash(Vector2(1, 0))

    def test_bad_args(self):
        with pytest.raises(TypeError):
            Vector2(1, 2, 3)


# ---------------------------------------------------------------------------
# Direction (2D axial vector)
# ---------------------------------------------------------------------------


class TestDirection:
    def test_default(self):
        d = Direction()
        assert d.direction == 0

    def test_from_angle(self):
        d = Direction(90)
        assert math.isclose(d.direction, 90)

    def test_from_two_args(self):
        d = Direction(1, 1)
        assert math.isclose(d.direction, 45)

    def test_from_string_x(self):
        assert Direction("x") == Direction(0)

    def test_from_string_y(self):
        assert Direction("y") == Direction(90)

    def test_repr(self):
        assert repr(Direction(45)) == "D:45"

    def test_axial_eq(self):
        assert Direction(45) == Direction(225)

    def test_axial_dot_abs(self):
        a = Direction(45)
        b = Direction(225)
        assert a.dot(b) >= 0

    def test_to_json(self):
        d = Direction(45)
        j = d.to_json()
        assert j["datatype"] == "Direction"
        assert len(j["args"]) == 1

    def test_aliases(self):
        assert dir2 is Direction


# ---------------------------------------------------------------------------
# Vector3
# ---------------------------------------------------------------------------


class TestVector3:
    def test_default(self):
        v = Vector3()
        assert v.x == 1
        assert v.y == 0
        assert v.z == 0

    def test_from_string_x(self):
        assert Vector3("x") == Vector3(1, 0, 0)

    def test_from_string_y(self):
        assert Vector3("y") == Vector3(0, 1, 0)

    def test_from_string_z(self):
        assert Vector3("z") == Vector3(0, 0, 1)

    def test_from_string_bad(self):
        with pytest.raises(TypeError):
            Vector3("w")

    def test_from_three_args(self):
        v = Vector3(1, 2, 3)
        assert v.x == 1
        assert v.y == 2
        assert v.z == 3

    def test_from_two_args_geo(self):
        v = Vector3(0, 90)
        assert math.isclose(v.x, 0, abs_tol=1e-10)
        assert math.isclose(v.y, 0, abs_tol=1e-10)
        assert math.isclose(v.z, 1, abs_tol=1e-10)

    def test_from_tuple(self):
        v = Vector3((1, 2, 3))
        assert v == Vector3(1, 2, 3)

    def test_from_vector3(self):
        v = Vector3(Vector3(1, 2, 3))
        assert v == Vector3(1, 2, 3)

    def test_repr(self):
        assert repr(Vector3(1, 2, 3)).startswith("Vector3(")

    def test_magnitude(self):
        assert math.isclose(Vector3(1, 2, 3).magnitude(), math.sqrt(14))

    def test_normalized(self):
        n = Vector3(3, 0, 0).normalized()
        assert n == Vector3(1, 0, 0)

    def test_dot(self):
        assert math.isclose(Vector3(1, 0, 0).dot(Vector3(0, 1, 0)), 0)
        assert math.isclose(Vector3(1, 2, 3).dot(Vector3(4, 5, 6)), 32)

    def test_cross(self):
        assert Vector3(1, 0, 0).cross(Vector3(0, 1, 0)) == Vector3(0, 0, 1)
        assert Vector3(0, 1, 0).cross(Vector3(1, 0, 0)) == Vector3(0, 0, -1)

    def test_cross_pow_operator(self):
        r = Vector3(1, 0, 0) ** Vector3(0, 1, 0)
        assert r == Vector3(0, 0, 1)

    def test_pow_scalar(self):
        r = Vector3(2, 3, 4) ** 2
        assert r == Vector3(4, 9, 16)

    def test_angle(self):
        a = Vector3(1, 0, 0)
        b = Vector3(0, 1, 0)
        assert math.isclose(a.angle(b), 90)
        assert math.isclose(a.angle(a), 0)

    def test_add(self):
        assert Vector3(1, 2, 3) + Vector3(4, 5, 6) == Vector3(5, 7, 9)

    def test_sub(self):
        assert Vector3(5, 7, 9) - Vector3(4, 5, 6) == Vector3(1, 2, 3)

    def test_mul(self):
        assert Vector3(1, 2, 3) * 2 == Vector3(2, 4, 6)

    def test_neg(self):
        assert -Vector3(1, -2, 3) == Vector3(-1, 2, -3)

    def test_eq(self):
        assert Vector3(1, 2, 3) == Vector3(1, 2, 3)
        assert Vector3(1, 2, 3) != Vector3(1, 2, 4)

    def test_matmul_vector(self):
        r = Vector3(1, 2, 3) @ Vector3(4, 5, 6)
        assert math.isclose(r, 32)

    def test_matmul_matrix(self):
        M = [[1, 0, 0], [0, 2, 0], [0, 0, 3]]
        r = Vector3(1, 2, 3) @ M
        assert isinstance(r, Vector3)

    def test_lower(self):
        assert Vector3(0, 0, -1).lower() == Vector3(0, 0, 1)
        assert Vector3(0, 0, 1).lower() == Vector3(0, 0, 1)

    def test_is_upper(self):
        assert Vector3(0, 0, -1).is_upper()
        assert not Vector3(0, 0, 1).is_upper()

    def test_geo(self):
        v = Vector3(0, 90)
        azi, inc = v.geo
        assert math.isclose(azi, 0)
        assert math.isclose(inc, 90)

    def test_rotate(self):
        v = Vector3(1, 0, 0)
        r = v.rotate(Vector3(0, 0, 1), 90)
        assert r == Vector3(0, 1, 0)

    def test_project(self):
        v = Vector3(3, 4, 0).project(Vector3(1, 0, 0))
        assert v == Vector3(3, 0, 0)

    def test_reject(self):
        v = Vector3(3, 4, 0).reject(Vector3(1, 0, 0))
        assert v == Vector3(0, 4, 0)

    def test_transform(self):
        F = [[1, 0, 0], [0, -1, 0], [0, 0, 1]]
        u = Vector3(1, 1, 1)
        assert u.transform(F) == Vector3(1, -1, 1)

    def test_transform_norm(self):
        F = [[2, 0, 0], [0, 2, 0], [0, 0, 2]]
        u = Vector3(1, 0, 0)
        r = u.transform(F, norm=True)
        assert math.isclose(r.magnitude(), 1)

    def test_is_unit(self):
        assert Vector3(1, 0, 0).is_unit()
        assert not Vector3(1, 2, 3).is_unit()

    def test_to_json(self):
        v = Vector3(1, 2, 3)
        j = v.to_json()
        assert j["datatype"] == "Vector3"
        assert j["args"] == ((1, 2, 3),)

    def test_random(self):
        v = Vector3.random()
        assert isinstance(v, Vector3)
        assert math.isclose(abs(v), 1, abs_tol=1e-10)

    def test_unit_x(self):
        assert Vector3.unit_x() == Vector3(1, 0, 0)

    def test_unit_y(self):
        assert Vector3.unit_y() == Vector3(0, 1, 0)

    def test_unit_z(self):
        assert Vector3.unit_z() == Vector3(0, 0, 1)

    def test_slerp(self):
        a = Vector3(1, 0, 0)
        b = Vector3(0, 1, 0)
        r = a.slerp(b, 0.5)
        assert math.isclose(abs(r), 1, abs_tol=1e-10)
        assert math.isclose(r.angle(a), 45)


# ---------------------------------------------------------------------------
# Lineation
# ---------------------------------------------------------------------------


class TestLineation:
    def test_default(self):
        v = Lineation()
        azi, inc = v.geo
        assert math.isclose(azi, 0)
        assert math.isclose(inc, 0)

    def test_from_two_args(self):
        v = Lineation(110, 26)
        azi, inc = v.geo
        assert math.isclose(azi, 110)
        assert math.isclose(inc, 26)

    def test_from_three_args(self):
        v = Lineation(0, 1, 0)
        assert v == Lineation(90, 0)

    def test_from_string_x(self):
        assert Lineation("x") == Lineation(0, 0)

    def test_from_string_y(self):
        assert Lineation("y") == Lineation(90, 0)

    def test_from_string_z(self):
        assert Lineation("z") == Lineation(0, 90)

    def test_from_quadrant_string(self):
        assert Lineation("N45E,30") == Lineation(45, 30)

    def test_from_string_bad(self):
        with pytest.raises(TypeError):
            Lineation("w")

    def test_repr(self):
        v = Lineation(110, 26)
        assert repr(v).startswith("L:")

    def test_repr_quadrant_notation(self):
        with apsg_conf_context(notation="quadrant"):
            assert repr(Lineation(45, 30)) == "L:N45E,30"

    def test_geo(self):
        v = Lineation(110, 26)
        azi, inc = v.geo
        assert math.isclose(azi, 110)
        assert math.isclose(inc, 26)

    def test_cross_returns_foliation(self):
        l1 = Lineation(0, 0)
        l2 = Lineation(90, 0)
        f = l1.cross(l2)
        assert isinstance(f, Foliation)

    def test_cross_result(self):
        l1 = Lineation(0, 0)
        l2 = Lineation(90, 0)
        f = l1.cross(l2)
        _, inc = f.geo
        assert math.isclose(inc, 0)

    def test_pow_operator(self):
        l1 = Lineation(0, 0)
        l2 = Lineation(90, 0)
        f = l1**l2
        assert isinstance(f, Foliation)

    def test_axial_eq_accepts_negative(self):
        l1 = Lineation(110, 26)
        l2 = Lineation(290, -26)
        assert l1 == l2

    def test_to_json(self):
        v = Lineation(110, 26)
        j = v.to_json()
        assert j["datatype"] == "Lineation"
        assert len(j["args"]) == 2

    def test_magnitude(self):
        v = Lineation(110, 26)
        assert math.isclose(v.magnitude(), 1, abs_tol=1e-10)

    def test_dot(self):
        l1 = Lineation(0, 0)
        l2 = Lineation(90, 0)
        assert math.isclose(l1.dot(l2), 0, abs_tol=1e-10)

    def test_aliases(self):
        assert lin is Lineation


# ---------------------------------------------------------------------------
# Foliation
# ---------------------------------------------------------------------------


class TestFoliation:
    def test_default(self):
        f = Foliation()
        azi, inc = f.geo
        assert math.isclose(azi, 180, abs_tol=1e-10)
        assert math.isclose(inc, 0, abs_tol=1e-10)

    def test_from_two_args_dd(self):
        f = Foliation(250, 30)
        azi, inc = f.geo
        assert math.isclose(azi, 250, abs_tol=1e-10)
        assert math.isclose(inc, 30, abs_tol=1e-10)

    def test_from_three_args(self):
        f = Foliation(0, 0, 1)
        azi, inc = f.geo
        assert math.isclose(azi, 180, abs_tol=1e-10)
        assert math.isclose(inc, 0, abs_tol=1e-10)

    def test_from_string_x(self):
        f = Foliation("x")
        assert isinstance(f, Foliation)

    def test_from_string_y(self):
        f = Foliation("z")
        assert isinstance(f, Foliation)

    def test_from_string_z(self):
        f = Foliation("y")
        assert isinstance(f, Foliation)

    def test_from_string_bad(self):
        with pytest.raises(TypeError):
            Foliation("w")

    def test_from_quadrant_string(self):
        with apsg_conf_context(notation="rhr"):
            assert Foliation("N30E,40NW") == Foliation(210, 40)

    def test_from_quadrant_string_bad(self):
        with pytest.raises(TypeError):
            Foliation("N30E,40")

    def test_repr(self):
        f = Foliation(250, 30)
        assert repr(f).startswith("S:")

    def test_repr_quadrant_notation(self):
        with apsg_conf_context(notation="quadrant"):
            assert repr(Foliation("N30E,40NW")) == "S:N30E,40NW"
            assert repr(Foliation("N80W,75SW")) == "S:N80W,75SW"

    def test_geo(self):
        f = Foliation(250, 30)
        azi, inc = f.geo
        assert math.isclose(azi, 250, abs_tol=1e-10)
        assert math.isclose(inc, 30, abs_tol=1e-10)

    def test_cross_returns_lineation(self):
        f1 = Foliation(0, 90)
        f2 = Foliation(90, 90)
        v = f1.cross(f2)
        assert isinstance(v, Lineation)

    def test_cross_result(self):
        f1 = Foliation(0, 90)
        f2 = Foliation(90, 90)
        v = f1.cross(f2)
        azi, inc = v.geo
        assert math.isclose(inc, 90, abs_tol=1e-10)

    def test_pow_operator(self):
        f1 = Foliation(0, 90)
        f2 = Foliation(90, 90)
        v = f1**f2
        assert isinstance(v, Lineation)

    def test_dipvec(self):
        f = Foliation(250, 30)
        v = f.dipvec()
        assert isinstance(v, Vector3)
        assert math.isclose(abs(v), 1, abs_tol=1e-10)

    def test_strike(self):
        f = Foliation(250, 30)
        s = f.strike()
        assert isinstance(s, Direction)
        strike = s.direction
        expected = (250 - 90) % 360
        assert math.isclose(strike, expected)

    def test_pole(self):
        f = Foliation(250, 30)
        p = f.pole()
        assert isinstance(p, Vector3)

    def test_rake(self):
        f = Foliation(250, 30)
        r = f.rake(45)
        assert isinstance(r, Vector3)
        assert math.isclose(abs(r), 1, abs_tol=1e-10)

    def test_transform(self):
        F = [[1, 0, 0], [0, -1, 0], [0, 0, 1]]
        f = Foliation(45, 20)
        r = f.transform(F)
        azi, inc = r.geo
        assert math.isclose(azi, 315)
        assert math.isclose(inc, 20)

    def test_to_json(self):
        f = Foliation(250, 30)
        j = f.to_json()
        assert j["datatype"] == "Foliation"
        assert len(j["args"]) == 2

    def test_axial_eq(self):
        f1 = Foliation(250, 90)
        f2 = Foliation(70, 90)
        assert f1 == f2

    def test_aliases(self):
        assert fol is Foliation


# ---------------------------------------------------------------------------
# Pair
# ---------------------------------------------------------------------------


class TestPair:
    def test_default(self):
        p = Pair()
        assert isinstance(p.fvec, Vector3)
        assert isinstance(p.lvec, Vector3)

    def test_from_four_args(self):
        p = Pair(140, 30, 110, 26)
        fazi, finc = p.fol.geo
        lazi, linc = p.lin.geo
        assert math.isclose(fazi, 140, abs_tol=1)
        assert math.isclose(lazi, 110, abs_tol=1)

    def test_from_fol_lin(self):
        f = Foliation(140, 30)
        v = Lineation(110, 26)
        p = Pair(f, v)
        assert isinstance(p, Pair)

    def test_from_tuple_six(self):
        p = Pair((1, 0, 0, 0, 1, 0))
        assert isinstance(p, Pair)

    def test_from_tuple_four(self):
        p = Pair((140, 30, 110, 26))
        assert isinstance(p, Pair)

    def test_from_pair(self):
        p1 = Pair(140, 30, 110, 26)
        p2 = Pair(p1)
        assert p1 == p2

    def test_repr(self):
        p = Pair(140, 30, 110, 26)
        assert repr(p).startswith("P:")

    def test_fol_property(self):
        p = Pair(140, 30, 110, 26)
        assert isinstance(p.fol, Foliation)

    def test_lin_property(self):
        p = Pair(140, 30, 110, 26)
        assert isinstance(p.lin, Lineation)

    def test_rax(self):
        p = Pair(140, 30, 110, 26)
        assert isinstance(p.rax, Vector3)

    def test_rake(self):
        p = Pair(140, 30, 110, 26)
        assert isinstance(p.rake, float)

    def test_misfit(self):
        p = Pair(140, 30, 110, 26)
        assert isinstance(p.misfit, float)

    def test_rotate(self):
        p = Pair(140, 30, 110, 26)
        axis = Lineation(40, 50)
        r = p.rotate(axis, 120)
        assert isinstance(r, Pair)

    def test_transform(self):
        F = [[2, 0, 0], [0, 1, 0], [0, 0, 1]]
        p = Pair(140, 30, 110, 26)
        r = p.transform(F)
        assert isinstance(r, Pair)

    def test_eq(self):
        p1 = Pair(140, 30, 110, 26)
        p2 = Pair(140, 30, 110, 26)
        assert p1 == p2

    def test_to_json(self):
        p = Pair(140, 30, 110, 26)
        j = p.to_json()
        assert j["datatype"] == "Pair"
        assert len(j["args"]) == 4

    def test_random(self):
        p = Pair.random()
        assert isinstance(p, Pair)

    def test_bad_args(self):
        with pytest.raises(TypeError):
            Pair(1, "bad")

    def test_misfit_warning(self):
        with pytest.warns(UserWarning, match="Misfit angle"):
            Pair(0, 0, 90, 80)

    def test_aliases(self):
        assert pair is Pair


# ---------------------------------------------------------------------------
# Fault
# ---------------------------------------------------------------------------


class TestFault:
    def test_default(self):
        f = Fault()
        assert isinstance(f.fvec, Vector3)
        assert isinstance(f.lvec, Vector3)

    def test_from_five_args(self):
        f = Fault(140, 30, 110, 26, -1)
        fazi, finc = f.fol.geo
        lazi, linc = f.lin.geo
        assert math.isclose(fazi, 140, abs_tol=1)
        assert math.isclose(lazi, 110, abs_tol=1)
        assert f.sense == -1

    def test_from_five_args_dip_slip(self):
        f = Fault(140, 30, 110, 26, 1)
        assert f.sense == 1

    def test_from_pair_with_sense(self):
        p = Pair(140, 30, 110, 26)
        f = Fault(p, -1)
        assert f.sense == -1

    def test_from_fol_lin_sense(self):
        f = Foliation(140, 30)
        v = Lineation(110, 26)
        fault = Fault(f, v, -1)
        assert fault.sense == -1

    def test_from_three_args_rake(self):
        f = Fault(140, 30, 45)
        assert isinstance(f, Fault)

    def test_from_five_tuple(self):
        f = Fault((140, 30, 110, 26, -1))
        assert f.sense == -1

    def test_sense_str(self):
        f = Fault(140, 30, 110, 26, -1)
        assert isinstance(f.sense_str, str)

    def test_calc_sense_string_s(self):
        f = Foliation(140, 30)
        v = Lineation(110, 26)
        res = Fault.calc_sense(f, v, "s")
        assert isinstance(res, int)

    def test_calc_sense_string_d(self):
        f = Foliation(140, 30)
        v = Lineation(110, 26)
        res = Fault.calc_sense(f, v, "d")
        assert isinstance(res, int)

    def test_calc_sense_string_n(self):
        f = Foliation(140, 30)
        v = Lineation(110, 26)
        res = Fault.calc_sense(f, v, "n")
        assert res == 1

    def test_calc_sense_string_r(self):
        f = Foliation(140, 30)
        v = Lineation(110, 26)
        res = Fault.calc_sense(f, v, "r")
        assert res == -1

    def test_p_vector(self):
        f = Fault(140, 30, 110, 26, -1)
        p = f.p_vector()
        assert isinstance(p, Vector3)

    def test_t_vector(self):
        f = Fault(140, 30, 110, 26, -1)
        t = f.t_vector()
        assert isinstance(t, Vector3)

    def test_p_property(self):
        f = Fault(140, 30, 110, 26, -1)
        assert isinstance(f.p, Lineation)

    def test_t_property(self):
        f = Fault(140, 30, 110, 26, -1)
        assert isinstance(f.t, Lineation)

    def test_m_property(self):
        f = Fault(140, 30, 110, 26, -1)
        assert isinstance(f.m, Foliation)

    def test_d_property(self):
        f = Fault(140, 30, 110, 26, -1)
        assert isinstance(f.d, Foliation)

    def test_repr(self):
        f = Fault(140, 30, 110, 26, -1)
        assert repr(f).startswith("F:")

    def test_eq(self):
        f1 = Fault(140, 30, 110, 26, -1)
        f2 = Fault(140, 30, 110, 26, -1)
        assert f1 == f2

    def test_to_json(self):
        f = Fault(140, 30, 110, 26, -1)
        j = f.to_json()
        assert j["datatype"] == "Fault"
        assert len(j["args"]) == 5

    def test_random(self):
        f = Fault.random()
        assert isinstance(f, Fault)

    def test_bad_args(self):
        with pytest.raises(TypeError):
            Fault(1, 2, 3, 4, 5, 6)

    def test_aliases(self):
        assert fault is Fault


# ---------------------------------------------------------------------------
# Cone
# ---------------------------------------------------------------------------


class TestCone:
    def test_default(self):
        c = Cone()
        assert isinstance(c.axis, Vector3)
        assert isinstance(c.secant, Vector3)
        assert c.revangle == 360

    def test_from_five_args(self):
        c = Cone(140, 30, 110, 26, 360)
        assert isinstance(c.axis, Vector3)
        assert isinstance(c.secant, Vector3)
        assert c.revangle == 360

    def test_from_five_args_short(self):
        c = Cone(140, 30, 110, 26)
        assert isinstance(c.axis, Vector3)
        assert c.revangle == 360

    def test_from_two_vectors(self):
        a = Lineation(140, 30)
        s = Lineation(110, 26)
        c = Cone(a, s, 360)
        assert isinstance(c, Cone)

    def test_from_two_args_apical(self):
        a = Lineation(140, 30)
        c = Cone(a, 20)
        assert isinstance(c, Cone)
        assert math.isclose(c.apical_angle(), 20)

    def test_from_cone(self):
        c1 = Cone(140, 30, 110, 26, 360)
        c2 = Cone(c1)
        assert c1 == c2

    def test_repr(self):
        c = Cone(140, 30, 110, 26, 360)
        assert repr(c).startswith("C:")

    def test_apical_angle(self):
        a = Lineation(0, 90)
        s = Lineation(0, 0)
        c = Cone(a, s, 360)
        assert math.isclose(c.apical_angle(), 90)

    def test_rotated_secant(self):
        c = Cone(0, 90, 0, 0, 180)
        rs = c.rotated_secant
        assert isinstance(rs, Vector3)

    def test_rotate(self):
        c = Cone(140, 30, 110, 26, 360)
        axis = Lineation(40, 50)
        r = c.rotate(axis, 120)
        assert isinstance(r, Cone)

    def test_eq(self):
        c1 = Cone(140, 30, 110, 26, 360)
        c2 = Cone(140, 30, 110, 26, 360)
        assert c1 == c2

    def test_to_json(self):
        c = Cone(140, 30, 110, 26, 360)
        j = c.to_json()
        assert j["datatype"] == "Cone"
        assert len(j["args"]) == 5

    def test_random(self):
        c = Cone.random()
        assert isinstance(c, Cone)

    def test_bad_args(self):
        with pytest.raises(TypeError):
            Cone(1, 2, 3, 4, 5, 6)

    def test_aliases(self):
        assert cone is Cone


# ---------------------------------------------------------------------------
# Cross-feature: JSON round-trip via feature_from_json
# ---------------------------------------------------------------------------


class TestJSONRoundtrip:
    def test_lineation(self):
        from apsg.feature import feature_from_json

        v = Lineation(110, 26)
        j = v.to_json()
        l2 = feature_from_json(j)
        assert v == l2

    def test_foliation(self):
        from apsg.feature import feature_from_json

        f = Foliation(250, 30)
        j = f.to_json()
        f2 = feature_from_json(j)
        assert f == f2

    def test_direction(self):
        from apsg.feature import feature_from_json

        d = Direction(45)
        j = d.to_json()
        d2 = feature_from_json(j)
        assert d == d2

    def test_pair(self):
        from apsg.feature import feature_from_json

        p = Pair(140, 30, 110, 26)
        j = p.to_json()
        p2 = feature_from_json(j)
        assert p == p2

    def test_fault(self):
        from apsg.feature import feature_from_json

        f = Fault(140, 30, 110, 26, -1)
        j = f.to_json()
        f2 = feature_from_json(j)
        assert f == f2

    def test_cone(self):
        from apsg.feature import feature_from_json

        c = Cone(140, 30, 110, 26, 360)
        j = c.to_json()
        c2 = feature_from_json(j)
        assert c == c2
