# -*- coding: utf-8 -*-

"""
Unit tests for `apsg` core module.

Use this steps for unit test:

- Arrange all necessary preconditions and inputs.
- Act on the object or method under test.
- Assert that the expected results have occurred.


Proper unit tests should fail for exactly one reason
(that’s why you usually should be using one assert per unit test.)
"""


import pytest
import numpy as np


from apsg import Vec3, Fol, Lin, Fault, Pair, Group, DefGrad, Ortensor, settings


# ############################################################################
# Helpers
# ############################################################################


def is_hashable(obj):
    try:
        hash(obj)
        return True
    except:
        return False


class TestVector:

    @pytest.fixture
    def x(self):
        return Vec3([1, 0, 0])

    @pytest.fixture
    def y(self):
        return Vec3([0, 1, 0])

    @pytest.fixture
    def z(self):
        return Vec3([0, 0, 1])

    @pytest.mark.skip
    def test_that_vector_is_hashable(self):
        assert is_hashable(Vec3([1, 2, 3]))

    def test_that_vec3_string_gets_three_digits_when_vec2dd_settings_is_false(self):
        settings["vec2dd"] = False

        vec = Vec3([1, 2, 3])

        current = str(vec)
        expects = "V(1.000, 2.000, 3.000)"

        assert current == expects

    def test_that_vec3_string_gets_dip_and_dir_when_vec2dd_settings_is_true(self):
        settings["vec2dd"] = True

        vec = Vec3([1, 2, 3])

        current = str(vec)
        expects = "V:63/53"

        assert current == expects

        settings["vec2dd"] = False

    # ``==`` operator

    def test_that_equality_operator_is_reflexive(self):
        u = Vec3(1, 2, 3)

        assert u == u

    def test_that_equality_operator_is_symetric(self):
        u = Vec3([1, 2, 3])
        v = Vec3([1, 2, 3])

        assert u == v and v == u

    def test_that_equality_operator_is_transitive(self):
        u = Vec3([1, 2, 3])
        v = Vec3([1, 2, 3])
        w = Vec3([1, 2, 3])

        assert u == v and v == w and u == w

    def test_that_equality_operator_precision_limits(self):
        """
        This is not the best method how to test a floating point precision limits,
        but I will keep it here for a future work.
        """
        lhs = Vec3([1.00000000000000001] * 3)
        rhs = Vec3([1.00000000000000009] * 3)

        assert lhs == rhs

    def test_that_equality_operator_returns_false_for_none(self):
        lhs = Vec3([1, 0, 0])
        rhs = None

        current = lhs == rhs
        expects = False

        assert current == expects

    # ``!=`` operator

    def test_inequality_operator(self):
        lhs = Vec3([1, 2, 3])
        rhs = Vec3([3, 2, 1])

        assert lhs != rhs

    # ``hash`` method

    @pytest.mark.skip
    def test_that_hash_is_same_for_identical_vectors(self):
        lhs = Vec3([1, 2, 3])
        rhs = Vec3([1, 2, 3])

        assert hash(lhs) == hash(rhs)

    @pytest.mark.skip
    def test_that_hash_is_not_same_for_different_vectors(self):
        lhs = Vec3([1, 2, 3])
        rhs = Vec3([3, 2, 1])

        assert not hash(lhs) == hash(rhs)

    # ``upper`` property

    def test_that_vector_is_upper(self):
        vec = Vec3([0, 0, -1])

        assert vec.upper

    def test_that_vector_is_not_upper(self):
        vec = Vec3([0, 0, 1])

        assert not vec.upper

    # ``flip`` property

    def test_that_vector_is_flipped(self):
        current = Vec3([0, 0, 1]).flip
        expects = Vec3([0, 0, -1])

        assert current == expects

    # ``abs`` operator

    def test_absolute_value(self):
        current = abs(Vec3([1, 2, 3]))
        expects = 3.7416573867739413

        assert current == expects

    # ``uv`` property

    def test_that_vector_is_normalized(self):
        current = Vec3([1, 2, 3]).uv
        expects = Vec3([0.26726124191242442, 0.5345224838248488, 0.8017837257372732])

        assert current == expects

    # ``dd`` property

    def test_dd_property(self):
        v = Vec3([1, 0, 0])

        current = v.dd
        expects = (0.0, 0.0)

        assert current == expects

    # ``aslin`` property

    def test_aslin_conversion(self):
        assert str(Vec3([1, 1, 1]).aslin) == str(Lin(45, 35))     # `Vec` to `Lin`
        assert str(Vec3(Lin(110, 37)).aslin) == str(Lin(110, 37)) # `Lin` to `Vec` to `Lin`

    # ``asfol`` property

    def test_asfol_conversion(self):
        assert str(Vec3([1, 1, 1]).asfol) == str(Fol(225, 55))    # `Vec` to `Fol`
        assert str(Vec3(Fol(213, 52)).asfol) == str(Fol(213, 52)) # `Fol` to `Vec` to `Fol`

    # ``asvec`` property

    @pytest.mark.skip
    def test_asvec_conversion(self):
        assert str(Vec3(1, 1, 1).asvec3) == str(Vec3([1, 1, 1]).uv)

    # ``angle`` property

    def test_that_angle_between_vectors_is_0_degrees_when_they_are_collinear(self):
        lhs = Vec3([1, 0, 0])
        rhs = Vec3([2, 0, 0])

        current = lhs.angle(rhs)
        expects = 0

        assert current == expects

    def test_that_angle_between_vectors_is_90_degrees_when_they_are_perpendicular(self):
        lhs = Vec3([1, 0, 0])
        rhs = Vec3([0, 1, 1])

        current = lhs.angle(rhs)
        expects = 90 # degrees

        assert current == expects

    def test_that_angle_between_vectors_is_180_degrees_when_they_are_opposite(self):
        lhs = Vec3([1, 0, 0])
        rhs = Vec3([-1, 0, 0])

        current = lhs.angle(rhs)
        expects = 180 # degrees

        assert current == expects

    # ``cross`` method

    def test_that_vector_product_is_anticommutative(self):
        lhs = Vec3(1, 0, 0)
        rhs = Vec3(0, 1, 0)

        assert lhs.cross(rhs) == (-rhs).cross(lhs)

    def test_that_vector_product_is_distributive_over_addition(self):
        a = Vec3(1, 0, 0)
        b = Vec3(0, 1, 0)
        c = Vec3(0, 0, 1)

        assert a.cross(b + c) == a.cross(b) + a.cross(c)

    def test_that_vector_product_is_zero_vector_when_they_are_collinear(self):
        lhs = Vec3([1, 0, 0])
        rhs = Vec3([2, 0, 0])

        current = lhs.cross(rhs)
        expects = Vec3([0, 0, 0])

        assert current == expects

    def test_that_vector_product_is_zero_vector_when_they_are_opposite(self):

        lhs = Vec3([1, 0, 0])
        rhs = Vec3([-1, 0, 0])

        current = lhs.cross(rhs)
        expects = Vec3([0, 0, 0])

        assert current == expects

    def test_vector_product_of_orthonormal_vectors(self):
        e1 = Vec3([1, 0, 0])
        e2 = Vec3([0, 1, 0])

        current = e1.cross(e2)
        expects = Vec3([0, 0, 1])

        assert current == expects

    # ``dot`` method

    def test_scalar_product_of_same_vectors(self):
        i = Vec3(1, 0, 0)
        j = Vec3(1, 0, 0)

        assert i.dot(j) == abs(i) == abs(i)

    def test_scalar_product_of_orthonornal_vectors(self):
        i = Vec3(1, 0, 0)
        j = Vec3(0, 1, 0)

        assert i.dot(j) == 0

    # ``rotate`` method

    def test_rotation_by_90_degrees_around_axis(self, z):
        v = Vec3([1, 1, 1])
        current = v.rotate(z, 90)
        expects = Vec3([-1, 1, 1])

        assert current == expects

    def test_rotation_by_180_degrees_around_axis(self, z):
        v = Vec3([1, 1, 1])
        current = v.rotate(z, 180)
        expects = Vec3([-1, -1, 1])

        assert current == expects

    def test_rotation_by_360_degrees_around_axis(self, z):
        v = Vec3([1, 1, 1])
        current = v.rotate(z, 360)
        expects = Vec3([1, 1, 1])

        assert current == expects

    # ``proj`` method

    def test_projection_of_xy_onto(self, z):
        xz = Vec3([1, 0, 1])
        current = xz.proj(z)
        expects = Vec3([0, 0, 1])

        assert current == expects

    # todo ``H``
    # todo ``transform``

    def test_add_operator(self):
        lhs = Vec3([1, 1, 1])
        rhs = Vec3([1, 1, 1])

        current = lhs + rhs
        expects = Vec3([2, 2, 2])

        assert current == expects

    def test_sub_operator(self):
        lhs = Vec3([1, 1, 1])
        rhs = Vec3([1, 1, 1])

        current = lhs - rhs
        expects = Vec3([0, 0, 0])

        assert current == expects

    # ``*`` operator aka dot product

    def test_mull_operator(self):
        lhs = Vec3([1, 1, 1])
        rhs = Vec3([1, 1, 1])

        current = lhs * rhs
        expects = lhs.dot(rhs)

        assert current == expects

    # ``**`` operator aka cross product

    def test_pow_operator_with_vector(self):
        lhs = Vec3([1, 0, 0])
        rhs = Vec3([0, 1, 0])

        current = lhs ** rhs
        expects = lhs.cross(rhs)

        assert current == expects

    def test_pow_operator_with_scalar(self):
        lhs = 2
        rhs = Vec3([1, 1, 1])

        current = lhs * rhs
        expects = Vec3([2, 2, 2])

        assert current == expects

    def test_length_method(self):
        u = Vec3([1])
        v = Vec3([1, 2])
        w = Vec3([1, 2, 3])

        len(u) == len(v) == len(w) == 3

    def test_getitem_operator(self):
        v = Vec3([1, 2, 3])
        assert all((v[0] == 1, v[1] == 2, v[2] == 3))


class TestLineation:
    """
    The lineation is represented as axial (pseudo) vector.
    """

    @pytest.fixture
    def x(self):
        return Lin(0, 0)

    @pytest.fixture
    def y(self):
        return Lin(90, 0)

    @pytest.mark.skip
    def test_repr(self, x):
        assert repr(x) == "Lin(1.0,0,0)"

    def test_str(self, x):
        assert str(x) == "L:0/0"

    def test_equality_for_oposite_dir_and_zero_dip(self, y):
        assert y == -y

    def test_that_azimuth_0_is_same_as_360(self):
        assert Lin(0, 0) == Lin(360, 0)

    def test_scalar_product(self):
        pass

    def test_vector_product(self):
        l1 = Lin(110, 22)
        l2 = Lin(163, 47)
        p = l1.cross(l2)

        assert np.allclose(p.angle(l1), p.angle(l2), 90)

    def test_vector_product_operator(self):
        l1 = Lin(110, 22)
        l2 = Lin(163, 47)
        p = l1 ** l2

        assert np.allclose(p.angle(l1), p.angle(l2), 90)

    def test_add_operator(self):
        l1, l2 = Group.randn_lin(2)

        assert l1.transform(l1.H(l2)) == l2

    def test_add_operator__simple(self):
        v1 = Lin(45, 0)
        v2 = Lin(315, 0)

        # The `v2` is converted from 2. quadrant to 4. quadrant.
        assert (v1 + v2).uv == Lin(90, 0)

        # The `v1` is converted from 1. quadrant to 3. quadrant.
        assert (v2 + v1).uv == Lin(270, 0)

        # Anyway, axial add is commutative.
        assert (v1 + v2) == (v2 + v1)

    def test_sub_operator__simple(self):
        v1 = Lin(45, 0)
        v2 = Lin(315, 0)

        assert (v1 - v2).uv == Lin(360, 0)

    def test_dd_property(self):
        l = Lin(120, 30)
        assert Lin(*l.V.dd) == l

# ############################################################################
# Group
# ############################################################################


def test_rotation_rdegree():
    g = Group.randn_lin()
    assert np.allclose(g.rotate(Lin(45, 45), 90).rdegree, g.rdegree)


def test_rotation_angle_lin():
    l1, l2 = Group.randn_lin(2)
    D = DefGrad.from_axis(Lin(45, 45), 60)
    assert np.allclose(l1.angle(l2), l1.transform(D).angle(l2.transform(D)))


def test_rotation_angle_fol():
    f1, f2 = Group.randn_fol(2)
    D = DefGrad.from_axis(Lin(45, 45), 60)
    assert np.allclose(f1.angle(f2), f1.transform(D).angle(f2.transform(D)))


def test_resultant_rdegree():
    g = Group.from_array([45, 135, 225, 315], [45, 45, 45, 45], Lin)
    c1 = g.R.uv == Lin(0, 90)
    c2 = np.allclose(abs(g.R), np.sqrt(8))
    c3 = np.allclose((g.rdegree/100 + 1)**2, 2)
    assert c1 and c2 and c3


def test_group_heterogenous_error():
    with pytest.raises(Exception) as exc:
        g = Group([1, 2, 3])
        assert "Data must be Fol, Lin or Vec3 type." ==  str(exc.exception)


# ############################################################################
# Lineation
# ############################################################################


def test_cross_product():
    l1 = Lin(110, 22)
    l2 = Lin(163, 47)
    p = l1**l2
    assert np.allclose(p.angle(l1), p.angle(l2), 90)


def test_axial_addition():
    l1, l2 = Group.randn_lin(2)
    assert l1.transform(l1.H(l2)) == l2


def test_vec_H():
    m = Lin(135, 10) + Lin(315, 10)
    assert m.uv == Lin(135, 0)


def test_group_heterogenous_error():
    with pytest.raises(Exception) as exc:
        g = Group([Fol(10, 10), Lin(20, 20)])
        assert "All data in group must be of same type." == str(exc.exception)


def test_pair_misfit():
    n, l = Group.randn_lin(2)
    f = n.asfol
    p = Pair.from_pair(f, f - l.proj(f))
    assert np.allclose(p.misfit, 0)


def test_pair_rotate():
    n, l = Group.randn_lin(2)
    f = n.asfol
    p = Pair.from_pair(f, f - l.proj(f))
    pr = p.rotate(Lin(45, 45), 120)
    assert np.allclose(p.fvec.angle(p.lvec), pr.fvec.angle(pr.lvec), 90)


def test_lin_vector_dd():
    l = Lin(120, 30)
    assert Lin(*l.V.dd) == l


# ############################################################################
# Foliation
# ############################################################################


def test_fol_vector_dd():
    f = Fol(120, 30)
    assert Lin(*f.V.dd).asfol == f


# ############################################################################
# Fault
# ############################################################################


def test_fault_rotation_sense():
    f = Fault(90, 30, 110, 28, -1)
    assert repr(f.rotate(Lin(220, 10), 60)) == 'F:343/37-301/29 +'


# ############################################################################
# Ortensor
# ############################################################################


def test_ortensor_orthogonal():
    f = Group.randn_fol(1)[0]
    assert np.allclose(*Ortensor(Group([f.V, f.rake(-45), f.rake(45)])).eigenvals)
