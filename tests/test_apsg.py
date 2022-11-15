# -*- coding: utf-8 -*-

"""
Unit tests for `apsg` module.

Use this steps for unit test:

- Arrange all necessary preconditions and inputs.
- Act on the object or method under test.
- Assert that the expected results have occurred.


Proper unit tests should fail for exactly one reason
(that’s why you usually should be using one assert per unit test.)
"""


import pytest
import numpy as np

from apsg.config import apsg_conf
from apsg import vec, fol, lin, fault, pair
from apsg import vecset, linset, folset
from apsg import defgrad

atol = 1e-05  # safe tests

# ############################################################################
# Vectors
# ############################################################################


class TestVector:
    @pytest.fixture
    def x(self):
        return vec(1, 0, 0)

    @pytest.fixture
    def y(self):
        return vec(0, 1, 0)

    @pytest.fixture
    def z(self):
        return vec(0, 0, 1)

    def test_that_vec_could_be_instatiated_from_single_ot_three_args(self):
        lhs = vec([1, 2, 3])
        rhs = vec(1, 2, 3)

        current = lhs == rhs
        expects = True

        assert current == expects

    def test_that_vec_string_gets_three_digits_when_vec2dd_settings_is_false(self):
        apsg_conf["vec2geo"] = False

        v = vec(1, 2, 3)

        current = str(v)
        expects = "Vector3(1, 2, 3)"

        assert current == expects

    def test_that_vec_string_gets_dip_and_dir_when_vec2dd_settings_is_true(self):
        apsg_conf["vec2geo"] = True

        v = vec(1, 2, 3)

        current = str(v)
        expects = "V:63/53"

        assert current == expects

        apsg_conf["vec2geo"] = False

    # ``==`` operator

    def test_that_equality_operator_is_reflexive(self):
        u = vec(1, 2, 3)

        assert u == u

    def test_that_equality_operator_is_symetric(self):
        u = vec(1, 2, 3)
        v = vec(1, 2, 3)

        assert u == v and v == u

    def test_that_equality_operator_is_transitive(self):
        u = vec(1, 2, 3)
        v = vec(1, 2, 3)
        w = vec(1, 2, 3)

        assert u == v and v == w and u == w

    def test_that_equality_operator_precision_limits(self):
        """
        This is not the best method how to test a floating point precision limits,
        but I will keep it here for a future work.
        """
        lhs = vec([1.00000000000000001] * 3)
        rhs = vec([1.00000000000000009] * 3)

        assert lhs == rhs

    # ``!=`` operator

    def test_inequality_operator(self):
        lhs = vec(1, 2, 3)
        rhs = vec(3, 2, 1)

        assert lhs != rhs

    # ``hash`` method

    def test_that_hash_is_same_for_identical_vectors(self):
        lhs = vec(1, 2, 3)
        rhs = vec(1, 2, 3)

        assert hash(lhs) == hash(rhs)

    def test_that_hash_is_not_same_for_different_vectors(self):
        lhs = vec(1, 2, 3)
        rhs = vec(3, 2, 1)

        assert not hash(lhs) == hash(rhs)

    # ``upper`` property

    def test_that_vector_is_upper(self):
        v = vec(0, 0, -1)

        assert v.is_upper()

    def test_that_vector_is_not_upper(self):
        v = vec(0, 0, 1)

        assert not v.is_upper()

    # ``abs`` operator

    def test_absolute_value(self):
        current = abs(vec(1, 2, 3))
        expects = 3.7416573867739413

        assert current == expects

    # ``uv`` property

    def test_that_vector_is_normalized(self):
        current = vec(1, 2, 3).normalized()
        current_alias = vec(1, 2, 3).uv()
        expects = vec(0.26726124191242442, 0.5345224838248488, 0.8017837257372732)

        assert current == current_alias == expects

    # ``geo`` property

    def test_geo_property(self):
        v = vec(1, 0, 0)

        current = v.geo
        expects = (0.0, 0.0)

        assert current == expects

    # ``aslin`` property

    def test_lin_conversion(self):
        assert str(lin(vec(1, 1, 1))) == str(lin(45, 35))  # `Vec` to `lin`
        assert str(lin(vec(lin(110, 37)))) == str(
            lin(110, 37)
        )  # `lin` to `Vec` to `lin`

    # ``asfol`` property

    def test_fol_conversion(self):
        assert str(fol(vec(1, 1, 1))) == str(fol(225, 55))  # `Vec` to `fol`
        assert str(fol(vec(fol(213, 52)))) == str(
            fol(213, 52)
        )  # `fol` to `Vec` to `fol`

    # ``asvec`` property

    def test_vec_geo_conversion(self):
        assert str(vec(lin(120, 10))) == str(vec(120, 10))

    def test_vec_scalar_multiplication(self):
        assert abs(vec(10 * vec(120, 50))) == 10

    # ``angle`` property
    def test_that_angle_between_vectors_is_0_degrees_when_they_are_collinear(self):
        lhs = vec(1, 0, 0)
        rhs = vec(2, 0, 0)

        current = lhs.angle(rhs)
        expects = 0

        assert current == expects

    def test_that_angle_between_vectors_is_90_degrees_when_they_are_perpendicular(self):
        lhs = vec(1, 0, 0)
        rhs = vec(0, 1, 1)

        current = lhs.angle(rhs)
        expects = 90  # degrees

        assert current == expects

    def test_that_angle_between_vectors_is_180_degrees_when_they_are_opposite(self):
        lhs = vec(1, 0, 0)
        rhs = vec(-1, 0, 0)

        current = lhs.angle(rhs)
        expects = 180  # degrees

        assert current == expects

    # ``cross`` method

    def test_that_vector_product_is_anticommutative(self):
        lhs = vec(1, 0, 0)
        rhs = vec(0, 1, 0)

        assert lhs.cross(rhs) == -rhs.cross(lhs)

    def test_that_vector_product_is_distributive_over_addition(self):
        x = vec("X")
        y = vec("Y")
        z = vec("Z")

        assert x.cross(y + z) == x.cross(y) + x.cross(z)

    def test_that_vector_product_is_zero_vector_when_they_are_collinear(self):
        lhs = vec(1, 0, 0)
        rhs = vec(2, 0, 0)

        current = lhs.cross(rhs)
        expects = vec(0, 0, 0)

        assert current == expects

    def test_that_vector_product_is_zero_vector_when_they_are_opposite(self):

        lhs = vec(1, 0, 0)
        rhs = vec(-1, 0, 0)

        current = lhs.cross(rhs)
        expects = vec(0, 0, 0)

        assert current == expects

    def test_vector_product_of_orthonormal_vectors(self):
        e1 = vec(1, 0, 0)
        e2 = vec(0, 1, 0)

        current = e1.cross(e2)
        expects = vec(0, 0, 1)

        assert current == expects

    # ``dot`` method

    def test_scalar_product_of_same_vectors(self):
        i = vec(1, 2, 3)

        assert np.isclose(i.dot(i), abs(i) ** 2, atol=atol)

    def test_scalar_product_of_orthonornal_vectors(self):
        i = vec(1, 0, 0)
        j = vec(0, 1, 0)

        assert np.isclose(i.dot(j), 0, atol=atol)

    # ``rotate`` method

    def test_rotation_by_90_degrees_around_axis(self, z):
        v = vec(1, 1, 1)
        current = v.rotate(z, 90)
        expects = vec(-1, 1, 1)

        assert current == expects

    def test_rotation_by_180_degrees_around_axis(self, z):
        v = vec(1, 1, 1)
        current = v.rotate(z, 180)
        expects = vec(-1, -1, 1)

        assert current == expects

    def test_rotation_by_360_degrees_around_axis(self, z):
        v = vec(1, 1, 1)
        current = v.rotate(z, 360)
        expects = vec(1, 1, 1)

        assert current == expects

    # ``proj`` method

    def test_projection_of_xy_onto(self, z):
        xz = vec(1, 0, 1)
        current = xz.proj(z)
        expects = vec(0, 0, 1)

        assert current == expects

    # ``H`` method

    def test_mutual_rotation(self, x, y, z):
        current = defgrad.from_two_vectors(x, y)
        expects = defgrad.from_axisangle(z, 90)

        assert current == expects

    # ``transform`` method

    def test_transform_method(self, x, y, z):
        F = defgrad.from_axisangle(z, 90)
        current = x.transform(F)
        expects = y

        assert current == expects

    def test_add_operator(self):
        lhs = vec(1, 1, 1)
        rhs = vec(1, 1, 1)

        current = lhs + rhs
        expects = vec(2, 2, 2)

        assert current == expects

    def test_sub_operator(self):
        lhs = vec(1, 2, 3)
        rhs = vec(3, 1, 2)

        current = lhs - rhs
        expects = vec(-2, 1, 1)

        assert current == expects

    def test_pow_operator_with_scalar(self):
        lhs = vec(1, 2, 3)
        rhs = 2

        current = lhs**rhs
        expects = vec(1, 4, 9)

        assert current == expects

    def test_length_method(self):
        w = vec(1, 2, 3)

        assert len(w) == 3


# ############################################################################
# lineation
# ############################################################################


class Testlineation:
    """
    The lineation is represented as axial (pseudo) vector.
    """

    @pytest.fixture
    def x(self):
        return lin.unit_x()

    def test_str(self, x):
        assert str(x) == "L:0/0"

    def test_equality_for_oposite_dir(self):
        l1 = lin.random()
        assert l1 == -l1

    def test_anlge_for_oposite_dir(self):
        l1 = lin.random()
        assert np.isclose(l1.angle(-l1), 0, atol=atol)

    def test_that_azimuth_0_is_same_as_360(self):
        assert lin(0, 20) == lin(360, 20)

    def test_scalar_product(self):
        l1 = lin.random()
        assert np.isclose(l1.dot(l1), 1, atol=atol)

    def test_cross_product(self):
        l1 = lin.random()
        l2 = lin.random()
        p = l1.cross(l2)

        assert np.allclose([p.angle(l1), p.angle(l2)], [90, 90])

    def test_mutual_rotation(self):
        l1 = lin.random()
        l2 = lin.random()
        F = defgrad.from_two_vectors(l1, l2)

        assert l1.transform(F) == l2

    def test_angle_under_rotation(self):
        l1 = lin.random()
        l2 = lin.random()
        D = defgrad.from_axisangle(lin(45, 45), 60)

        assert np.isclose(
            l1.angle(l2), l1.transform(D).angle(l2.transform(D)), atol=atol
        )

    def test_add_operator__simple(self):
        l1 = lin.random()
        l2 = lin.random()

        assert l1 + l2 == l1 + (-l2)

        # Anyway, axial add is commutative.
        assert l1 + l2 == l2 + l1

    def test_sub_operator__simple(self):
        l1 = lin.random()
        l2 = lin.random()

        assert l1 - l2 == l1 - (-l2)

        # Anyway, axial sub is commutative.
        assert l1 - l2 == l2 - l1

    def test_geo_property(self):
        l1 = lin(120, 30)
        assert lin(*l1.geo) == l1


# ############################################################################
# foliation
# ############################################################################


class Testfoliation:
    """
    The foliation is represented as axial (pseudo) vector.
    """

    @pytest.fixture
    def x(self):
        return fol(0, 0)

    def test_str(self, x):
        assert str(x) == "S:0/0"

    def test_equality_for_oposite_dir(self):
        f = fol.random()
        assert f == -f

    def test_anlge_for_oposite_dir(self):
        f = fol.random()
        assert np.isclose(f.angle(-f), 0, atol=atol)

    def test_that_azimuth_0_is_same_as_360(self):
        assert fol(0, 20) == fol(360, 20)

    def test_scalar_product(self):
        f = fol.random()
        assert np.isclose(f.dot(f), 1, atol=atol)

    def test_cross_product(self):
        f1 = fol.random()
        f2 = fol.random()
        p = f1**f2

        assert np.allclose([p.angle(f1), p.angle(f2)], [90, 90])

    def test_foliation_product(self):
        f1 = fol.random()
        f2 = fol.random()
        p = f1.cross(f2)

        assert np.allclose([p.angle(f1), p.angle(f2)], [90, 90])

    def test_foliation_product_operator(self):
        f1 = fol.random()
        f2 = fol.random()

        assert f1.cross(f2) == f1**f2

    def test_mutual_rotation(self):
        f1 = fol.random()
        f2 = fol.random()
        F = defgrad.from_two_vectors(f1, f2)

        assert f1.transform(F) == f2

    def test_angle_under_rotation(self):
        f1 = fol.random()
        f2 = fol.random()
        D = defgrad.from_axisangle(lin(45, 45), 60)

        assert np.isclose(
            f1.angle(f2), f1.transform(D).angle(f2.transform(D)), atol=atol
        )

    def test_add_operator__simple(self):
        f1 = fol.random()
        f2 = fol.random()

        assert f1 + f2 == f1 + (-f2)

        # Anyway, axial add is commutative.
        assert f1 + f2 == f2 + f1

    def test_sub_operator__simple(self):
        f1 = fol.random()
        f2 = fol.random()

        assert f1 - f2 == f1 - (-f2)

        # Anyway, axial sub is commutative.
        assert f1 - f2 == f2 - f1

    def test_geo_property(self):
        f = fol(120, 30)

        assert fol(*f.geo) == f


# ############################################################################
# FeatureSets
# ############################################################################


class TestVector3Set:
    def test_rdegree_under_rotation(self):
        g = vecset.random_fisher()
        assert np.isclose(g.rotate(lin(45, 45), 90).rdegree(), g.rdegree(), atol=atol)

    def test_resultant_rdegree(self):
        g = vecset.from_array([45, 135, 225, 315], [45, 45, 45, 45])
        c1 = g.R().uv() == vec(0, 90)
        c2 = np.isclose(abs(g.R()), np.sqrt(8), atol=atol)
        c3 = np.isclose((g.rdegree() / 100 + 1) ** 2, 2, atol=atol)
        assert c1 and c2 and c3

    def test_group_type_error(self):
        with pytest.raises(Exception) as exc:
            vecset([1, 2, 3])
            assert "Data must be instances of Vector3" == str(exc.exception)

    def test_centered_group(self):
        g = vecset.random_fisher(position=lin(40, 50))
        gc = g.centered()
        el = gc.ortensor().eigenlins
        assert el[0] == vec("x") and el[1] == vec("y") and el[2] == vec("z")


class TestLineationSet:
    def test_rdegree_under_rotation(self):
        g = linset.random_fisher()
        assert np.allclose(g.rotate(lin(45, 45), 90).rdegree(), g.rdegree())

    def test_resultant_rdegree(self):
        g = linset.from_array([45, 135, 225, 315], [45, 45, 45, 45])
        c1 = g.R().uv() == lin(0, 90)
        c2 = np.isclose(abs(g.R()), np.sqrt(8), atol=atol)
        c3 = np.isclose((g.rdegree() / 100 + 1) ** 2, 2, atol=atol)
        assert c1 and c2 and c3

    def test_group_type_error(self):
        with pytest.raises(Exception) as exc:
            linset([1, 2, 3])
            assert "Data must be instances of Lineation" == str(exc.exception)

    def test_group_heterogenous_error(self):
        with pytest.raises(Exception) as exc:
            linset([fol(10, 10), lin(20, 20)])
            assert "Data must be instances of Lineation" == str(exc.exception)

    def test_centered_group(self):
        g = linset.random_fisher(position=lin(40, 50))
        gc = g.centered()
        el = gc.ortensor().eigenlins
        assert el[0] == vec("x") and el[1] == vec("y") and el[2] == vec("z")


# ############################################################################
# pair
# ############################################################################


class Testpair:
    def test_pair_misfit(self):
        p = pair.random()
        assert np.isclose(p.misfit, 0, atol=atol)

    def test_pair_rotate(self):
        p = pair.random()
        pr = p.rotate(lin(45, 45), 120)
        assert np.allclose([p.fvec.angle(p.lvec), pr.fvec.angle(pr.lvec)], [90, 90])


# ############################################################################
# fault
# ############################################################################


class Testfault:
    def test_fault_flip(self):
        f = fault(90, 30, 110, 28, -1)
        fr = f.rotate(f.rax, 180)
        assert (f.p == fr.p) and (f.t == fr.t)

    def test_fault_rotation_sense(self):
        f = fault(90, 30, 110, 28, -1)
        assert repr(f.rotate(lin(220, 10), 60)) == "F:343/37-301/29 +"

    def test_fault_t_axis(self):
        f = fault(150, 60, 150, 60, 1)
        assert f.t == lin(150, 15)

    def test_fault_p_axis(self):
        f = fault(150, 30, 150, 30, -1)
        assert f.p == lin(330, 15)
