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


from apsg import Vec3, Fol, Lin, Fault, Pair, Group, FaultSet, settings
from apsg import Ortensor, DefGrad, VelGrad, Stress


# ############################################################################
# Helpers
# ############################################################################


def is_hashable(obj):
    try:
        hash(obj)
        return True
    except TypeError:
        return False


# ############################################################################
# Vectors
# ############################################################################


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
        u = Vec3([1, 2, 3])

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
        assert str(Vec3([1, 1, 1]).aslin) == str(Lin(45, 35))      # `Vec` to `Lin`
        assert str(Vec3(Lin(110, 37)).aslin) == str(Lin(110, 37))  # `Lin` to `Vec` to `Lin`

    # ``asfol`` property

    def test_asfol_conversion(self):
        assert str(Vec3([1, 1, 1]).asfol) == str(Fol(225, 55))     # `Vec` to `Fol`
        assert str(Vec3(Fol(213, 52)).asfol) == str(Fol(213, 52))  # `Fol` to `Vec` to `Fol`

    # ``asvec`` property

    def test_asvec_conversion(self):
        assert str(Lin(120, 10).asvec3) == str(Vec3(120, 10, 1))

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
        expects = 90  # degrees

        assert current == expects

    def test_that_angle_between_vectors_is_180_degrees_when_they_are_opposite(self):
        lhs = Vec3([1, 0, 0])
        rhs = Vec3([-1, 0, 0])

        current = lhs.angle(rhs)
        expects = 180  # degrees

        assert current == expects

    # ``cross`` method

    def test_that_vector_product_is_anticommutative(self):
        lhs = Vec3([1, 0, 0])
        rhs = Vec3([0, 1, 0])

        assert lhs.cross(rhs) == -rhs.cross(lhs)

    def test_that_vector_product_is_distributive_over_addition(self):
        a = Vec3([1, 0, 0])
        b = Vec3([0, 1, 0])
        c = Vec3([0, 0, 1])

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
        i = Vec3([1, 2, 3])

        assert np.allclose(i.dot(i), abs(i)**2)

    def test_scalar_product_of_orthonornal_vectors(self):
        i = Vec3([1, 0, 0])
        j = Vec3([0, 1, 0])

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
        lhs = Vec3([1, 2, 3])
        rhs = Vec3([3, 1, 2])

        current = lhs - rhs
        expects = Vec3([-2, 1, 1])

        assert current == expects

    # ``*`` operator aka dot product

    def test_mull_operator(self):
        lhs = Vec3([1, 1, 1])
        rhs = Vec3([1, 1, 1])

        current = lhs * rhs
        expects = lhs.dot(rhs)

        assert np.allclose(current, expects)

    # ``**`` operator aka cross product

    def test_pow_operator_with_vector(self):
        lhs = Vec3([1, 0, 0])
        rhs = Vec3([0, 1, 0])

        current = lhs ** rhs
        expects = lhs.cross(rhs)

        assert current == expects

    def test_pow_operator_with_scalar(self):
        lhs = Vec3([1, 1, 1])
        rhs = 2

        current = lhs ** rhs
        expects = np.dot(lhs, lhs)

        assert np.allclose(current, expects)

    def test_length_method(self):
        w = Vec3([1, 2, 3])

        assert len(w) == 3

    def test_getitem_operator(self):
        v = Vec3([1, 2, 3])

        assert all((v[0] == 1, v[1] == 2, v[2] == 3))


# ############################################################################
# Lineation
# ############################################################################


class TestLineation:
    """
    The lineation is represented as axial (pseudo) vector.
    """

    @pytest.fixture
    def x(self):
        return Lin(0, 0)

    @pytest.mark.skip
    def test_repr(self, x):
        assert repr(x) == "Lin(1.0,0,0)"

    def test_str(self, x):
        assert str(x) == "L:0/0"

    def test_equality_for_oposite_dir(self):
        lin = Lin.rand()
        assert lin == -lin

    def test_that_azimuth_0_is_same_as_360(self):
        assert Lin(0, 20) == Lin(360, 20)

    def test_scalar_product(self):
        lin = Lin.rand()
        assert np.allclose(lin * lin, 1)

    def test_cross_product(self):
        l1, l2 = Lin.rand(), Lin.rand()
        p = l1**l2

        assert np.allclose([p.angle(l1), p.angle(l2)], [90, 90])

    def test_lineation_product(self):
        l1, l2 = Lin.rand(), Lin.rand()
        p = l1.cross(l2)

        assert np.allclose([p.angle(l1), p.angle(l2)], [90, 90])

    def test_lineation_product_operator(self):
        l1, l2 = Lin.rand(), Lin.rand()

        assert l1.cross(l2) == l1 ** l2

    def test_mutual_rotation(self):
        l1, l2 = Lin.rand(), Lin.rand()

        assert l1.transform(l1.H(l2)) == l2

    def test_angle_under_rotation(self):
        l1, l2 = Lin.rand(), Lin.rand()
        D = DefGrad.from_axis(Lin(45, 45), 60)

        assert np.allclose(l1.angle(l2), l1.transform(D).angle(l2.transform(D)))

    def test_add_operator__simple(self):
        l1, l2 = Lin.rand(), Lin.rand()

        assert l1 + l2 == l1 + (-l2)

        # Anyway, axial add is commutative.
        assert l1 + l2 == l2 + l1

    def test_sub_operator__simple(self):
        l1, l2 = Lin.rand(), Lin.rand()

        assert l1 - l2 == l1 - (-l2)

        # Anyway, axial sub is commutative.
        assert l1 - l2 == l2 - l1

    def test_dd_property(self):
        lin = Lin(120, 30)
        assert Lin(*lin.dd) == lin


# ############################################################################
# Foliation
# ############################################################################


class TestFoliation:

    def test_angle_under_rotation(self):
        f1, f2 = Fol.rand(), Fol.rand()
        D = DefGrad.from_axis(Lin(45, 45), 60)

        assert np.allclose(f1.angle(f2), f1.transform(D).angle(f2.transform(D)))


# ############################################################################
# Group
# ############################################################################

class TestGroup:

    def test_rdegree_under_rotation(self):
        g = Group.randn_lin()
        assert np.allclose(g.rotate(Lin(45, 45), 90).rdegree, g.rdegree)

    def test_resultant_rdegree(self):
        g = Group.from_array([45, 135, 225, 315], [45, 45, 45, 45], Lin)
        c1 = g.R.uv == Lin(0, 90)
        c2 = np.allclose(abs(g.R), np.sqrt(8))
        c3 = np.allclose((g.rdegree / 100 + 1)**2, 2)
        assert c1 and c2 and c3

    def test_group_type_error(self):
        with pytest.raises(Exception) as exc:
            Group([1, 2, 3])
            assert "Data must be Fol, Lin or Vec3 type." == str(exc.exception)

    def test_group_heterogenous_error(self):
        with pytest.raises(Exception) as exc:
            Group([Fol(10, 10), Lin(20, 20)])
            assert "All data in group must be of same type." == str(exc.exception)

    def test_centered_group(self):
        g = Group.randn_lin(mean=Lin(40, 50))
        gc = g.centered
        el = gc.ortensor.eigenlins
        assert el[0] == Lin(0, 90) and el[1] == Lin(90, 0) and el[2] == Lin(0, 0)

    def test_group_examples(self):
        exlist = Group.examples()
        for ex in exlist:
            g = Group.examples(ex)

# ############################################################################
# Pair
# ############################################################################


def test_pair_misfit():
    p = Pair.rand()
    assert np.allclose(p.misfit, 0)


def test_pair_rotate():
    p = Pair.rand()
    pr = p.rotate(Lin(45, 45), 120)
    assert np.allclose([p.fvec.angle(p.lvec), pr.fvec.angle(pr.lvec)], [90, 90])


def test_pair_equal():
    n, lin = Lin.rand(), Lin.rand()
    fol = n ** lin
    p = Pair.from_pair(fol, lin)
    assert p == Pair.from_pair(fol, lin)
    assert p == Pair.from_pair(fol, -lin)
    assert p == Pair.from_pair(-fol, lin)
    assert p == Pair.from_pair(-fol, -lin)


# ############################################################################
# Axial->Vector->DD
# ############################################################################


def test_lin_vector_dd():
    lin = Lin(120, 30)
    assert Lin(*lin.V.dd) == lin


def test_fol_vector_dd():
    fol = Fol(120, 30)
    assert Lin(*fol.V.dd).asfol == fol


# ############################################################################
# Fault
# ############################################################################


def test_fault_flip():
    f = Fault(90, 30, 110, 28, -1)
    fr = f.rotate(f.rax, 180)
    assert (f.p == fr.p) and (f.t == fr.t)


def test_fault_rotation_sense():
    f = Fault(90, 30, 110, 28, -1)
    assert repr(f.rotate(Lin(220, 10), 60)) == 'F:343/37-301/29 +'

def test_faultset_examples():
    exlist = FaultSet.examples()
    for ex in exlist:
        g = FaultSet.examples(ex)

# ############################################################################
# Ortensor
# ############################################################################


def test_ortensor_uniform():
    f = Fol.rand()
    assert np.allclose(Ortensor.from_group(Group([f.V, f.rake(-45), f.rake(45)]))._evals, np.ones(3) / 3)

# ############################################################################
# DefGrad
# ############################################################################


def test_orthogonality_rotation_matrix():
    lin = Lin.rand()
    a = np.random.randint(180)
    R = DefGrad.from_axis(lin, a)
    assert np.allclose(R * R.T, np.eye(3))


# ############################################################################
# Stress
# ############################################################################


def test_stress_component():
    S = Stress.from_comp(xx=2, yy=2, zz=1, xy=1, xz=3, yz=-2)
    n = Vec3([1, 2, -2]).uv
    sn, tau = S.stress_comp(n)
    assert np.allclose([abs(sn), abs(tau)], [abs(S.normal_stress(n)), S.shear_stress(n)])


def test_stress_invariants_calculation():
    S = Stress.from_comp(xx=4, yy=6, zz=8, xy=1, xz=2)
    assert np.allclose([S.I1, S.I2, S.I3], [18, 99, 160])


def test_stress_invariants_under_rotation():
    S = Stress.from_comp(xx=4, yy=6, zz=8, xy=1, xz=2)
    lin = Lin.rand()
    a = np.random.randint(180)
    Sr = S.rotate(lin, a)
    assert np.allclose([S.I1, S.I2, S.I3], [Sr.I1, Sr.I2, Sr.I3])
