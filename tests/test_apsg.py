# -*- coding: utf-8 -*-

"""
Unit tests for `apsg` core module.

Use this steps for unit test:
    
- Arrange all necessary preconditions and inputs.
- Act on the object or method under test.
- Assert that the expected results have occurred.


Proper unit tests should fail for exactly one reason
(that’s why you should be using one assert per unit test.)
"""


import pytest
import numpy as np


from apsg import Vec3, Fol, Lin, Fault, Pair, Group, DefGrad, Ortensor, settings


# ############################################################################
# Vector
# ############################################################################


def test_that_vec3_string_gets_three_digits_when_vec2dd_settings_is_false():
    settings["vec2dd"] = False

    vec = Vec3([1, 2, 3])

    current = str(vec)
    expects = "V(1.000, 2.000, 3.000)"    
    
    assert current == expects


def test_that_vec3_string_gets_dip_and_dir_when_vec2dd_settings_is_true():
    settings["vec2dd"] = True
    
    vec = Vec3([1, 2, 3])
    
    current = str(vec)
    expects = "V:63/53"

    assert current == expects
    
    settings["vec2dd"] = False


def test_that_equality_operator_works():
    lhs = Vec3([1.000] * 3)
    rhs = Vec3([1.000] * 3)

    assert lhs == rhs


def test_that_equality_operator_precision_limits():
    """
    This is not the best method how to test a floating point precision limits, 
    but I will keep it here for a future work.
    """
    lhs = Vec3([1.00000000000000001] * 3)
    rhs = Vec3([1.00000000000000009] * 3)

    assert lhs == rhs


def test_that_equality_operator_is_reflexive():
    u = Vec3(1, 2, 3)

    assert u == u


def test_that_equality_operator_is_symetric():
    u = Vec3([1, 2, 3])
    v = Vec3([1, 2, 3])

    assert u == v and v == u


def test_that_equality_operator_is_transitive():
    u = Vec3([1, 2, 3])
    v = Vec3([1, 2, 3])
    w = Vec3([1, 2, 3])

    assert u == v and v == w and u == w 


def teste_inequality_operator():
    lhs = Vec3([1, 2, 3])
    rhs = Vec3([3, 2, 1])

    assert lhs != rhs


@pytest.mark.skip
def test_that_hash_is_same_for_identical_vectors():
    lhs = Vec3([1, 2, 3])
    rhs = Vec3([1, 2, 3])
    
    assert hash(lhs) == hash(rhs)    


@pytest.mark.skip
def test_that_hash_is_not_same_for_different_vectors():
    lhs = Vec3([1, 2, 3])
    rhs = Vec3([3, 2, 1])
    
    assert not hash(lhs) == hash(rhs)    


def test_lin_to_vec_to_lin():
    assert Vec3(Lin(110, 37)).aslin == Lin(110, 37)


def test_fol_to_vec_to_fol():
    assert Vec3(Fol(213, 52)).asfol == Fol(213, 52)


def test_that_vector_is_upper():
    vec = Vec3([0, 0, -1])

    assert vec.upper


def test_that_vector_is_not_upper():
    vec = Vec3([0, 0, 1])

    assert not vec.upper


def test_that_vector_is_flipped():
    current = Vec3([0, 0, 1]).flip
    expects = Vec3([0, 0, -1])

    assert current == expects


def test_absolute_value():
    current = abs(Vec3([1, 2, 3]))
    expects = 3.7416573867739413

    assert current == expects


def test_that_vector_is_normalized():
    current = Vec3([1, 2, 3]).uv
    expects = Vec3([0.26726124191242442, 0.5345224838248488, 0.8017837257372732])

    assert current == expects


def test_that_mul_operator_applied_to_same_vectors_returns_proper_dot_product():
    i = Vec3(1, 0, 0)

    assert (i * i) == abs(i)


def test_that_mul_operator_applied_to_orthogonal_vectors_returns_proper_dot_product():
    i = Vec3(1, 0, 0)
    j = Vec3(0, 1, 0)
    
    assert (i * j) == 0


# TODO #

# dd

# aslin

# asfol

# asvec

# angle

# cross

# rotate

# project

# H

# transform


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


