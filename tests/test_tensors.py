# -*- coding: utf-8 -*-

"""
Test the tensor classes.
"""

import numpy as np

from apsg.math import Matrix3
from apsg import vec
from apsg import lin, fol, vecset
from apsg import defgrad, velgrad, stress, ortensor
from apsg import defgrad2, velgrad2

# Matrix3 type is value object => structural equality


def test_that_tensors_are_equal_and_has_same_hash(helpers):
    lhs = Matrix3([[1, 1, 1], [2, 2, 3], [3, 3, 3]])
    rhs = Matrix3([[1, 1, 1], [2, 2, 3], [3, 3, 3]])

    assert helpers.has_same_hash_when_value_objects_are_equals(lhs, rhs)


def test_that_tensors_are_not_equal_and_has_different_hash(helpers):
    lhs = Matrix3([[1, 1, 1], [2, 2, 3], [3, 3, 3]])
    rhs = Matrix3([[3, 3, 3], [2, 2, 3], [1, 1, 1]])

    assert helpers.has_not_same_hash_when_value_objects_are_not_equals(lhs, rhs)


def test_tensor_repr_and_str():
    assert "Matrix3\n[[2 1 1]\n [1 2 1]\n [1 1 2]]" == str(
        Matrix3([[2, 1, 1], [1, 2, 1], [1, 1, 2]])
    )


# Ortensor


def test_ortensor_uniform():
    f = fol.random()
    g = vecset([f.pole(), f.rake(-45), f.rake(45)])
    ot = ortensor.from_features(g)
    assert np.allclose(ot.eigenvalues(), np.ones(3) / 3)


# DefGrad


def test_orthogonality_rotation_matrix():
    k = lin.random()
    a = np.random.randint(180)
    R = defgrad.from_axisangle(k, a)
    assert np.allclose(R @ R.T, np.eye(3))


def test_defgrad_derivation():
    F = defgrad.from_comp(xx=2, zz=0.5)
    L = velgrad.from_comp(xx=np.log(2), zz=-np.log(2))
    F.velgrad() == L


# DefGrad2


def test_rotation_decomposition2():
    F = defgrad2.from_comp(xy=1)
    assert defgrad2.from_angle(F.angle()) == F.R


# VelGrad


def test_velgrad_integration():
    F = defgrad.from_comp(xx=2, zz=0.5)
    L = velgrad.from_comp(xx=np.log(2), zz=-np.log(2))
    L.defgrad() == F


# Stress


def test_stress_component():
    S = stress.from_comp(xx=2, yy=2, zz=1, xy=1, xz=3, yz=-2)
    n = vec(1, 2, -2).uv()
    sn, tau = S.stress_comp(n)
    assert np.allclose(
        [abs(sn), abs(tau)], [abs(S.normal_stress(n)), S.shear_stress(n)]
    )


def test_stress_invariants_calculation():
    S = stress.from_comp(xx=4, yy=6, zz=8, xy=1, xz=2)
    assert np.allclose([S.I1, S.I2, S.I3], [18, 99, 160])


def test_stress_invariants_under_rotation():
    S = stress.from_comp(xx=4, yy=6, zz=8, xy=1, xz=2)
    k = lin.random()
    a = np.random.randint(180)
    R = defgrad.from_axisangle(k, a)
    Sr = S.transform(R)
    assert np.allclose([S.I1, S.I2, S.I3], [Sr.I1, Sr.I2, Sr.I3])
