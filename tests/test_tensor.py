# -*- coding: utf-8 -*-

"""
Test the tensor classes.
"""

import numpy as np

from apsg import Tensor, DefGrad, VelGrad, Stress, Ortensor, Lin, Vec3, Fol, Group

# Tensor type is value object => structural equality

def test_that_tensors_are_equal_and_has_same_hash(helpers):
    lhs = Tensor([[1, 1, 1], [2, 2, 3], [3, 3, 3]])
    rhs = Tensor([[1, 1, 1], [2, 2, 3], [3, 3, 3]])

    assert helpers.has_same_hash_when_value_objects_are_equals(lhs, rhs)


def test_that_tensors_are_not_equal_and_has_different_hash(helpers):
    lhs = Tensor([[1, 1, 1], [2, 2, 3], [3, 3, 3]])
    rhs = Tensor([[3, 3, 3], [2, 2, 3], [1, 1, 1]])

    assert helpers.has_not_same_hash_when_value_objects_are_not_equals(lhs, rhs)


def test_tensor_repr_and_str():
    assert "Tensor: T Kind: L\n(E1:2,E2:1,E3:1)\n[[2 1 1]\n [1 2 1]\n [1 1 2]]" == str(Tensor([[2, 1, 1], [1, 2, 1], [1,1, 2]], name='T'))


# Ortensor


def test_ortensor_uniform():
    f = Fol.rand()
    assert np.allclose(Ortensor.from_group(Group([f.V, f.rake(-45), f.rake(45)]))._evals, np.ones(3) / 3)

# DefGrad


def test_orthogonality_rotation_matrix():
    lin = Lin.rand()
    a = np.random.randint(180)
    R = DefGrad.from_axis(lin, a)
    assert np.allclose(R * R.T, np.eye(3))


def test_defgrad_derivation():
    F = DefGrad.from_comp(xx=2, zz=0.5)
    L = VelGrad.from_comp(xx=np.log(2), zz=-np.log(2))
    F.velgrad() == L


# VelGrad


def test_velgrad_integration():
    F = DefGrad.from_comp(xx=2, zz=0.5)
    L = VelGrad.from_comp(xx=np.log(2), zz=-np.log(2))
    L.defgrad() == F


# Stress


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
