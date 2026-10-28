"""
Tests of rotation sets, the matrix Fisher distribution and the frame tests.
"""

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from apsg import pairset, rotation
from apsg.feature import (
    G,
    Lisle,
    MatrixFisher,
    Pair,
    PairSet,
    RotationSet,
    Vector3,
    feature_from_json,
)
from apsg.feature._distributions import matrix_fisher_log_norm, proper_svd


def _F(psi=(3, 1.5, -0.5), seed=1):
    U = Rotation.random(random_state=seed).as_matrix()
    V = Rotation.random(random_state=seed + 1).as_matrix()
    return U @ np.diag(psi) @ V.T


def test_proper_svd():
    A = _F()  # det < 0, as psi_3 < 0
    U, d, V = proper_svd(A)
    assert np.isclose(np.linalg.det(U), 1) and np.isclose(np.linalg.det(V), 1)
    assert np.allclose(U @ np.diag(d) @ V.T, A)
    assert d[2] < 0


def test_normalizer_monte_carlo():
    F = _F()
    R = Rotation.random(200000, random_state=3).as_matrix()
    mc = np.log(np.mean(np.exp(np.einsum("ij,nij->n", F, R))))
    assert np.isclose(matrix_fisher_log_norm(proper_svd(F)[1]), mc, atol=0.02)
    assert np.isclose(matrix_fisher_log_norm((0, 0, 0)), 0)


def test_sampler_mean_matches_expectation():
    d = MatrixFisher(_F())
    R = Rotation.random(200000, random_state=4).as_matrix()
    w = np.exp(np.einsum("ij,nij->n", d.F, R))
    expected = (w[:, None, None] * R).sum(axis=0) / w.sum()
    sample = np.asarray(d.rvs(4000, rng=1))
    assert np.abs(sample.mean(axis=0) - expected).max() < 0.02


def test_fit_recovers_parameters():
    d = MatrixFisher(_F((8, 4, 2)))
    fitted = MatrixFisher.fit(d.rvs(3000, rng=2))
    assert np.allclose(fitted.concentrations, d.concentrations, rtol=0.1)
    M = np.asarray(d.mean).T @ np.asarray(fitted.mean)
    assert np.allclose(M, np.eye(3), atol=0.03)
    assert fitted.fit_info.n == 3000


def test_fit_is_rotation_equivariant():
    X = np.asarray(MatrixFisher(_F((5, 3, 1))).rvs(300, rng=3))
    Q = Rotation.random(random_state=9).as_matrix()
    fitted = MatrixFisher.fit(X)
    rotated = MatrixFisher.fit(Q @ X)
    assert np.allclose(Q @ np.asarray(fitted.mean), np.asarray(rotated.mean), atol=1e-6)
    assert np.isclose(fitted.fit_info.loglik, rotated.fit_info.loglik)


def test_matrix_fisher_interface():
    d = MatrixFisher.from_mode(rotation.from_axisangle(Vector3(1, 1, 0), 40), 10)
    assert np.allclose(d.concentrations, 10)
    assert feature_from_json(d.to_json()) == d
    sample = d.rvs(5, rng=7)
    assert isinstance(sample, RotationSet)
    assert np.allclose(np.asarray(d.rvs(5, rng=7)), np.asarray(sample))
    assert isinstance(d.pdf(sample[0]), float)
    assert d.pdf(sample).shape == (5,)
    with pytest.raises(ValueError, match="3x3"):
        MatrixFisher(np.eye(2))
    with pytest.raises(ValueError, match="orthogonal"):
        d.pdf(np.full((3, 3), 0.5))


def test_rotation_set():
    rs = RotationSet.random(20)
    assert repr(rs).startswith("R(20)")
    assert isinstance(G(list(rs)), RotationSet)
    assert np.allclose(np.asarray(feature_from_json(rs.to_json())), np.asarray(rs))
    assert np.allclose(rs.angle(rs), 0, atol=1e-5)
    single = rotation.from_axisangle(Vector3(0, 0, 1), 30)
    assert np.allclose(RotationSet([single]).angle(), 30)
    tight = MatrixFisher.from_mode(single, 200).rvs(100, rng=1)
    assert tight.mean().angle() == pytest.approx(30, abs=1)
    with pytest.raises(TypeError):
        RotationSet([Vector3(1, 0, 0)])


def _pairs_from_rotations(R):
    return PairSet([Pair(Vector3(r[:, 2]), Vector3(r[:, 0])) for r in R])


def test_frame_uniformity_test():
    rng = np.random.default_rng(0)
    rejected = sum(
        _pairs_from_rotations(Rotation.random(40, random_state=rng).as_matrix())
        .uniformity_test()
        .reject()
        for _ in range(200)
    )
    assert 0.02 <= rejected / 200 <= 0.09
    r = Lisle(Pair(120, 40, 100, 38), 5, 5).rvs(50, rng=1).uniformity_test()
    assert r.df == 10 and r.reject()
    assert pairset.random(5).uniformity_test().df == 10
