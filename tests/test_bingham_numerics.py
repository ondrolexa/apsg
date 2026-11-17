"""Normalizer, moments and maximum-likelihood fit of the Bingham distribution."""

import numpy as np
import pytest
from scipy import special
from scipy.integrate import quad
from scipy.optimize import minimize

from apsg import lin
from apsg.feature import Bingham, BinghamMixture
from apsg.feature._distributions._axial import _scatter_eigen
from apsg.feature._distributions._special import (
    _bingham_terms,
    bingham_fit_kappas,
    bingham_log_norm,
    bingham_moments,
)

CONCENTRATIONS = [
    (0, 0),
    (-0.001, -0.002),
    (-1, -2),
    (-5, -20),
    (-3, -3),
    (-30, -30),
    (-0.5, -200),
    (-100, -400),
    (-1e4, -1e4),
    (-1e6, -1e6),
    (-1e5, -1e6),
]


def _reference_integral(k_axis, k_other, power=0):
    """Adaptive quadrature of the Bingham normalizer, the previous implementation."""

    def integrand(u):
        return (
            u**power * np.exp(k_axis * u * u) * special.i0e(-k_other * (1 - u * u) / 2)
        )

    points = [5 / np.sqrt(-k_axis)] if k_axis < -25 else None
    value, _ = quad(integrand, 0, 1, points=points, limit=200, epsabs=0)
    return 4 * np.pi * value


@pytest.mark.parametrize("k_major, k_minor", CONCENTRATIONS)
def test_normalizer_and_moments_match_quadrature(k_major, k_minor):
    F = _reference_integral(k_minor, k_major)
    assert bingham_log_norm(k_major, k_minor) == pytest.approx(np.log(F), abs=1e-10)
    m_major, m_minor = bingham_moments(k_major, k_minor)
    assert m_major == pytest.approx(
        _reference_integral(k_major, k_minor, 2) / F, rel=1e-8, abs=1e-12
    )
    assert m_minor == pytest.approx(
        _reference_integral(k_minor, k_major, 2) / F, rel=1e-8, abs=1e-12
    )


def test_uniform_normalizer():
    assert bingham_log_norm(0, 0) == pytest.approx(np.log(4 * np.pi))
    assert bingham_moments(0, 0) == pytest.approx((1 / 3, 1 / 3))


@pytest.mark.parametrize("k_major, k_minor", [(-3, -11), (-0.2, -0.5), (-40, -90)])
def test_hessian_is_finite_difference_of_gradient(k_major, k_minor):
    _, _, H = _bingham_terms(k_major, k_minor, hessian=True)
    eps = 1e-5 * max(1, abs(k_minor))
    num = np.empty((2, 2))
    for i, d in enumerate(np.eye(2)):
        hi = _bingham_terms(*(np.array([k_major, k_minor]) + eps * d))[1]
        lo = _bingham_terms(*(np.array([k_major, k_minor]) - eps * d))[1]
        num[:, i] = (hi - lo) / (2 * eps)
    assert np.allclose(H, num, rtol=1e-5, atol=1e-8)
    assert np.all(np.linalg.eigvalsh(H) > 0)  # log F is convex


def _nelder_mead(tau_major, tau_minor):
    """Reference maximum likelihood by Nelder-Mead, the previous implementation."""

    def nll(p):
        k_major = -np.exp(p[0])
        k_minor = k_major - np.exp(p[1])
        return -(k_major * tau_major + k_minor * tau_minor) + np.log(
            _reference_integral(k_minor, k_major)
        )

    k_major = -1 / (2 * max(tau_major, 1e-6))
    k_minor = min(-1 / (2 * max(tau_minor, 1e-6)), k_major - 1e-3)
    res = minimize(
        nll,
        [np.log(-k_major), np.log(k_major - k_minor)],
        method="Nelder-Mead",
        options={"xatol": 1e-9, "fatol": 1e-12, "maxiter": 2000},
    )
    k_major = -np.exp(res.x[0])
    return k_major, k_major - np.exp(res.x[1])


@pytest.mark.parametrize(
    "k_major, k_minor",
    [
        (-1, -2),
        (-0.2, -0.5),
        (-5, -20),
        (-30, -80),
        (-3, -3),
        (-0.05, -4),
        (-300, -900),
    ],
)
@pytest.mark.parametrize("n", [6, 40, 400])
def test_newton_matches_nelder_mead(k_major, k_minor, n):
    rng = np.random.default_rng(n)
    X = np.asarray(Bingham(lin(10, 80), lin(100, 0), k_major, k_minor).rvs(n, rng=rng))
    taus, _ = _scatter_eigen(X / np.linalg.norm(X, axis=1, keepdims=True))
    km, kn, converged = bingham_fit_kappas(taus[1], taus[2])
    assert converged and kn < km < 0
    ref_km, ref_kn = _nelder_mead(taus[1], taus[2])
    assert km == pytest.approx(ref_km, rel=1e-3)
    assert kn == pytest.approx(ref_kn, rel=1e-3)
    # the likelihood equations hold
    assert bingham_moments(km, kn) == pytest.approx((taus[1], taus[2]), rel=1e-6)


def test_warm_start_gives_the_same_fit():
    cold = bingham_fit_kappas(0.2, 0.05)
    warm = bingham_fit_kappas(0.2, 0.05, start=(-2.0, -9.0))
    assert warm[:2] == pytest.approx(cold[:2], rel=1e-8)
    # an infeasible start is ignored
    assert bingham_fit_kappas(0.2, 0.05, start=(-9.0, -2.0))[:2] == pytest.approx(
        cold[:2], rel=1e-8
    )


def test_degenerate_data_give_huge_but_finite_concentration():
    # all axes on one great circle: the concentration along the pole is infinite
    km, kn, _ = bingham_fit_kappas(0.5, 0.0)
    assert np.isfinite(km) and np.isfinite(kn) and kn < -1e5
    # axes of one point
    assert not bingham_fit_kappas(0.0, 0.0)[2]


def test_fit_reports_convergence_and_degenerate_data():
    d = Bingham(lin(10, 80), lin(100, 0), -5, -20).rvs(200, rng=1)
    fitted = Bingham.fit(d)
    assert fitted.fit_info.converged
    with pytest.raises(ValueError):
        Bingham.fit(np.array([[0, 0, 1.0]] * 5))


def test_em_components_are_weighted_bingham_fits():
    """At convergence each component is the weighted fit given the memberships."""
    truth = BinghamMixture(
        [
            Bingham(lin(10, 80), lin(100, 0), -8, -20),
            Bingham(lin(120, 10), lin(30, 5), -15, -25),
        ],
        [0.6, 0.4],
    )
    data = truth.rvs(300, rng=np.random.default_rng(3))
    w = np.random.default_rng(4).uniform(0.5, 2, 300)
    m = BinghamMixture.fit(data, 2, weights=w, n_init=3, tol=1e-10, rng=0)
    R = m.responsibilities(data)
    for j, comp in enumerate(m.components):
        again = Bingham.fit(data, weights=w * R[:, j])
        assert comp.mean.angle(again.mean) < 0.1
        assert comp.kappa_major == pytest.approx(again.kappa_major, rel=2e-3)
        assert comp.kappa_minor == pytest.approx(again.kappa_minor, rel=2e-3)
    assert m.weights.sum() == pytest.approx(1)
    total = w / w.sum()
    assert m.weights == pytest.approx((total[:, None] * R).sum(axis=0), abs=2e-3)
