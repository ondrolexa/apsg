"""Normalizer, statistics and maximum-likelihood fit of the Lisle distribution."""

import numpy as np
import pytest
from scipy import special
from scipy.integrate import quad
from scipy.optimize import minimize

from apsg import pair
from apsg.feature import Lisle
from apsg.feature._distributions import _special
from apsg.feature._distributions._frames import lisle_moments, pair_arrays
from apsg.feature._distributions._special import (
    _lisle_terms,
    lisle_fit_kappas,
    lisle_log_norm,
    restricted_eigen,
)

GRID = [
    (0, 0),
    (0, 1e-3),
    (1e-3, 0),
    (1, 2),
    (5, 3),
    (10, 10),
    (30, 30),
    (100, 100),
    (0, 300),
    (0.1, 1000),
    (1000, 10),
    (0, 1e4),
]


def _dense_reference(kappa1, kappa2):
    """The original construction (quadrature over u, trapezoid in the azimuth)
    with a dense azimuth grid and tight tolerances, where its 128 points fail."""
    lam = np.array([kappa1 + kappa2, kappa2, 0.0])
    top = lam[0]
    psi = np.linspace(0, 2 * np.pi, 4097)[:-1]

    def along(u):
        r = np.sqrt(max(1 - u * u, 0.0))
        L = np.column_stack([np.full_like(psi, u), r * np.cos(psi), r * np.sin(psi)])
        q, s, d = restricted_eigen(lam, L)
        return 2 * np.pi * np.mean(np.exp(q - (s - d) / 2 - top) * special.i0e(d / 2))

    width = 1 / max(lam[0] - lam[1], 1.0)
    points = sorted({max(0.0, 1 - 30 * width)} - {0.0}) or None
    value, _ = quad(along, 0, 1, points=points, limit=400, epsabs=0, epsrel=1e-13)
    return float(np.log(2 * value * 2 * np.pi / (8 * np.pi**2)) + top)


@pytest.mark.parametrize("kappas", GRID)
def test_normalizer_matches_dense_reference(kappas):
    assert lisle_log_norm(*kappas) == pytest.approx(_dense_reference(*kappas), abs=1e-8)


def test_uniform_normalizer_and_statistics():
    log_c, stats_ = _lisle_terms(0, 0, gradient=True)
    assert log_c == pytest.approx(0, abs=1e-12)
    assert stats_ == pytest.approx((0, 0), abs=1e-12)


def test_normalizer_is_symmetric_in_the_frame_roles():
    # exchanging the roles of kappa1 and kappa2 exchanges the two statistics
    _, a = _lisle_terms(50, 5, gradient=True)
    _, b = _lisle_terms(5, 50, gradient=True)
    assert a[0] == pytest.approx(b[1])
    assert a[1] == pytest.approx(b[0])


def test_node_counts_are_converged(monkeypatch):
    """The tiered rules agree with much larger ones."""
    default = {k: lisle_log_norm(*k) for k in GRID}
    monkeypatch.setattr(_special, "_LISLE_TIERS", ((np.inf, 192, 128),))
    monkeypatch.setattr(_special, "_LISLE_RULES", {})
    for k, value in default.items():
        assert lisle_log_norm(*k) == pytest.approx(value, abs=5e-9)


@pytest.mark.parametrize(
    "kappas", [(0.5, 0.5), (2, 3), (10, 10), (50, 5), (5, 50), (200, 200), (0.01, 0.02)]
)
def test_statistics_are_derivatives_of_the_normalizer(kappas):
    _, g = _lisle_terms(*kappas, gradient=True)
    k = np.array(kappas, dtype=float)
    for i in range(2):
        h = 1e-5 * max(1.0, k[i])
        up, down = k.copy(), k.copy()
        up[i] += h
        down[i] -= h
        num = (lisle_log_norm(*up) - lisle_log_norm(*down)) / (2 * h)
        assert g[i] == pytest.approx(num, abs=1e-7)


def _nelder_mead(tau0, tau1):
    """Reference maximum likelihood by Nelder-Mead, the previous implementation."""

    def nll(p):
        k1, k2 = np.exp(p)
        return -((k1 + k2) * tau0 + k2 * tau1 - lisle_log_norm(k1, k2))

    start = [np.log(1 / max(1 - tau0, 1e-3)), np.log(1 / max(1 - (tau0 + tau1), 1e-3))]
    res = minimize(
        nll,
        start,
        method="Nelder-Mead",
        options={"xatol": 1e-8, "fatol": 1e-10, "maxiter": 2000},
    )
    return np.exp(res.x)


@pytest.mark.parametrize(
    "kappas", [(0.5, 0.5), (2, 3), (10, 10), (50, 5), (5, 50), (100, 100)]
)
@pytest.mark.parametrize("n", [10, 60, 400])
def test_newton_matches_nelder_mead(kappas, n):
    data = Lisle(pair(), *kappas).rvs(n, rng=np.random.default_rng(n))
    taus, _, _ = lisle_moments(pair_arrays(data))
    k1, k2, converged = lisle_fit_kappas(taus[0], taus[1])
    assert converged and k1 > 0 and k2 > 0
    ref = _nelder_mead(taus[0], taus[1])
    assert k1 == pytest.approx(ref[0], rel=1e-3)
    assert k2 == pytest.approx(ref[1], rel=1e-3)
    # the likelihood equations hold
    _, stats_ = _lisle_terms(k1, k2, gradient=True)
    assert stats_ == pytest.approx((taus[0], taus[0] + taus[1]), rel=1e-6)


def test_fit_uses_newton_and_reports_convergence():
    fitted = Lisle.fit(Lisle(pair(), 20, 10).rvs(300, rng=1))
    assert fitted.fit_info.converged
    assert fitted.kappa1 == pytest.approx(20.81208, rel=1e-5)
    assert fitted.kappa2 == pytest.approx(9.59500, rel=1e-5)


def test_warm_start_gives_the_same_fit():
    cold = lisle_fit_kappas(0.9, 0.05)
    warm = lisle_fit_kappas(0.9, 0.05, start=(3.0, 3.0))
    assert warm[:2] == pytest.approx(cold[:2], rel=1e-7)
    # an infeasible start is ignored
    assert lisle_fit_kappas(0.9, 0.05, start=(-1.0, 3.0))[:2] == pytest.approx(
        cold[:2], rel=1e-7
    )


def test_degenerate_data_stay_finite():
    # identical frames: the concentration is infinite, the fit stops at a huge one
    k1, k2, _ = lisle_fit_kappas(1.0, 0.0)
    assert np.isfinite(k1) and np.isfinite(k2) and k1 > 1e6 and k2 > 1e6
    # nearly uniform frames: both concentrations are small and positive
    for taus in [(1e-3, 1e-4), (0.0, 0.0)]:
        k1, k2, _ = lisle_fit_kappas(*taus)
        assert 0 < k1 < 0.1 and 0 < k2 < 0.1
