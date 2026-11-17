"""Weighted orientation statistics and Terzaghi weights."""

import numpy as np
import pytest

from apsg import (
    dist_bingham,
    dist_fisher,
    dist_kent,
    dist_watson,
    fol,
    lin,
)
from apsg.feature import (
    FoliationSet,
    LineationSet,
    OrientationTensor3,
    Vector3,
    Vector3Set,
    terzaghi_weights,
)


@pytest.fixture
def axes():
    return LineationSet.random_bingham(60, rng=np.random.default_rng(1))


@pytest.fixture
def twice(axes):
    """Weights 2 on the first 10 data, and the same data repeated."""
    w = np.ones(len(axes))
    w[:10] = 2
    return w, LineationSet(list(axes) + list(axes[:10]))


def _same(a, b):
    assert a.keys() == b.keys()
    for k, v in a.items():
        if hasattr(v, "angle"):
            assert v.angle(b[k]) < 1e-6
        else:
            assert np.allclose(v, b[k], rtol=1e-5), k


@pytest.mark.parametrize("model", [dist_bingham, dist_watson])
def test_axial_fit_weights_equal_replication(model, axes, twice):
    w, rep = twice
    _same(model.fit(axes, weights=w).params, model.fit(rep).params)


def test_fisher_fit_weights_equal_replication():
    v = Vector3Set.random_fisher(40, rng=np.random.default_rng(2))
    w = np.ones(40)
    w[:5] = 3
    rep = Vector3Set(list(v) + list(v[:5]) * 2)
    _same(dist_fisher.fit(v, weights=w).params, dist_fisher.fit(rep).params)


@pytest.mark.parametrize("model", [dist_bingham, dist_watson, dist_fisher])
def test_unit_weights_are_unweighted(model, axes):
    data = axes if model is not dist_fisher else Vector3Set.random_fisher(30)
    if model is dist_fisher:
        a = model.fit(data)
        b = model.fit(data, weights=np.ones(30))
    else:
        a = model.fit(data)
        b = model.fit(data, weights=np.ones(len(data)))
    _same(a.params, b.params)
    assert b.fit_info.n_eff == pytest.approx(len(data))


def test_scaling_and_zero_weight(axes):
    w = np.random.default_rng(3).uniform(0.5, 2, len(axes))
    _same(
        dist_bingham.fit(axes, weights=w).params,
        dist_bingham.fit(axes, weights=7 * w).params,
    )
    w0 = w.copy()
    w0[:5] = 0
    _same(
        dist_bingham.fit(axes, weights=w0).params,
        dist_bingham.fit(axes[5:], weights=w[5:]).params,
    )


def test_n_eff_is_kish(axes):
    w = np.random.default_rng(4).uniform(0.1, 2, len(axes))
    info = dist_watson.fit(axes, weights=w).fit_info
    assert info.n == len(axes)
    assert info.n_eff == pytest.approx(w.sum() ** 2 / (w**2).sum())
    assert info.n_eff < info.n


@pytest.mark.parametrize(
    "bad", [np.ones(3), -np.ones(60), np.zeros(60), np.full(60, np.nan)]
)
def test_bad_weights_raise(axes, bad):
    with pytest.raises(ValueError):
        dist_bingham.fit(axes, weights=bad)


def test_unsupported_model_raises():
    v = Vector3Set.random_fisher(30)
    with pytest.raises(NotImplementedError):
        dist_kent.fit(v, weights=np.ones(30))


def test_weighted_ortensor(axes, twice):
    w, rep = twice
    assert np.allclose(axes.ortensor(w), rep.ortensor())
    assert np.allclose(axes.ortensor(np.ones(len(axes))), axes.ortensor())
    assert np.allclose(
        OrientationTensor3.from_features(axes, weights=w), rep.ortensor()
    )


def test_weighted_statistics(axes, twice):
    w, rep = twice
    a, b = axes.watson_statistics(weights=w), rep.watson_statistics()
    assert a["k"] == pytest.approx(b["k"], rel=1e-5)
    assert a["mu"].angle(b["mu"]) < 1e-6
    a, b = axes.bingham_statistics(weights=w), rep.bingham_statistics()
    assert a["k"] == pytest.approx(b["k"], rel=1e-4)
    assert a["n_eff"] < len(rep)
    assert "n_eff" not in axes.bingham_statistics()
    # fewer effective data than replicates, so a wider region
    assert all(x >= y for x, y in zip(a["gamma"], b["gamma"]))


def test_weighted_fisher_statistics():
    v = Vector3Set.random_fisher(40, rng=np.random.default_rng(5))
    w = np.ones(40)
    w[:4] = 3
    a = v.fisher_statistics(weights=w)
    b = Vector3Set(list(v) + list(v[:4]) * 2).fisher_statistics()
    assert a["mu"].angle(b["mu"]) < 1e-6
    assert a["alpha"] >= b["alpha"]
    unit = v.fisher_statistics(weights=np.ones(40))
    ref = v.fisher_statistics()
    assert unit["k"] == pytest.approx(ref["k"])
    assert unit["alpha"] == pytest.approx(ref["alpha"])


def test_weighted_bootstrap(axes):
    np.random.seed(0)
    r = axes.bootstrap_statistics(n=200, weights=np.ones(len(axes)))
    assert r["mu"].angle(axes.ortensor().eigenvectors(0)) < 1e-6
    assert 0 < r["alpha"] < 90


# Terzaghi weights


def test_terzaghi_line():
    poles = FoliationSet([fol(0, 0), fol(0, 90), fol(0, 60)])
    w = poles.terzaghi_weights(lin(0, 90), max_weight=None)
    # pole of a horizontal fracture is parallel to a vertical borehole
    assert w[0] == pytest.approx(1)
    assert np.isinf(w[1])
    assert w[2] == pytest.approx(1 / np.cos(np.radians(60)))
    assert poles.terzaghi_weights(lin(0, 90))[1] == 10


def test_terzaghi_plane_is_complementary():
    poles = FoliationSet([fol(0, 0), fol(0, 90), fol(0, 60)])
    w = poles.terzaghi_weights(fol(0, 0), max_weight=None)
    # a plane parallel to the outcrop leaves no trace
    assert np.isinf(w[0])
    assert w[1] == pytest.approx(1)
    assert w[2] == pytest.approx(1 / np.sin(np.radians(60)))


def test_terzaghi_options():
    poles = FoliationSet([fol(10, 30), fol(200, 70)])
    north = Vector3(1, 0, 0)
    with pytest.raises(ValueError):
        terzaghi_weights(poles, north)  # kind unknown for a plain vector
    with pytest.raises(ValueError):
        poles.terzaghi_weights(lin(0, 0), kind="area")
    with pytest.raises(ValueError):
        poles.terzaghi_weights(lin(0, 0), per="set")
    w = poles.terzaghi_weights(lin(0, 0), per="mean")
    assert w[0] == w[1]
    assert np.allclose(
        poles.terzaghi_weights(north, kind="line"),
        poles.terzaghi_weights(lin(0, 0)),
    )


def test_terzaghi_recovers_orientation():
    """Weighting undoes the borehole bias of a two-set sample."""
    rng = np.random.default_rng(7)
    a = FoliationSet.random_fisher(2000, mean=fol(0, 5), kappa=40, rng=rng)
    b = FoliationSet.random_fisher(2000, mean=fol(90, 85), kappa=40, rng=rng)
    both = FoliationSet(list(a) + list(b))
    bh = lin(0, 90)
    p = np.abs(np.asarray(both.normalized()) @ np.asarray(bh))
    keep = rng.uniform(size=len(both)) < p
    sample = both[np.flatnonzero(keep)]
    n_a = sum(1 for i in np.flatnonzero(keep) if i < 2000)
    n_b = keep.sum() - n_a
    assert n_b < 0.2 * n_a  # the vertical set is nearly lost
    w = sample.terzaghi_weights(bh)
    is_a = np.flatnonzero(keep) < 2000
    assert w[~is_a].mean() > 3 * w[is_a].mean()
    # weighted counts move toward the true 1:1 ratio
    assert w[~is_a].sum() / w[is_a].sum() > 5 * n_b / n_a
