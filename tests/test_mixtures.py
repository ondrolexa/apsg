"""Mixtures of Bingham distributions fitted by EM."""

import numpy as np
import pytest

from apsg import lin
from apsg.feature import (
    Bingham,
    BinghamClusters,
    BinghamMixture,
    Foliation,
    FoliationSet,
    LineationSet,
    Vector3Set,
    feature_from_json,
)
from apsg.feature._distributions import get_distribution


@pytest.fixture(scope="module")
def truth():
    return BinghamMixture(
        [
            Bingham(lin(10, 80), lin(100, 0), -8, -20),
            Bingham(lin(120, 10), lin(30, 5), -15, -25),
            Bingham(lin(250, 20), lin(340, 0), -10, -12),
        ],
        [0.4, 0.3, 0.3],
    )


@pytest.fixture(scope="module")
def data(truth):
    return truth.rvs(240, rng=np.random.default_rng(0))


@pytest.fixture(scope="module")
def fitted(data):
    return BinghamMixture.fit(data, 3, n_init=3, rng=1)


def _nearest(comp, others):
    return min(others, key=lambda o: comp.mean.angle(o.mean))


def test_recovers_components(truth, fitted):
    assert len(fitted) == 3
    assert np.isclose(fitted.weights.sum(), 1)
    # components are ordered by decreasing weight
    assert list(fitted.weights) == sorted(fitted.weights, reverse=True)
    for c, w in zip(fitted.components, fitted.weights):
        t = _nearest(c, truth.components)
        i = truth.components.index(t)
        assert c.mean.angle(t.mean) < 8
        assert abs(w - truth.weights[i]) < 0.1


def test_fit_info(fitted, data):
    info = fitted.fit_info
    assert info.method == "em"
    assert info.n == len(data)
    assert info.n_eff == pytest.approx(len(data))
    assert info.converged
    assert info.stats["converged_starts"] >= 1
    assert info.loglik == pytest.approx(fitted.loglik(data))


def test_reproducible(data):
    a = BinghamMixture.fit(data, 2, n_init=2, rng=5)
    b = BinghamMixture.fit(data, 2, n_init=2, rng=5)
    assert a == b


def test_select_picks_three(data):
    best, table = BinghamMixture.select(data, range(1, 5), n_init=3, rng=1)
    assert len(best) == 3
    assert [row["k"] for row in table] == [1, 2, 3, 4]
    bics = [row["bic"] for row in table]
    assert bics[2] == min(bics)
    assert table[2]["bic"] == pytest.approx(best.bic(data))
    aic_best, _ = BinghamMixture.select(
        data, range(2, 4), criterion="aic", n_init=2, rng=1
    )
    assert len(aic_best) in (2, 3)
    with pytest.raises(ValueError):
        BinghamMixture.select(data, criterion="icl")


def test_one_component_is_bingham(data):
    m = BinghamMixture.fit(data, 1)
    b = Bingham.fit(data)
    assert len(m) == 1 and m.weights[0] == 1
    assert m.components[0] == b
    assert m.fit_info.loglik == pytest.approx(b.fit_info.loglik)


def test_density_integrates_to_one(truth):
    rng = np.random.default_rng(3)
    X = rng.standard_normal((400000, 3))
    X /= np.linalg.norm(X, axis=1, keepdims=True)
    assert 4 * np.pi * np.mean(truth.pdf(X)) == pytest.approx(1, abs=0.03)


def test_density_is_axial_and_weighted_sum(truth):
    x = np.array([0.3, -0.5, 0.8])
    x = x / np.linalg.norm(x)
    assert truth.pdf(x) == pytest.approx(truth.pdf(-x))
    expected = sum(w * c.pdf(x) for w, c in zip(truth.weights, truth.components))
    assert truth.pdf(x) == pytest.approx(expected)


def test_responsibilities_and_labels(fitted, data):
    r = fitted.responsibilities(data)
    assert r.shape == (len(data), 3)
    assert np.allclose(r.sum(axis=1), 1)
    labels = fitted.labels(data)
    assert np.array_equal(labels, r.argmax(axis=1))
    parts = fitted.split(data)
    assert len(parts) == 3 and sum(len(p) for p in parts) == len(data)
    assert all(isinstance(p, LineationSet) for p in parts)
    # the dominant component captures about its share
    assert abs(len(parts[0]) / len(data) - fitted.weights[0]) < 0.1


def test_sampling(truth):
    s = truth.rvs(3000, rng=7)
    assert len(s) == 3000 and isinstance(s, LineationSet)
    assert np.allclose(
        np.asarray(truth.rvs(10, rng=7)), np.asarray(truth.rvs(10, rng=7))
    )
    labels = truth.labels(s)
    share = np.bincount(labels, minlength=3) / len(s)
    assert np.allclose(share, truth.weights, atol=0.06)


def test_json_round_trip(fitted):
    again = feature_from_json(fitted.to_json())
    assert isinstance(again, BinghamMixture)
    assert again == fitted
    assert again.n_params == fitted.n_params == 3 * 5 + 2


def test_registered_by_name(truth):
    assert get_distribution("bingham_mixture") is BinghamMixture
    assert get_distribution("BinghamMixture", dim=3) is BinghamMixture
    d = truth.rvs(120, rng=np.random.default_rng(8))
    m = d.fit("bingham_mixture", k=2, n_init=2, rng=0)
    assert isinstance(m, BinghamMixture) and len(m) == 2
    s = LineationSet.random(
        50, model="bingham_mixture", components=truth.components, rng=1
    )
    assert len(s) == 50
    s = LineationSet.random_bingham_mixture(50, truth.components, truth.weights, rng=1)
    assert isinstance(s, LineationSet) and len(s) == 50


def test_validation(data):
    with pytest.raises(ValueError):
        BinghamMixture([])
    with pytest.raises(TypeError):
        BinghamMixture([object()])
    with pytest.raises(ValueError):
        BinghamMixture.fit(data, 0)
    with pytest.raises(ValueError):
        BinghamMixture.fit(data[:2], 3)
    with pytest.raises(ValueError):
        BinghamMixture.fit(data, 2, weights=np.ones(3))


def test_weights_match_replication(truth):
    d = truth.rvs(150, rng=np.random.default_rng(11))
    w = np.ones(len(d))
    w[:30] = 2
    rep = LineationSet(list(d) + list(d[:30]))
    a = BinghamMixture.fit(d, 3, weights=w, n_init=3, rng=2)
    b = BinghamMixture.fit(rep, 3, n_init=3, rng=2)
    assert a.fit_info.n_eff < len(rep)
    assert np.allclose(a.weights, b.weights, atol=0.02)
    for c in a.components:
        assert c.mean.angle(_nearest(c, b.components).mean) < 2


def test_set_method(data):
    m = data.fit_mixture(3, n_init=2, rng=0)
    assert isinstance(m, BinghamMixture) and len(m) == 3
    m = data.fit_mixture(k_max=3, n_init=2, rng=0)
    assert 1 <= len(m) <= 3
    f = FoliationSet([Foliation(np.asarray(x)) for x in data])
    assert len(f.fit_mixture(2, n_init=2, rng=0)) == 2
    with pytest.raises(TypeError):
        Vector3Set.random_fisher(30).fit_mixture(2)


def test_cluster_bingham(data):
    c = data.cluster_bingham(3, n_init=2, rng=0)
    assert isinstance(c, BinghamClusters) and len(c) == 3
    assert sum(len(g) for g in c.groups) == len(data)
    assert np.array_equal(c.idx, c.mixture.labels(data))
    assert np.allclose(c.responsibilities.sum(axis=1), 1)
    assert len(c.unassigned) == 0
    assert c[0] is c.groups[0]
    assert "3 clusters" in repr(c)
    rows = c.summary()
    assert len(rows) == 3
    assert {"weight", "n", "mean", "major", "kappa_major", "kappa_minor"} <= set(
        rows[0]
    )
    assert sum(r["n"] for r in rows) == len(data)
    assert sum(r["weight"] for r in rows) == pytest.approx(1)
    assert rows[0]["ellipse"]["mu"].angle(rows[0]["mean"]) < 15
    assert len(c.R) == 3


def test_cluster_bingham_selects_k_and_weights(data):
    c = data.cluster_bingham(k_max=3, n_init=2, rng=0)
    assert 1 <= len(c) <= 3
    w = np.ones(len(data))
    w[:40] = 3
    cw = data.cluster_bingham(3, weights=w, n_init=2, rng=0)
    assert cw.mixture.fit_info.n_eff < len(data)


def test_cluster_bingham_threshold(data):
    loose = data.cluster_bingham(3, n_init=2, rng=0)
    strict = data.cluster_bingham(3, n_init=2, rng=0, threshold=0.99)
    assert len(strict.unassigned) >= 1
    assert (strict.idx == -1).sum() == len(strict.unassigned)
    assert sum(len(g) for g in strict.groups) + len(strict.unassigned) == len(data)
    assert "unassigned" in repr(strict)
    assert len(loose.unassigned) == 0
    for bad in (0, -0.1, 1.5):
        with pytest.raises(ValueError):
            data.cluster_bingham(3, threshold=bad)


def test_cluster_bingham_types():
    with pytest.raises(TypeError):
        Vector3Set.random_fisher(30).cluster_bingham(2)
    f = FoliationSet.random_bingham(60, rng=1)
    assert len(f.cluster_bingham(2, n_init=2, rng=0)) == 2


@pytest.fixture(scope="module")
def clusters(data):
    w = np.ones(len(data))
    w[:60] = 3
    return data.cluster_bingham(3, weights=w, n_init=2, rng=0)


def test_table(clusters, data):
    c = clusters
    rows = c.table()
    assert [r["set"] for r in rows] == [0, 1, 2]
    assert sum(r["weight"] for r in rows) == pytest.approx(1)
    assert sum(r["n"] for r in rows) == len(data) - len(c.unassigned)
    for r in rows:
        assert 0 < r["n_eff"] <= len(data)
        assert r["kappa_minor"] <= r["kappa_major"] <= 0
        assert 0 <= r["cone"] < 90 and r["gamma_major"] > 0
        assert r["level"] == 0.95
    # weights lower the effective size below the plain count of memberships
    plain = data.cluster_bingham(3, n_init=2, rng=0).table()
    assert sum(r["n_eff"] for r in rows) < sum(r["n_eff"] for r in plain)


def test_table_small_cluster_is_nan(data):
    c = data.cluster_bingham(3, n_init=2, rng=0, threshold=0.999)
    rows = c.table()
    small = [r for r in rows if r["n"] < 5]
    if small:  # a cluster with too few data has no ellipse
        assert all(np.isnan(r["gamma_major"]) for r in small)
    empty = BinghamClusters(data, c.mixture, threshold=1.0).table()
    assert all(np.isnan(r["gamma_minor"]) for r in empty if r["n"] < 5)
    assert all(np.isnan(r["fisher_kappa"]) for r in empty if r["n"] < 2)


def test_table_notation():
    from apsg import apsg_conf_context

    f = FoliationSet(
        [Foliation(np.asarray(x)) for x in LineationSet.random_bingham(40, rng=2)]
    )
    c = f.cluster_bingham(1)
    dd = c.table(notation="dd")[0]
    rhr = c.table(notation="rhr")[0]
    assert (dd["mean_azimuth"] - rhr["mean_azimuth"]) % 360 == pytest.approx(90)
    assert dd["mean_inclination"] == pytest.approx(rhr["mean_inclination"])
    assert dd["pole_trend"] == pytest.approx(rhr["pole_trend"])
    with apsg_conf_context(notation="rhr"):
        assert c.table()[0]["mean_azimuth"] == pytest.approx(rhr["mean_azimuth"])
        assert c.to_json()["notation"] == "rhr"
    # the temporary notation is restored
    assert c.table()[0]["mean_azimuth"] == pytest.approx(dd["mean_azimuth"])


def test_exports(clusters, tmp_path):
    import json

    import pandas as pd

    c = clusters
    frame = c.to_dataframe()
    assert list(frame.columns) == list(c.table()[0])
    assert len(frame) == 3
    c.to_csv(tmp_path / "sets.csv")
    back = pd.read_csv(tmp_path / "sets.csv")
    pd.testing.assert_frame_equal(back, frame)

    obj = c.to_json(tmp_path / "sets.json")
    assert json.loads((tmp_path / "sets.json").read_text()) == json.loads(
        json.dumps(obj)
    )
    assert obj["datatype"] == "BinghamClusters"
    assert obj["n"] == len(c.data)
    assert obj["n_eff"] < obj["n"]
    assert len(obj["sets"]) == 3 and obj["unassigned"] == 0
    assert feature_from_json(obj["mixture"]) == c.mixture
    assert all(
        v is None or not (isinstance(v, float) and np.isnan(v))
        for row in obj["sets"]
        for v in row.values()
    )
