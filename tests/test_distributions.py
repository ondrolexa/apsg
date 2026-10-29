"""
Tests of the statistics of orientation distributions.

The snapshot tests freeze the outputs of the samplers and of the statistics
methods for fixed seeds, so the planned refactoring of the distribution API
can be checked against them.
"""

import warnings

import numpy as np
import pytest
from scipy import stats

from apsg import fol, lin, linset, pair, vec, vec2, vecset
from apsg.feature import (
    Direction,
    Foliation,
    Lineation,
    Vector2,
    Vector3,
    feature_from_json,
)
from apsg.feature._container import (
    Direction2Set,
    FoliationSet,
    LineationSet,
    Vector2Set,
    Vector3Set,
)
from apsg.feature._distributions import (
    ACG,
    ESAG,
    AxialUniform2,
    AxialUniform3,
    AxialVonMises,
    Bingham,
    Cardioid,
    JonesPewsey,
    Kent,
    OrientationDistribution,
    Uniform2,
    Uniform3,
    VonMises,
    VonMisesFisher,
    Watson,
    WrappedCauchy,
    WrappedNormal,
    as_rng,
    get_distribution,
)
from apsg.feature._distributions._base import _DISTRIBUTION_REGISTRY
from apsg.feature._distributions._circular import double_angles
from apsg.plotting import RosePlot
from apsg.plotting._confidence import confidence_curve

# Snapshots ------------------------------------------------------------------


def test_snapshot_random_fisher():
    np.random.seed(7)
    d = linset.random_fisher(5, position=lin(120, 40), kappa=30)
    # scipy's sampler since 2.1.0
    expected = [
        [-0.08201306, 0.48967249, 0.86804073],
        [-0.49302955, 0.65107414, 0.5770826],
        [-0.21811803, 0.80397454, 0.55321737],
        [-0.32668279, 0.57774697, 0.7479885],
        [-0.41619153, 0.66207666, 0.62324883],
    ]
    assert np.allclose(np.array(d), expected, atol=1e-7)


def test_snapshot_random_normal():
    np.random.seed(7)
    with pytest.warns(FutureWarning, match="deprecated"):
        d = linset.random_normal(5, position=lin(120, 40), sigma=15)
    expected = [
        [-0.40185584, 0.6483937, 0.64660459],
        [0.33677883, 0.35965442, 0.8701889],
        [-0.39884725, 0.63217057, 0.66429003],
        [-0.41763382, 0.69630042, 0.58373601],
        [-0.35306356, 0.21396449, 0.91080477],
    ]
    assert np.allclose(np.array(d), expected, atol=1e-7)


def test_snapshot_random_vonmises():
    np.random.seed(7)
    d = Vector2Set.random_vonmises(5, position=30, kappa=8)
    expected = [27.5617465, 7.43919488, 39.06333172, 32.10435331, 37.04380619]
    assert np.allclose(d.direction, expected, atol=1e-6)


def test_snapshot_random_kent():
    np.random.seed(7)
    d = linset.random_kent(pair(150, 40, 150, 40), n=5, kappa=30, beta=10)
    # sampler of Kent since 2.1.0
    expected = [
        [-0.88339855, -0.1618029, 0.43980316],
        [-0.50650218, 0.37653347, 0.77567911],
        [-0.81669372, 0.26050929, 0.51492357],
        [-0.61784764, 0.41820285, 0.66586085],
        [-0.40103452, 0.58814618, 0.70232143],
    ]
    assert np.allclose(np.array(d), expected, atol=1e-7)


# random_fisher(40, position=vec(200, 30), kappa=15) with seed 11, from apsg 2.0
FISHER_SAMPLE = [
    [-0.9694825325, 0.0928113336, 0.2269133657],
    [-0.8704452191, -0.3020331830, 0.3887172199],
    [-0.7307592353, -0.2736280490, 0.6253947800],
    [-0.9750241378, -0.2215501597, 0.0156031212],
    [-0.8059997446, -0.2495990859, 0.5367165993],
    [-0.6798419965, -0.5274397044, 0.5095313710],
    [-0.8436290481, -0.2926880067, 0.4501374900],
    [-0.8183450988, -0.4322224155, 0.3788074481],
    [-0.9838832145, 0.0118382625, 0.1784199421],
    [-0.3638358761, -0.7596793541, 0.5389904770],
    [-0.7415160805, -0.0962892687, 0.6639896679],
    [-0.9384978925, -0.2810717820, 0.2005501411],
    [-0.7920315347, 0.2400179836, 0.5613175710],
    [-0.7587506322, -0.3808000137, 0.5284778404],
    [-0.7856586727, -0.4749392604, 0.3964506892],
    [-0.7660680853, -0.2586727669, 0.5884114957],
    [-0.9696125724, -0.2206043958, 0.1057599170],
    [-0.2820280104, -0.7963923891, 0.5349947327],
    [-0.5998669415, -0.1786720821, 0.7798948259],
    [-0.3832515295, -0.5454087568, 0.7454177038],
    [-0.6969041981, -0.0451849488, 0.7157393793],
    [-0.6068755707, -0.2140693136, 0.7654256141],
    [-0.9890394637, -0.0509434117, 0.1385846600],
    [-0.9315678501, -0.3564973549, 0.0713510799],
    [-0.8613553489, -0.0899538136, 0.4999752738],
    [-0.5747147075, -0.6362584991, 0.5146631202],
    [-0.3852141618, -0.2095645552, 0.8987172785],
    [-0.5301799715, -0.5442066491, 0.6501909880],
    [-0.9114200701, -0.1357380258, 0.3884438752],
    [-0.9413661465, -0.2162234204, 0.2589926848],
    [-0.8820156535, 0.3078658622, 0.3567450039],
    [-0.7236109119, -0.1362136776, 0.6766336396],
    [-0.9306884776, -0.1426012126, 0.3368736437],
    [-0.8958362623, -0.2115774242, 0.3907843199],
    [-0.9149661725, 0.1022032961, 0.3903733974],
    [-0.6630259810, -0.7387582850, 0.1209658828],
    [-0.6419681501, -0.2915495114, 0.7091373468],
    [-0.6104641674, -0.7176538841, 0.3351214749],
    [-0.8845064804, -0.1044444943, 0.4546863026],
    [-0.8260581402, -0.3178170489, 0.4654248300],
]


@pytest.fixture
def fisher_sample():
    return vecset.from_xyz(*np.array(FISHER_SAMPLE).T)


def test_snapshot_statistics(fisher_sample):
    d = fisher_sample
    L = d.to_lin()
    mu = [-0.82306889, -0.28902261, 0.48890034]
    with pytest.warns(FutureWarning, match="fisher_statistics"):
        lin_stats = L.fisher_statistics()
    for s in (d.fisher_statistics(), lin_stats):
        assert np.allclose(s["mu"], mu, atol=1e-7)
        # Fisher's precision since 2.1.0, the maximum-likelihood kappa before
        assert np.isclose(s["k"], 13.03567314)
        assert np.isclose(s["kappa"], 13.36992117)
        assert np.isclose(s["alpha"], 6.513292)
    axis = [-0.82648815, -0.28251089, 0.48693422]
    s = L.watson_statistics()
    assert np.allclose(s["mu"], axis, atol=1e-7)
    assert np.isclose(s["k"], 7.92720421)
    # cone of Fisher, Lewis & Embleton (1987) eq. 6.13 since 2.1.0
    assert np.isclose(s["alpha"], 6.48739747)
    s = L.bingham_statistics()
    assert np.allclose(s["mu"], axis, atol=1e-7)
    # large-sample eigenvector ellipse since 2.1.0 (was 20.06, 20.91)
    assert np.allclose(s["gamma"], [7.36256505, 5.47678308])
    assert np.isclose(d.uniformity_test().statistic, 102.72056908)
    # Bingham test since 2.1.0
    assert np.isclose(L.uniformity_test().statistic, 125.33462399)
    assert np.isclose(d.csd(), 22.43459779)
    assert np.isclose(d.var(), 0.07479476)
    assert np.isclose(d.delta(), 22.30067569)


@pytest.mark.parametrize(
    "method, total",
    [
        ("fisher", [-294.33999386, -103.35819459, 174.96706708]),
        ("watson", [-295.57847256, -101.03488949, 174.27253664]),
        ("bingham", [-148.74020817, -50.72509423, 87.53757821]),
    ],
)
def test_snapshot_confidence_curves(fisher_sample, method, total):
    if method == "fisher":
        with pytest.warns(FutureWarning, match="fisher_statistics"):
            curve = confidence_curve(fisher_sample.to_lin(), method=method)
    else:
        curve = confidence_curve(fisher_sample.to_lin(), method=method)
    assert np.allclose(np.nansum(curve, axis=0), total, atol=1e-6)


# Circular data ----------------------------------------------------------------


def test_vonmises_interval_is_upton_approximation():
    # N = 20 directions with R = 15 (Rbar = 0.75 <= 0.9)
    N, R = 20, 15.0
    a = np.degrees(np.arccos(R / N))
    v = Vector2Set.from_directions([a, -a] * (N // 2))
    chi2 = 3.841458820694124
    cos_d = np.sqrt(2 * N * (2 * R**2 - N * chi2) / (4 * N - chi2)) / R
    assert np.isclose(v.fisher_statistics()["alpha"], np.degrees(np.arccos(cos_d)))


def test_vonmises_interval_too_dispersed_is_full_circle():
    v = Vector2Set.from_directions([0, 90, 180, 270, 10])
    assert v.fisher_statistics()["alpha"] == 180.0


def test_vonmises_interval_coverage():
    rng = np.random.default_rng(5)
    n_trials = 1000
    covered = 0
    for _ in range(n_trials):
        v = Vector2Set.from_directions(np.degrees(rng.vonmises(0, 10, 20)))
        s = v.fisher_statistics()
        covered += abs((s["mu"].direction + 180) % 360 - 180) <= s["alpha"]
    assert 0.93 <= covered / n_trials <= 0.97


def test_circular_standard_deviation():
    np.random.seed(3)
    v = Vector2Set.random_vonmises(200, position=40, kappa=4)
    rbar = abs(v.normalized().R(mean=True))
    assert np.isclose(v.csd(), np.degrees(np.sqrt(-2 * np.log(rbar))))


def test_axial_mean_uses_doubled_angles():
    rng = np.random.default_rng(8)
    a = rng.uniform(0, 180, 30)
    d = Direction2Set.from_directions(a)
    doubled = Vector2Set.from_directions(2 * a).R()
    assert np.isclose(d.R().direction % 180, (doubled.direction / 2) % 180)
    assert np.isclose(abs(d.R()), abs(doubled))


def test_axial_statistics_do_not_depend_on_sign():
    rng = np.random.default_rng(9)
    a = np.degrees(rng.vonmises(np.radians(60), 6, 25)) / 2 + 30
    flipped = a + 180 * rng.integers(0, 2, len(a))
    s1 = Direction2Set.from_directions(a).fisher_statistics()
    s2 = Direction2Set.from_directions(flipped).fisher_statistics()
    assert np.isclose(s1["mu"].direction % 180, s2["mu"].direction % 180)
    assert np.isclose(s1["k"], s2["k"])
    assert np.isclose(s1["alpha"], s2["alpha"])
    assert s1["alpha"] <= 90


def test_axial_csd_and_delta_are_halved():
    rng = np.random.default_rng(10)
    a = rng.uniform(0, 60, 30)
    d = Direction2Set.from_directions(a)
    doubled = Vector2Set.from_directions(2 * a)
    assert np.isclose(d.csd(), doubled.csd() / 2)
    assert np.isclose(d.delta(), doubled.delta() / 2)


def test_muci_axial_bootstrap_interval():
    np.random.seed(5)
    d = Direction2Set.from_directions([10, 20, 30, 15, 25, 190])
    stats = d.bootstrap_statistics(level=0.95, n=200)
    # 190 is the axis 10, so the mean axis lies between 10 and 30
    assert 10 < stats["mu"].direction < 30
    assert stats["low"] < stats["mu"].direction < stats["high"]
    r = RosePlot()
    r.muci(d, method="bootstrap", level=0.95, n_resamples=50)
    r.init_figure()
    r._render()


# Watson -----------------------------------------------------------------------


def test_watson_girdle():
    np.random.seed(2)
    azi = np.random.uniform(0, 360, 200)
    g = linset([lin(az, abs(np.random.normal(0, 5))) for az in azi])
    s = g.watson_statistics()
    assert s["k"] < 0
    assert abs(s["mu"].normalized()[2]) > 0.99
    assert 0 < s["alpha"] < 5


def test_watson_bipolar_kappa_solves_likelihood_equation(fisher_sample):
    from scipy.integrate import quad

    L = fisher_sample.to_lin()
    k = L.watson_statistics()["k"]
    num = quad(lambda u: u**2 * np.exp(k * (u**2 - 1)), 0, 1)[0]
    den = quad(lambda u: np.exp(k * (u**2 - 1)), 0, 1)[0]
    assert np.isclose(num / den, L.ortensor().eigenvalues(0))


def test_watson_uniform_data_have_small_kappa():
    np.random.seed(4)
    s = linset.random(500).watson_statistics()
    assert abs(s["k"]) < 1


# Kent -------------------------------------------------------------------------


def test_random_kent_beta():
    p = pair(150, 40, 150, 40)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        linset.random_kent(p, n=5, kappa=20)
    with pytest.warns(UserWarning, match="unimodal"):
        linset.random_kent(p, n=5, kappa=20, beta=10)
    with pytest.raises(ValueError):
        linset.random_kent(p, n=5, kappa=20, beta=-1)


# Common interface -------------------------------------------------------------
#
# Every registered distribution needs an entry in EXAMPLES: instances used by
# the shared tests and, for the goodness-of-fit test, a function mapping a
# sample array to values with a known CDF.


def _angle_gof(dist, X):
    return np.arctan2(X[:, 1], X[:, 0]) % (2 * np.pi), stats.uniform(0, 2 * np.pi).cdf


def _z_gof(dist, X):
    return X[:, 2], stats.uniform(-1, 2).cdf


def _vonmises_gof(dist, X):
    mean = np.asarray(dist.mean, dtype=float)
    if dist.axial:
        X, mean = double_angles(X), double_angles(mean[None, :])[0]
    theta = np.arctan2(X[:, 1], X[:, 0]) - np.arctan2(mean[1], mean[0])
    theta = (theta + np.pi) % (2 * np.pi) - np.pi
    return theta, stats.vonmises(dist.kappa).cdf


def _wrapped_normal_gof(dist, X):
    mean = np.asarray(dist.mean, dtype=float)
    theta = np.arctan2(X[:, 1], X[:, 0]) - np.arctan2(mean[1], mean[0])
    theta = (theta + np.pi) % (2 * np.pi) - np.pi
    k = np.arange(-20, 21)

    def cdf(t):
        t = np.asarray(t)[..., None]
        upper = stats.norm.cdf((t + 2 * np.pi * k) / dist.sigma)
        lower = stats.norm.cdf((-np.pi + 2 * np.pi * k) / dist.sigma)
        return (upper - lower).sum(axis=-1)

    return theta, cdf


def _numeric_cdf_gof(dist, X):
    # CDF of the angle from the mean by quadrature of the density
    mean = np.asarray(dist.mean, dtype=float)
    theta = np.arctan2(X[:, 1], X[:, 0]) - np.arctan2(mean[1], mean[0])
    theta = (theta + np.pi) % (2 * np.pi) - np.pi
    grid = np.linspace(-np.pi, np.pi, 20001)
    pts = np.column_stack([np.cos(grid), np.sin(grid)]) @ _rotation_to(mean).T
    pdf = dist.pdf(pts)
    cdf = np.concatenate([[0], np.cumsum((pdf[1:] + pdf[:-1]) / 2 * np.diff(grid))])
    return theta, lambda t: np.interp(t, grid, cdf / cdf[-1])


def _rotation_to(mean):
    c, s = mean / np.linalg.norm(mean)
    return np.array([[c, -s], [s, c]])


_SPHERE_GRID = np.array(Vector3Set.sfs(200000))


def _mean_marginal_gof(dist, X):
    """``u = mean.x`` with its CDF integrated numerically from the density."""
    mean = np.asarray(dist.mean, dtype=float)
    u_grid = _SPHERE_GRID @ mean
    order = np.argsort(u_grid)
    mass = np.cumsum(dist.pdf(_SPHERE_GRID)[order]) * 4 * np.pi / len(_SPHERE_GRID)

    def cdf(t):
        return np.interp(t, u_grid[order], mass, left=0, right=1)

    return X @ mean, cdf


EXAMPLES = {
    Uniform2: ([Uniform2()], _angle_gof),
    AxialUniform2: ([AxialUniform2()], _angle_gof),
    Uniform3: ([Uniform3()], _z_gof),
    AxialUniform3: ([AxialUniform3()], _z_gof),
    VonMises: ([VonMises(30, 4), VonMises(250, 1.5)], _vonmises_gof),
    WrappedNormal: (
        [WrappedNormal(30, 0.8), WrappedNormal(250, 0.3)],
        _wrapped_normal_gof,
    ),
    AxialVonMises: ([AxialVonMises(120, 3), AxialVonMises(10, 1.5)], _vonmises_gof),
    WrappedCauchy: (
        [WrappedCauchy(30, 0.8), WrappedCauchy(250, 0.4)],
        _numeric_cdf_gof,
    ),
    Cardioid: ([Cardioid(30, 0.4), Cardioid(250, 0.3)], _numeric_cdf_gof),
    JonesPewsey: (
        [JonesPewsey(30, 2, 0.5), JonesPewsey(250, 4, -0.8)],
        _numeric_cdf_gof,
    ),
    VonMisesFisher: (
        [VonMisesFisher((1, 2, 3), 15), VonMisesFisher((0, 0, -1), 2.5)],
        _mean_marginal_gof,
    ),
    Kent: (
        [Kent((0, 0, 1), (1, 0, 0), 20, 8), Kent((1, 1, 0), (0, 0, 1), 5, 2)],
        _mean_marginal_gof,
    ),
    ESAG: (
        [ESAG((1, 0, 2), (0, 1, 0), 0.4), ESAG((0, -3, 4), (1, 0, 0), 0.7)],
        _mean_marginal_gof,
    ),
    Watson: ([Watson((1, 2, 3), 8), Watson((0, 0, 1), -10)], _mean_marginal_gof),
    Bingham: (
        [
            Bingham((0, 0, 1), (1, 0, 0), -3, -12),
            Bingham((1, 1, 0), (0, 0, 1), -2, -30),
        ],
        _mean_marginal_gof,
    ),
    ACG: (
        [
            ACG((0, 0, 1), (1, 0, 0), (2, 0.6, 0.4)),
            ACG((1, 0, 1), (0, 1, 0), (2.5, 0.4, 0.1)),
        ],
        _mean_marginal_gof,
    ),
}

INSTANCES = [d for examples, _ in EXAMPLES.values() for d in examples]


def _rotation(dim, seed=0):
    rng = np.random.default_rng(seed)
    Q, R = np.linalg.qr(rng.standard_normal((dim, dim)))
    Q = Q * np.sign(np.diag(R))
    return Q if np.linalg.det(Q) > 0 else -Q


def _grid(dim, n=20000):
    """Quadrature points and weight for integrating over the circle or sphere."""
    if dim == 2:
        theta = 2 * np.pi * (np.arange(n) + 0.5) / n
        return np.column_stack([np.cos(theta), np.sin(theta)]), 2 * np.pi / n
    return np.array(Vector3Set.sfs(n)), 4 * np.pi / n


def test_every_registered_distribution_has_examples():
    # distributions of vectors and axes; pairs (Lisle) have their own tests
    vectors = {c for c in _DISTRIBUTION_REGISTRY.values() if c.dim in (2, 3)}
    assert vectors == set(EXAMPLES)


@pytest.mark.parametrize("dist", INSTANCES, ids=repr)
def test_pdf_integrates_to_one(dist):
    X, w = _grid(dist.dim)
    assert np.isclose(dist.pdf(X).sum() * w, 1, rtol=1e-3)


@pytest.mark.parametrize("dist", [d for d in INSTANCES if d.axial], ids=repr)
def test_axial_pdf_is_symmetric(dist):
    X = as_rng(1).standard_normal((50, dist.dim))
    assert np.allclose(dist.logpdf(X), dist.logpdf(-X))


@pytest.mark.parametrize("dist", INSTANCES, ids=repr)
def test_fit_recovers_parameters(dist):
    fitted = type(dist).fit(dist.rvs(5000, rng=2))
    assert fitted.fit_info.n == 5000
    assert np.isclose(fitted.fit_info.loglik, fitted.loglik(dist.rvs(5000, rng=2)))
    for key, value in dist.params.items():
        if isinstance(value, Vector2 | Vector3):
            angle = value.angle(fitted.params[key])
            if key == "major":  # an axis, its sign is arbitrary
                angle = min(angle, 180 - angle)
            assert angle < 3
            assert np.isclose(abs(fitted.params[key]), abs(value), rtol=0.1)
        else:
            assert np.allclose(fitted.params[key], value, rtol=0.1)


@pytest.mark.parametrize("dist", INSTANCES, ids=repr)
def test_fit_is_rotation_equivariant(dist):
    X = dist.rvs(500, rng=3)
    Q = _rotation(dist.dim)
    fitted = type(dist).fit(X)
    rotated = type(dist).fit(np.array(X) @ Q.T)
    assert np.isclose(fitted.fit_info.loglik, rotated.fit_info.loglik)
    if fitted.mean is not None:
        expected = Q @ np.asarray(fitted.mean, dtype=float)
        assert abs(np.dot(expected, rotated.mean)) > 1 - 1e-8


@pytest.mark.parametrize("dist", INSTANCES, ids=repr)
def test_rvs_goodness_of_fit(dist):
    values, cdf = EXAMPLES[type(dist)][1](dist, np.array(dist.rvs(2000, rng=4)))
    assert stats.kstest(values, cdf).pvalue > 0.01


@pytest.mark.parametrize("dist", INSTANCES, ids=repr)
def test_rvs_reproducibility(dist):
    assert np.allclose(dist.rvs(5, rng=7), dist.rvs(5, rng=7))
    np.random.seed(7)
    first = np.array(dist.rvs(5))
    np.random.seed(7)
    assert np.allclose(dist.rvs(5), first)


@pytest.mark.parametrize("dist", INSTANCES, ids=repr)
def test_json_round_trip(dist):
    assert feature_from_json(dist.to_json()) == dist


@pytest.mark.parametrize("dist", INSTANCES, ids=repr)
def test_single_and_set_evaluation(dist):
    sample = dist.rvs(4, rng=5)
    values = dist.pdf(sample)
    assert values.shape == (4,)
    assert isinstance(dist.pdf(sample[0]), float)
    assert np.isclose(dist.pdf(sample[0]), values[0])
    assert np.isclose(dist.loglik(sample), np.log(values).sum())


def test_rvs_feature_types():
    assert isinstance(Uniform2().rvs(3), Vector2Set)
    assert isinstance(AxialUniform2().rvs(3), Direction2Set)
    assert isinstance(Uniform3().rvs(3), Vector3Set)
    assert isinstance(AxialUniform3().rvs(3), LineationSet)
    assert isinstance(AxialUniform3().rvs(3, feature=Foliation), FoliationSet)
    assert isinstance(Uniform3().rvs(3, feature=Lineation), LineationSet)
    with pytest.raises(TypeError):
        AxialUniform3().rvs(3, feature=Vector3)
    with pytest.raises(TypeError):
        AxialUniform3().rvs(3, feature=Direction)


def test_axial_samples_have_random_sign():
    X = np.array(AxialUniform3().rvs(2000, rng=6))
    assert abs(np.mean(X[:, 2] > 0) - 0.5) < 0.05


def test_inputs():
    d = AxialUniform3()
    assert np.isclose(d.pdf(lin(120, 30)), 1 / (4 * np.pi))
    assert np.isclose(d.pdf(fol(120, 30)), 1 / (4 * np.pi))
    assert np.isclose(d.pdf([0, 0, 5]), 1 / (4 * np.pi))
    assert np.isclose(Uniform2().pdf(vec2(30)), 1 / (2 * np.pi))
    with pytest.raises(ValueError):
        d.pdf(vec2(30))
    with pytest.raises(ValueError):
        d.pdf([0, 0, 0])


def test_information_criteria():
    data = AxialUniform3().rvs(50, rng=1)
    d = AxialUniform3.fit(data)
    assert np.isclose(d.aic(data), -2 * d.loglik(data))
    assert np.isclose(d.bic(data), -2 * d.loglik(data))


def test_mean_confidence_needs_fit():
    with pytest.raises(ValueError):
        Uniform3().mean_confidence()
    with pytest.raises(NotImplementedError):
        Uniform3.fit(Uniform3().rvs(10)).mean_confidence()


def test_get_distribution():
    assert get_distribution("Uniform3") is Uniform3
    assert get_distribution("uniform", dim=2) is Uniform2
    assert get_distribution("axial_uniform", dim=3) is AxialUniform3
    assert get_distribution(Uniform2) is Uniform2
    with pytest.raises(ValueError, match="ambiguous"):
        get_distribution("uniform")
    with pytest.raises(ValueError, match="Unknown"):
        get_distribution("nonsense")


def test_distributions_are_orientation_distributions():
    assert all(issubclass(c, OrientationDistribution) for c in EXAMPLES)


# Circle models ----------------------------------------------------------------


def test_vonmises_logpdf_matches_scipy():
    d = VonMises(250, 3.5)
    theta = np.linspace(-np.pi, np.pi, 50)
    X = np.column_stack([np.cos(theta), np.sin(theta)])
    expected = stats.vonmises.logpdf(theta, 3.5, loc=np.radians(250))
    assert np.allclose(d.logpdf(X), expected)


def test_axial_vonmises_is_vonmises_of_doubled_angles():
    d = AxialVonMises(70, 2)
    theta = np.linspace(0, 2 * np.pi, 50)
    X = np.column_stack([np.cos(theta), np.sin(theta)])
    expected = stats.vonmises.logpdf(2 * theta, 2, loc=np.radians(140))
    assert np.allclose(d.logpdf(X), expected)


def test_vonmises_mle_matches_scipy():
    rng = np.random.default_rng(12)
    theta = stats.vonmises.rvs(2.5, loc=1.0, size=300, random_state=rng)
    d = VonMises.fit(np.column_stack([np.cos(theta), np.sin(theta)]))
    kappa, loc, _ = stats.vonmises.fit(theta, fscale=1)
    assert np.isclose(d.kappa, kappa, rtol=1e-4)
    assert np.isclose(np.radians(d.mean.direction), loc % (2 * np.pi), atol=1e-4)


def test_vonmises_fit_methods():
    data = VonMises(30, 4).rvs(500, rng=13)
    mle = VonMises.fit(data)
    approx = VonMises.fit(data, method="approx")
    assert mle.fit_info.loglik >= approx.fit_info.loglik
    assert np.isclose(mle.kappa, approx.kappa, rtol=0.02)
    with pytest.raises(ValueError):
        VonMises.fit(data, method="nonsense")


def test_vonmises_mean_confidence_is_upton_interval():
    data = VonMises(30, 4).rvs(40, rng=14)
    d = VonMises.fit(data)
    region = d.mean_confidence(0.9)
    assert region.mu == d.mean
    assert np.isclose(region.alpha, data.fisher_statistics(level=0.9)["alpha"])
    axial = AxialVonMises.fit(Direction2Set.from_directions(data.direction / 2))
    assert np.isclose(axial.mean_confidence(0.9).alpha, region.alpha / 2)


def test_fisher_statistics_2d_wraps_vonmises_fit():
    data = VonMises(200, 2).rvs(30, rng=15)
    s = data.fisher_statistics()
    d = VonMises.fit(data, method="approx")
    assert s["mu"] == d.mean
    assert s["k"] == d.kappa
    axial = Direction2Set.from_directions(data.direction)
    s = axial.fisher_statistics()
    d = AxialVonMises.fit(axial, method="approx")
    assert isinstance(s["mu"], Direction)
    assert s["mu"] == d.mean
    assert s["k"] == d.kappa


def test_fisher_statistics_2d_identical_data():
    # the concentration of identical data is infinite, so it can't be estimated
    with pytest.raises(ValueError, match="identical"):
        Vector2Set.from_directions([40, 40, 40]).fisher_statistics()


def test_random_vonmises_direction2set():
    np.random.seed(16)
    d = Direction2Set.random_vonmises(5, position=30, kappa=8)
    assert isinstance(d, Direction2Set)
    np.random.seed(16)
    v = Vector2Set.random_vonmises(5, position=30, kappa=8)
    assert np.allclose(np.array(d), np.array(v))


def test_vonmises_kde():
    from apsg.feature._distributions import vonmises_kde

    theta = np.linspace(-np.pi, np.pi, 7)
    centers = np.array([0.3, 2.0])
    expected = stats.vonmises.pdf(theta[:, None], 5, loc=centers).mean(axis=1)
    assert np.allclose(vonmises_kde(theta, centers, 5), expected)
    axial = vonmises_kde(theta, centers, 5, axial=True)
    assert np.allclose(axial, vonmises_kde(theta + np.pi, centers, 5, axial=True))


def test_wrapped_normal_density_is_wrapped_sum():
    theta = np.linspace(-np.pi, np.pi, 25)
    X = np.column_stack([np.cos(theta), np.sin(theta)])
    k = np.arange(-50, 51)
    for rho in (0.95, 0.5, 0.05):
        d = WrappedNormal(0, rho)
        z = (theta[:, None] + 2 * np.pi * k) / d.sigma
        expected = np.exp(-(z**2) / 2).sum(axis=1) / (d.sigma * np.sqrt(2 * np.pi))
        assert np.allclose(d.pdf(X), expected)
    assert np.allclose(WrappedNormal(0, 0).pdf(X), 1 / (2 * np.pi))


def test_wrapped_normal_fit_methods():
    data = WrappedNormal(80, 0.6).rvs(400, rng=17)
    moments = WrappedNormal.fit(data, method="moments")
    assert np.isclose(moments.rho, abs(data.normalized().R(mean=True)))
    assert np.isclose(moments.mean.direction, data.R().direction)
    mle = WrappedNormal.fit(data)
    assert mle.fit_info.converged
    assert mle.fit_info.loglik >= moments.fit_info.loglik
    assert np.isclose(mle.rho, moments.rho, atol=0.02)
    assert np.isclose(WrappedNormal(0, 0.5).sigma, np.sqrt(-2 * np.log(0.5)))
    with pytest.raises(ValueError):
        WrappedNormal(0, 1)


def test_wrapped_normal_mean_confidence_is_dispersion_interval():
    data = WrappedNormal(80, 0.7).rvs(60, rng=18)
    d = WrappedNormal.fit(data, method="moments")
    theta = np.radians(data.direction - d.mean.direction)
    rbar = abs(data.normalized().R(mean=True))
    delta = (1 - np.mean(np.cos(2 * theta))) / (2 * rbar**2)
    expected = np.degrees(np.arcsin(stats.norm.ppf(0.975) * np.sqrt(delta / 60)))
    region = d.mean_confidence()
    assert region.mu == d.mean
    assert np.isclose(region.alpha, expected)


def test_wrapped_normal_mean_confidence_coverage():
    rng = np.random.default_rng(19)
    n_trials = 500
    covered = 0
    for _ in range(n_trials):
        d = WrappedNormal.fit(WrappedNormal(0, 0.6).rvs(40, rng=rng))
        covered += (
            abs((d.mean.direction + 180) % 360 - 180) <= d.mean_confidence().alpha
        )
    assert 0.92 <= covered / n_trials <= 0.98


# Sphere, vectorial ------------------------------------------------------------


def test_fisher_logpdf_matches_scipy():
    d = VonMisesFisher((1, 2, -1), 7.5)
    X = Uniform3().rvs_array(20, as_rng(3))
    mean = np.asarray(d.mean)
    assert np.allclose(d.logpdf(X), stats.vonmises_fisher(mean, 7.5).logpdf(X))
    big = VonMisesFisher((0, 0, 1), 5000)
    assert np.isfinite(big.logpdf([0, 0, 1]))


def test_fisher_mle_matches_scipy():
    X = np.array(VonMisesFisher((1, 0, 1), 6).rvs(300, rng=20))
    d = VonMisesFisher.fit(X)
    mu, kappa = stats.vonmises_fisher.fit(X)
    assert np.isclose(d.kappa, kappa, rtol=1e-6)
    assert np.allclose(np.asarray(d.mean), mu)


def test_fisher_statistics_wraps_fisher_fit(fisher_sample):
    s = fisher_sample.fisher_statistics(level=0.9)
    d = VonMisesFisher.fit(fisher_sample)
    assert s["mu"] == d.mean
    assert s["kappa"] == d.kappa
    assert s["k"] == VonMisesFisher.fit(fisher_sample, method="precision").kappa
    assert s["alpha"] == d.mean_confidence(0.9).alpha


def test_fisher_statistics_identical_data():
    with pytest.raises(ValueError, match="identical"):
        vecset([vec(120, 30)] * 3).fisher_statistics()


def test_kent_normalizer_matches_series():
    from apsg.feature._distributions._special import kent_log_norm

    # log c(kappa, beta) from Kent's (1982) Bessel series
    series = [
        (10, 0, 9.535291971354146),
        (10, 3, 9.67553142435901),
        (30, 14, 29.0104941094839),
        (2, 0.5, 3.150465490010348),
        (5, 4, 5.965315094850729),
        (200, 60, 196.7563578252084),
    ]
    for kappa, beta, expected in series:
        assert np.isclose(kent_log_norm(kappa, beta), expected, rtol=1e-10)
    assert np.isfinite(kent_log_norm(5000, 100))


def test_kent_moment_estimates():
    X = np.array(Kent((0, 0, 1), (1, 0, 0), 20, 8).rvs(500, rng=21))
    d = Kent.fit(X, method="moments")
    rbar = np.linalg.norm(X.mean(axis=0))
    _, g2, g3 = (np.asarray(a) for a in d.axes)
    B = np.array([[np.mean((X @ a) * (X @ b)) for b in (g2, g3)] for a in (g2, g3)])
    assert np.isclose(B[0, 1], 0, atol=1e-12)
    r2 = B[0, 0] - B[1, 1]
    a, b = 2 - 2 * rbar - r2, 2 - 2 * rbar + r2
    assert np.isclose(d.kappa, 1 / a + 1 / b)
    assert np.isclose(d.beta, (1 / a - 1 / b) / 2)
    mle = Kent.fit(X)
    assert mle.fit_info.loglik >= d.fit_info.loglik
    assert 2 * mle.beta < mle.kappa


def test_kent_concentrated_sampling_is_fast():
    # nearly normal in the tangent plane, variances 1/(kappa -+ 2 beta)
    d = Kent((0, 0, 1), (1, 0, 0), 2000, 900)
    X = np.array(d.rvs(4000, rng=22))
    assert np.isclose(np.var(X[:, 0]), 1 / 200, rtol=0.1)
    assert np.isclose(np.var(X[:, 1]), 1 / 3800, rtol=0.1)


def test_random_kent_wraps_kent():
    p = pair(150, 40, 150, 40)
    np.random.seed(23)
    a = np.array(linset.random_kent(p, n=10, kappa=30, beta=10))
    d = Kent(p.lvec, p.fvec.cross(p.lvec), 30, 10)
    np.random.seed(23)
    assert np.allclose(a, d.rvs_array(10, np.random.mtrand._rand))


def test_esag_density_is_projected_normal():
    from scipy.integrate import quad

    d = ESAG((1, -0.5, 2), (0, 1, 0), 0.5)
    V = np.linalg.inv(d._Vinv)
    mvn = stats.multivariate_normal(np.asarray(d.params["mu"]), V)
    for x in Uniform3().rvs_array(5, as_rng(24)):
        radial, _ = quad(lambda r, x=x: r**2 * mvn.pdf(r * x), 0, np.inf)
        assert np.isclose(d.pdf(x), radial)


def test_esag_isotropic_has_rho_one():
    d = ESAG((0, 0, 3), (1, 0, 0), 1)
    X = Uniform3().rvs_array(10, as_rng(25))
    assert np.allclose(d._Vinv, np.eye(3))
    other = ESAG((0, 0, 3), (0, 1, 0), 1)
    assert np.allclose(d.logpdf(X), other.logpdf(X))


def test_mean_direction_ellipse():
    X = np.array(Kent((0, 0, 1), (1, 0, 0), 20, 8).rvs(80, rng=26))
    region = Kent.fit(X).mean_confidence(0.9)
    assert region.kind == "ellipse"
    rbar = np.linalg.norm(X.mean(axis=0))
    chi2 = stats.chi2.ppf(0.9, 2)
    for axis, gamma in zip(region.axes, region.gamma, strict=True):
        moment = np.mean((X @ np.asarray(axis)) ** 2)
        expected = np.degrees(np.arcsin(np.sqrt(chi2 * moment / (80 * rbar**2))))
        assert np.isclose(gamma, expected)
    assert region.gamma[0] > region.gamma[1]
    esag = ESAG.fit(X).mean_confidence(0.9)
    assert np.allclose(esag.gamma, region.gamma)


# Sphere, axial ----------------------------------------------------------------


def test_watson_normalizer_is_kummer_function():
    from scipy.special import hyp1f1

    X = Uniform3().rvs_array(10, as_rng(30))
    for kappa in (-20, -1, 0.5, 15):
        d = Watson((0, 1, 1), kappa)
        u = X @ np.asarray(d.mean)
        expected = np.exp(kappa * u**2) / (4 * np.pi * hyp1f1(0.5, 1.5, kappa))
        assert np.allclose(d.pdf(X), expected)


def test_watson_is_bingham_with_equal_concentrations():
    X = Uniform3().rvs_array(10, as_rng(31))
    bipolar = Bingham((0, 0, 1), (1, 0, 0), -6, -6)
    assert np.allclose(Watson((0, 0, 1), 6).logpdf(X), bipolar.logpdf(X))
    girdle = Bingham((1, 0, 0), (0, 1, 0), 0, -6)
    assert np.allclose(Watson((0, 0, 1), -6).logpdf(X), girdle.logpdf(X))


def test_watson_statistics_wraps_watson_fit(fisher_sample):
    L = fisher_sample.to_lin()
    s = L.watson_statistics(level=0.9)
    d = Watson.fit(L)
    assert s["k"] == d.kappa
    assert s["alpha"] == d.mean_confidence(0.9).alpha
    assert s["mu"] == Lineation(d.mean)


def test_bingham_fit_matches_moments():
    from apsg.feature._distributions._special import bingham_moments

    X = np.array(Bingham((0, 0, 1), (1, 0, 0), -3, -12).rvs(400, rng=32))
    d = Bingham.fit(X)
    taus = d.fit_info.stats["eigenvalues"]
    moments = bingham_moments(d.kappa_major, d.kappa_minor)
    assert np.allclose(moments, taus[1:], atol=1e-6)


def test_bingham_sample_moments():
    from apsg.feature._distributions._special import bingham_moments

    d = Bingham((0, 0, 1), (1, 0, 0), -2, -30)
    X = np.array(d.rvs(20000, rng=33))
    assert np.allclose(np.mean(X**2, axis=0)[:2], bingham_moments(-2, -30), rtol=0.03)


def test_bingham_envelope_bounds_density():
    from scipy.optimize import brentq

    # exp(-x'Ax) <= M (x'Omega x)^(-3/2) on the whole sphere
    X = np.array(Vector3Set.sfs(20000))
    for lam in ([0, 2, 30], [0, 0.1, 0.2], [0, 50, 400]):
        lam = np.array(lam, dtype=float)
        b = brentq(lambda b, lam=lam: np.sum(1 / (b + 2 * lam)) - 1, 1e-12, 3)
        omega = 1 + 2 * lam / b
        log_m = -(3 - b) / 2 + 1.5 * np.log(3 / b)
        log_ratio = -(X**2 @ lam) + 1.5 * np.log(X**2 @ omega) - log_m
        assert log_ratio.max() <= 1e-12


def test_bingham_statistics_wraps_bingham():
    np.random.seed(34)
    L = linset.random_fisher(60, position=lin(120, 40), kappa=10)
    s = L.bingham_statistics(level=0.9)
    d = Bingham.fit(L)
    region = d.mean_confidence(0.9)
    assert np.allclose(s["gamma"], region.gamma)
    assert s["k"] == (d.kappa_major, d.kappa_minor)
    with pytest.raises(ValueError):
        L.bingham_statistics(which=3)


def test_acg_density_is_projected_normal():
    from scipy.integrate import quad

    d = ACG((1, 0, 1), (0, 1, 0), (2.5, 0.4, 0.1))
    mvn = stats.multivariate_normal(np.zeros(3), d.matrix)
    for x in Uniform3().rvs_array(5, as_rng(35)):
        radial, _ = quad(lambda r, x=x: r**2 * mvn.pdf(r * x), 0, np.inf)
        # the projected normal density of the axis, both signs
        assert np.isclose(d.pdf(x), radial)


def test_acg_fit_is_tyler_fixed_point():
    X = np.array(ACG((1, 0, 1), (0, 1, 0), (2.5, 0.4, 0.1)).rvs(300, rng=36))
    d = ACG.fit(X)
    L = d.matrix
    w = 1 / np.einsum("ij,jk,ik->i", X, np.linalg.inv(L), X)
    assert np.allclose(L, 3 * (X * w[:, None]).T @ X / len(X), atol=1e-8)
    assert np.isclose(np.trace(L), 3)
    assert ACG.from_matrix(2 * L) == d


# Feature set integration ------------------------------------------------------


def test_short_aliases():
    import apsg

    aliases = {
        "acg": ACG,
        "axial_vonmises": AxialVonMises,
        "bingham": Bingham,
        "esag": ESAG,
        "fisher": VonMisesFisher,
        "kent": Kent,
        "vonmises": VonMises,
        "watson": Watson,
        "wrapped_normal": WrappedNormal,
    }
    for alias, cls in aliases.items():
        assert getattr(apsg, f"dist_{alias}") is cls
        assert f"dist_{alias}" in apsg.__all__
        assert not hasattr(apsg, alias)
        assert get_distribution(alias) is cls


@pytest.mark.parametrize("dist", [d for d in INSTANCES if d.n_params], ids=repr)
def test_featureset_fit(dist):
    data = dist.rvs(200, rng=40)
    for model in (dist.name, type(dist).__name__, type(dist)):
        fitted = data.fit(model)
        assert fitted == type(dist).fit(data)


def test_featureset_fit_checks_dimension():
    from apsg import pairset

    data = Uniform3().rvs(20, rng=41)
    with pytest.raises(ValueError, match="not .*3D"):
        data.fit("vonmises")
    with pytest.raises(ValueError, match="not .*3D"):
        data.fit(VonMises)
    assert isinstance(data.fit("uniform"), Uniform3)
    with pytest.raises(ValueError, match="not pair"):
        pairset([pair(150, 40, 150, 40)]).fit("fisher")
    with pytest.raises(ValueError, match="not 3D"):
        data.fit("lisle")


def test_vectorial_model_of_axial_data_warns():
    np.random.seed(42)
    L = linset.random_fisher(100, position=lin(120, 40), kappa=20)
    with pytest.warns(UserWarning, match="halfspace") as record:
        d = L.fit("fisher")
    assert record[0].filename == __file__
    assert d == VonMisesFisher.fit(np.asarray(L.halfspace()))
    with pytest.warns(UserWarning, match="halfspace"):
        Kent.fit(L)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        L.fit("watson")
        L.halfspace().to_vec().fisher_statistics()
        vecset(list(L)).fit("watson")


def test_model_selection():
    data = Kent((0, 0, 1), (1, 0, 0), 20, 8).rvs(300, rng=43)
    fits = [data.fit(m) for m in ("fisher", "kent", "esag")]
    best = min(fits, key=lambda d: d.bic(data))
    assert isinstance(best, Kent | ESAG)
    assert fits[1].aic(data) < fits[0].aic(data)


# Resultant statistics of axial data -------------------------------------------

_AXIAL_SAMPLES = [
    Bingham((0, 0, 1), (1, 0, 0), -1, -20),  # elongated
    Watson((0, 0, 1), -10),  # girdle
    Watson((1, 1, 1), 5),  # cluster
]


@pytest.mark.parametrize("dist", _AXIAL_SAMPLES, ids=repr)
def test_axial_statistics_do_not_depend_on_stored_signs(dist):
    X = np.array(dist.rvs(100, rng=50))
    rng = np.random.default_rng(51)
    reference = None
    for _ in range(20):
        L = linset.from_xyz(*(X * rng.choice([-1, 1], size=(len(X), 1))).T)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", FutureWarning)
            values = (
                np.asarray(L.R()),
                np.array(L.halfspace()),
                L.var(),
                L.rdegree(),
                L.fisher_statistics()["k"],
            )
        if reference is None:
            reference = values
        for a, b in zip(values, reference, strict=True):
            assert np.allclose(a, b)


@pytest.mark.parametrize("dist", _AXIAL_SAMPLES, ids=repr)
def test_axial_resultant_is_principal_axis(dist):
    L = dist.rvs(100, rng=52)
    R = np.asarray(L.R())
    e1 = np.asarray(L.ortensor().eigenvectors(0))
    assert np.isclose(abs(R @ e1), np.linalg.norm(R))
    assert R[2] >= 0
    X = np.array(L)
    folded = X * np.where(X @ e1 < 0, -1, 1)[:, None]
    assert np.isclose(abs(L.R()), np.linalg.norm(folded.sum(axis=0)))
    assert np.all(np.array(L.halfspace()) @ R >= 0)


def test_axial_resultant_of_clustered_data_is_unchanged(fisher_sample):
    # the iterative flipping of apsg 2.0 and the principal axis agree
    L = fisher_sample.to_lin()
    assert np.isclose(abs(L.R()), abs(fisher_sample.R()))
    assert L.R().angle(fisher_sample.R()) < 0.5


@pytest.mark.parametrize(
    "name", ["fisher_statistics", "var", "delta", "rdegree", "csd"]
)
@pytest.mark.parametrize("to_set", ["to_lin", "to_fol"])
def test_resultant_statistics_of_axial_data_are_deprecated(fisher_sample, name, to_set):
    axial = getattr(fisher_sample, to_set)()
    with pytest.warns(FutureWarning, match=rf"{name}\(\) of axial data") as record:
        deprecated = getattr(axial, name)()
    assert len(record) == 1
    assert record[0].filename == __file__
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        getattr(fisher_sample, name)()
        converted = getattr(axial.halfspace().to_vec(), name)()
    if isinstance(deprecated, dict):
        deprecated, converted = deprecated["k"], converted["k"]
    assert np.isclose(deprecated, converted)


def test_elbow_uses_orientation_tensor_for_axes():
    from apsg import cluster

    L = Bingham((0, 0, 1), (1, 0, 0), -1, -20).rvs(60, rng=53)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        _, variances = cluster(L).elbow(no_plot=True)
    assert np.isclose(variances[0][0], 100 * (1 - L.ortensor().eigenvalues(0)))


def test_hotelling_mean_difference_is_of_stored_vectors():
    from apsg.feature._container._vectorsets import _hotelling_test

    a = Watson((0, 0, 1), 20).rvs(40, rng=54)
    b = Watson((0, 0.2, 1), 20).rvs(40, rng=55)
    a, b = a.halfspace(), linset.from_xyz(*np.abs(np.array(b)).T)
    X, Y = np.array(a), np.array(b)
    statistic, p_value, _ = a.similarity(b, method="hotelling")
    expected = _hotelling_test(X, Y, X.mean(axis=0) - Y.mean(axis=0))
    assert np.isclose(statistic, expected[0])
    assert np.isclose(p_value, expected[1])


def test_bingham_ellipse_formula():
    from apsg.feature._distributions import axial_cross_moments, bingham_ellipse

    X = np.array(Bingham((0, 0, 1), (1, 0, 0), -3, -15).rvs(60, rng=60))
    w, V = np.linalg.eigh(X.T @ X / len(X))
    taus, vecs = w[::-1], list(V[:, ::-1].T)
    chi2 = stats.chi2.ppf(0.9, 2)
    for which in (0, 1, 2):
        region = bingham_ellipse(
            taus, vecs, axial_cross_moments(X, vecs), len(X), 0.9, which
        )
        others = [i for i in range(3) if i != which]
        for j, gamma in zip(others, region.gamma, strict=True):
            m = np.mean((X @ vecs[which]) ** 2 * (X @ vecs[j]) ** 2)
            arg = np.sqrt(chi2 * m / (len(X) * (taus[which] - taus[j]) ** 2))
            assert np.isclose(
                gamma, np.degrees(np.arcsin(min(arg, 1))) if arg < 1 else 90
            )


def test_bingham_ellipse_coverage_and_orientation():
    rng = np.random.default_rng(61)
    d = Bingham((0, 0, 1), (1, 0, 0), -3, -15)
    covered = 0
    n_trials = 300
    for _ in range(n_trials):
        sample = d.rvs(60, rng=rng)
        s = sample.bingham_statistics()
        mu, (a, b), (ga, gb) = np.asarray(s["mu"]), s["axes"], np.radians(s["gamma"])
        t = np.array([0, 0, 1.0]) * np.sign(mu[2])
        x, y = np.arcsin(t @ np.asarray(a)), np.arcsin(t @ np.asarray(b))
        covered += (x / ga) ** 2 + (y / gb) ** 2 <= 1
    assert 0.9 <= covered / n_trials <= 0.98
    # the long axis of the ellipse lies along the major axis of the data
    s = d.rvs(200, rng=62).bingham_statistics()
    assert s["gamma"][0] > s["gamma"][1]
    assert abs(np.asarray(s["axes"][0]) @ [1, 0, 0]) > 0.9


# Pairs (Lisle) ----------------------------------------------------------------

# Lisle (1989), table 2: bedding/cleavage intersection lineations and bedding
LISLE_LINEATIONS = [(58, 20), (46, 53), (32, 32), (79, 10)]
LISLE_BEDDING = [(30, 23), (129, 86), (34, 32), (145, 24)]


def _lisle_pairs():
    from apsg import pairset

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return pairset(
            [
                pair(*b, *ln)
                for b, ln in zip(LISLE_BEDDING, LISLE_LINEATIONS, strict=True)
            ]
        )


def test_lisle_1989_table_2():
    from apsg import fol, ortensor

    ps = _lisle_pairs()
    s = ps.lisle_statistics()
    assert np.allclose(4 * np.array(s["eigenvalues"]), [3.21, -0.87, -2.34], atol=0.005)
    assert s["mean"].lin.angle(lin(54, 25)) < 0.6
    assert s["mean"].fol.angle(fol(53, 25)) < 0.6  # pole 233/65
    ot = ortensor.from_pairs(ps, shift=False)
    assert np.allclose(ot.eigenvalues(), s["eigenvalues"])
    # the raw measurements, as in the paper
    X = np.array([np.asarray(lin(*ln)) for ln in LISLE_LINEATIONS])
    Z = np.array([np.asarray(fol(*b)) for b in LISLE_BEDDING])
    taus = np.linalg.eigvalsh(X.T @ X - Z.T @ Z)[::-1]
    assert np.allclose(taus, [3.21, -0.87, -2.34], atol=0.005)


def test_shifted_lisle_tensor_weights_axes():
    from apsg import ortensor, pairset

    ps = pairset.random(30)
    L, N = np.array(ps.lvec), np.array(ps.fvec)
    M = np.cross(N, L)
    Tx, Tm = L.T @ L / 30, M.T @ M / 30
    assert np.allclose(np.asarray(ortensor.from_pairs(ps)), (2 * Tx + Tm) / 3)


def _uniform_frames(n=200000, seed=1):
    from scipy.spatial.transform import Rotation

    R = Rotation.random(n, random_state=seed).as_matrix()
    return np.stack([R[:, :, 0], R[:, :, 2]], axis=1)


@pytest.mark.parametrize("kappas", [(0, 0), (1, 2), (5, 3)])
def test_lisle_normalizer(kappas):
    from apsg.feature import Lisle

    X = _uniform_frames()
    d = Lisle(pair(120, 40, 160, 35), *kappas)
    # the density with respect to the uniform distribution of frames averages to 1
    assert np.isclose(d.pdf(X).mean(), 1, rtol=0.02)


def test_lisle_density_symmetry_and_mode():
    from apsg.feature import Lisle

    d = Lisle(pair(120, 40, 160, 35), 8, 3)
    X = _uniform_frames(50)
    for sl, sn in [(-1, 1), (1, -1), (-1, -1)]:
        Y = X * np.array([sl, sn])[None, :, None]
        assert np.allclose(d.logpdf(Y), d.logpdf(X))
    assert np.all(d.logpdf(X) <= d.logpdf(d.mean) + 1e-12)


def test_lisle_sample_moments_match_density():
    from apsg.feature import Lisle

    d = Lisle(pair(120, 40, 160, 35), 6, 2)
    X = _uniform_frames()
    w = d.pdf(X)[:, None]
    frame = d._frame
    expected_l = np.mean(w * (X[:, 0] @ frame.T) ** 2, axis=0)
    expected_n = np.mean(w * (X[:, 1] @ frame.T) ** 2, axis=0)
    S = d.rvs_array(20000, as_rng(4))
    assert np.allclose(np.mean((S[:, 0] @ frame.T) ** 2, axis=0), expected_l, atol=0.01)
    assert np.allclose(np.mean((S[:, 1] @ frame.T) ** 2, axis=0), expected_n, atol=0.01)
    assert np.allclose(np.sum(S[:, 0] * S[:, 1], axis=1), 0)


@pytest.mark.parametrize("kappas", [(30, 10), (2, 40), (5, 3)])
def test_lisle_fit_recovers_parameters(kappas):
    from apsg.feature import Lisle

    d = Lisle(pair(120, 40, 160, 35), *kappas)
    sample = d.rvs(3000, rng=5)
    fitted = Lisle.fit(sample)
    assert fitted.fit_info.converged
    assert np.allclose([fitted.kappa1, fitted.kappa2], kappas, rtol=0.1)
    assert fitted.mean.lin.angle(d.mean.lin) < 3
    assert fitted.mean.fol.angle(d.mean.fol) < 3
    # the maximum-likelihood mean is Lisle's mean
    assert fitted.mean == sample.lisle_statistics()["mean"]
    assert sample.fit("lisle") == fitted


def test_lisle_fit_is_rotation_equivariant():
    from apsg.feature import Lisle

    sample = Lisle(pair(120, 40, 160, 35), 6, 2).rvs(300, rng=6)
    axis, angle = vec(30, 20), 50
    rotated = sample.rotate(axis, angle)
    a, b = Lisle.fit(sample), Lisle.fit(rotated)
    assert np.isclose(a.kappa1, b.kappa1, rtol=1e-5)
    assert np.isclose(a.kappa2, b.kappa2, rtol=1e-5)
    assert b.mean == a.mean.rotate(axis, angle)


def test_lisle_rvs_and_json():
    from apsg.feature import Lisle
    from apsg.feature._container import PairSet

    d = Lisle(pair(120, 40, 160, 35), 6, 2)
    s = d.rvs(500, rng=7)
    assert isinstance(s, PairSet)
    assert np.allclose(np.array(d.rvs(5, rng=8)), np.array(d.rvs(5, rng=8)))
    lz = np.array(s.lvec)[:, 2]
    assert 0.4 < np.mean(lz > 0) < 0.6  # random signs
    assert feature_from_json(d.to_json()) == d
    assert isinstance(d.pdf(s[0]), float)


def test_lisle_confidence_coverage():
    rng = np.random.default_rng(9)
    from apsg.feature import Lisle

    d = Lisle(pair(120, 40, 160, 35), 20, 8)
    true = {"lin": np.asarray(d.mean.lvec), "fol": np.asarray(d.mean.fvec)}
    covered = {"lin": 0, "fol": 0}
    n_trials = 300
    for _ in range(n_trials):
        s = d.rvs(40, rng=rng).lisle_statistics()
        for key in covered:
            region = s[key]
            t = true[key] * np.sign(true[key] @ np.asarray(region["mu"]))
            a, b = (np.asarray(v) for v in region["axes"])
            g0, g1 = np.radians(region["gamma"])
            x, y = np.arcsin(t @ a), np.arcsin(t @ b)
            covered[key] += (x / g0) ** 2 + (y / g1) ** 2 <= 1
    for count in covered.values():
        assert 0.9 <= count / n_trials <= 0.98
    fitted = Lisle.fit(d.rvs(40, rng=10))
    assert fitted.mean_confidence(which="fol").mu == Lineation(fitted.mean.fvec)
    with pytest.raises(ValueError):
        fitted.mean_confidence(which="nonsense")


# Pairs, faults and degenerate tensors -----------------------------------------


def test_pair_equality_ignores_signs():
    from apsg.feature import Fault

    p = pair(120, 40, 160, 35)
    assert p == pair(-p.fvec, p.lvec) == pair(p.fvec, -p.lvec) == pair(-p.fvec, -p.lvec)
    f = Fault(p.fvec, p.lvec, 1)
    assert Fault(-f.fvec, f.lvec) == f
    assert Fault(f.fvec, -f.lvec) != f  # the slip direction sets the sense


def test_random_pairs_are_unit():
    from apsg import faultset, pairset

    for s in (pairset.random(20), faultset.random(20)):
        assert np.allclose(np.linalg.norm(np.array(s.fvec), axis=1), 1)


def test_degenerate_orientation_tensor_properties():
    from apsg import ortensor, pairset

    names = [
        "strength", "shape", "e1", "e2", "e3", "K", "k", "d", "D", "r", "goct",
        "eoct", "lode", "P_j", "T", "P", "G", "R", "B", "Intensity", "MAD",
    ]  # fmt: skip
    tensors = [
        ortensor.from_features(linset([lin(10, 20)] * 5)),
        ortensor.from_pairs(pairset([pair(120, 40, 160, 35)] * 5)),
    ]
    for ot in tensors:
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            for name in names:
                getattr(ot, name)
    assert tensors[0].shape == np.inf and tensors[0].lode == -1
    assert tensors[1].lode == 1 and np.allclose(
        tensors[1].eigenvalues(), [2 / 3, 1 / 3, 0]
    )


def test_jones_pewsey_special_cases():
    X = np.array(Uniform2().rvs(50, rng=1))
    k = 1.3
    # psi = 1: cardioid with rho = tanh(kappa) / 2, psi = -1: wrapped Cauchy
    # with rho = tanh(kappa / 2) (Ley & Verdebout, 2017, sec. 2.2.4)
    assert np.allclose(
        JonesPewsey(40, k, 1).logpdf(X), Cardioid(40, np.tanh(k) / 2).logpdf(X)
    )
    assert np.allclose(
        JonesPewsey(40, k, -1).logpdf(X), WrappedCauchy(40, np.tanh(k / 2)).logpdf(X)
    )
    assert np.allclose(JonesPewsey(40, k, 0).logpdf(X), VonMises(40, k).logpdf(X))


def test_circular_models_moment_fits():
    data = WrappedCauchy(70, 0.6).rvs(3000, rng=2)
    assert np.isclose(WrappedCauchy.fit(data, method="moments").rho, 0.6, atol=0.03)
    data = Cardioid(70, 0.3).rvs(3000, rng=2)
    assert np.isclose(Cardioid.fit(data, method="moments").rho, 0.3, atol=0.03)
    region = WrappedCauchy.fit(data).mean_confidence()
    assert 0 < region.alpha < 20
