"""
Tests of the hypothesis tests and statistics against worked examples.

The printed values come from the sources, as reproduced in the
OrientationData knowledge base (Calculation notes).
"""

import numpy as np
import pytest

from apsg import linset, ortensor, vecset
from apsg.feature import (
    AxialUniform3,
    Bingham,
    Direction2Set,
    EllipsoidSet,
    Kent,
    Lineation,
    TestResult,
    Uniform3,
    Vector2Set,
    VonMises,
    VonMisesFisher,
    Watson,
    anisotropy_test,
    bingham_test,
    jelinek_shape,
    rayleigh_test,
)
from apsg.feature._distributions import FitInfo
from apsg.feature._distributions._circular import vonmises_kappa_corrected
from apsg.feature._hypothesis import (
    axial_common_mean_test,
    common_mean_test,
    equal_kappa_test,
    hodges_ajne_test,
    likelihood_ratio_test,
    watson_symmetry_test,
)
from apsg.feature._tensor3 import Ellipsoid

# Fisher (1953) Table 1, nine specimens, as (North, West, Up) in Mardia & Jupp
# (2000) Table 9.1
FISHER_1953 = np.array(
    [
        [0.388, 0.117, -0.914],
        [0.171, -0.321, -0.932],
        [0.272, -0.204, -0.940],
        [0.123, -0.062, -0.991],
        [0.182, 0.003, -0.983],
        [0.291, -0.029, -0.956],
        [0.225, -0.272, -0.935],
        [0.518, 0.022, -0.855],
        [0.449, -0.433, -0.782],
    ]
)

# Mardia & Jupp (2000) Ex. 10.3: scatter matrix of 30 calcite c-axes
CALCITE_T = np.array(
    [[0.471, 0.000, 0.085], [0.000, 0.353, 0.057], [0.085, 0.057, 0.176]]
)


# TestResult --------------------------------------------------------------------


def test_testresult_reject_and_dict():
    r = TestResult("demo", 3.0, p_value=0.01, df=2, critical=2.0, level=0.95)
    assert r.reject()
    assert not r.reject(0.995)
    d = r.as_dict()
    assert d["reject"] is True and d["statistic"] == 3.0


def test_testresult_critical_value_only():
    r = TestResult("demo", 3.0, critical=2.0, level=0.95)
    assert r.reject()
    with pytest.raises(ValueError, match="critical value"):
        r.reject(0.99)


def test_testresult_legacy_keys_warn():
    r = rayleigh_test(np.eye(3)[[0, 0, 1]])
    with pytest.warns(FutureWarning, match="TestResult"):
        assert r["uniform"] is (not r.reject())
    with pytest.warns(FutureWarning):
        assert r["R"] == r.extra["R"]
    with pytest.raises(KeyError):
        r["nonsense"]


# uniformity --------------------------------------------------------------------


def test_rayleigh_statistic_and_modification():
    # S = p n Rbar^2, S* of Mardia & Jupp (2000) eq. 10.4.6
    r = rayleigh_test(FISHER_1953 / np.linalg.norm(FISHER_1953, axis=1)[:, None])
    n, p = FISHER_1953.shape
    Rbar = r.extra["Rbar"]
    assert np.isclose(r.statistic, p * n * Rbar**2)
    S = r.statistic
    assert np.isclose(r.extra["modified"], (1 - 1 / (2 * n)) * S + S**2 / (2 * n * 5))
    assert r.reject()


def test_bingham_test_calcite():
    # Mardia & Jupp (2000) Ex. 10.12: S = 14.61, S* = 14.87, reject at 5 %
    r = bingham_test(CALCITE_T, 30)
    assert abs(r.statistic - 14.61) < 0.05
    assert abs(r.extra["modified"] - 14.87) < 0.06
    assert r.df == 5
    assert r.reject()


def test_bingham_test_equals_lisle_intensity():
    np.random.seed(3)
    L = linset.random_fisher(40, kappa=5)
    assert np.isclose(L.uniformity_test().statistic, len(L) * L.ortensor().Intensity)


def test_bingham_test_size():
    rng = np.random.default_rng(11)
    rejected = sum(
        AxialUniform3().rvs(50, rng=rng).uniformity_test().reject() for _ in range(400)
    )
    assert 0.02 <= rejected / 400 <= 0.09


# Fisher statistics ---------------------------------------------------------------


def test_fisher_cone_fisher_1953():
    # Fisher (1953): alpha95 = 8.8 deg for the nine specimens
    s = vecset.from_xyz(*FISHER_1953.T).fisher_statistics()
    assert abs(s["alpha"] - 8.8) < 0.05
    n, R = (
        9,
        np.linalg.norm(
            (FISHER_1953 / np.linalg.norm(FISHER_1953, axis=1)[:, None]).sum(axis=0)
        ),
    )
    assert np.isclose(s["k"], (n - 1) / (n - R))


# Watson cone ---------------------------------------------------------------------


@pytest.mark.parametrize("kappa, n", [(10, 50), (-10, 50)])
def test_watson_cone_coverage(kappa, n):
    rng = np.random.default_rng(1)
    true = Watson(Lineation(120, 40), kappa)
    covered = 0
    for _ in range(300):
        d = Watson.fit(true.rvs(n, rng=rng))
        region = d.mean_confidence()
        covered += region.mu.angle(true.mean) <= region.alpha
    assert 0.91 <= covered / 300 <= 0.98


# fabric shape --------------------------------------------------------------------


def test_woodcock_strength_of_orientation_tensor():
    ot = ortensor(np.diag([0.6, 0.3, 0.1]))
    assert np.isclose(ot.strength, np.log(6))
    assert np.isclose(ot.shape, np.log(2) / np.log(3))


def test_jelinek_shape():
    # Jelinek (1981) eq. 4 with the mean of the logarithms: P' = 6.09
    s = jelinek_shape([0.1, 0.6, 0.3])
    assert np.isclose(s["Pj"], 6.09, atol=0.005)
    assert np.isclose(s["P"], 6) and np.isclose(s["L"], 2) and np.isclose(s["F"], 3)
    eta = np.log([0.6, 0.3, 0.1])
    assert np.isclose(s["T"], (2 * eta[1] - eta[0] - eta[2]) / (eta[0] - eta[2]))
    assert np.isclose(s["U"], (0.6 - 0.6 - 0.1) / 0.5)
    # Ellipsoid.P_j takes the square roots of the eigenvalues
    assert np.isclose(Ellipsoid(np.diag([0.6, 0.3, 0.1])).P_j ** 2, s["Pj"])
    with pytest.raises(ValueError, match="positive"):
        jelinek_shape([1, 0, -1])


def test_anisotropy_test():
    rng = np.random.default_rng(4)

    def sample(true, n):
        noise = rng.normal(scale=0.03, size=(n, 3, 3))
        return np.diag(true) + (noise + noise.transpose(0, 2, 1)) / 2

    assert anisotropy_test(sample((1.1, 1.0, 0.9), 12)).reject()
    rejected = sum(
        anisotropy_test(sample((1.0, 1.0, 1.0), 12)).reject() for _ in range(300)
    )
    assert 0.02 <= rejected / 300 <= 0.09
    with pytest.raises(ValueError, match="more than 5"):
        anisotropy_test(sample((1, 1, 1), 5))


def test_mean_tensor_intervals_and_test():
    rng = np.random.default_rng(5)
    noise = rng.normal(scale=0.02, size=(10, 3, 3))
    mats = np.diag([1.2, 1.0, 0.8]) + (noise + noise.transpose(0, 2, 1)) / 2
    r = EllipsoidSet([Ellipsoid(m) for m in mats]).mean_tensor()
    lo, hi = r["intervals"].T
    assert np.all(lo < r["eigenvalues"]) and np.all(r["eigenvalues"] < hi)
    assert r["anisotropy_test"].reject()
    assert np.isclose(r["shape"]["P"], r["eigenvalues"][0] / r["eigenvalues"][2])


# circular data (Fisher, 1993, Appendix B) -----------------------------------------

B4 = [2, 9, 18, 24, 30, 35, 35, 39, 39, 44, 44, 49, 56, 70, 76, 76, 81, 86, 91, 112,
      121, 127, 133, 134, 138, 147, 152, 157, 166, 171, 177, 187, 206, 210, 211, 215,
      238, 246, 269, 270, 285, 292, 305, 315, 325, 328, 329, 343, 354, 359]  # fmt: skip
B5 = [1, 1, 2, 2, 3, 8, 9, 12, 16, 17, 19, 23, 28, 28, 34, 34, 35, 36, 36, 37, 41, 45,
      49, 50, 51, 53, 58, 68, 69, 70, 72, 72, 76, 78, 80, 85, 97, 97, 99, 101, 105,
      121, 125, 126, 133, 141, 143, 149, 152, 156, 160, 163, 167, 168, 170, 171, 172,
      174, 175, 176]  # fmt: skip
B6_2 = [294, 301, 329, 315, 277, 281, 254, 245, 272, 242, 177, 257, 177, 229, 250,
        166, 232, 245, 224, 186, 257, 267, 241, 239, 287, 229, 290, 214, 215, 224]  # fmt: skip
# Fisher (1993) B12 with 185 for the misprinted 285
B12 = [85, 135, 135, 140, 145, 150, 150, 150, 160, 185, 200, 210, 220, 225, 270]


def test_rayleigh_fisher_ex_4_12():
    r = Vector2Set.from_directions(B4).uniformity_test()
    assert np.isclose(r.extra["Rbar"], 0.1798, atol=5e-5)
    assert np.isclose(r.statistic / 2, 1.617, atol=5e-4)  # Z = n Rbar^2
    assert np.isclose(r.p_value, 0.20, atol=0.005)


def test_axial_rayleigh_and_kuiper_fisher_ex_4_8_4_11():
    d = Direction2Set.from_directions(B5)
    r = d.uniformity_test()
    assert np.isclose(r.extra["Rbar"], 0.2370, atol=5e-5)
    assert np.isclose(r.p_value, 0.0337, atol=5e-4)
    k = d.uniformity_test(method="kuiper")
    assert np.isclose(k.extra["D+"], 0.1389, atol=5e-5)
    assert np.isclose(k.extra["D-"], 0.0611, atol=5e-5)
    assert np.isclose(k.statistic, 1.586, atol=5e-4)
    assert np.isclose(k.critical, 1.747, atol=5e-3)


@pytest.mark.parametrize("method", ["kuiper", "watson", "hodges_ajne", "rao"])
def test_circular_uniformity_tests(method):
    rng = np.random.default_rng(6)
    uniform = Vector2Set.from_directions(rng.uniform(0, 360, 100))
    assert not uniform.uniformity_test(method=method).reject()
    clustered = Vector2Set.from_directions(rng.normal(40, 30, 100))
    assert clustered.uniformity_test(method=method).reject()
    with pytest.raises(ValueError, match="Unknown method"):
        uniform.uniformity_test(method="nonsense")


def test_circular_uniformity_test_sizes():
    rng = np.random.default_rng(7)
    for method in ("kuiper", "watson"):
        rejected = sum(
            Vector2Set.from_directions(rng.uniform(0, 360, 30))
            .uniformity_test(method=method)
            .reject()
            for _ in range(300)
        )
        assert 0.02 <= rejected / 300 <= 0.09


def test_hodges_ajne_exact_probability():
    # all data in one half-circle: m = 0, P = 2^(1-n) n
    r = hodges_ajne_test(np.radians([10, 20, 30, 40, 50, 60]))
    assert r.statistic == 0
    assert np.isclose(r.p_value, 2.0**-5 * 6)


def test_circular_dispersion_interval_fisher_ex_4_16():
    v = Vector2Set.from_directions(B6_2)
    assert np.isclose(np.sqrt(v.circular_dispersion() / len(v)), 0.1314, atol=5e-5)
    d = VonMises.fit(v)
    assert np.isclose(d.mean.direction, 247.6, atol=0.05)
    assert np.isclose(d.mean_confidence(method="dispersion").alpha, 14.9, atol=0.05)


def test_vonmises_intervals_fisher_ex_4_21_4_22():
    d = VonMises.fit(Vector2Set.from_directions(B12))
    assert np.isclose(d.mean.direction, 168.5, atol=0.05)
    lo, hi = d.kappa_confidence()
    assert round(lo, 1) == 1.0 and round(hi, 1) == 3.6
    # von Mises interval with the book's kappa 1.80: (142.3, 194.7)
    book = VonMises(d.mean, 1.80)
    book.fit_info = d.fit_info
    h = book.mean_confidence(method="vonmises").alpha
    assert abs(d.mean.direction - h - 142.3) < 0.06
    assert abs(d.mean.direction + h - 194.7) < 0.06
    with pytest.raises(ValueError, match="Unknown method"):
        d.mean_confidence(method="nonsense")


def test_vonmises_kappa_interval_warns_for_dispersed_data():
    d = VonMises.fit(VonMises(0, 0.5).rvs(40, rng=1))
    with pytest.warns(UserWarning, match="kappa >= 2"):
        d.kappa_confidence()


def test_vonmises_corrected_fisher_ex_4_18():
    # Best & Fisher (1981) correction of F93 eq. 4.41: 1.55 -> 1.54 for n = 100
    assert np.isclose(vonmises_kappa_corrected(1.5507, 100), 1.54, atol=0.006)
    assert vonmises_kappa_corrected(0.1, 5) == 0
    data = VonMises(30, 4).rvs(10, rng=3)
    assert VonMises.fit(data, method="corrected").kappa < VonMises.fit(data).kappa


def test_circular_descriptive_statistics():
    rng = np.random.default_rng(12)
    theta = rng.vonmises(np.radians(50), 3, 300)
    v = Vector2Set.from_directions(np.degrees(theta))
    Rbar, mean = v.moment(1)
    assert np.isclose(Rbar, abs(np.mean(np.exp(1j * theta))))
    assert np.isclose(mean, np.degrees(np.angle(np.mean(np.exp(1j * theta)))) % 360)
    assert np.isclose(v.angular_deviation(), np.degrees(np.sqrt(2 * (1 - Rbar))))
    assert abs(v.skewness()) < 0.3  # von Mises is symmetric
    assert abs((v.median().direction - 50 + 180) % 360 - 180) < 10
    # symmetric data about 0: zero skewness, median 0
    s = Vector2Set.from_directions([-30, -10, 0, 10, 30])
    assert np.isclose(s.skewness(), 0, atol=1e-12)
    assert np.isclose(s.median().direction % 360, 0)


def test_axial_median_and_moments_use_doubled_angles():
    d = Direction2Set.from_directions([170, 175, 5, 10, 185])
    assert abs((d.median().direction - 176) % 180 - 4) < 6
    doubled = Vector2Set.from_directions(2 * np.array([170, 175, 5, 10, 185]))
    assert np.isclose(d.moment(1)[0], doubled.moment(1)[0])


# spherical data -----------------------------------------------------------------


def test_fisher_kappa_interval_fle_ex_5_24():
    # n = 26, R = 25.77: (70.3, 155.3) by FLE eq. 5.37
    d = VonMisesFisher((0, 0, 1), 100)
    d.fit_info = FitInfo(n=26, method="mle", loglik=0.0, stats={"Rbar": 25.77 / 26})
    lo, hi = d.kappa_confidence()
    assert abs(lo - 70.3) < 0.1 and abs(hi - 155.3) < 0.1


def test_kappa_confidence_needs_fit():
    with pytest.raises(ValueError, match="fitted"):
        VonMisesFisher().kappa_confidence()
    with pytest.raises(NotImplementedError):
        Kent.fit(Kent(kappa=20, beta=3).rvs(50, rng=1)).kappa_confidence()


def test_fisher_precision_methods():
    data = VonMisesFisher((0, 0, 1), 15).rvs(12, rng=4)
    X = np.asarray(data, dtype=float)
    R = np.linalg.norm(X.sum(axis=0))
    assert np.isclose(VonMisesFisher.fit(data, method="precision").kappa, 11 / (12 - R))
    mle = VonMisesFisher.fit(data).kappa
    assert np.isclose(
        VonMisesFisher.fit(data, method="corrected").kappa, mle * (11 / 12) ** 2
    )


def test_fisher_goodness_of_fit():
    rng = np.random.default_rng(8)
    rejected = np.zeros(2)
    for _ in range(300):
        gof = VonMisesFisher.goodness_of_fit(
            VonMisesFisher((0, 0, 1), 20).rvs(60, rng=rng)
        )
        rejected += [gof["colatitude"].reject(), gof["longitude"].reject()]
    assert np.all((0.02 <= rejected / 300) & (rejected / 300 <= 0.09))
    fisher = VonMisesFisher((0, 0, 1), 20).rvs(60, rng=rng)
    kent = Kent(kappa=20, beta=9).rvs(200, rng=rng)
    assert VonMisesFisher.goodness_of_fit(kent)["longitude"].reject()
    with pytest.raises(ValueError, match="level"):
        VonMisesFisher.goodness_of_fit(fisher, level=0.8)


def test_fisher_kent_score_test():
    rng = np.random.default_rng(9)
    rejected = sum(
        Kent.score_test(VonMisesFisher((0, 0, 1), 20).rvs(50, rng=rng)).reject()
        for _ in range(300)
    )
    assert 0.02 <= rejected / 300 <= 0.09
    assert Kent.score_test(Kent(kappa=20, beta=8).rvs(100, rng=rng)).reject()


def test_watson_symmetry_test_calcite():
    # Mardia & Jupp (2000) Ex. 10.13: w_g = 1.33, w_b = 9.97
    r = watson_symmetry_test((0.495, 0.365, 0.139), (0, -0.68, -3.62), 30)
    assert np.isclose(r["girdle"].statistic, 1.33, atol=0.01)
    assert np.isclose(r["bipolar"].statistic, 9.97, atol=0.01)
    assert not r["girdle"].reject() and r["bipolar"].reject()


def test_watson_symmetry_test_on_data():
    rng = np.random.default_rng(10)
    r = Watson.symmetry_test(Watson(Lineation(0, 90), 10).rvs(150, rng=rng))
    assert not r["bipolar"].reject()
    assert r["girdle"].reject()


def test_watson_kappa_interval_fle_ex_6_19():
    # n = 33, tau_1 = 0.9494: (13.1, 26.4) by FLE eq. 6.26
    d = Watson((0, 0, 1), 20)
    d.fit_info = FitInfo(
        n=33, method="mle", loglik=0.0, stats={"eigenvalues": (0.9494, 0.03, 0.0206)}
    )
    lo, hi = d.kappa_confidence()
    assert abs(lo - 13.1) < 0.06 and abs(hi - 26.4) < 0.06
    girdle = Watson.fit(Watson((0, 0, 1), -10).rvs(50, rng=2))
    with pytest.raises(ValueError, match="bipolar"):
        girdle.kappa_confidence()


def test_likelihood_ratio_test():
    rng = np.random.default_rng(11)
    data = Kent(kappa=25, beta=10).rvs(150, rng=rng)
    r = likelihood_ratio_test(VonMisesFisher.fit(data), Kent.fit(data), data)
    assert r.df == 2 and r.reject()
    with pytest.raises(ValueError, match="more parameters"):
        likelihood_ratio_test(Kent.fit(data), VonMisesFisher.fit(data), data)
    axes = Watson((0, 0, 1), 5).rvs(50, rng=rng)
    with pytest.raises(ValueError, match="nested"):
        likelihood_ratio_test(Uniform3(), Bingham.fit(axes), axes)


def test_bootstrap_ellipse():
    np.random.seed(2)
    v = VonMisesFisher((0, 0, 1), 30).rvs(40, rng=5)
    s = v.bootstrap_statistics(n=200, method="ellipse")
    assert set(s) == {"mu", "axes", "gamma"}
    assert all(abs(a.dot(s["mu"])) < 1e-9 for a in s["axes"])
    assert s["gamma"][0] >= s["gamma"][1] > 0
    with pytest.raises(ValueError, match="Unknown method"):
        v.bootstrap_statistics(method="nonsense")


# comparing samples --------------------------------------------------------------


def _circle(angles):
    a = np.radians(angles)
    return np.column_stack([np.cos(a), np.sin(a)])


# Mardia & Jupp (2000) Ex. 8.2 and 8.4: control and experimental pigeons
CONTROL = [50, 285, 295, 295, 305, 320, 330, 330, 335, 340, 340, 355]
EXPERIMENTAL = [70, 155, 190, 195, 215, 235, 235, 240, 255, 260, 290, 300, 300, 300]


def test_two_sample_u2_and_uniform_scores_mj_ex_8_2_8_4():
    samples = [_circle(CONTROL), _circle(EXPERIMENTAL)]
    assert np.isclose(
        common_mean_test(samples, "watson_u2").statistic, 0.320, atol=1e-3
    )
    assert np.isclose(
        common_mean_test(samples, "uniform_scores").statistic, 10.60, atol=5e-3
    )


@pytest.mark.parametrize(
    "dist, method, k",
    [
        (VonMisesFisher((0, 0, 1), 10), None, 2),
        (VonMisesFisher((0, 0, 1), 10), "mcfadden_lowes", 3),
        (VonMisesFisher((0, 0, 1), 10), "watson_debiche", 2),
        (VonMises(0, 3), None, 2),
        (VonMises(0, 3), "uniform_scores", 3),
        (VonMises(0, 3), "watson_u2", 2),
    ],
)
def test_common_mean_test_size_and_power(dist, method, k):
    rng = np.random.default_rng(13)
    rejected = sum(
        common_mean_test(
            [np.asarray(dist.rvs(25, rng=rng)) for _ in range(k)], method=method
        ).reject()
        for _ in range(300)
    )
    assert 0.015 <= rejected / 300 <= 0.09
    # the same distribution with the mean turned by 90 degrees
    mean = 90 if dist.dim == 2 else (1, 0, 0)
    shifted = type(dist)(**(dist.params | {"mean": mean}))
    samples = [np.asarray(dist.rvs(25, rng=rng)), np.asarray(shifted.rvs(25, rng=rng))]
    assert common_mean_test(samples, method=method).reject()


def test_common_mean_test_errors():
    x = np.asarray(VonMisesFisher().rvs(10, rng=1))
    with pytest.raises(ValueError, match="two samples"):
        common_mean_test([x])
    with pytest.raises(ValueError, match="Unknown method"):
        common_mean_test([x, x], method="watson_williams")
    with pytest.raises(ValueError, match="dimension"):
        common_mean_test([x, x[:, :2]])


@pytest.mark.parametrize("k", [2, 3])
def test_equal_kappa_test(k):
    rng = np.random.default_rng(14)
    rejected = sum(
        equal_kappa_test(
            [
                np.asarray(VonMisesFisher((0, 0, 1), 10).rvs(25, rng=rng))
                for _ in range(k)
            ]
        ).reject()
        for _ in range(300)
    )
    assert 0.015 <= rejected / 300 <= 0.09
    samples = [
        np.asarray(VonMisesFisher((0, 0, 1), c).rvs(30, rng=rng)) for c in (5, 50)
    ][:k] + [np.asarray(VonMisesFisher((0, 0, 1), 5).rvs(30, rng=rng))] * (k - 2)
    assert equal_kappa_test(samples).reject()


def test_axial_common_mean_test_size():
    rng = np.random.default_rng(15)
    rejected = sum(
        axial_common_mean_test(
            [np.asarray(Watson((0, 0, 1), 10).rvs(30, rng=rng)) for _ in range(2)]
        ).reject()
        for _ in range(300)
    )
    # 0.003 with FLE's printed factor 2 in g
    assert 0.02 <= rejected / 300 <= 0.09


def test_set_comparison_methods():
    rng = np.random.default_rng(16)
    a = VonMisesFisher((0, 0, 1), 20).rvs(30, rng=rng)
    b = VonMisesFisher((1, 0, 1), 20).rvs(30, rng=rng)
    assert a.common_mean_test(b).reject()
    assert a.common_mean_test(b, method="watson_debiche").reject()
    assert isinstance(a.equal_kappa_test(b), TestResult)
    la, lb = a.to_lin(), b.to_lin()
    assert la.common_mean_test(lb).name == "axial common mean"
    with pytest.raises(ValueError, match="axial"):
        la.common_mean_test(lb, method="watson")
    with pytest.raises(TypeError, match="axes"):
        la.equal_kappa_test(lb)
    with pytest.raises(TypeError, match="Cannot compare"):
        a.common_mean_test(lb)
    d1 = Direction2Set.from_directions(rng.normal(10, 5, 20) % 180)
    d2 = Direction2Set.from_directions(rng.normal(100, 5, 20) % 180)
    assert d1.common_mean_test(d2).reject()
