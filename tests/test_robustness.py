"""Invalid, extreme and degenerate input: fail early with a clear error, or work.

Regression tests of the findings of a chaos-testing run, grouped by the part of
apsg they concern.
"""

import matplotlib

matplotlib.use("Agg")

import numpy as np
import pytest

from apsg import (
    G,
    RosePlot,
    StereoGrid,
    StereoNet,
    apsg_conf_context,
    cone,
    defgrad,
    dir2,
    dir2set,
    dist_acg,
    dist_axial_vonmises,
    dist_bingham,
    dist_esag,
    dist_fisher,
    dist_kent,
    dist_lisle,
    dist_vonmises,
    dist_watson,
    dist_wrapped_normal,
    ellipse,
    ellipsoid,
    ellipsoidset,
    fault,
    fol,
    folset,
    lin,
    linset,
    ortensor,
    ortensor2,
    pair,
    pairset,
    rotation,
    stress,
    vec,
    vec2,
    vec2set,
    vecset,
)
from apsg.config import apsg_conf
from apsg.feature import (
    FoliationSet,
    Lineation,
    LineationSet,
    Vector3Set,
    feature_from_json,
)

nan, inf = float("nan"), float("inf")

# -- distributions and samplers -------------------------------------------


@pytest.mark.parametrize(
    "make",
    [
        lambda k: dist_vonmises(0, k),
        lambda k: dist_axial_vonmises(0, k),
        lambda k: dist_wrapped_normal(0, k),
        lambda k: dist_fisher(lin(1, 2), k),
        lambda k: dist_kent(vec(0, 0, 1), vec(1, 0, 0), k, 0),
        lambda k: dist_esag(vec(0, 0, 3), vec(1, 0, 0), k),
        lambda k: dist_watson(lin(1, 2), k),
        lambda k: dist_bingham(vec(0, 0, 1), vec(1, 0, 0), -1, k),
        lambda k: dist_lisle(pair(120, 40, 162, 28), k, 1),
    ],
)
@pytest.mark.parametrize("value", [nan, inf, -inf])
def test_non_finite_parameters_raise(make, value):
    # kappa=inf used to hang the Watson and Bingham samplers forever
    with pytest.raises(ValueError, match="finite"):
        make(value)


@pytest.mark.parametrize("value", ["5", None, True, [1, 2]])
def test_non_numeric_parameters_raise(value):
    with pytest.raises(TypeError, match="kappa must be a number"):
        dist_fisher(lin(1, 2), value)


def test_acg_shape_must_be_finite():
    with pytest.raises(ValueError, match="finite"):
        dist_acg(vec(0, 0, 1), vec(1, 0, 0), (inf, 1, 0.5))


@pytest.mark.parametrize("mean", [vec(0, 0, 0), np.array([nan, 0, 1])])
def test_zero_or_non_finite_mean_raises(mean):
    with pytest.raises(ValueError, match="mean|finite"):
        dist_fisher(mean, 10)
    with pytest.raises(ValueError, match="mean|finite"):
        dist_watson(mean, 10)


def test_random_vonmises_infinite_kappa_raises():
    # used to loop forever inside scipy's legacy sampler
    with pytest.raises(ValueError, match="finite"):
        vec2set.random_vonmises(kappa=inf)


def test_bingham_sampler_never_spins_forever():
    # the rejection loop accepts nothing with an infinite envelope, which the
    # constructor rules out; the sampler still refuses rather than looping
    d = dist_bingham(vec(0, 0, 1), vec(1, 0, 0), -1e300, -1e300)
    assert len(d.rvs(5, rng=0)) == 5
    d.kappa_minor = -inf
    with pytest.raises(ValueError, match="too large"):
        d.rvs(5, rng=0)


@pytest.mark.parametrize(
    "dist,cls",
    [
        (dist_fisher(lin(1, 2), 5), "Vector3Set"),
        (dist_watson(lin(1, 2), 5), "LineationSet"),
        (dist_vonmises(30, 2), "Vector2Set"),
        (dist_axial_vonmises(30, 2), "Direction2Set"),
        (dist_lisle(pair(120, 40, 162, 28), 2, 2), "PairSet"),
    ],
)
def test_rvs_zero_returns_empty_set(dist, cls):
    sample = dist.rvs(0, rng=0)
    assert type(sample).__name__ == cls
    assert len(sample) == 0


@pytest.mark.parametrize(
    "n,error",
    [(-1, ValueError), (1.5, TypeError), (None, TypeError), ("10", TypeError)],
)
def test_invalid_sample_size_raises(n, error):
    with pytest.raises(error, match="n must be"):
        dist_fisher(lin(1, 2), 5).rvs(n)
    with pytest.raises(error, match="n must be"):
        linset.random_fisher(n=n)
    with pytest.raises(error, match="n must be"):
        vec2set.random_vonmises(n=n)
    with pytest.raises(error, match="n must be"):
        pairset.random(n=n)


def test_g_of_empty_list():
    with pytest.raises(ValueError, match="feature class"):
        G([])
    empty = G([], feature=Lineation)
    assert isinstance(empty, LineationSet)
    assert len(empty) == 0


@pytest.mark.parametrize("n", [1, 5])
def test_fitting_identical_data_raises(n):
    with pytest.raises(ValueError, match="at least 2|identical"):
        dist_fisher.fit(vecset([vec(120, 30)] * n))
    with pytest.raises(ValueError, match="at least 2|identical"):
        dist_watson.fit(linset([lin(120, 30)] * n))
    with pytest.raises(ValueError, match="at least 2|identical"):
        dist_vonmises.fit(vec2set.from_directions([40] * n))


def test_statistics_of_identical_data_raise():
    with pytest.raises(ValueError, match="identical"):
        folset.from_array([120] * 4, [30] * 4).watson_statistics()


def test_fitting_a_perfect_girdle_raises():
    # all axes on one great circle: the fitted Watson concentration is -inf
    girdle = linset([lin(a, 0) for a in range(0, 180, 20)])
    with pytest.raises(ValueError, match="degenerate"):
        dist_watson.fit(girdle)


# -- density grid ----------------------------------------------------------


def test_kamb_density_is_the_same_in_chunks(monkeypatch):
    # the grid x features kernel matrix is summed over chunks of features, so
    # 10**6 features don't need 22 GiB
    from apsg import StereoGrid
    from apsg.plotting._stereo_engine import _stereogrid

    data = linset.random_fisher(n=500, kappa=10)
    whole = StereoGrid(n=300)
    whole.calculate_density(data, method="kamb")
    monkeypatch.setattr(_stereogrid, "_KAMB_CHUNK", 300 * 7)  # 7 features a chunk
    chunked = StereoGrid(n=300)
    chunked.calculate_density(data, method="kamb")
    assert np.allclose(whole.values, chunked.values)


# -- feature constructors --------------------------------------------------


@pytest.mark.parametrize(
    "make",
    [
        lambda: vec2("abc", 30),
        lambda: vec2(None, 30),
        lambda: dir2({}, 30),
        lambda: dir2("north"),
        lambda: vec(1 + 2j, 0, 0),
        lambda: vec(True, 1, 1),
        lambda: lin(None, 30),
        lambda: lin(30, "45"),
        lambda: fol([], 30),
    ],
)
def test_non_numeric_components_raise(make):
    # these used to build features holding garbage, failing much later
    with pytest.raises(TypeError, match="real number|Not valid arguments"):
        make()


@pytest.mark.parametrize(
    "make",
    [
        lambda: vec2(inf, 0),
        lambda: dir2(inf),
        lambda: dir2(nan, 30),
        lambda: vec(nan, 1, 1),
        lambda: lin(nan, 30),
        lambda: fol(30, inf),
        lambda: vec(10**400, 1, 1),
        lambda: lin(10, 20) + nan,
        lambda: lin(10, 20) / 0,
    ],
)
@pytest.mark.filterwarnings("ignore:divide by zero:RuntimeWarning")
def test_non_finite_components_raise(make):
    # a NaN lineation used to be created, and then failed to print
    with pytest.raises(ValueError, match="finite|too large"):
        make()


def test_plunge_and_dip_ranges():
    with pytest.raises(ValueError, match=r"Lineation plunge must be in \[-90, 90\]"):
        lin(10, 95)
    with pytest.raises(ValueError, match=r"Foliation dip must be in \[0, 180\]"):
        fol(10, 181)
    with pytest.raises(ValueError, match="dip"):
        fol(10, -1)
    assert lin(10, -30) == lin(190, 30)  # negative plunge is the upper hemisphere
    assert lin(-710, 30) == lin(10, 30)  # azimuths are periodic
    with apsg_conf_context(notation="rhr"), pytest.raises(ValueError, match="dip"):
        fol(10, 200)


def test_zero_length_vectors():
    assert abs(vec(0, 0, 0)) == 0  # a zero vector is a valid vector
    for make in (lambda: lin(0, 0, 0), lambda: fol(0, 0, 0), lambda: dir2(0, 0)):
        with pytest.raises(ValueError, match="zero vector"):
            make()
    with pytest.raises(ValueError, match="zero vector"):
        lin(10, 20) - lin(10, 20)
    with pytest.raises(ValueError, match="zero vector"):
        vec(0, 0, 0).angle(vec(1, 0, 0))  # used to be 90
    with pytest.raises(ValueError, match="zero vector"):
        vec2(0, 0).angle(vec2(1, 0))
    with pytest.raises(ValueError, match="zero vector"):
        lin(10, 20).rotate(vec(0, 0, 0), 30)  # used to be a silent no-op


def test_cone_apical_angle():
    axis = lin(120, 30)
    for angle in (-10, 500):  # used to wrap to 10 and 40 degrees
        with pytest.raises(ValueError, match=r"apical angle must be in \[0, 180\]"):
            cone(axis, angle)
    for angle in (0, 20, 75, 120, 180):  # over the vertical too
        c = cone(axis, angle)
        assert np.isclose(c.axis.angle(c.secant), min(angle, 180 - angle))


def test_pair_and_fault_arguments():
    with pytest.raises(ValueError, match="must not be zero"):
        pair(vec(0, 0, 0), vec(1, 0, 0))
    with pytest.raises(ValueError, match="parallel"):
        pair(fol(10, 20), fol(10, 20))
    # a numeric sense is taken by its sign; 7 used to give a reverse fault
    assert fault(120, 30, 120, 30, 7).sense == 1
    assert fault(120, 30, 120, 30, -0.5).sense == -1
    with pytest.raises(ValueError, match="positive or negative"):
        fault(120, 30, 120, 30, 0)


def test_length_mismatch_raises():
    # zip() used to drop the extra items silently
    with pytest.raises(ValueError, match="same length"):
        linset.from_array([10, 20, 30], [5])
    a = linset([lin(10, 20), lin(30, 40), lin(50, 60)])
    with pytest.raises(ValueError, match="same length"):
        a.angle(linset([lin(10, 20)]))
    with pytest.raises(ValueError, match="same length"):
        a.cross(linset([lin(10, 20)]))
    with pytest.raises(ValueError, match="same length"):
        vec2set([vec2(1, 0)]).angle(vec2set([vec2(1, 0), vec2(0, 1)]))
    with pytest.raises(ValueError, match="same length"):
        pairset.random(3).angle(pairset.random(2))


@pytest.mark.parametrize(
    "obj,error",
    [
        (None, TypeError),
        ({}, TypeError),
        ("Lineation", TypeError),
        ({"datatype": "nope"}, ValueError),
        ({"datatype": "feature_from_json", "args": []}, ValueError),
        ({"datatype": "Lineation"}, TypeError),
        ({"datatype": "Lineation", "args": "x"}, TypeError),  # used to give L:0/0
        ({"datatype": "Lineation", "args": [[nan, 0, 1]]}, ValueError),
    ],
)
def test_feature_from_json_rejects_bad_input(obj, error):
    with pytest.raises(error):
        feature_from_json(obj)


@pytest.mark.parametrize(
    "value,error",
    [
        (np.full((3, 3), nan), ValueError),
        (np.diag([inf, 1, 1]), ValueError),
        (np.eye(3) * 1j, TypeError),
        ([["a"] * 3] * 3, TypeError),
    ],
)
def test_tensor_entries_must_be_finite_real(value, error):
    with pytest.raises(error, match="entries"):
        defgrad(value)
    with pytest.raises(error, match="entries"):
        stress(value)


# -- empty and too small sets ----------------------------------------------


@pytest.mark.parametrize("empty", [vecset([]), linset([]), vec2set([]), dir2set([])])
@pytest.mark.parametrize(
    "method", ["R", "var", "delta", "rdegree", "csd", "uniformity_test", "ortensor"]
)
@pytest.mark.filterwarnings("ignore::FutureWarning")
def test_statistics_of_an_empty_set_raise(empty, method):
    # these used to fail with ZeroDivisionError, AttributeError or LinAlgError
    with pytest.raises(ValueError, match="empty"):
        getattr(empty, method)()


@pytest.mark.filterwarnings("ignore::FutureWarning")
def test_fitted_statistics_of_an_empty_set_raise():
    with pytest.raises(ValueError, match="at least 2"):
        vecset([]).fisher_statistics()
    with pytest.raises(ValueError, match="at least 2"):
        linset([]).watson_statistics()
    with pytest.raises(ValueError, match="at least 2"):
        linset([]).fit("bingham")
    with pytest.raises(ValueError, match="empty"):
        pairset([]).ortensor()


def test_empty_sets_still_work():
    assert len(linset([]).halfspace()) == 0
    assert len(list(linset([]))) == 0
    s = StereoNet()
    s.point(linset([]))
    s.great_circle(folset([]))
    s.init_figure()
    s._render()


def test_pairwise_cross_of_one_item_is_empty():
    # used to fail with IndexError
    product = linset([lin(10, 20)]).cross()
    assert isinstance(product, FoliationSet)
    assert len(product) == 0
    assert isinstance(vecset([vec(1, 0, 0)]).cross(), Vector3Set)
    assert len(linset([lin(10, 20), lin(50, 10)]).cross()) == 1


@pytest.mark.parametrize(
    "n,error", [(0, ValueError), (-1, ValueError), (2.5, TypeError)]
)
def test_spiral_sizes(n, error):
    # gss(0) used to divide by zero and gss(-1) returned an empty set
    with pytest.raises(error, match="n must be"):
        linset.uniform_gss(n)
    with pytest.raises(error, match="n must be"):
        vecset.uniform_sfs(n)
    assert len(linset.uniform_gss(1)) == 1


def test_mean_tensor_needs_three_tensors():
    tensors = [
        ellipsoid.from_defgrad(defgrad.from_comp(xx=2 + i, zz=0.5)) for i in range(3)
    ]
    with pytest.raises(ValueError, match="at least 3"):
        ellipsoidset(tensors[:2]).mean_tensor()
    ellipsoidset(tensors).mean_tensor()


# -- printing --------------------------------------------------------------


@pytest.mark.parametrize(
    "make",
    [
        lambda: dir2set([dir2(30)]).ortensor(),  # used to raise ZeroDivisionError
        lambda: ortensor2(np.zeros((2, 2))),
        lambda: ellipse(np.zeros((2, 2))),
        lambda: ortensor(np.zeros((3, 3))),
        lambda: ellipsoid(np.zeros((3, 3))),
        lambda: defgrad(np.eye(3) * 1e308),  # used to print inf
        lambda: stress(np.eye(3) * 1e308),
        lambda: linset([]),
        lambda: vec(0, 0, 0),
        lambda: cone(lin(10, 20), 0),
    ],
)
def test_repr_of_degenerate_objects(make):
    text = repr(make())
    assert "inf]" not in text and "inf " not in text.split("(")[0]


# -- plot parameters -------------------------------------------------------


@pytest.mark.parametrize(
    "kwargs",
    [
        {"grid_step": 0},  # used to divide by zero at render
        {"grid_step": -15},
        {"grid_step": nan},
        {"clip_pole": -5},
        {"clip_pole": 100},
        {"figsize": (0, 0)},
        {"figsize": "big"},
        {"kind": "x"},
        {"hemisphere": "side"},
        {"legend_position": "inside"},
    ],
)
def test_stereonet_parameters_are_validated(kwargs):
    with pytest.raises(ValueError, match=next(iter(kwargs))):
        StereoNet(**kwargs)


@pytest.mark.parametrize(
    "kwargs",
    [{"bins": 0}, {"bins": 1.5}, {"pdf_res": 1}, {"kappa": 0}, {"figsize": (5,)}],
)
def test_roseplot_parameters_are_validated(kwargs):
    with pytest.raises(ValueError, match=next(iter(kwargs))):
        RosePlot(**kwargs)


@pytest.mark.parametrize("kwargs", [{"n": 0}, {"n": 5}, {"n": "many"}, {"type": "x"}])
def test_stereogrid_parameters_are_validated(kwargs):
    with pytest.raises(ValueError, match=next(iter(kwargs))):
        StereoGrid(**kwargs)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"levels": 0},  # used to divide by zero at render
        {"levels": []},
        {"sigma": 0},
        {"sigma": -1},  # used to be silently accepted
        {"method": "x"},
        {"colorbar_bounds": (1, 2)},
    ],
)
def test_contour_parameters_are_validated_when_called(kwargs):
    with pytest.raises(ValueError, match=next(iter(kwargs))):
        StereoNet().contour(linset.random_fisher(n=10), **kwargs)
    with pytest.raises(ValueError, match=next(iter(kwargs))):
        StereoNet().density(dist_fisher(lin(1, 2), 5), **kwargs)


def test_failed_render_closes_its_figure(tmp_path, monkeypatch):
    import matplotlib.pyplot as plt

    s = StereoNet()
    s.point(lin(10, 20))

    def broken():
        raise RuntimeError("broken artist")

    monkeypatch.setattr(s, "_plot_artists", broken)
    before = plt.get_fignums()
    with pytest.raises(RuntimeError, match="broken"):
        s.savefig(str(tmp_path / "x.png"))
    assert plt.get_fignums() == before


# -- configuration ---------------------------------------------------------


@pytest.mark.parametrize(
    "key,value",
    [
        ("notation", "xx"),  # used to be stored, then every fol() failed
        ("ndigits", "x"),
        ("ndigits", -5),
        ("figsize", "x"),
        ("dpi", 0),
        ("stress_convention", "x"),
    ],
)
def test_config_values_are_validated_on_assignment(key, value):
    saved = apsg_conf[key]
    with pytest.raises(ValueError, match=key):
        setattr(apsg_conf, key, value)
    assert apsg_conf[key] == saved
    with pytest.raises(ValueError, match=key):
        apsg_conf.update({key: value})
    assert apsg_conf[key] == saved


def test_nested_config_values_are_validated():
    with pytest.raises(ValueError, match="kind"):
        apsg_conf.update({"stereonet": {"kind": "x"}})
    assert apsg_conf.stereonet.kind == "equal-area"
    with pytest.raises(ValueError, match="grid_step"):
        apsg_conf.stereonet.grid_step = 0


def test_unknown_config_keys_raise():
    with pytest.raises(AttributeError, match="not a valid configuration key"):
        apsg_conf.notatoin = "rhr"
    with pytest.raises(KeyError):
        apsg_conf.update({"notatoin": "rhr"})


def test_config_context_restores_on_invalid_value():
    # the values set before the invalid one used to stay changed
    with (
        pytest.raises(ValueError, match="ndigits"),
        apsg_conf_context(notation="rhr", ndigits="x"),
    ):
        pass
    assert apsg_conf.notation == "dd"
    assert apsg_conf.ndigits == 3


# -- CSV files -------------------------------------------------------------


@pytest.mark.parametrize(
    "content,match",
    [
        ("", "is empty"),  # used to raise csv.Error
        ("10\n20\n", "Can't detect the CSV dialect"),
        ("10,20\n30\n40,50\n", "Line 2 .* has 1 columns|dialect"),
        # no header detected for a text data row, so line 1 is reported
        ("azi,inc\n10,abc\n", "Line [12] .*(abc|azi)"),
        ("azi,inc\n10,\n", "Line 2"),
        ("azi,inc\n1e400,30\n", "Line 2 .*not a finite number"),
        ("\x00\x01\x02\xff", "Can't detect the CSV dialect"),  # was an empty set
    ],
)
def test_malformed_csv_raises_value_error(tmp_path, content, match):
    p = tmp_path / "data.csv"
    p.write_text(content, encoding="utf-8")
    with pytest.raises(ValueError, match=match):
        linset.from_csv(str(p))


def test_csv_edge_cases_that_work(tmp_path):
    p = tmp_path / "header.csv"
    p.write_text("azi,inc\n")
    assert len(linset.from_csv(str(p))) == 0  # only a header: an empty set
    p = tmp_path / "single.csv"
    p.write_text("10\n20\n")
    with pytest.raises(ValueError, match="Line 1 .* has 1 columns"):
        linset.from_csv(str(p), delimiter=",")
    p = tmp_path / "pairs.csv"
    p.write_text("1,2,3\n")
    with pytest.raises(ValueError, match="has 3 columns, column 3"):
        pairset.from_csv(str(p))  # used to raise IndexError


# -- rotations -------------------------------------------------------------


def test_rotation_must_be_proper():
    with pytest.raises(ValueError, match="proper rotation"):
        rotation(np.diag([1, 1, -1.0]))  # was accepted with det -1
    with pytest.raises(ValueError, match="orthogonal"):
        rotation(np.eye(3) * 2)
    R = rotation.from_axisangle(lin(120, 30), 40)
    assert isinstance(R @ R, type(R))
    # a product with a reflection is no rotation, but still a valid tensor
    reflected = R @ defgrad(np.diag([1, 1, -1.0]))
    assert not isinstance(reflected, type(R))
    assert np.isclose(np.linalg.det(reflected), -1)


# -- pairwise operations ---------------------------------------------------


@pytest.mark.parametrize(
    "make",
    [
        lambda: vecset.random(40),
        lambda: linset.random(40),
        lambda: folset.random(40),
        lambda: vec2set.random_vonmises(n=40),
        lambda: dir2set.from_directions(np.random.rand(40) * 360),
    ],
)
def test_vectorized_pairwise_operations_match_item_by_item(make):
    from itertools import combinations

    s = make()
    pairs = list(combinations(s.data, 2))
    assert np.allclose(s.angle(), [e.angle(f) for e, f in pairs], atol=1e-9)
    crossed = s.cross()
    expected = np.asarray([e.cross(f) for e, f in pairs], dtype=float)
    assert np.allclose(np.asarray(list(crossed), dtype=float), expected)


def test_pairwise_angle_of_zero_vector_raises():
    with pytest.raises(ValueError, match="zero vector"):
        vecset([vec(0, 0, 0), vec(1, 0, 0)]).angle()


def test_density_of_no_data_raises():
    # used to divide by zero and leave NaN values
    with pytest.raises(ValueError, match="no data"):
        StereoGrid().calculate_density(linset([]))
    with pytest.raises(ValueError, match="no data"):
        StereoNet().contour(linset([]))
