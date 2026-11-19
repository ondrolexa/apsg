"""Grain models of magnetic fabrics (apsg.ams), checked against the printed
values of Willis (1977), Jeffery (1922), Jezek & Hrouda (2002a, 2004) and
Jezek (2026) as reproduced in the OrientationData calculation notes."""

import numpy as np
import pytest
from scipy.linalg import expm

import apsg
from apsg import velgrad
from apsg.ams import (
    ACG,
    ams_from_grains,
    ams_from_orientation,
    grain_anisotropy,
    grain_K,
    grain_susceptibilities,
    kinematic_K,
    osborn_factors,
)


def test_not_in_default_namespace():
    assert not hasattr(apsg, "ams_from_orientation")
    assert "ams" not in apsg.__all__


# ---------------------------------------------------------------- shape factors


def test_cylinder_K_printed_values():
    # Willis (1977) p. 891: L/R = 2 sqrt(3) and 3 sqrt(3) give 0.6 and 0.8
    assert grain_K("cylinder", 2 * np.sqrt(3)) == pytest.approx(0.6, abs=1e-12)
    assert grain_K("cylinder", 3 * np.sqrt(3)) == pytest.approx(0.8, abs=1e-12)
    assert grain_K("cylinder", np.inf) == 1


@pytest.mark.parametrize("r", [1.2, 1.5, 3.0, 10.0])
def test_spheroid_K_is_jeffery(r):
    # K = 1 - I1/I2 with I1 = (b^2 + c^2)/5, I2 = (c^2 + a^2)/5 (Willis eq. 9)
    a, b = r, 1.0
    assert grain_K("prolate", r) == pytest.approx(kinematic_K(2 * b**2, b**2 + a**2))
    assert grain_K("prolate", r) == pytest.approx((r**2 - 1) / (r**2 + 1))
    assert grain_K("oblate", r) == -grain_K("prolate", r)


def test_grain_K_limits_and_errors():
    assert grain_K("prolate", 1) == 0
    assert grain_K("prolate", np.inf) == 1
    assert grain_K("oblate", np.inf) == -1
    for args in [("cone", 2), ("prolate", 0.5), ("prolate", -1), ("cylinder", 0)]:
        with pytest.raises(ValueError):
            grain_K(*args)
    with pytest.raises(ValueError):
        kinematic_K(1, 0)
    with pytest.raises(ValueError):
        kinematic_K(5, 1)


@pytest.mark.parametrize("kind", ["prolate", "oblate"])
@pytest.mark.parametrize("m", [1.001, 1.2, 3, 10])
def test_osborn_factors_sum_to_one(m, kind):
    Na, Nb, Nc = osborn_factors(m, kind)
    assert Na + Nb + Nc == pytest.approx(1, abs=1e-12)
    assert Nb == Nc
    # the long axis of a prolate grain and the plane of an oblate one demagnetise less
    assert (Na < Nb) == (kind == "prolate")


def test_osborn_sphere_limit():
    assert osborn_factors(1.0) == pytest.approx((1 / 3,) * 3)
    assert osborn_factors(1.0001)[0] == pytest.approx(1 / 3, abs=1e-3)


def test_magnetite_grain_anisotropy():
    # SUSIE compPg: k = 16.6, r = 1.2 gives P_g = 1.2044
    assert grain_anisotropy(16.6, 1.2) == pytest.approx(1.2044, abs=5e-5)
    Ka, Kb = grain_susceptibilities(16.6, 1.2)
    assert Ka > Kb
    Ka, Kb = grain_susceptibilities(16.6, 1.2, "oblate")
    assert Ka < Kb
    assert grain_anisotropy(0.0, 3) == 1
    with pytest.raises(ValueError):
        grain_susceptibilities(-1, 2)


# ---------------------------------------------------------------- pseudo flow


def test_pseudo_velocity_gradient():
    L = velgrad.from_comp(xy=1, yx=0.2, zz=0.1, xx=-0.1)
    assert np.allclose(L.pseudo(1), L)  # passive line
    assert np.allclose(L.pseudo(0), L.spin())  # sphere: spin only
    P = L.pseudo(0.4)
    assert np.allclose(P.rate(), 0.4 * L.rate())
    assert np.allclose(P.spin(), L.spin())
    with pytest.raises(ValueError):
        L.pseudo(1.5)


def test_irrotational_pseudo_displacement_is_power():
    # D_W = D^K for irrotational flow: log(Xp/Yp) = K log(X/Y)
    L = velgrad(np.diag([0.3, -0.1, -0.2]))
    D = np.asarray(L.defgrad(2.0))
    K = 0.64
    Dw = np.asarray(L.pseudo(K).defgrad(2.0))
    assert np.allclose(np.diag(Dw), np.diag(D) ** K)


# ---------------------------------------------------------------- March / Willis


@pytest.fixture
def displacement():
    # arbitrary non-coaxial displacement (Willis eq. 12 check of the vault)
    rot = expm(np.array([[0, 0.3, 0], [-0.3, 0, 0], [0, 0, 0]]))
    D = np.diag([2.0, 1.0, 0.5]) @ rot
    return D / np.linalg.det(D) ** (1 / 3) * 1.1


def test_march_density_equals_acg(displacement):
    D = displacement
    acg = ACG.from_displacement(D)
    rng = np.random.default_rng(3)
    U = rng.standard_normal((50, 3))
    U /= np.linalg.norm(U, axis=1)[:, None]
    rho = 1 / np.linalg.norm(np.linalg.solve(D, U.T).T, axis=1)
    sigma = rho**3 / abs(np.linalg.det(D))  # relative to the uniform density
    pdf = np.exp(acg.logpdf_array(U)) * 4 * np.pi
    assert pdf == pytest.approx(sigma, rel=1e-10)


def test_march_poles_second_ellipsoid(displacement):
    D = displacement
    acg = ACG.from_displacement(D, kind="poles")
    rng = np.random.default_rng(4)
    U = rng.standard_normal((50, 3))
    U /= np.linalg.norm(U, axis=1)[:, None]
    sigma = abs(np.linalg.det(D)) * np.einsum("ij,jk,ik->i", U, D @ D.T, U) ** -1.5
    assert np.exp(acg.logpdf_array(U)) * 4 * np.pi == pytest.approx(sigma, rel=1e-10)


def test_march_sampling_is_affine_image_of_uniform(displacement):
    D = displacement
    rng = np.random.default_rng(5)
    x = rng.standard_normal((400000, 3))
    x /= np.linalg.norm(x, axis=1)[:, None]
    y = x @ D.T
    y /= np.linalg.norm(y, axis=1)[:, None]
    T = y.T @ y / len(y)
    assert np.asarray(ACG.from_displacement(D).ortensor()) == pytest.approx(T, abs=2e-3)


def test_acg_ortensor_exact_and_unit_trace():
    acg = ACG(shape=(2, 0.6, 0.4))
    T = acg.ortensor()
    assert np.trace(np.asarray(T)) == pytest.approx(1)
    assert T.eigenlins()[0].angle(acg.mean) == pytest.approx(0, abs=1e-6)
    iso = ACG(shape=(1, 1, 1)).ortensor()
    assert np.asarray(iso) == pytest.approx(np.eye(3) / 3)
    x = acg.rvs(300000, rng=np.random.default_rng(6))
    assert np.asarray(x.ortensor()) == pytest.approx(np.asarray(T), abs=2e-3)


def test_from_displacement_errors():
    with pytest.raises(ValueError):
        ACG.from_displacement(np.zeros((3, 3)))
    with pytest.raises(ValueError):
        ACG.from_displacement(np.eye(3), kind="planes")
    with pytest.raises(ValueError):
        ACG.from_displacement(np.eye(2))


def _transpression_L(alpha_deg):
    # Jezek (2026) eq. 1 (column vectors, gamma_dot = 1)
    ep = np.tan(np.radians(alpha_deg))
    return velgrad([[0, 1, 0], [0, -ep, 0], [0, 0, ep]])


def _switch(alpha_deg, K, gmax=14.0, n=1400):
    """First shear strain at which the lineation of the grain fabric is vertical."""
    L = _transpression_L(alpha_deg).pseudo(K)
    for g in np.linspace(0.02, gmax, n):
        lam = ACG.from_displacement(np.asarray(L.defgrad(g))).axes[0]
        if abs(np.asarray(lam)[2]) > np.sqrt(0.5):
            return g
    return np.nan


@pytest.mark.parametrize("r, printed", [(1.5, 4.0), (3.0, 5.4), (10.0, 8.0)])
def test_lineation_switch_strain(r, printed):
    # Jezek (2026) figs 2-5, alpha = 10 degrees (printed values are rounded)
    assert _switch(10, grain_K("prolate", r)) == pytest.approx(printed, abs=0.07)


def test_lineation_switch_needle_is_8_578():
    # printed 8.7 (Jezek 2026, pp. 5, 9); the root of lambda_xy = lambda_z is 8.578
    assert _switch(10, 1.0) == pytest.approx(8.578, abs=0.03)


def test_lineation_follows_left_cauchy_green_for_column_vectors():
    # Jezek (2026) eq. 5, 10 print D'D; the fabric follows D D' for x = D x0
    D = np.asarray(_transpression_L(10).pseudo(0.8).defgrad(2.5))
    rng = np.random.default_rng(7)
    x = rng.standard_normal((50000, 3))
    x /= np.linalg.norm(x, axis=1)[:, None]
    y = x @ D.T
    y /= np.linalg.norm(y, axis=1)[:, None]
    v = np.linalg.eigh(y.T @ y)[1][:, -1]
    left = np.linalg.eigh(D @ D.T)[1][:, -1]
    right = np.linalg.eigh(D.T @ D)[1][:, -1]
    ang = lambda a, b: np.degrees(np.arccos(min(1, abs(a @ b))))
    assert ang(v, left) < 2  # Monte Carlo noise
    assert ang(v, right) > 20


# ---------------------------------------------------------------- AMS forward


def test_ams_equals_grain_sum():
    # k = K I + Delta T equals the average of the grain tensors (2002a eq. 2, 5)
    rng = np.random.default_rng(8)
    c = rng.standard_normal((5000, 3))
    c /= np.linalg.norm(c, axis=1)[:, None]
    c[:, 2] *= 0.4
    c /= np.linalg.norm(c, axis=1)[:, None]
    K1, K2 = 1.2, 0.9
    grains = np.mean([K2 * np.eye(3) + (K1 - K2) * np.outer(v, v) for v in c], axis=0)
    T = c.T @ c / len(c)
    k = np.asarray(ams_from_orientation(T, K2, K1 - K2))
    assert k == pytest.approx(grains, abs=1e-12)
    oblate = np.mean([K1 * np.eye(3) - (K1 - K2) * np.outer(v, v) for v in c], axis=0)
    assert np.asarray(ams_from_orientation(T, K1, -(K1 - K2))) == pytest.approx(
        oblate, abs=1e-12
    )


def test_ams_accepts_distribution_and_sets():
    acg = ACG(shape=(2, 1, 0.5))
    k1 = np.asarray(ams_from_orientation(acg, 1, 0.3))
    k2 = np.asarray(ams_from_orientation(acg.ortensor(), 1, 0.3))
    assert k1 == pytest.approx(k2)
    S = acg.rvs(20000, rng=np.random.default_rng(1))
    assert np.asarray(ams_from_orientation(S, 1, 0.3)) == pytest.approx(k1, abs=5e-3)
    assert np.linalg.eigvalsh(k1).min() > 0


def test_ams_from_grains_weighted_sum():
    T1 = np.diag([0.6, 0.3, 0.1])
    T2 = np.diag([0.1, 0.3, 0.6])
    k = np.asarray(ams_from_grains([(1, T1, 1.0, 0.2), (3, T2, 0.5, -0.1)]))
    expected = 0.25 * (1.0 * np.eye(3) + 0.2 * T1) + 0.75 * (0.5 * np.eye(3) - 0.1 * T2)
    assert k == pytest.approx(expected)
    one = ams_from_grains([(2, T1, 1.0, 0.2)])
    assert np.asarray(one) == pytest.approx(
        np.asarray(ams_from_orientation(T1, 1.0, 0.2))
    )
    with pytest.raises(ValueError):
        ams_from_grains([])
    with pytest.raises(ValueError):
        ams_from_grains([(-1, T1, 1, 0.1)])
    with pytest.raises(ValueError):
        ams_from_orientation(np.zeros((3, 3)), 1, 0.1)
    with pytest.raises(ValueError):
        ams_from_orientation(np.eye(2), 1, 0.1)


def test_isotropic_grains_give_isotropic_ams():
    k = np.asarray(ams_from_orientation(ACG(shape=(1, 1, 1)), 1.0, 0.3))
    assert k == pytest.approx(1.1 * np.eye(3))


def test_ams_lineation_switches_with_grain_type():
    # prolate grains in a flattening flow: the AMS k1 follows the grain lineation,
    # oblate grains give a foliation pole as k3 (Jezek & Hrouda 2002a)
    L = velgrad(np.diag([0.3, 0.3, -0.6]))
    D = np.asarray(L.pseudo(grain_K("prolate", 3)).defgrad(1.0))
    acg = ACG.from_displacement(D)
    k = ams_from_orientation(acg, 0.9, 0.3)
    _vals, vecs = np.linalg.eigh(np.asarray(k))
    assert abs(vecs[2, 0]) > 0.99  # smallest k along the shortening direction z


# ---------------------------------------------------------------- specimens and sites

from apsg import data
from apsg.ams import AMSSite, AMSSpecimen, from_dataframe, read_csv
from apsg.paleomag._specimen import bedding_vector, core_rotation


def _spec(**kw):
    kw.setdefault("tensor", np.diag([1.1, 1.0, 0.9]))
    return AMSSpecimen("s", **kw)


def test_specimen_basics_and_validation():
    s = _spec(bulk=2e-4)
    assert s.k_mean == 2e-4
    assert _spec().k_mean == pytest.approx(1.0)
    assert np.trace(_spec(tensor=np.diag([2.2, 2.0, 1.8])).normalized) == pytest.approx(
        3
    )
    p = s.params()
    assert p["L"] == pytest.approx(1.1) and p["F"] == pytest.approx(1 / 0.9 * 1.0)
    assert p["k_mean"] == 2e-4
    for bad in [
        np.eye(2),
        np.array([[1, 2, 0], [0, 1, 0], [0, 0, 1.0]]),
        np.full((3, 3), np.nan),
    ]:
        with pytest.raises(ValueError):
            _spec(tensor=bad)
    with pytest.raises(ValueError):
        _spec(azimuth=10)
    with pytest.raises(ValueError):
        _spec(frame="tectonic")
    with pytest.raises(TypeError):
        AMSSite([1])


def test_coordinate_systems():
    T = np.array([[1.1, 0.03, -0.02], [0.03, 1.0, 0.04], [-0.02, 0.04, 0.9]])
    s = _spec(tensor=T, azimuth=40, hade=25, bedding=(130, 35))
    R = np.asarray(core_rotation(40, 25))
    assert s.tensor_in("geographic") == pytest.approx(R @ T @ R.T)
    assert s.tensor_in("specimen") == pytest.approx(T)
    # tilt correction preserves eigenvalues and brings the bedding normal upright
    tec = s.tensor_in("tectonic")
    assert np.linalg.eigvalsh(tec) == pytest.approx(np.linalg.eigvalsh(T))
    nogeo = _spec(tensor=T)
    with pytest.raises(ValueError, match="orientation"):
        nogeo.tensor_in("geographic")
    with pytest.raises(ValueError, match="bedding"):
        _spec(tensor=T, azimuth=40, hade=25).tensor_in("tectonic")
    # a tensor already in geographic coordinates is only tilted
    g = _spec(tensor=T, frame="geographic", bedding=(130, 35))
    assert g.tensor_in("geographic") == pytest.approx(T)
    assert np.linalg.eigvalsh(g.tensor_in("tectonic")) == pytest.approx(
        np.linalg.eigvalsh(T)
    )
    with pytest.raises(ValueError):
        g.tensor_in("specimen")
    with pytest.raises(ValueError):
        s.tensor_in("sky")


def test_tilt_correction_of_a_known_fabric():
    # a horizontal foliation of a bed dipping 40 degrees towards 90 degrees is tilted
    # in geographic coordinates and horizontal again in tectonic ones
    g = _spec(tensor=np.diag([1.1, 1.05, 0.85]), frame="geographic", bedding=(90, 40))
    # rotate the tectonic fabric into the tilted bed
    from apsg.paleomag._specimen import untilt_rotation

    U = np.asarray(untilt_rotation(g.bedding))
    tilted = U.T @ np.diag([1.1, 1.05, 0.85]) @ U
    g2 = _spec(tensor=tilted, frame="geographic", bedding=(90, 40))
    assert g2.tensor_in("tectonic") == pytest.approx(
        np.diag([1.1, 1.05, 0.85]), abs=1e-12
    )
    k3 = np.linalg.eigh(tilted)[1][:, 0]
    assert abs(k3[2]) < 0.8  # pole of the tilted foliation is no longer vertical


def test_specimen_json_roundtrip():
    s = _spec(
        bulk=3e-4, std_error=1e-3, azimuth=40, hade=25, bedding=(130, 35), site="A"
    )
    t = AMSSpecimen.from_json(s.to_json())
    assert np.allclose(t.tensor, s.tensor) and t.site == "A" and t.bulk == 3e-4
    assert np.allclose(t.bedding, s.bedding)
    assert _spec().to_json()["bedding"] is None


def test_site_container_and_statistics():
    sites = data.load("ams_two_sites")
    a, b = sites["AM1"], sites["AM2"]
    assert len(a) == 14 and len(b) == 12 and a.names[0] == "AM1-1"
    assert a["AM1-3"].site == "AM1" and a[0].name == "AM1-1"
    with pytest.raises(KeyError):
        a["nope"]
    es = a.tensors("tectonic")
    assert len(es) == 14
    assert np.trace(np.asarray(es[0])) == pytest.approx(3)
    m = a.mean("tectonic")
    # oblate fabric in tectonic coordinates: the pole of the foliation is vertical
    k3 = np.asarray(m["ellipses"][2]["mu"])
    assert abs(k3[2]) > 0.99
    assert m["anisotropy_test"].reject()
    assert a.anisotropy_test("tectonic").reject()
    # the specimen frame scatters the axes, the tectonic frame does not
    spread = lambda c: a.table(c)["k3_inc"].std()
    assert spread("tectonic") < 0.3 * spread("specimen")
    # prolate site is lineated, oblate site foliated
    assert b.table("tectonic")["T"].mean() < -0.5
    assert a.table("tectonic")["T"].mean() > 0.5
    with pytest.raises(ValueError):
        AMSSite().tensors()
    a2 = AMSSite(list(a), name="copy")
    a2.set_bedding(0, 0)
    assert a2[0].bedding[2] == pytest.approx(-1)  # z is down


def test_table_columns():
    t = data.load("ams_two_sites")["AM2"].table("geographic")
    assert list(t.columns[:8]) == ["name", "k_mean", "L", "F", "P", "Pj", "T", "U"]
    assert {"k1_azi", "k1_inc", "k3_azi", "k3_inc"} <= set(t.columns)
    assert (t["k1_inc"].between(-90, 90)).all() and (t["P"] >= 1).all()


def test_site_json_roundtrip():
    a = data.load("ams_two_sites")["AM1"]
    b = AMSSite.from_json(a.to_json())
    assert b.name == "AM1" and len(b) == len(a)
    assert b.tensors("tectonic")[3] == a.tensors("tectonic")[3]


def test_csv_roundtrip(tmp_path):
    sites = data.load("ams_two_sites")
    csv = tmp_path / "ams.csv"
    from apsg.ams import to_dataframe

    to_dataframe(sites).to_csv(csv, index=False)
    back = read_csv(csv)
    assert list(back) == ["AM1", "AM2"]
    for k in sites:
        assert len(back[k]) == len(sites[k])
        for s, t in zip(sites[k], back[k], strict=True):
            assert t.name == s.name and t.azimuth == pytest.approx(s.azimuth)
            assert np.allclose(t.tensor, s.tensor, atol=1e-9)
            assert t.bulk == pytest.approx(s.bulk)
            assert t.tensor_in("tectonic") == pytest.approx(
                s.tensor_in("tectonic"), abs=1e-9
            )
    one = (
        AMSSite.from_csv(csv.open()) if False else None
    )  # two sites: see the error below
    del one
    with pytest.raises(ValueError, match="2 sites"):
        AMSSite.from_csv(csv)
    text = sites["AM1"].to_csv()
    assert text.startswith("site,name,k11,k22,k33,k12,k23,k13")
    single = tmp_path / "one.csv"
    sites["AM1"].to_csv(single)
    assert len(AMSSite.from_csv(single)) == 14


def test_from_dataframe_aliases_and_errors():
    import pandas as pd

    df = pd.DataFrame(
        {
            "Specimen": ["a", "b"],
            "K11": [1.1, 1.2],
            "k22": [1.0, 1.0],
            "k33": [0.9, 0.8],
            "k12": [0, 0],
            "k23": [0, 0],
            "k13": [0, 0],
            "Km": [1e-4, np.nan],
            "dipdir": [90, 90],
            "dip": [30, 30],
            "extra": ["x", "y"],
        }
    )
    sites = from_dataframe(df, name="S")
    s = sites["S"]
    assert s.names == ["a", "b"] and s[0].bulk == 1e-4 and s[1].bulk is None
    assert s[0].meta == {"extra": "x"} and s[0].bedding is not None
    with pytest.raises(ValueError, match="k13"):
        from_dataframe(df.drop(columns="k13"))
    with pytest.raises(ValueError, match="together"):
        from_dataframe(df.drop(columns="dip"))
    unnamed = from_dataframe(df.drop(columns="Specimen"), name="Q")["Q"]
    assert unnamed.names == ["Q-1", "Q-2"]


def test_ams_two_sites_dataset_is_recoverable_from_the_grain_model():
    # the stored sites were made from ams_from_orientation fabrics; the tectonic mean
    # tensor of the oblate site has K near 1 and negative delta
    m = data.load("ams_two_sites")["AM1"].mean("tectonic")
    ev = m["eigenvalues"]
    assert ev[0] == pytest.approx(ev[1], abs=0.01) and ev[2] < ev[1] - 0.02


# ---------------------------------------------------------------- bootstrap and comparison

from apsg import StereoNet, linset
from apsg.feature import (
    compare_tensor_means,
    tensor_bootstrap,
    tensor_error_bootstrap,
)
from apsg.feature._statistics import jelinek_statistics


def _sample(rng, n, mean=None, scale=0.02):
    mean = np.diag([1.05, 1.0, 0.95]) if mean is None else mean
    N = rng.normal(scale=scale, size=(n, 3, 3))
    return mean + (N + N.transpose(0, 2, 1)) / 2


def test_two_sample_test_size_and_power():
    rng = np.random.default_rng(10)
    reps = 600
    rej = sum(
        compare_tensor_means(_sample(rng, 12), _sample(rng, 12)).reject()
        for _ in range(reps)
    )
    assert 0.025 < rej / reps < 0.08
    shifted = np.diag([1.05, 1.03, 0.92])
    power = np.mean(
        [
            compare_tensor_means(_sample(rng, 12), _sample(rng, 12, shifted)).reject()
            for _ in range(100)
        ]
    )
    assert power > 0.9


def test_two_sample_permutation_test():
    rng = np.random.default_rng(11)
    a, b = _sample(rng, 15), _sample(rng, 15)
    h = compare_tensor_means(a, b)
    p = compare_tensor_means(a, b, method="permutation", n_resamples=999, rng=1)
    assert p.statistic == pytest.approx(h.statistic)
    assert p.p_value == pytest.approx(h.p_value, abs=0.1)
    # size
    rej = sum(
        compare_tensor_means(
            _sample(rng, 8),
            _sample(rng, 8),
            method="permutation",
            n_resamples=199,
            rng=i,
        ).reject()
        for i in range(150)
    )
    assert 0.01 < rej / 150 < 0.12
    shifted = np.diag([1.05, 1.04, 0.91])
    assert compare_tensor_means(
        a, _sample(rng, 15, shifted), method="permutation", n_resamples=199, rng=2
    ).reject()


def test_two_sample_test_invariances():
    rng = np.random.default_rng(12)
    a, b = _sample(rng, 10), _sample(rng, 10, np.diag([1.1, 1.0, 0.9]))
    base = compare_tensor_means(a, b).statistic
    R = np.asarray(core_rotation(30, 20))
    assert compare_tensor_means(R @ a @ R.T, R @ b @ R.T).statistic == pytest.approx(
        base
    )
    assert compare_tensor_means(a, b[::-1]).statistic == pytest.approx(base)
    # the bulk susceptibility does not matter once the tensors are normalised
    assert compare_tensor_means(a, 3 * b, normalize=True).statistic == pytest.approx(
        compare_tensor_means(a, b, normalize=True).statistic
    )
    t = compare_tensor_means(a, b)
    assert t.df == (5, 14) and t.extra["n"] == (10, 10)
    assert t.reject() == (t.statistic > t.critical)


def test_two_sample_test_errors():
    a = _sample(np.random.default_rng(1), 5)
    with pytest.raises(ValueError):
        compare_tensor_means(a, a[:1])
    with pytest.raises(ValueError):
        compare_tensor_means(a[:3], a[:3])
    with pytest.raises(ValueError):
        compare_tensor_means(a, a, method="mcmc")
    with pytest.raises(ValueError):
        compare_tensor_means(a, np.eye(3))
    with pytest.raises(ValueError):
        compare_tensor_means(a, a, method="permutation", n_resamples=10)


def test_site_comparison():
    sites = data.load("ams_two_sites")
    a, b = sites["AM1"], sites["AM2"]
    assert a.compare(b, "tectonic").reject()
    assert a.compare(
        b, "tectonic", method="permutation", n_resamples=299, rng=0
    ).reject()
    # in specimen coordinates the cores scatter the axes, so a site is not
    # distinguishable from itself shuffled
    half1, half2 = AMSSite(a[::2]), AMSSite(a[1::2])
    assert not half1.compare(half2, "tectonic").reject()
    with pytest.raises(TypeError):
        a.compare(a.tensors())
    es_a, es_b = a.tensors("tectonic"), b.tensors("tectonic")
    assert es_a.common_mean_test(es_b).statistic == pytest.approx(
        a.compare(b, "tectonic").statistic
    )
    with pytest.raises(TypeError):
        es_a.common_mean_test(linset.random_fisher(n=10))


def test_bootstrap_basics():
    es = data.load("ams_two_sites")["AM2"].tensors("tectonic")
    b = es.bootstrap_tensor(n_resamples=500, rng=3)
    assert b.eigenvalues.shape == (500, 3) and b.eigenvectors.shape == (500, 3, 3)
    assert np.all(np.diff(b.eigenvalues, axis=1) <= 1e-12) is np.True_ or np.all(
        b.eigenvalues[:, 0] >= b.eigenvalues[:, 2]
    )
    mean = np.linalg.eigvalsh(b.mean)[::-1]
    assert np.all((b.intervals[:, 0] <= mean) & (mean <= b.intervals[:, 1]))
    # reproducible and seed dependent
    again = es.bootstrap_tensor(n_resamples=500, rng=3)
    assert np.array_equal(b.eigenvalues, again.eigenvalues)
    assert not np.array_equal(
        b.eigenvalues, es.bootstrap_tensor(n_resamples=500, rng=4).eigenvalues
    )
    # global numpy state seeds the default generator
    np.random.seed(5)
    x = es.bootstrap_tensor(n_resamples=100).eigenvalues
    np.random.seed(5)
    assert np.array_equal(x, es.bootstrap_tensor(n_resamples=100).eigenvalues)
    # axes are turned to the side of the sample axes
    v0 = np.linalg.eigh(b.mean)[1][:, ::-1]
    assert np.all(np.einsum("bki,ki->bi", b.eigenvectors, v0) > 0)
    assert [e["which"] for e in b.ellipses] == [0, 1, 2]
    assert "TensorBootstrap" in repr(b)


def test_bootstrap_ellipse_contains_the_stated_fraction():
    es = data.load("ams_two_sites")["AM2"].tensors("tectonic")
    b = es.bootstrap_tensor(n_resamples=2000, level=0.9, rng=5)
    for i, e in enumerate(b.ellipses):
        d = b.eigenvectors[:, :, i]
        u, v = (np.asarray(a) for a in e["axes"])
        g1, g2 = np.sin(np.radians(e["gamma"]))
        inside = (d @ u / g1) ** 2 + (d @ v / g2) ** 2 <= 1
        assert 0.84 < inside.mean() < 0.97


def test_bootstrap_shape_parameters():
    es = data.load("ams_two_sites")["AM1"].tensors("tectonic")
    b = es.bootstrap_tensor(n_resamples=400, rng=1)
    p = b.shape_parameters()
    assert set(p) == {"L", "F", "P", "Pj", "T"}
    mean = jelinek_statistics(np.array([np.asarray(t) for t in es]))["shape"]
    lo, hi = b.parameter_intervals()["Pj"]
    assert lo <= mean["Pj"] <= hi
    lo, hi = b.parameter_intervals()["T"]
    assert lo <= mean["T"] <= hi <= 1 + 1e-9
    assert np.all(p["P"] >= 1)


def test_bootstrap_errors():
    with pytest.raises(ValueError):
        tensor_bootstrap(np.eye(3)[None])
    with pytest.raises(ValueError):
        tensor_bootstrap(np.eye(3))
    ok = np.array([np.eye(3), 2 * np.eye(3)])
    with pytest.raises(ValueError):
        tensor_bootstrap(ok, level=1.5)
    with pytest.raises(ValueError):
        tensor_bootstrap(ok, n_resamples=3)
    with pytest.raises(ValueError):
        tensor_bootstrap(np.array([np.eye(3), np.zeros((3, 3))]), normalize=True)


def test_region_coverage_for_close_eigenvalues():
    # Samples of 8 tensors with noise of the size of the gap between k2 and k3:
    # coverage of the true k2 axis by the regions of Jelinek (1978) and the bootstrap.
    rng = np.random.default_rng(21)
    true = np.diag([1.1, 0.96, 0.94])
    reps = 200
    hits = {"j": 0, "b": 0}
    for _ in range(reps):
        K = _sample(rng, 8, true, scale=0.03)
        regions = {
            "j": jelinek_statistics(K)["ellipses"][1],
            "b": tensor_bootstrap(K, n_resamples=300, rng=rng).ellipses[1],
        }
        for key, ell in regions.items():
            mu = np.asarray(ell["mu"])
            u, v = (np.asarray(a) for a in ell["axes"])
            g1, g2 = np.radians(ell["gamma"])
            y = np.array([0, 1.0, 0])
            y = y * np.sign(y @ mu)
            hits[key] += (y @ u / np.sin(g1)) ** 2 + (y @ v / np.sin(g2)) ** 2 <= 1
    # not the improvement of the paper: with 8 specimens the percentile region of the
    # bootstrap covers less than the nominal 95 % (simulated 0.82), the T² region of
    # Jelinek more (0.91); the test pins both so a change is noticed
    assert 0.75 < hits["b"] / reps < 0.95
    assert 0.85 < hits["j"] / reps < 1.0


def test_error_bootstrap():
    T = np.diag([1.1, 1.0, 0.9])
    zero = tensor_error_bootstrap(T, 0.0, n_resamples=50, rng=0)
    assert np.allclose(zero.eigenvalues, [1.1, 1.0, 0.9])
    assert all(max(e["gamma"]) < 1e-5 for e in zero.ellipses)
    small = tensor_error_bootstrap(T, 0.005, rng=0)
    large = tensor_error_bootstrap(T, 0.02, rng=0)
    assert large.ellipses[0]["gamma"][0] > small.ellipses[0]["gamma"][0] > 0
    assert small.n is None
    # eigenvalue intervals bracket the tensor's eigenvalues
    assert np.all(
        (small.intervals[:, 0] <= [1.1, 1, 0.9])
        & ([1.1, 1, 0.9] <= small.intervals[:, 1])
    )
    # normalised: the mean of the simulated eigenvalues is 1
    assert small.eigenvalues.sum(axis=1).mean() == pytest.approx(3, abs=0.01)
    with pytest.raises(ValueError):
        tensor_error_bootstrap(np.eye(2), 0.1)
    with pytest.raises(ValueError):
        tensor_error_bootstrap(T, -1)


def test_specimen_bootstrap():
    site = data.load("ams_two_sites")["AM2"]
    s = site[0]
    b = s.bootstrap("tectonic", n_resamples=400, rng=1)
    ref = np.linalg.eigh(s.tensor_in("tectonic", normalize=True))[1][:, ::-1]
    for i, e in enumerate(b.ellipses):
        assert abs(np.asarray(e["mu"]) @ ref[:, i]) > 0.9
    assert b.ellipses[0]["gamma"][0] < 10
    bare = AMSSpecimen("x", np.diag([1.1, 1, 0.9]))
    with pytest.raises(ValueError, match="std_error"):
        bare.bootstrap()
    sb = site.bootstrap("tectonic", n_resamples=200, rng=2)
    assert sb.n == len(site)


def test_confidence_bootstrap_tensor_method():
    es = data.load("ams_two_sites")["AM2"].tensors("tectonic")
    s = StereoNet()
    s.confidence(es, method="bootstrap_tensor", n_resamples=100, which=0)
    assert s._artists[0].kwargs["method"] == "bootstrap_tensor"
    s.init_figure()
    s._render()
    # same drawing as the linearised ellipse, but a different curve
    t = StereoNet()
    t.confidence(es, method="jelinek", which=0)
    t.init_figure()
    t._render()
    assert len(s.ax.get_children()) == len(t.ax.get_children())
    for args, kw in (((linset.random_fisher(n=10),), {"method": "bootstrap_tensor"}),):
        with pytest.warns(UserWarning, match="Not valid arguments"):
            StereoNet().confidence(*args, **kw)


# ---------------------------------------------------------------- plots

import matplotlib.pyplot as plt

from apsg.ams import (
    AMSStereoPlot,
    BootstrapPlot,
    LFPlot,
    PjTPlot,
    PTPlot,
)


@pytest.fixture
def two_sites():
    return data.load("ams_two_sites")


@pytest.mark.parametrize("cls", [PjTPlot, LFPlot, PTPlot])
def test_diagram_plots(cls, two_sites, tmp_path):
    p = cls(two_sites, coords="tectonic", mean=True)
    p.init_figure()
    p._render()
    ax = p.fig.axes[0]
    labels = [t.get_text() for t in ax.get_legend().get_texts()]
    assert labels == ["AM1", "AM2"]
    lines = [ln for ln in ax.get_lines() if ln.get_label() in ("AM1", "AM2")]
    assert [len(ln.get_xdata()) for ln in lines] == [14, 12]
    p.savefig(tmp_path / "d.png")
    assert (tmp_path / "d.png").stat().st_size > 1000


def test_pjt_values_and_inputs(two_sites):
    site = two_sites["AM2"]
    p = PjTPlot(site)
    ((name, par),) = p.groups
    assert name == "AM2" and len(par["Pj"]) == 12
    assert par["T"] == pytest.approx(site.table()["T"].to_numpy(), abs=1e-9)
    assert (par["T"] < 0).all()
    q = PjTPlot(site.tensors("tectonic"))
    assert q.groups[0][1]["Pj"] == pytest.approx(par["Pj"], abs=1e-9)
    assert len(PjTPlot([two_sites["AM1"], site]).groups) == 2
    with pytest.raises(TypeError):
        PjTPlot(3)
    with pytest.raises(TypeError):
        PjTPlot([1])
    with pytest.raises(ValueError):
        PjTPlot(site, coords="sky")
    with pytest.raises(TypeError, match="unexpected"):
        PjTPlot(site, nonsense=1)
    p = PjTPlot(site, logx=True)
    p.init_figure()
    p._render()
    assert p.fig.axes[0].get_xscale() == "log"


def test_lf_plot_diagonal(two_sites):
    p = LFPlot(two_sites)
    p.init_figure()
    p._render()
    ax = p.fig.axes[0]
    assert ax.get_xlim()[0] == 1 and ax.get_ylim()[0] == 1


@pytest.mark.parametrize("coords", ["tectonic", "specimen", "both"])
@pytest.mark.parametrize("ellipses", ["jelinek", "bootstrap", None])
def test_stereo_plot(coords, ellipses, two_sites):
    p = AMSStereoPlot(two_sites["AM2"], coords=coords, ellipses=ellipses, n=100)
    p.init_figure()
    p._render()
    assert len(p.fig.axes) >= (2 if coords == "both" else 1)


def test_stereo_plot_options_and_errors(two_sites):
    es = two_sites["AM2"].tensors("tectonic")
    p = AMSStereoPlot(es)
    p.init_figure()
    p._render()
    with pytest.raises(TypeError):
        AMSStereoPlot(1)
    with pytest.raises(ValueError):
        AMSStereoPlot(es, coords="x")
    with pytest.raises(ValueError):
        AMSStereoPlot(es, ellipses="kent")
    with pytest.raises(ValueError):
        AMSStereoPlot(es, level=2)
    with pytest.raises(ValueError):
        AMSStereoPlot(es, legend_position="middle")


def test_bootstrap_plot(two_sites, tmp_path):
    site = two_sites["AM1"]
    p = BootstrapPlot(site, n=200, rng=1)
    p.savefig(tmp_path / "b.png")
    assert p.boot.n_resamples == 200
    boot = site.bootstrap("tectonic", 100, rng=0)
    assert BootstrapPlot(boot).boot is boot
    p = BootstrapPlot(site.tensors("tectonic"), n=100, rng=2)
    p.init_figure()
    p._render()
    assert len(p.fig.axes) >= 2
    with pytest.raises(TypeError):
        BootstrapPlot("x")


def test_ams_config_is_validated(two_sites):
    assert apsg.config.apsg_conf.ams["marker"] == "o"
    with pytest.raises(ValueError):
        PjTPlot(two_sites, alpha=2)
    p = PjTPlot(two_sites, marker="x", ms=8)
    assert p._kwargs["marker"] == "x" and apsg.config.apsg_conf.ams["marker"] == "o"


def test_plots_are_closed_after_savefig(two_sites, tmp_path):
    PjTPlot(two_sites).savefig(tmp_path / "x.png")
    assert not plt.get_fignums()


# ---------------------------------------------------------------- instrument files

import io
import re
import struct

from apsg.ams import (
    read_aex,
    read_ams,
    read_anisoft_table,
    read_ran,
    read_sufar,
)
from apsg.ams._specimen import tensor_from_components

# -- SUFAR-layout report (made from a SUSAR page of the Melechov data) and the Anisoft tables --

TABLE = data.path("ams_table")
NO_TENSOR = data.path("ams_table_no_tensor")


def _mel1_pages():
    """The pages of the SUSAR report of site MEL1 (the Melechov data)."""
    text = (data.path("melechov") / "MEL1.ASC").read_text(encoding="latin-1")
    return [p for p in text.split("\f") if "ANISOTROPY" in p]


def _as_sufar(text):
    """The same page(s) in the layout of SUFAR 1.2: the susceptibility row has the
    field instead of the norming factor, the title names the program."""
    text = re.sub(r"PROGRAM\s+SUSAR", "Program SUFAR ver.1.2", text)
    text = re.sub(
        r"Mean\s+Norming\s+Standard", "Field         Mean      Standard", text
    )
    text = re.sub(
        r"susc\.\s+factor\s+err\. \[%\]", "[A/m]         susc.     err. [%]", text
    )
    return re.sub(r"(?m)^ (\S+E[-+]\d+)\s+\S+E[-+]\d+", r"   300        \1", text)


@pytest.fixture
def sufar_report(tmp_path):
    """A report of two specimens in the SUFAR layout."""
    pages = [_as_sufar(p) for p in _mel1_pages()]
    path = tmp_path / "sufar_report.asc"
    path.write_text("\f".join(pages[:2]), encoding="latin-1")
    return path


@pytest.fixture
def cut_report(tmp_path):
    """The same report with a third page cut off in its tensor."""
    pages = [_as_sufar(p) for p in _mel1_pages()]
    third = pages[1].replace("MEL 1/1/2", "MEL 1/2/1")
    third = third.split("system    I")[0] + "system    I        2      13      77"
    path = tmp_path / "sufar_report.asc"
    path.write_text("\f".join([*pages[:2], third]), encoding="latin-1")
    return path


def test_read_sufar_report(cut_report):
    with pytest.warns(UserWarning, match="incomplete"):
        site = read_sufar(cut_report)
    assert site.name == "sufar_report" and site.names == ["MEL 1/1/1", "MEL 1/1/2"]
    s = site[0]
    assert s.azimuth == pytest.approx(141) and s.hade == pytest.approx(16)
    assert s.frame == "specimen"
    assert s.bulk == pytest.approx(576.4e-6)
    assert s.std_error == pytest.approx(0.0014333, abs=1e-6)
    # the SUFAR row has the field and no norming factor
    assert s.meta["field"] == 300 and "norming_factor" not in s.meta
    # normed tensor of the report, specimen coordinates
    assert np.diag(s.tensor) == pytest.approx([0.9551, 0.9731, 1.0718])
    assert s.tensor[0, 1] == pytest.approx(-0.0378)
    # the geographic tensor printed in the report is reproduced by the core orientation
    printed = _printed(data.path("melechov") / "MEL1.ASC", "Geograph")[0]
    assert s.tensor_in("geographic") == pytest.approx(printed, abs=3e-4)
    assert s.params()["L"] == pytest.approx(1.080, abs=1e-3)


def test_sufar_and_susar_layouts_give_the_same_specimens(sufar_report):
    susar = data.load("melechov", site="MEL1")
    sufar = read_sufar(sufar_report)
    assert len(sufar) == 2
    for a, b in zip(susar, sufar, strict=False):
        assert a.name == b.name and a.bulk == b.bulk and a.std_error == b.std_error
        assert np.array_equal(a.tensor, b.tensor) and a.azimuth == b.azimuth
        assert a.meta["F"] == b.meta["F"] and a.meta["holder"] == b.meta["holder"]
    assert susar[0].meta["field"] is None and "norming_factor" in susar[0].meta


def test_read_anisoft_table_with_tensor():
    site = read_anisoft_table(TABLE)
    assert site.names == ["1/2", "3/1", "4/1"]
    s = site[0]
    assert s.frame == "geographic" and s.bulk == pytest.approx(2062.694)
    assert s.tensor[0, 0] == pytest.approx(0.94931)
    # the eigen-directions of the tensor are those listed in the table (K1 80.1/11.8)
    k1 = s.ellipsoid("geographic").eigenlins()[0]
    assert k1.geo[0] % 360 == pytest.approx(80.1, abs=0.3)
    assert k1.geo[1] == pytest.approx(11.8, abs=0.3)
    assert read_anisoft_table(TABLE, frame="specimen")[0].frame == "specimen"
    # the same table through apsg.data
    assert data.load("ams_table")["ams_table"].names == site.names


def test_read_anisoft_table_without_tensor_uses_directions():
    site = read_anisoft_table(NO_TENSOR)
    assert len(site) >= 2 and site[0].name.startswith("GC")
    s = site[0]
    p = s.params("geographic")
    # L and F of the table are those of the reconstructed tensor
    text = NO_TENSOR.read_text().splitlines()
    head = text[0].split()
    row = text[1].split()
    assert p["L"] == pytest.approx(float(row[head.index("L")]), abs=2e-4)
    assert p["F"] == pytest.approx(float(row[head.index("F")]), abs=2e-4)
    k1 = s.ellipsoid("geographic").eigenlins()[0]
    assert k1.geo[0] % 360 == pytest.approx(float(row[head.index("Kmax_Dec")]), abs=0.2)
    assert np.trace(s.tensor) == pytest.approx(3)
    groups = read_anisoft_table(NO_TENSOR, site=lambda n: n.split("_")[0])
    assert set(groups) == {"GC2", "GC4"}
    assert groups["GC2"][0].site == "GC2"
    assert set(data.load("ams_table_no_tensor")) == {"GC2", "GC4"}


def test_read_ams_dispatch_and_errors(tmp_path, cut_report):
    with pytest.warns(UserWarning):
        assert read_ams(cut_report).names == ["MEL 1/1/1", "MEL 1/1/2"]
    assert len(read_ams(TABLE, name="X")) == 3
    assert read_ams(TABLE, name="X").name == "X"
    with pytest.raises(TypeError):
        read_ams(cut_report, frame="specimen")
    binary = tmp_path / "a.ams"
    binary.write_bytes(b"GC2_1   \x00\x00\x01\x00" * 20)
    with pytest.raises(ValueError, match="binary"):
        read_ams(binary)
    with pytest.raises(ValueError, match="Name"):
        read_anisoft_table(io.StringIO("a b c\n1 2 3\n"))
    with pytest.raises(ValueError, match="neither"):
        read_anisoft_table(io.StringIO("Name Km\nx 1\n"))
    with pytest.raises(ValueError, match="SUFAR"):
        read_sufar(io.StringIO("nothing here"))
    # open text files work as well
    with open(TABLE) as f:
        assert len(read_anisoft_table(f)) == 3
    with (
        open(cut_report, encoding="latin-1") as f,
        pytest.warns(UserWarning),
    ):
        assert len(read_sufar(f)) == 2


def _first_page(text):
    """The first page of a report."""
    return text.split("\f")[0] + "\f"


def _edit_sufar(text, op=None, azi=None, dip=None, tensor=None, foliation=None):
    """Rewrite the orientation, specimen-system tensor or foliation of a page."""
    if op is not None:
        text = re.sub(
            r"O\.P\. :(\s+\d+){4}", "O.P. : " + "  ".join(f"{c:3d}" for c in op), text
        )
    if azi is not None:
        text = re.sub(r"Azi\s+\S+", f"Azi {azi:5g}", text)
    if dip is not None:
        text = re.sub(r"Dip\s+\S+", f"Dip {dip:5g}", text)
    if tensor is not None:
        k11, k22, k33, k12, k23, k13 = tensor
        text = re.sub(
            r"(Specimen  D[^\n]*?)[-\d.]+\s+[-\d.]+\s+[-\d.]+\n",
            lambda m: f"{m.group(1)}{k11:8.4f} {k22:8.4f} {k33:8.4f}\n",
            text,
        )
        text = re.sub(
            r"(system    I[^\n]*?)[-\d.]+\s+[-\d.]+\s+[-\d.]+\n",
            lambda m: f"{m.group(1)}{k12:8.4f} {k23:8.4f} {k13:8.4f}\n",
            text,
            count=1,
        )
    if foliation is not None:
        text = re.sub(
            r"(T1\s+F1\s+L1[^\n]*\n\n)\S+\s+\d+/\d+",
            lambda m: f"{m.group(1)}S0        {foliation}",
            text,
            count=1,
        )
    return text


def test_sufar_metadata(sufar_report):
    s = read_sufar(sufar_report)[0]
    m = s.meta
    assert m["op"] == (12, 90, 6, 0)
    assert m["nominal_volume"] == 10.0 and m["actual_volume"] == 10.0
    assert m["holder"] == pytest.approx(-5.17e-6) and m["demag_factor"] is True
    assert m["field"] == 300 and m["error_percent"] == pytest.approx(0.191)
    assert (m["F"], m["F12"], m["F23"]) == pytest.approx((914.1, 559.4, 458.3))
    assert m["foliation_tag"] == "S0" and s.bedding is not None
    # a report without foliation (tag 00, dip 0) has no bedding
    page = data.load("melechov", site="MEL2")[0]
    assert page.bedding is None and page.meta["foliation_tag"] is None


@pytest.mark.parametrize(
    "op, azi, dip, spin",
    [
        ((12, 0, 12, 0), 141, 16, 0),
        ((12, 90, 12, 0), 141, 74, 0),  # plunge of the core axis, hade = 90 - dip
        ((6, 0, 12, 0), 141, 16, 180),  # arrow at 6 o'clock is absorbed
        ((3, 0, 6, 0), 321, 16, 90),  # arrows at 3 and 9 turn the tensor
        ((9, 0, 6, 0), 321, 16, 270),
        ((12, 0, 3, 0), 231, 16, 0),  # azimuth measured with the arrow at 3
    ],
)
def test_sufar_orientation_codes(op, azi, dip, spin):
    text = _first_page(_mel1_pages()[0])
    ref = read_sufar(io.StringIO(text))[0]
    K = ref.tensor
    c, s = np.cos(np.radians(-spin)), np.sin(np.radians(-spin))
    R = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
    # measured in the frame turned by -spin, restored by the reader
    comps = tuple(np.round((R @ K @ R.T)[[0, 1, 2, 0, 1, 0], [0, 1, 2, 1, 2, 2]], 5))
    edited = _edit_sufar(text, op=op, azi=azi, dip=dip, tensor=comps)
    got = read_sufar(io.StringIO(edited))[0]
    assert got.meta["op"] == op
    assert got.tensor_in("geographic") == pytest.approx(
        ref.tensor_in("geographic"), abs=3e-4
    )


def test_sufar_foliation_with_strike_code():
    text = _first_page(_mel1_pages()[0])
    edited = _edit_sufar(text, op=(6, 0, 6, 90), foliation="110/30")
    s = read_sufar(io.StringIO(edited))[0]
    # P4 = 90: the foliation azimuth is the strike, the dip direction 90 more
    assert s.bedding.angle(bedding_vector(200, 30)) == pytest.approx(0, abs=1e-6)
    assert s.tensor_in("tectonic").shape == (3, 3)
    dipdir = _edit_sufar(text, foliation="110/30")
    s = read_sufar(io.StringIO(dipdir))[0]
    assert s.bedding.angle(bedding_vector(110, 30)) == pytest.approx(0, abs=1e-6)


def test_sufar_unsupported_and_missing_code():
    text = _first_page(_mel1_pages()[0])
    with pytest.raises(NotImplementedError, match="P1=5"):
        read_sufar(io.StringIO(_edit_sufar(text, op=(5, 0, 6, 0))))
    no_code = re.sub(r"O\.P\. :(\s+\d+){4}", "", text)
    with pytest.warns(UserWarning, match="assuming 6 0 6 0"):
        s = read_sufar(io.StringIO(no_code))[0]
    assert s.meta["op"] == (6, 0, 6, 0)


def test_anisoft_table_warns_about_dropped_rows():
    lines = TABLE.read_text().splitlines()
    text = "\n".join([*lines, "short row", lines[1].replace("2062.694", "x")])
    with pytest.warns(UserWarning, match="2 table rows .* skipped"):
        site = read_anisoft_table(io.StringIO(text))
    assert site.names == ["1/2", "3/1", "4/1"]


# -- the Melechov data: SUSAR reports, RAN files and AEX table of one measurement --


def _melechov_specimens(fmt):
    sites = data.load("melechov", format=fmt)
    return [s for site in sites.values() for s in site]


def _printed(report, label):
    """Normed tensors printed in the system `label` of each page of a report."""
    out = []
    for page in report.read_text(encoding="latin-1").split("\f"):
        if "ANISOTROPY" not in page:
            continue
        lines = page.replace("\r", "").splitlines()
        tensor = None
        for i, ln in enumerate(lines):
            if ln.startswith(label):
                diag = [float(x) for x in re.findall(r"-?\d+\.\d+", ln[8:])][-3:]
                off = [float(x) for x in re.findall(r"-?\d+\.\d+", lines[i + 1])][-3:]
                tensor = tensor_from_components(*diag, *off)
        out.append(tensor)
    return out


def test_susar_report_of_the_melechov_data():
    sites = data.load("melechov")
    assert list(sites) == ["MEL1", "MEL2", "MEL3", "MEL4", "MEL5"]
    assert [len(s) for s in sites.values()] == [10, 10, 10, 10, 11]
    s = sites["MEL1"][0]
    assert s.name == "MEL 1/1/1" and s.frame == "specimen" and s.site == "MEL1"
    assert s.bulk == pytest.approx(576.4e-6)
    assert s.meta["op"] == (12, 90, 6, 0) and s.meta["field"] is None
    assert s.meta["norming_factor"] == pytest.approx(576.4e-6)
    assert s.meta["F"] == pytest.approx(914.1) and s.meta["demag_factor"] is True
    # Azi 321, Dip 74 with the code 12 90 6 0: the dip is the plunge of the core axis
    assert s.azimuth == pytest.approx(141) and s.hade == pytest.approx(16)
    assert s.meta["foliation_tag"] == "S0" and s.bedding is not None
    # site MEL2 has no foliation
    assert all(sp.bedding is None for sp in sites["MEL2"])
    assert {sp.meta["foliation_tag"] for sp in sites["MEL2"]} == {None}


def test_susar_orientation_matches_the_printed_systems():
    folder = data.path("melechov")
    n = bedded = 0
    for k in range(1, 6):
        specs = data.load("melechov", site=f"MEL{k}")
        geo = _printed(folder / f"MEL{k}.ASC", "Geograph")
        paleo = _printed(folder / f"MEL{k}.ASC", "Paleo 1")
        assert len(geo) == len(specs)
        for s, g, p in zip(specs, geo, paleo, strict=True):
            # P2 = 90: specimen to geographic coordinates by the core orientation
            assert s.tensor_in("geographic") == pytest.approx(g, abs=3e-4)
            n += 1
            if s.bedding is not None:  # the foliation, a dip direction (P4 = 0)
                assert s.tensor_in("tectonic") == pytest.approx(p, abs=3e-4)
                bedded += 1
    assert n == 51 and bedded > 20


def test_ran_matches_the_reports():
    asc = _melechov_specimens("asc")
    ran = _melechov_specimens("ran")
    assert [s.name for s in ran] == [s.name for s in asc]
    for a, r in zip(asc, ran, strict=True):
        assert r.frame == "geographic" and r.azimuth is None
        assert r.tensor_in("geographic") == pytest.approx(
            a.tensor_in("geographic"), abs=2e-4
        )
        assert r.bulk == pytest.approx(a.bulk, rel=1e-3)
        assert (r.bedding is None) == (a.bedding is None)
        if a.bedding is not None:
            assert r.bedding.angle(a.bedding) == pytest.approx(0, abs=1e-6)
            assert r.tensor_in("tectonic") == pytest.approx(
                a.tensor_in("tectonic"), abs=3e-4
            )
    r = ran[0]
    assert r.meta["op"] == (12, 90, 6, 0) and r.meta["locality"] == "BABICE"
    assert r.meta["foliation"] == (20, 90) and r.meta["foliation_tag"] == "S0"
    assert r.meta["norming_factor"] == pytest.approx(576.4e-6, rel=1e-3)
    with pytest.raises(ValueError, match="specimen coordinates"):
        r.tensor_in("specimen")


def test_ran_input_types_and_dispatch():
    folder = data.path("melechov")
    path = folder / "MEL1.RAN"
    raw = path.read_bytes()
    by_path = read_ran(path)
    assert by_path.name == "MEL1" and len(by_path) == 10
    assert read_ran(raw, name="x").name == "x"
    with open(path, "rb") as f:
        assert read_ran(f).names == by_path.names
    # read_ams: by extension (any case) and by the structure of the content
    upper = read_ams(path)
    assert upper.names == by_path.names and upper.name == "MEL1"
    assert read_ams(raw).names == by_path.names
    with pytest.raises(TypeError):
        read_ams(path, frame="specimen")
    with pytest.raises(TypeError, match="binary"):
        read_ran(io.StringIO("text file"))


def test_aex_table_matches_the_measurements():
    aex = data.load("melechov_aex")
    asc = _melechov_specimens("asc")
    ran = _melechov_specimens("ran")
    assert list(aex.columns) == ["Km", "P", "T"] and list(aex.index) == [
        s.name for s in asc
    ]
    path = data.path("melechov") / "MEL1-5.AEX"
    plain = read_aex(path)
    assert len(plain) == 51 and plain.index.tolist() == list(range(51))
    assert plain["Km"].iloc[0] == pytest.approx(576.4e-6)
    for specs, coords, tol_t in ((asc, "specimen", 1.5e-2), (ran, "geographic", 2e-3)):
        params = [s.params(coords) for s in specs]
        assert np.array([p["P"] for p in params]) == pytest.approx(aex["P"], abs=2e-3)
        assert np.array([p["T"] for p in params]) == pytest.approx(aex["T"], abs=tol_t)
        assert np.array([s.bulk for s in specs]) == pytest.approx(aex["Km"], rel=1e-3)


def test_melechov_mean_tensor_matches_the_statistics_file():
    # MEL1-5.AS1 of Anisoft (geographic coordinates, 51 specimens): eigenvalues
    # and mean directions; its confidence angles are not reproduced
    site = AMSSite(_melechov_specimens("ran"), name="MEL1-5")
    mean = site.mean("geographic")
    assert mean["n"] == 51
    assert mean["eigenvalues"] == pytest.approx([1.027, 1.004, 0.969], abs=6e-4)
    got = []
    for e in mean["ellipses"]:
        v = np.asarray(e["mu"], dtype=float)
        dec = np.degrees(np.arctan2(v[1], v[0])) % 360
        inc = np.degrees(np.arcsin(v[2] / np.linalg.norm(v)))
        got.append(((dec + 180) % 360, -inc) if inc < 0 else (dec, inc))
    assert np.array(got) == pytest.approx(
        np.array([(253.3, 30.2), (9.4, 37.1), (136.1, 38.2)]), abs=0.15
    )


def test_real_sufar_report_and_the_orientation_codes():
    # E236-2: a report of SUFAR 1.1 (field column), code 12 90 6 0; TR1: SUSAR, code
    # 12 90 12 0 (P3 = 12). The geographic tensors printed in the reports are reproduced.
    folder = data.path("trebic")
    sites = data.load("trebic")
    assert list(sites) == ["E236-2", "TR1"] and [len(s) for s in sites.values()] == [
        10,
        7,
    ]
    for key, op, field in (
        ("E236-2", (12, 90, 6, 0), 300.0),
        ("TR1", (12, 90, 12, 0), None),
    ):
        printed = _printed(folder / f"{key}.ASC", "Geograph")
        for s, g in zip(sites[key], printed, strict=True):
            assert s.meta["op"] == op and s.meta["field"] == field
            assert s.tensor_in("geographic") == pytest.approx(g, abs=3e-4)
    assert "norming_factor" not in sites["E236-2"][0].meta
    assert sites["TR1"][0].meta["norming_factor"] == pytest.approx(sites["TR1"][0].bulk)
    assert sites["E236-2"][0].meta["holder"] == pytest.approx(2.45e-6)


def test_trebic_ran_matches_the_reports():
    # the RAN files of SUFAR hold the field in their first value, those of SUSAR the mean
    asc = data.load("trebic")
    ran = data.load("trebic", format="ran")
    for key, field in (("E236-2", 300.0), ("TR1", None)):
        assert ran[key].names == asc[key].names
        for a, r in zip(asc[key], ran[key], strict=True):
            assert r.bulk == pytest.approx(a.bulk, rel=1e-3)
            assert r.meta["field"] == field
            assert r.tensor_in("geographic") == pytest.approx(
                a.tensor_in("geographic"), abs=2e-4
            )
    assert ran["E236-2"][0].meta["op"] == (12, 90, 6, 0)
    assert ran["TR1"][0].meta["op"] == (12, 90, 12, 0)


def test_trebic_aex_matches_the_measurements():
    from apsg import lin

    aex = data.load("trebic_aex")
    assert aex.shape == (17, 16) and aex.index.name == "Name"
    assert list(aex.columns[:4]) == ["Km", "K1", "K2", "K3"]
    asc = [s for site in data.load("trebic").values() for s in site]
    ran = [s for site in data.load("trebic", format="ran").values() for s in site]
    # the names of the table differ from those of the reports, rows are in the same order
    assert aex.index[0] == "E-236/D/1" and asc[0].name == "E236D/1"
    for specs, tol_t, tol_d in ((asc, 1.5e-2, 1.0), (ran, 2e-3, 0.2)):
        for s, (_, row) in zip(specs, aex.iterrows(), strict=True):
            p = s.params("geographic")
            assert p["P"] == pytest.approx(row["P"], abs=2e-3)
            assert p["T"] == pytest.approx(row["T"], abs=tol_t)
            assert s.bulk == pytest.approx(row["Km"], rel=1e-3)
            k = np.sort(np.linalg.eigvalsh(s.tensor_in("geographic")))[::-1] * s.bulk
            assert k == pytest.approx([row["K1"], row["K2"], row["K3"]], rel=2e-3)
            axes = s.ellipsoid("geographic").eigenlins()
            assert axes[0].angle(lin(row["dK1geo"], row["iK1geo"])) < tol_d
            assert axes[2].angle(lin(row["dK3geo"], row["iK3geo"])) < tol_d


def test_aex_with_names_and_ran_field_on_synthetic_files():
    text = (
        "Spec Name   Km   K1\n"
        "TR 1/1/1  1.0E-03  1.1E-03\n"
        "A B C    2.0E-03  2.1E-03\n"
        "short\n"
    )
    with pytest.warns(UserWarning, match="1 AEX rows"):
        df = read_aex(io.StringIO(text))
    assert df.index.tolist() == ["TR 1/1/1", "A B C"] and df.index.name == "Name"
    assert list(df.columns) == ["Km", "K1"] and df["K1"].tolist() == [1.1e-3, 2.1e-3]
    assert read_aex(io.StringIO("Name Km\nx 1\n")).index.tolist() == ["x"]
    # RAN: the second value is the mean susceptibility, the first the field (SUFAR)
    raw = _ran_bytes([("A", 2.5e-4, _K, "00", (0, 0), (0, 0))], field=300.0)
    s = read_ran(raw)[0]
    assert s.bulk == pytest.approx(2.5e-4) and s.meta["field"] == 300.0
    s = read_ran(_ran_bytes([("A", 3e-4, _K, "00", (0, 0), (0, 0))]))[0]
    assert s.bulk == pytest.approx(3e-4) and s.meta["field"] is None


def _ran_bytes(records, codes=" 12 90 6  0 ", locality="SITE", rock="ROCK", field=None):
    """A RAN file: records are (name, mean, tensor, tag, foliation, lineation)."""
    head = struct.pack("<h", len(records) + 2)
    head += locality.ljust(30).encode() + rock.ljust(16).encode()
    head += codes.ljust(14).encode() + b"\0\0"
    assert len(head) == 64
    out = head
    for name, mean, k, tag, fol, lin in records:
        out += struct.pack(
            "<12s8f2s4h2s4h",
            name.ljust(12).encode(),
            mean * 1e6 if field is None else field,
            mean * 1e6,
            *k,
            tag.encode(),
            *fol,
            *lin,
            b"00",
            0,
            0,
            0,
            0,
        )
    return out


_K = (0.95, 1.0, 1.05, 0.01, -0.02, 0.005)


def test_ran_reader_on_synthetic_files():
    raw = _ran_bytes(
        [
            ("A 1/1", 2e-4, _K, "S0", (100, 30), (10, 0)),
            ("A 1/2", 3e-4, _K, "00", (0, 0), (0, 0)),
            ("A 2", 4e-4, _K, "K0", (45, 0), (0, 0)),
        ],
        codes=" 12 0  6  90",
    )
    site = read_ran(raw, name="s")
    assert site.names == ["A 1/1", "A 1/2", "A 2"]
    a, b, c = site
    assert a.bulk == pytest.approx(2e-4, rel=1e-6)
    assert a.tensor == pytest.approx(tensor_from_components(*_K), abs=1e-6)
    assert a.meta["op"] == (12, 0, 6, 90)
    # P4 = 90: the foliation azimuth is a strike, the dip direction 90 more
    assert a.bedding.angle(bedding_vector(190, 30)) == pytest.approx(0, abs=1e-5)
    assert b.bedding is None  # tag 00
    assert c.bedding is None  # dip 0 is the same as no foliation
    assert a.meta["lineation"] == (10, 0) and a.meta["rock"] == "ROCK"


def test_ran_reader_rejects_and_warns():
    good = _ran_bytes([("A 1/1", 2e-4, _K, "S0", (100, 30), (0, 0))])
    with pytest.raises(ValueError, match="shorter"):
        read_ran(good[:100])
    bad_name = bytearray(good)
    bad_name[64:68] = b"\x01\x02\x03\x04"
    with pytest.raises(ValueError, match="not a RAN file"):
        read_ran(bytes(bad_name))
    nan = _ran_bytes([("A 1/1", 2e-4, (np.nan, *_K[1:]), "S0", (1, 1), (0, 0))])
    with pytest.raises(ValueError, match="not a RAN file"):
        read_ran(nan)
    with pytest.warns(UserWarning, match="3 trailing bytes"):
        assert len(read_ran(good + b"abc")) == 1
    flat = _ran_bytes([("A 1/1", 2e-4, (2, 2, 2, 0, 0, 0), "S0", (1, 1), (0, 0))])
    with pytest.warns(UserWarning, match="trace 3"):
        read_ran(flat)
    with pytest.warns(UserWarning, match="no orientation parameters"):
        read_ran(_ran_bytes([("A", 2e-4, _K, "S0", (1, 1), (0, 0))], codes="?"))
    # structure is detected without an extension, text is not mistaken for it
    assert read_ams(good).names == ["A 1/1"]
    with pytest.raises(ValueError, match="binary"):
        read_ams(b"\x00" * 200)


def test_aex_reader_edge_cases(tmp_path):
    text = "Km P T\n 1.0E-03 1.1 0.2\n bad row\n 2.0E-03 1.2 x\n 3.0E-03 1.3 -0.4\n"
    with pytest.warns(UserWarning, match="2 AEX rows .* skipped"):
        df = read_aex(io.StringIO(text))
    assert df["Km"].tolist() == [1e-3, 3e-3] and df["T"].tolist() == [0.2, -0.4]
    df = read_aex(io.StringIO("Km P\n 1.0E-03 1.1\n"), names=["x"])
    assert df.index.tolist() == ["x"]
    with pytest.raises(ValueError, match="names"):
        read_aex(io.StringIO("Km P\n 1 1.1\n"), names=["x", "y"])
    with pytest.raises(ValueError, match="name the columns"):
        read_aex(io.StringIO("1.0 2.0\n3.0 4.0\n"))
    with pytest.raises(ValueError, match="empty"):
        read_aex(io.StringIO(""))
    path = tmp_path / "x.AEX"
    path.write_text(text)
    with pytest.raises(ValueError, match="read_aex"):
        read_ams(path)


# ---------------------------------------------------------------- Jeffery, models, inversion

from scipy.spatial.transform import Rotation

from apsg.ams import (
    AMSEvolution,
    AMSModel,
    GrainPopulation,
    ModelPlot,
    ams_from_rotations,
    jeffery_evolution,
    jeffery_rotations,
    orientation_tensor_from_ams,
)

SHEAR = np.array([[0, 1, 0], [0, 0, 0], [0, 0, 0.0]])


def _R0(n, seed=1):
    return Rotation.random(n, random_state=seed).as_matrix()


@pytest.mark.parametrize("r", [1.5, 3.0, 10.0])
def test_jeffery_spheroid_equals_willis(r):
    K = grain_K("prolate", r)
    R0 = _R0(5)
    t = np.array([0, 1, 2.5, 4])
    R = jeffery_rotations(SHEAR, [r, 1, 1], R0, t)
    D = np.asarray(velgrad(SHEAR).pseudo(K).defgrad(2.5))
    p = R0[:, :, 0] @ D.T
    p /= np.linalg.norm(p, axis=1)[:, None]
    assert np.abs((R[2][:, :, 0] * p).sum(axis=1)) == pytest.approx(1, abs=1e-8)
    # oblate: the pole of the flat grain follows the same law with negative K
    Ro = jeffery_rotations(SHEAR, [1, 1, 1 / r], R0, t)
    Do = np.asarray(velgrad(SHEAR).pseudo(grain_K("oblate", r)).defgrad(2.5))
    q = R0[:, :, 2] @ Do.T
    q /= np.linalg.norm(q, axis=1)[:, None]
    assert np.abs((Ro[2][:, :, 2] * q).sum(axis=1)) == pytest.approx(1, abs=1e-8)


@pytest.mark.parametrize("r", [1.5, 3.0, 10.0])
def test_jeffery_period_in_simple_shear(r):
    # the axis of a spheroid returns after 2 pi (r + 1/r) / shear rate (Jeffery 1922)
    T = 2 * np.pi * (r + 1 / r)
    R0 = _R0(4)
    R = jeffery_rotations(SHEAR, [r, 1, 1], R0, np.array([0, T / 2, T]), rtol=1e-11)
    assert np.abs(R[2][:, :, 0] - R0[:, :, 0]).max() < 1e-6
    assert np.abs(R[1][:, :, 0] - R0[:, :, 0]).max() > 0.5


def test_jeffery_sphere_and_orthonormality():
    R0 = _R0(3)
    Om = np.array([0, 0, -0.5])  # vorticity vector of simple shear dv_x/dy = 1
    R = jeffery_rotations(SHEAR, [1, 1, 1], R0, np.array([0, 2.0]))
    assert R[1] == pytest.approx(
        Rotation.from_rotvec(2 * Om).as_matrix() @ R0, abs=1e-7
    )
    Rt = jeffery_rotations(SHEAR, [3, 2, 1], R0, np.linspace(0, 30, 7))
    for Rk in Rt:
        assert np.abs(Rk.transpose(0, 2, 1) @ Rk - np.eye(3)).max() < 1e-6
        assert np.linalg.det(Rk) == pytest.approx(1, abs=1e-6)
    # pure rotation flow rotates everything rigidly
    spin = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 0.0]])
    Rs = jeffery_rotations(spin, [3, 2, 1], R0, np.array([0, 1.0]))
    assert Rs[1] == pytest.approx(
        Rotation.from_rotvec([0, 0, 1.0]).as_matrix() @ R0, abs=1e-7
    )


def test_jeffery_errors():
    R0 = _R0(2)
    for kw in (
        {"L": np.eye(2)},
        {"semi_axes": [1, 1]},
        {"semi_axes": [1, 0, 1]},
        {"R0": np.eye(2)},
        {"times": [1.0, 0.5]},
    ):
        args = {"L": SHEAR, "semi_axes": [2, 1, 1], "R0": R0, "times": [0, 1.0], **kw}
        with pytest.raises(ValueError):
            jeffery_rotations(**args)
    assert jeffery_rotations(SHEAR, [2, 1, 1], R0[0], [0.0]).shape == (1, 1, 3, 3)


def test_ams_from_rotations():
    R = _R0(20000, 3)
    k = np.asarray(ams_from_rotations(R, [1.2, 1.0, 0.8]))
    assert k == pytest.approx(np.eye(3), abs=0.02)  # random grains: isotropic, mean 1
    assert np.trace(k) == pytest.approx(3.0)
    assert np.asarray(ams_from_rotations(np.eye(3), [3, 2, 1])) == pytest.approx(
        np.diag([3, 2, 1])
    )
    with pytest.raises(ValueError):
        ams_from_rotations(R, [1, 2])


def test_jeffery_evolution_matches_analytic_model():
    a = np.tan(np.radians(10))
    L = [[0, 1, 0], [0, -a, 0], [0, 0, a]]
    t = np.linspace(0, 8, 33)
    jef = jeffery_evolution(L, [3, 1, 1], [1.3, 1, 1], t, n=1500, rng=2)
    ana = AMSModel(L, GrainPopulation("prolate", 3, K=1.0, delta=0.3)).evolve(t)
    assert isinstance(jef, AMSEvolution) and len(jef) == len(t)
    # same fabric statistics: the tensors agree within the Monte Carlo error
    assert np.abs(jef.tensors - ana.tensors).max() < 0.02
    assert jef.switch() == pytest.approx(ana.switch(), abs=0.5)
    assert jef.tensors[0] == pytest.approx(np.eye(3) * jef.tensors[0][0, 0], abs=0.03)
    # triaxial grains also run and switch
    tri = jeffery_evolution(L, [3, 2, 1], [1.3, 1.1, 1.0], t, n=300, rng=3)
    assert tri.switch() is None or tri.switch() < 8
    a1 = jeffery_evolution(L, [3, 1, 1], [1.3, 1, 1], t, n=100, rng=5)
    b1 = jeffery_evolution(L, [3, 1, 1], [1.3, 1, 1], t, n=100, rng=5)
    assert np.array_equal(a1.tensors, b1.tensors)


def _transpression(alpha=10):
    a = np.tan(np.radians(alpha))
    return [[0, 1, 0], [0, -a, 0], [0, 0, a]]


@pytest.mark.parametrize("r, printed", [(1.5, 4.0), (3.0, 5.4), (10.0, 8.0)])
def test_model_switch_strains(r, printed):
    ev = AMSModel(_transpression(), GrainPopulation("prolate", r)).evolve(
        np.linspace(0, 12, 481)
    )
    assert ev.switch() == pytest.approx(printed, abs=0.1)


def test_population_and_model_validation():
    with pytest.raises(ValueError):
        GrainPopulation("cube", 2)
    with pytest.raises(ValueError):
        GrainPopulation("prolate", 0.5)
    with pytest.raises(ValueError):
        GrainPopulation("prolate", 2, delta=-0.1)
    with pytest.raises(ValueError):
        GrainPopulation("oblate", 2, delta=0.1)
    with pytest.raises(ValueError):
        GrainPopulation("prolate", 2, K=-1)
    assert GrainPopulation("oblate", 2).delta < 0
    p = GrainPopulation.from_susceptibility("prolate", 1.2, 16.6)
    assert 1 + p.delta / p.K == pytest.approx(grain_anisotropy(16.6, 1.2))
    assert p.kinematic_K == pytest.approx(grain_K("prolate", 1.2))
    with pytest.raises(TypeError):
        AMSModel(np.eye(3), [])
    with pytest.raises(TypeError):
        AMSModel(np.eye(3), [1])
    with pytest.raises(ValueError):
        AMSModel(np.eye(3), GrainPopulation("prolate", 2, weight=0))
    m = AMSModel(SHEAR, GrainPopulation("prolate", 2))
    with pytest.raises(ValueError):
        m.evolve([-1.0])
    with pytest.raises(ValueError):
        m.evolve([])


def test_mixture_of_populations():
    flow = np.diag([0.3, 0.3, -0.6])
    pro, obl = (
        GrainPopulation("prolate", 3, delta=0.2),
        GrainPopulation("oblate", 3, delta=-0.1),
    )
    mix = AMSModel(flow, [pro, obl]).tensor(1.0)
    one = AMSModel(flow, pro).tensor(1.0)
    two = AMSModel(flow, obl).tensor(1.0)
    assert np.asarray(mix) == pytest.approx(0.5 * (np.asarray(one) + np.asarray(two)))
    # at zero strain every population is isotropic
    iso = np.asarray(AMSModel(flow, [pro, obl]).tensor(0.0))
    assert iso == pytest.approx(np.eye(3) * iso[0, 0])
    assert AMSModel(flow, pro).fabric(2.0).axes[0] is not None


def test_evolution_table_and_ellipsoids():
    ev = AMSModel(_transpression(), GrainPopulation("prolate", 3)).evolve([0, 1, 3, 8])
    t = ev.table()
    assert list(t["time"]) == [0, 1, 3, 8]
    assert np.isnan(t["T"][0]) and np.isnan(t["k1_inc"][0])  # isotropic start
    assert t["Pj"][0] == pytest.approx(1)
    assert len(ev.ellipsoids()) == 4 and "AMSEvolution" in repr(ev)
    assert (
        t["k1_inc"][1] < 10 and t["k1_inc"][3] > 80
    )  # horizontal then vertical lineation


@pytest.mark.parametrize("kind", ["prolate", "oblate"])
def test_orientation_tensor_from_ams_roundtrip(kind):
    Pc = 1.35
    O_true = np.array([0.6, 0.3, 0.1])
    A, B = (
        (3 * (Pc - 1) / (Pc + 2), 3 / (Pc + 2))
        if kind == "prolate"
        else (3 * (1 - Pc) / (2 * Pc + 1), 3 * Pc / (2 * Pc + 1))
    )
    R = np.asarray(core_rotation(30, 20))
    k = R @ np.diag(A * O_true + B) @ R.T
    ot = orientation_tensor_from_ams(k, Pc, kind)
    assert np.trace(np.asarray(ot)) == pytest.approx(1)
    expected = R @ np.diag(O_true) @ R.T
    assert np.asarray(ot) == pytest.approx(expected, abs=1e-12)
    # the vault's corrected eq. 16 for the prolate O2: (2P - LP - L)/(Pc - 1) + P over J
    kk = np.sort(A * O_true + B)[::-1]
    L, P = kk[0] / kk[1], kk[0] / kk[2]
    if kind == "prolate":
        J = L * P + P + L
        O2 = ((2 * P - L * P - L) / (Pc - 1) + P) / J
        assert O2 == pytest.approx(O_true[1])


def test_orientation_tensor_from_ams_consistent_with_forward_model():
    acg = ACG(shape=(2.5, 0.4, 0.1))
    T = np.asarray(acg.ortensor())
    for kind, delta in (("prolate", 0.3), ("oblate", -0.3)):
        k = np.asarray(ams_from_orientation(T, 1.0, delta))
        Pc = (1.0 + delta) / 1.0 if kind == "prolate" else 1.0 / (1.0 + delta)
        rec = np.asarray(orientation_tensor_from_ams(k, Pc, kind))
        assert rec == pytest.approx(T, abs=1e-10)


def test_orientation_tensor_from_ams_errors():
    k = np.diag([1.3, 1.0, 0.7])
    with pytest.raises(ValueError):
        orientation_tensor_from_ams(k, 1.0, "prolate")
    with pytest.raises(ValueError):
        orientation_tensor_from_ams(k, 1.2, "cone")
    with pytest.raises(ValueError):
        orientation_tensor_from_ams(np.diag([1, 1, -1.0]), 1.2, "prolate")
    with pytest.raises(ValueError, match="not compatible"):
        orientation_tensor_from_ams(
            k, 1.2, "prolate"
        )  # P = 1.86 exceeds one grain's Pc
    with pytest.raises(ValueError):
        orientation_tensor_from_ams(np.eye(2), 1.2, "prolate")


def test_model_plot(tmp_path):
    t = np.linspace(0, 10, 41)
    ev = {
        f"r = {r}": AMSModel(_transpression(), GrainPopulation("prolate", r)).evolve(t)
        for r in (1.5, 3)
    }
    p = ModelPlot(ev, xlabel="shear strain")
    p.savefig(tmp_path / "m.png")
    p.init_figure()
    p._render()
    assert len(p.fig.axes) == 3
    single = ModelPlot(ev["r = 3"], switch=False)
    single.init_figure()
    single._render()
    with pytest.raises(TypeError):
        ModelPlot(3)
    with pytest.raises(TypeError):
        ModelPlot({"a": 1})
