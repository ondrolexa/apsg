import json
import sys
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import apsg
from apsg import data
from apsg.feature import FaultSet, FeatureSet
from apsg.feature._hypothesis import fisher_goodness_of_fit

ROOT = Path(__file__).parent.parent
KINDS = {
    "vectors",
    "lines",
    "planes",
    "faults",
    "circular",
    "grouped",
    "table",
    "tensors",
    "stress",
    "groups",
    "paleomag",
}


def test_not_in_default_namespace():
    assert "data" not in apsg.__all__


def test_catalog_is_complete_and_consistent():
    cat = data.catalog()
    assert list(cat.index) == data.names()
    assert len(cat) >= 35
    assert set(cat["kind"]) <= KINDS
    for name, row in cat.iterrows():
        assert row["description"], name
        assert row["source"], name
        assert row["n"] > 0, name
        assert data.info(name).name == name
    assert "Fisher" in str(data.info("fisher_1953"))


@pytest.mark.parametrize("name", data.names())
def test_every_dataset_loads_with_the_declared_size(name):
    info = data.info(name)
    obj = data.load(name)
    if info.kind == "paleomag":
        assert sum(len(s) for s in obj.values()) == info.n
    elif info.kind == "groups":
        assert sum(len(v) for v in obj.values()) == info.n
        assert all(isinstance(v, FeatureSet) for v in obj.values())
    elif info.kind == "grouped":
        assert isinstance(obj, pd.DataFrame)
        assert obj["count"].sum() == info.n
    elif info.kind == "table":
        assert isinstance(obj, pd.DataFrame)
        assert len(obj) == info.n
    else:
        assert isinstance(obj, FeatureSet)
        assert len(obj) == info.n


def test_unknown_names_and_arguments():
    with pytest.raises(KeyError, match="available"):
        data.load("no_such_data")
    with pytest.raises(KeyError):
        data.info("no_such_data")
    with pytest.raises(TypeError):
        data.load("fisher_1953", site="x")
    with pytest.raises(KeyError):
        data.load("pmdata", site="PM99")
    with pytest.raises(AttributeError):
        _ = data.no_such_data


def test_shortcuts_and_dir():
    a, b = data.fisher_1953(), data.load("fisher_1953")
    assert np.array_equal(np.asarray(a), np.asarray(b))
    assert "fisher_1953" in dir(data) and "load" in dir(data)
    assert "fisher_1953" in data.fisher_1953.__doc__


def test_loads_return_independent_objects():
    a = data.load("fisher_1953")
    n = len(a)
    a.data = a.data[:3]
    assert len(data.load("fisher_1953")) == n
    s1, s2 = data.load("pmdata", site="PM01"), data.load("pmdata", site="PM01")
    s1[0].interpret("X", [0, 1, 2])
    assert not s2[0].components


def test_paths():
    p = data.path("mele")
    assert p.is_file()
    assert np.array_equal(
        np.asarray(FaultSet.from_csv(p).fol), np.asarray(data.load("mele").fol)
    )
    assert data.path("structures").name == "structures.csv"
    folder = data.path("pmdata")
    assert len(list(folder.glob("*.pmd"))) == 71
    spec = apsg.paleomag.Specimen.from_pmd(data.pmd_file("PM01.1a"))
    assert spec.name == "PM01.1a"
    with pytest.raises(KeyError):
        data.pmd_file("PM99.1a")
    with pytest.raises(ValueError):
        data.path("fisher_1953")


def test_pmdata_is_one_compact_file():
    z = data.path("pmdata").parent.parent  # temp dir, only to make sure it is cached
    assert z.exists()
    archive = Path(data.__file__).parent / "files" / "pmdata.zip"
    assert archive.stat().st_size < 100_000
    with zipfile.ZipFile(archive) as zf:
        names = zf.namelist()
    assert len(names) == 72 and "sites.csv" in names
    sites = data.load("pmdata_sites")
    assert list(sites.index) == [f"PM{i:02d}" for i in range(1, 11)]
    assert sites["n"].sum() == 71


def test_tutorial_tables():
    assert list(data.load("structures").columns) == ["site", "structure", "azi", "inc"]
    wide = data.load("structures_wide")
    assert "S1_azi" in wide.columns and wide["S1_azi"].isna().any()
    assert len(data.load("mele")) == 94


# published data reproduce the printed statistics ---------------------------------


def test_fisher_1953_statistics():
    # Fisher (1953): R = 8.772, k = 35.1, 95 % cone 8.8 deg
    st = data.fisher_1953().fisher_statistics()
    assert st["k"] == pytest.approx(35.1, abs=0.1)
    assert st["alpha"] == pytest.approx(8.8, abs=0.05)


def test_tauxe_table_4_2_modes():
    # Tauxe (2003) Example 4.9: Fisher means of the two modes
    g = data.load("tauxe_table_4_2")
    assert (len(g["normal"]), len(g["reversed"])) == (25, 24)
    assert np.allclose(
        g["normal"].fisher_statistics()["mu"].geo, (26.5, 39.1), atol=0.05
    )
    assert np.allclose(
        g["reversed"].fisher_statistics()["mu"].geo, (189.2, -44.3), atol=0.05
    )


def test_fle_b2_goodness_of_fit():
    # FLE Example 5.17 (B2, n = 26): M_E = 0.5619, M_U = 0.7798
    gof = fisher_goodness_of_fit(np.asarray(data.fle_b2()))
    assert gof["colatitude"].statistic == pytest.approx(0.5619, abs=5e-4)
    assert gof["longitude"].statistic == pytest.approx(0.7798, abs=5e-4)


def test_circular_sets():
    wind = data.load("gorleston_wind")
    assert [len(v) for v in wind.values()] == [12, 12, 13, 12]
    b12 = data.fisher_1993_b12()
    angles = np.degrees(np.arctan2(*np.asarray(b12).T[::-1])) % 360
    assert 185 in np.round(angles) and 285 not in np.round(angles)
    assert "285" in data.info("fisher_1993_b12").note
    assert data.load("mallard")["count"].sum() == 714


def test_lisle_pairs_are_paired():
    g = data.load("lisle_1989")
    assert len(g["lineations"]) == len(g["bedding"]) == 4
    assert g["lineations"][0].geo == pytest.approx((58, 20))


# synthetic data have the properties they claim -----------------------------------


def test_synthetic_sets_have_their_structure():
    k50 = data.fisher_k50().fisher_statistics()
    assert k50["k"] == pytest.approx(50, rel=0.3)
    assert data.fisher_weak().fisher_statistics()["k"] < 10
    ev = data.watson_girdle().ortensor().eigenvalues()
    assert ev[1] / ev[0] > 0.5 and ev[2] / ev[1] < 0.3  # girdle: two large, one small
    assert data.watson_bipolar().ortensor().eigenvalues()[0] > 0.8
    assert len(data.ams_sites()) == 20 and len(data.stress_tensors()) == 15
    fold = data.fold_data()
    assert len(fold["bedding"]) == 60
    faults = data.synthetic_faults()
    assert len(faults) == 60


# reproducibility ---------------------------------------------------------------------


def _scripts():
    path = ROOT / "scripts"
    if not (path / "make_data.py").is_file():
        pytest.skip("generator scripts not available")
    sys.path.insert(0, str(path))
    try:
        import make_data
        import make_pmdata
    finally:
        sys.path.remove(str(path))
    return make_data, make_pmdata


def _same(a, b):
    if isinstance(a, dict):
        assert a.keys() == b.keys()
        for k in a:
            _same(a[k], b[k])
    elif isinstance(a, list) and a and isinstance(a[0], (list, float, int)):
        assert np.allclose(np.asarray(a, float), np.asarray(b, float), atol=0.15)
    else:
        assert a == b


def test_synthetic_files_are_reproducible():
    make_data, _ = _scripts()
    stored = json.loads(
        (Path(data.__file__).parent / "files" / "synthetic.json").read_text()
    )
    _same(json.loads(json.dumps(make_data.make())), stored)


def test_pmdata_archive_is_reproducible(tmp_path):
    _, make_pmdata = _scripts()
    out = tmp_path / "pmdata.zip"
    make_pmdata.write(out)
    stored = Path(data.__file__).parent / "files" / "pmdata.zip"
    with zipfile.ZipFile(out) as a, zipfile.ZipFile(stored) as b:
        assert a.namelist() == b.namelist()
        for n in a.namelist():
            assert a.read(n) == b.read(n), n


def test_docs_catalog_table_is_current():
    page = ROOT / "docs" / "api" / "data.md"
    if not page.is_file():
        pytest.skip("docs not available")
    text = page.read_text()
    table = text.split("<!-- catalog:start -->\n")[1].split("\n<!-- catalog:end -->")[0]
    listed = [ln.split("`")[1] for ln in table.splitlines()[2:]]
    assert listed == data.names()
