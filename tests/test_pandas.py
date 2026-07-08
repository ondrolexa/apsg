import warnings

import numpy as np
import pandas as pd
import pytest

import apsg.pandas  # noqa: F401 - registers extension dtypes and accessors
from apsg import fault, fol, lin, vec, vecset, folset, linset, faultset
from apsg.pandas import FaultArray, FolArray, LinArray, Vec3Array


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def vec_df():
    return pd.DataFrame(
        {
            "x": [1.0, 0.0],
            "y": [0.0, 1.0],
            "z": [0.0, 0.0],
        }
    )


@pytest.fixture
def geo_df():
    return pd.DataFrame(
        {
            "azi": [0, 90],
            "inc": [0, 0],
        }
    )


@pytest.fixture
def fault_df():
    return pd.DataFrame(
        {
            "fazi": [140, 250],
            "finc": [30, 45],
            "lazi": [110, 200],
            "linc": [26, 30],
            "sense": [-1, 1],
        }
    )


@pytest.fixture
def vecset_f():
    return vecset([vec(1, 0, 0), vec(0, 1, 0)])


@pytest.fixture
def folset_f():
    return folset([fol(250, 30), fol(100, 60)])


@pytest.fixture
def linset_f():
    return linset([lin(110, 26), lin(200, 45)])


@pytest.fixture
def faultset_f():
    return faultset(
        [
            fault(140, 30, 110, 26, -1),
            fault(250, 45, 200, 30, 1),
        ]
    )


@pytest.fixture
def vec_array():
    return Vec3Array([vec(1, 0, 0), vec(0, 1, 0)])


# ---------------------------------------------------------------------------
# Vec3Array — extension array
# ---------------------------------------------------------------------------


class TestVec3Array:
    def test_construct(self):
        arr = Vec3Array([vec(1, 0, 0), vec(0, 1, 0)])
        assert len(arr) == 2

    def test_construct_empty(self):
        arr = Vec3Array([])
        assert len(arr) == 0

    def test_dtype(self, vec_array):
        assert vec_array.dtype.name == "vec"

    def test_len(self, vec_array):
        assert len(vec_array) == 2

    def test_getitem_int(self, vec_array):
        v = vec_array[0]
        assert isinstance(v, vec)
        assert v == vec(1, 0, 0)

    def test_getitem_slice(self, vec_array):
        sub = vec_array[0:1]
        assert isinstance(sub, Vec3Array)
        assert len(sub) == 1

    def test_getitem_bool_array(self, vec_array):
        mask = np.array([True, False])
        sub = vec_array[mask]
        assert isinstance(sub, Vec3Array)
        assert len(sub) == 1
        assert sub[0] == vec(1, 0, 0)

    def test_eq_scalar(self, vec_array):
        result = vec_array == vec(1, 0, 0)
        assert isinstance(result, np.ndarray)
        assert result.tolist() == [True, False]

    def test_eq_array(self, vec_array):
        other = Vec3Array([vec(1, 0, 0), vec(0, 1, 0)])
        result = vec_array == other
        assert result.tolist() == [True, True]

    def test_neq(self, vec_array):
        result = vec_array != vec(42, 0, 0)
        assert result.tolist() == [True, True]

    def test_eq_notimplemented_for_pandas_containers(self, vec_array):
        result = vec_array.__eq__(pd.Index([]))
        assert result is NotImplemented

    def test_array_conversion(self, vec_array):
        a = np.asarray(vec_array)
        assert isinstance(a, np.ndarray)
        assert len(a) == 2

    def test_from_sequence(self):
        arr = Vec3Array._from_sequence([vec(1, 0, 0)])
        assert len(arr) == 1

    def test_concat_same_type(self, vec_array):
        other = Vec3Array([vec(0, 0, 1)])
        result = Vec3Array._concat_same_type([vec_array, other])
        assert len(result) == 3
        assert result[2] == vec(0, 0, 1)

    def test_nbytes(self, vec_array):
        assert vec_array.nbytes > 0

    def test_isna(self):
        arr = Vec3Array([vec(1, 0, 0), vec(float("nan"), 0, 0)])
        result = arr.isna()
        assert result.tolist() == [False, True]

    def test_take(self, vec_array):
        result = vec_array.take([1, 0])
        assert result[0] == vec(0, 1, 0)
        assert result[1] == vec(1, 0, 0)

    def test_copy(self, vec_array):
        c = vec_array.copy()
        assert len(c) == 2
        assert c[0] == vec_array[0]
        assert c is not vec_array


# ---------------------------------------------------------------------------
# FolArray, LinArray, FaultArray
# ---------------------------------------------------------------------------


class TestFolArray:
    def test_construct(self):
        arr = FolArray([fol(250, 30)])
        assert len(arr) == 1

    def test_dtype(self):
        arr = FolArray([fol(250, 30)])
        assert arr.dtype.name == "fol"


class TestLinArray:
    def test_construct(self):
        arr = LinArray([lin(110, 26)])
        assert len(arr) == 1

    def test_dtype(self):
        arr = LinArray([lin(110, 26)])
        assert arr.dtype.name == "lin"


class TestFaultArray:
    def test_construct(self):
        arr = FaultArray([fault(140, 30, 110, 26, -1)])
        assert len(arr) == 1

    def test_dtype(self):
        arr = FaultArray([fault(140, 30, 110, 26, -1)])
        assert arr.dtype.name == "fault"


# ---------------------------------------------------------------------------
# APSGAccessor (.apsg)
# ---------------------------------------------------------------------------


class TestAPSGAccessor:
    def test_create_vecs(self, vec_df):
        res = vec_df.apsg.create_vecs()
        assert "vecs" in res.columns
        assert isinstance(res["vecs"].array, Vec3Array)

    def test_create_vecs_custom_columns(self, geo_df):
        res = geo_df.apsg.create_vecs(columns=["azi", "inc"], name="v")
        assert "v" in res.columns
        assert len(res) == 2

    def test_create_vecs_custom_name(self, vec_df):
        res = vec_df.apsg.create_vecs(name="myvecs")
        assert "myvecs" in res.columns

    def test_add_vecs(self, vec_df, vecset_f):
        res = vec_df.apsg.add_vecs(vecset_f)
        assert "vecs" in res.columns
        assert isinstance(res["vecs"].array, Vec3Array)

    def test_add_vecs_bad_type(self, vec_df):
        with pytest.raises(TypeError, match="must be Vector3Set"):
            vec_df.apsg.add_vecs("not_a_set")

    def test_create_fols(self, geo_df):
        res = geo_df.apsg.create_fols()
        assert "fols" in res.columns
        assert isinstance(res["fols"].array, FolArray)

    def test_add_fols(self, geo_df, folset_f):
        res = geo_df.apsg.add_fols(folset_f)
        assert "fols" in res.columns

    def test_add_fols_bad_type(self, geo_df):
        with pytest.raises(TypeError, match="must be FoliationSet"):
            geo_df.apsg.add_fols("bad")

    def test_create_lins(self, geo_df):
        res = geo_df.apsg.create_lins()
        assert "lins" in res.columns
        assert isinstance(res["lins"].array, LinArray)

    def test_add_lins(self, geo_df, linset_f):
        res = geo_df.apsg.add_lins(linset_f)
        assert "lins" in res.columns

    def test_add_lins_bad_type(self, geo_df):
        with pytest.raises(TypeError, match="must be LineationSet"):
            geo_df.apsg.add_lins("bad")

    def test_create_faults(self, fault_df):
        res = fault_df.apsg.create_faults()
        assert "faults" in res.columns
        assert isinstance(res["faults"].array, FaultArray)

    def test_add_faults(self, fault_df, faultset_f):
        res = fault_df.apsg.add_faults(faultset_f)
        assert "faults" in res.columns

    def test_add_faults_bad_type(self, fault_df):
        with pytest.raises(TypeError, match="must be FaultSet"):
            fault_df.apsg.add_faults("bad")

    def test_create_does_not_mutate_original(self, vec_df):
        orig_cols = list(vec_df.columns)
        _ = vec_df.apsg.create_vecs()
        assert list(vec_df.columns) == orig_cols

    def test_add_does_not_mutate_original(self, vec_df, vecset_f):
        orig_cols = list(vec_df.columns)
        _ = vec_df.apsg.add_vecs(vecset_f)
        assert list(vec_df.columns) == orig_cols


# ---------------------------------------------------------------------------
# GAccessor — Series accessor .G()
# ---------------------------------------------------------------------------


class TestVec3Accessor:
    def test_G(self, vec_df):
        s = vec_df.apsg.create_vecs()["vecs"]
        g = s.G()
        assert isinstance(g, vecset)
        assert g.name == "vecs"

    def test_G_fails_on_non_feature_series(self):
        s = pd.Series([1, 2, 3])
        with pytest.raises(AttributeError, match="APSG feature array"):
            s.G()

    def test_R(self, vec_df):
        s = vec_df.apsg.create_vecs()["vecs"]
        r = s.G().R()
        assert isinstance(r, vec)

    def test_fisher_k(self, geo_df):
        s = geo_df.apsg.create_lins()["lins"]
        k = s.G().fisher_statistics()["k"]
        assert isinstance(k, float)

    def test_fisher_csd(self, geo_df):
        s = geo_df.apsg.create_lins()["lins"]
        csd = s.G().fisher_statistics()["csd"]
        assert isinstance(csd, float)

    def test_fisher_alpha(self, geo_df):
        s = geo_df.apsg.create_lins()["lins"]
        alpha = s.G().fisher_statistics()["alpha"]
        assert isinstance(alpha, float)

    def test_fisher_alpha_custom_level(self, geo_df):
        s = geo_df.apsg.create_lins()["lins"]
        alpha = s.G().fisher_statistics(level=0.99)["alpha"]
        assert isinstance(alpha, float)

    def test_var(self, geo_df):
        s = geo_df.apsg.create_lins()["lins"]
        v = s.G().var()
        assert isinstance(v, float)

    def test_delta(self, geo_df):
        s = geo_df.apsg.create_lins()["lins"]
        d = s.G().delta()
        assert isinstance(d, float)

    def test_rdegree(self, geo_df):
        s = geo_df.apsg.create_lins()["lins"]
        r = s.G().rdegree()
        assert isinstance(r, float)

    def test_ortensor(self, geo_df):
        s = geo_df.apsg.create_lins()["lins"]
        ot = s.G().ortensor()
        assert ot is not None


class TestFolAccessor:
    def test_G(self, geo_df):
        s = geo_df.apsg.create_fols()["fols"]
        g = s.G()
        assert isinstance(g, folset)
        assert g.name == "fols"

    def test_R(self, geo_df):
        s = geo_df.apsg.create_fols()["fols"]
        r = s.G().R()
        assert isinstance(r, fol)

    def test_stat_methods(self, geo_df):
        s = geo_df.apsg.create_fols()["fols"]
        assert isinstance(s.G().R(), fol)
        assert isinstance(s.G().fisher_statistics()["k"], float)


class TestLinAccessor:
    def test_G(self, geo_df):
        s = geo_df.apsg.create_lins()["lins"]
        g = s.G()
        assert isinstance(g, linset)
        assert g.name == "lins"

    def test_R(self, geo_df):
        s = geo_df.apsg.create_lins()["lins"]
        r = s.G().R()
        assert isinstance(r, lin)

    def test_stat_methods(self, geo_df):
        s = geo_df.apsg.create_lins()["lins"]
        assert isinstance(s.G().R(), lin)
        assert isinstance(s.G().fisher_statistics()["k"], float)


class TestFaultAccessor:
    def test_G(self, fault_df):
        s = fault_df.apsg.create_faults()["faults"]
        g = s.G()
        assert isinstance(g, faultset)
        assert g.name == "faults"

    def test_ortensor(self, fault_df):
        s = fault_df.apsg.create_faults()["faults"]
        ot = s.G().ortensor()
        assert ot is not None


# ---------------------------------------------------------------------------
# Multi-column scenario
# ---------------------------------------------------------------------------


class TestMultiColumn:
    def test_two_lin_columns(self, geo_df):
        df = geo_df.apsg.create_lins(name="L1").apsg.create_lins(name="L2")
        assert isinstance(df.L1.G(), linset)
        assert isinstance(df.L2.G(), linset)
        assert df.L1.G().name == "L1"
        assert df.L2.G().name == "L2"

    def test_lin_and_fol(self, geo_df):
        df = geo_df.apsg.create_lins(name="L1").apsg.create_fols(name="F1")
        assert isinstance(df.L1.G(), linset)
        assert isinstance(df.F1.G(), folset)

    def test_mixed_types(self, vec_df, geo_df):
        df = pd.concat([vec_df, geo_df], axis=1)
        df = df.apsg.create_vecs(name="V").apsg.create_lins(name="L")
        assert isinstance(df.V.G(), vecset)
        assert isinstance(df.L.G(), linset)


# ---------------------------------------------------------------------------
# Invalid name warning
# ---------------------------------------------------------------------------


class TestInvalidNameWarning:
    def test_warning_on_non_identifier_name(self, geo_df):
        with pytest.warns(UserWarning, match="not a valid Python identifier"):
            geo_df.apsg.create_lins(name="my lins")

    def test_no_warning_on_identifier_name(self, geo_df):
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            geo_df.apsg.create_lins(name="lins")
            assert len(w) == 0

    def test_bracket_access_on_non_identifier_name(self, geo_df):
        df = geo_df.apsg.create_lins(name="my lins")
        g = df["my lins"].G()
        assert isinstance(g, linset)
