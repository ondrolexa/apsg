import numpy as np
import pandas as pd
import pytest

import apsg.pandas  # noqa: F401 - registers extension dtypes and accessors
from apsg import (
    dir2,
    dir2set,
    fault,
    fol,
    lin,
    vec,
    vec2,
    vec2set,
    vecset,
    folset,
    linset,
    faultset,
)
from apsg.pandas import (
    DirArray,
    FaultArray,
    FolArray,
    LinArray,
    Vec2Array,
    Vec3Array,
    gbf,
)


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
def vec2_df():
    return pd.DataFrame(
        {
            "x": [1.0, 0.0],
            "y": [0.0, 1.0],
        }
    )


@pytest.fixture
def dir_df():
    return pd.DataFrame(
        {
            "angle": [0, 90],
        }
    )


@pytest.fixture
def vec2set_f():
    return vec2set([vec2(1, 0), vec2(0, 1)])


@pytest.fixture
def dir2set_f():
    return dir2set([dir2(0), dir2(90)])


@pytest.fixture
def vec2_array():
    return Vec2Array([vec2(1, 0), vec2(0, 1)])


@pytest.fixture
def dir_array():
    return DirArray([dir2(0), dir2(90)])


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


@pytest.fixture
def structure_df():
    return pd.DataFrame(
        {
            "azi": [113, 118, 42, 42, 145, 163],
            "inc": [47, 42, 79, 73, 12, 9],
            "structure": ["L3", "L3", "S1", "S1", "L3", "S1"],
        }
    )


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
        arr = Vec3Array([vec(1, 0, 0), None], mask=np.array([False, True]))
        result = arr.isna()
        assert result.tolist() == [False, True]

    def test_isna_all_valid(self, vec_array):
        assert not vec_array.isna().any()

    def test_isna_from_sequence(self):
        arr = Vec3Array._from_sequence([vec(1, 0, 0), None, pd.NA])
        assert arr.isna().tolist() == [False, True, True]

    def test_getitem_na(self):
        arr = Vec3Array([vec(1, 0, 0), None], mask=np.array([False, True]))
        assert arr[0] == vec(1, 0, 0)
        assert arr[1] is pd.NA

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
    def test_bracket_access_on_non_identifier_name(self, geo_df):
        df = geo_df.apsg.create_lins(name="my lins")
        g = df["my lins"].G()
        assert isinstance(g, linset)


# ---------------------------------------------------------------------------
# GAccessorGroupBy — SeriesGroupBy accessor .G.apply()
# ---------------------------------------------------------------------------


class TestGAccessorGroupBy:
    def test_apply_lin(self, structure_df):
        df = structure_df.apsg.create_lins()

        def first_lin(s):
            return s.G().ortensor().eigenlins(which=0)

        result = df.groupby("structure")["lins"].G.apply(first_lin)
        assert result.dtype.name == "lin"

    def test_apply_fol(self, structure_df):
        df = structure_df.apsg.create_fols()

        def first_fol(s):
            return s.G().ortensor().eigenfols(which=0)

        result = df.groupby("structure")["fols"].G.apply(first_fol)
        assert result.dtype.name == "fol"

    def test_apply_vec(self, structure_df):
        df = structure_df.apsg.create_vecs(columns=["azi", "inc"])

        def first_vec(s):
            return s.G().R()

        result = df.groupby("structure")["vecs"].G.apply(first_vec)
        assert result.dtype.name == "vec"

    def test_apply_non_apsg_result(self, structure_df):
        df = structure_df.apsg.create_lins()
        result = df.groupby("structure")["lins"].G.apply(lambda s: len(s))
        assert result.dtype.name == "int64"

    def test_apply_empty_group(self, structure_df):
        df = structure_df.apsg.create_lins()
        sub = df[df["structure"] == "L3"]
        result = sub.groupby("structure")["lins"].G.apply(lambda s: s.G().R())
        assert isinstance(result, pd.Series)

    def test_apply_axis_label_preserved(self, structure_df):
        df = structure_df.apsg.create_lins()
        result = df.groupby("structure")["lins"].G.apply(lambda s: s.G().R())
        assert list(result.index) == ["L3", "S1"]

    def test_apply_with_args(self, structure_df):
        df = structure_df.apsg.create_lins()

        def pick(s, which):
            return s.G().ortensor().eigenlins(which=which)

        result = df.groupby("structure")["lins"].G.apply(pick, which=0)
        assert result.dtype.name == "lin"

    # ------------------------------------------------------------------ #
    # aggregate / agg
    # ------------------------------------------------------------------ #

    def test_aggregate_lin(self, structure_df):
        df = structure_df.apsg.create_lins()

        def first_lin(s):
            return s.G().ortensor().eigenlins(which=0)

        result = df.groupby("structure")["lins"].G.aggregate(first_lin)
        assert result.dtype.name == "lin"

    def test_aggregate_non_apsg_result(self, structure_df):
        df = structure_df.apsg.create_lins()
        result = df.groupby("structure")["lins"].G.aggregate(len)
        assert result.dtype.name == "int64"

    def test_agg_lin(self, structure_df):
        df = structure_df.apsg.create_lins()

        def first_lin(s):
            return s.G().ortensor().eigenlins(which=0)

        result = df.groupby("structure")["lins"].G.agg(first_lin)
        assert result.dtype.name == "lin"

    def test_agg_list_of_functions(self, structure_df):
        df = structure_df.apsg.create_lins()
        result = df.groupby("structure")["lins"].G.agg([len, "count"])
        assert isinstance(result, pd.DataFrame)
        assert list(result.columns) == ["len", "count"]

    # ------------------------------------------------------------------ #
    # transform
    # ------------------------------------------------------------------ #

    def test_transform_scalar_result(self, structure_df):
        df = structure_df.apsg.create_lins()

        def my_transform_scalar(s):
            return s.G().angle(s.G().R())

        result = df.groupby("structure")["lins"].G.transform(my_transform_scalar)
        assert result.dtype.name == "float64"
        assert len(result) == len(df)

    def test_transform_vector_result(self, structure_df):
        df = structure_df.apsg.create_lins()

        def my_transform_vector(s):
            return s.G().cross(s.G().R())

        result = df.groupby("structure")["lins"].G.transform(my_transform_vector)
        assert result.dtype.name == "fol"
        assert len(result) == len(df)


# ---------------------------------------------------------------------------
# Vec2Array — extension array
# ---------------------------------------------------------------------------


class TestVec2Array:
    def test_construct(self):
        arr = Vec2Array([vec2(1, 0), vec2(0, 1)])
        assert len(arr) == 2

    def test_construct_empty(self):
        arr = Vec2Array([])
        assert len(arr) == 0

    def test_dtype(self, vec2_array):
        assert vec2_array.dtype.name == "vec2"

    def test_len(self, vec2_array):
        assert len(vec2_array) == 2

    def test_getitem_int(self, vec2_array):
        v = vec2_array[0]
        assert isinstance(v, vec2)
        assert v == vec2(1, 0)

    def test_getitem_slice(self, vec2_array):
        sub = vec2_array[0:1]
        assert isinstance(sub, Vec2Array)
        assert len(sub) == 1

    def test_getitem_bool_array(self, vec2_array):
        mask = np.array([True, False])
        sub = vec2_array[mask]
        assert isinstance(sub, Vec2Array)
        assert len(sub) == 1
        assert sub[0] == vec2(1, 0)

    def test_eq_scalar(self, vec2_array):
        result = vec2_array == vec2(1, 0)
        assert isinstance(result, np.ndarray)
        assert result.tolist() == [True, False]

    def test_eq_other_array(self, vec2_array):
        other = Vec2Array([vec2(1, 0), vec2(0, 1)])
        result = vec2_array == other
        assert isinstance(result, np.ndarray)
        assert result.tolist() == [True, True]

    def test_neq(self, vec2_array):
        result = vec2_array != vec2(1, 0)
        assert isinstance(result, np.ndarray)
        assert (result == [False, True]).all()

    def test_copy(self, vec2_array):
        c = vec2_array.copy()
        assert len(c) == 2
        assert c[0] == vec2(1, 0)

    def test_isna(self, vec2_array):
        assert not vec2_array.isna().any()

    def test_take(self, vec2_array):
        result = vec2_array.take([0])
        assert len(result) == 1
        assert result[0] == vec2(1, 0)

    def test_repr(self, vec2_array):
        assert "Vec2Array" in repr(vec2_array)

    def test_concat_same_type(self, vec2_array):
        other = Vec2Array([vec2(1, 1)])
        result = Vec2Array._concat_same_type([vec2_array, other])
        assert len(result) == 3
        assert result[-1] == vec2(1, 1)


# ---------------------------------------------------------------------------
# DirArray — extension array
# ---------------------------------------------------------------------------


class TestDirArray:
    def test_construct(self):
        arr = DirArray([dir2(0), dir2(90)])
        assert len(arr) == 2

    def test_construct_empty(self):
        arr = DirArray([])
        assert len(arr) == 0

    def test_dtype(self, dir_array):
        assert dir_array.dtype.name == "dir"

    def test_len(self, dir_array):
        assert len(dir_array) == 2

    def test_getitem_int(self, dir_array):
        v = dir_array[0]
        assert isinstance(v, dir2)
        assert v == dir2(0)

    def test_getitem_slice(self, dir_array):
        sub = dir_array[0:1]
        assert isinstance(sub, DirArray)
        assert len(sub) == 1

    def test_eq_scalar(self, dir_array):
        result = dir_array == dir2(0)
        assert isinstance(result, np.ndarray)
        assert result.tolist() == [True, False]

    def test_copy(self, dir_array):
        c = dir_array.copy()
        assert len(c) == 2
        assert c[0] == dir2(0)


# ---------------------------------------------------------------------------
# Vec2Accessor — G accessor
# ---------------------------------------------------------------------------


class TestVec2Accessor:
    def test_G(self, vec2_df):
        s = vec2_df.apsg.create_vecs2()["vecs2"]
        g = s.G()
        assert isinstance(g, vec2set)

    def test_G_create(self, vec2_df):
        s = vec2_df.apsg.create_vecs2()["vecs2"]
        assert isinstance(s.array, Vec2Array)
        assert len(s) == 2

    def test_G_add(self, vec2set_f):
        df = pd.DataFrame({"id": [1, 2]})
        df = df.apsg.add_vecs2(vec2set_f)
        assert isinstance(df.vecs2.array, Vec2Array)
        assert len(df.vecs2) == 2

    def test_R(self, vec2_df):
        s = vec2_df.apsg.create_vecs2()["vecs2"]
        r = s.G().R()
        assert isinstance(r, vec2)

    def test_var(self, vec2_df):
        s = vec2_df.apsg.create_vecs2()["vecs2"]
        v = s.G().var()
        assert isinstance(v, float)

    def test_ortensor(self, vec2_df):
        s = vec2_df.apsg.create_vecs2()["vecs2"]
        ot = s.G().ortensor()
        assert ot is not None


# ---------------------------------------------------------------------------
# DirAccessor — G accessor
# ---------------------------------------------------------------------------


class TestDirAccessor:
    def test_G(self, dir_df):
        s = dir_df.apsg.create_dirs()["dirs"]
        g = s.G()
        assert isinstance(g, dir2set)

    def test_G_create(self, dir_df):
        s = dir_df.apsg.create_dirs()["dirs"]
        assert isinstance(s.array, DirArray)
        assert len(s) == 2

    def test_G_add(self, dir2set_f):
        df = pd.DataFrame({"id": [1, 2]})
        df = df.apsg.add_dirs(dir2set_f)
        assert isinstance(df.dirs.array, DirArray)
        assert len(df.dirs) == 2

    def test_R(self, dir_df):
        s = dir_df.apsg.create_dirs()["dirs"]
        r = s.G().R()
        assert isinstance(r, dir2)

    def test_var(self, dir_df):
        s = dir_df.apsg.create_dirs()["dirs"]
        v = s.G().var()
        assert isinstance(v, float)


# ---------------------------------------------------------------------------
# Vec2 GroupBy
# ---------------------------------------------------------------------------


class TestVec2GroupBy:
    def test_apply_vec2(self, vec2_df):
        df = vec2_df.apsg.create_vecs2()
        df["group"] = ["a", "b"]
        result = df.groupby("group")["vecs2"].G.apply(lambda s: s.G().R())
        assert result.dtype.name == "vec2"
        assert len(result) == 2

    def test_aggregate_vec2(self, vec2_df):
        df = vec2_df.apsg.create_vecs2()
        df["group"] = ["a", "a"]
        result = df.groupby("group")["vecs2"].G.aggregate(lambda s: s.G().R())
        assert result.dtype.name == "vec2"

    def test_aggregate_non_apsg(self, vec2_df):
        df = vec2_df.apsg.create_vecs2()
        df["group"] = ["a", "b"]
        result = df.groupby("group")["vecs2"].G.aggregate(len)
        assert result.dtype.name == "int64"

    def test_transform(self, vec2_df):
        df = vec2_df.apsg.create_vecs2()
        df["group"] = ["a", "a"]

        def angle_to_r(s):
            return s.G().angle(s.G().R())

        result = df.groupby("group")["vecs2"].G.transform(angle_to_r)
        assert result.dtype.name == "float64"
        assert len(result) == 2


# ---------------------------------------------------------------------------
# gbf.mean
# ---------------------------------------------------------------------------


class TestGBFMean:
    def test_mean_vec3(self, vec_df):
        df = vec_df.apsg.create_vecs()
        df["g"] = ["a", "a"]
        result = df.groupby("g")["vecs"].G.apply(gbf.mean)
        assert isinstance(result.iloc[0], vec)

    def test_mean_vec2(self, vec2_df):
        df = vec2_df.apsg.create_vecs2()
        df["g"] = ["a", "a"]
        result = df.groupby("g")["vecs2"].G.apply(gbf.mean)
        assert isinstance(result.iloc[0], vec2)

    def test_mean_dir(self, dir_df):
        df = dir_df.apsg.create_dirs()
        df["g"] = ["a", "a"]
        result = df.groupby("g")["dirs"].G.apply(gbf.mean)
        assert isinstance(result.iloc[0], float)

    def test_mean_lin(self, geo_df):
        df = geo_df.apsg.create_lins()
        df["g"] = ["a", "a"]
        result = df.groupby("g")["lins"].G.apply(gbf.mean)
        assert isinstance(result.iloc[0], lin)

    def test_mean_fol(self, geo_df):
        df = geo_df.apsg.create_fols()
        df["g"] = ["a", "a"]
        result = df.groupby("g")["fols"].G.apply(gbf.mean)
        assert isinstance(result.iloc[0], fol)

    def test_mean_fault_raises(self, fault_df):
        df = fault_df.apsg.create_faults()
        df["g"] = ["a", "a"]
        with pytest.raises(TypeError, match="mean is not defined"):
            df.groupby("g")["faults"].G.apply(gbf.mean)

    def test_mean_custom_function(self, geo_df):
        df = geo_df.apsg.create_lins()
        df["g"] = ["a", "a"]
        result = df.groupby("g")["lins"].G.agg([gbf.mean, gbf.resultant_magnitude])
        assert isinstance(result, pd.DataFrame)
        assert "mean" in result.columns

    def test_angle_to_mean_vec3(self, vec_df):
        df = vec_df.apsg.create_vecs()
        df["g"] = ["a", "a"]
        result = df.groupby("g")["vecs"].G.transform(gbf.angle_to_mean)
        assert isinstance(result, pd.Series)
        assert result.dtype == float

    def test_angle_to_mean_vec2(self, vec2_df):
        df = vec2_df.apsg.create_vecs2()
        df["g"] = ["a", "a"]
        result = df.groupby("g")["vecs2"].G.transform(gbf.angle_to_mean)
        assert isinstance(result, pd.Series)
        assert result.dtype == float

    def test_angle_to_mean_dir(self, dir_df):
        df = dir_df.apsg.create_dirs()
        df["g"] = ["a", "a"]
        result = df.groupby("g")["dirs"].G.transform(gbf.angle_to_mean)
        assert isinstance(result, pd.Series)
        assert result.dtype == float

    def test_angle_to_mean_lin(self, geo_df):
        df = geo_df.apsg.create_lins()
        df["g"] = ["a", "a"]
        result = df.groupby("g")["lins"].G.transform(gbf.angle_to_mean)
        assert isinstance(result, pd.Series)
        assert result.dtype == float

    def test_angle_to_mean_fol(self, geo_df):
        df = geo_df.apsg.create_fols()
        df["g"] = ["a", "a"]
        result = df.groupby("g")["fols"].G.transform(gbf.angle_to_mean)
        assert isinstance(result, pd.Series)
        assert result.dtype == float

    def test_angle_to_mean_fault_raises(self, fault_df):
        df = fault_df.apsg.create_faults()
        df["g"] = ["a", "a"]
        with pytest.raises(TypeError):
            df.groupby("g")["faults"].G.transform(gbf.angle_to_mean)


# ---------------------------------------------------------------------------
# GAccessor plot
# ---------------------------------------------------------------------------


class TestGAccessorPlot:
    def test_plot_vec2_default(self, vec2_df):
        s = vec2_df.apsg.create_vecs2()["vecs2"]
        p = s.G.plot(show=False)
        assert p is not None

    def test_plot_vec2_kind(self, vec2_df):
        s = vec2_df.apsg.create_vecs2()["vecs2"]
        p = s.G.plot(kind="pdf", show=False)
        assert p is not None

    def test_plot_vec2_dir_default(self, dir_df):
        s = dir_df.apsg.create_dirs()["dirs"]
        p = s.G.plot(show=False)
        assert p is not None

    def test_plot_vec2_dir_kind(self, dir_df):
        s = dir_df.apsg.create_dirs()["dirs"]
        p = s.G.plot(kind="bar", show=False)
        assert p is not None

    def test_plot_vec2_plot_kws(self, vec2_df):
        s = vec2_df.apsg.create_vecs2()["vecs2"]
        p = s.G.plot(plot_kws={"bins": 12}, kind="bar", show=False)
        assert p is not None

    def test_plot_vec2_dir_plot_kws(self, dir_df):
        s = dir_df.apsg.create_dirs()["dirs"]
        p = s.G.plot(plot_kws={"bins": 12}, show=False)
        assert p is not None

    def test_plot_vec3_default(self, vec_df):
        s = vec_df.apsg.create_vecs()["vecs"]
        p = s.G.plot(show=False)
        assert p is not None

    def test_plot_vec3_kind(self, vec_df):
        s = vec_df.apsg.create_vecs()["vecs"]
        p = s.G.plot(kind="vector", show=False)
        assert p is not None

    def test_plot_foliation(self, geo_df):
        s = geo_df.apsg.create_fols()["fols"]
        p = s.G.plot(show=False)
        assert p is not None

    def test_plot_lineation(self, geo_df):
        s = geo_df.apsg.create_lins()["lins"]
        p = s.G.plot(show=False)
        assert p is not None
