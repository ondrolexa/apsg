import numpy as np
import pandas as pd
import pytest

import apsg.pandas  # noqa: F401 - registers extension dtypes and accessors
from apsg import (
    dir2,
    dir2set,
    fault,
    fol,
    folset,
    lin,
    linset,
    vec,
    vec2,
    vec2set,
    vecset,
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
def vec2_array():
    return Vec2Array([vec2(1, 0), vec2(0, 1)])


@pytest.fixture
def dir_array():
    return DirArray([dir2(0), dir2(90)])


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
# VecAccessor (.vec)
# ---------------------------------------------------------------------------


class TestVecAccessor:
    def test_default_call(self, vec_df):
        fs = vec_df.vec()
        assert isinstance(fs, vecset)
        assert len(fs) == 2

    def test_set_columns_custom(self, geo_df):
        # vec accessor can be repointed at azi/inc-named columns too
        geo_df.vec.set_columns(x="azi", y="inc", z="azi")
        fs = geo_df.vec()
        assert isinstance(fs, vecset)
        assert len(fs) == 2

    def test_set_columns_invalid_kwarg_raises(self, vec_df):
        with pytest.raises(TypeError, match="Unknown column parameter"):
            vec_df.vec.set_columns(bogus="x")

    def test_set_columns_sticks_across_accesses(self, vec_df):
        vec_df.vec.set_columns(x="y", y="x", z="z")
        assert vec_df.vec._columns == {"x": "y", "y": "x", "z": "z"}

    def test_set_columns_independent_per_copy(self, vec_df):
        other = vec_df.copy()
        other.vec.set_columns(x="y")
        assert vec_df.vec._columns["x"] == "x"
        assert other.vec._columns["x"] == "y"

    def test_nan_row_skipped(self):
        df = pd.DataFrame({"x": [1.0, np.nan], "y": [0.0, 1.0], "z": [0.0, 0.0]})
        fs = df.vec()
        assert len(fs) == 1

    def test_plot_default(self, vec_df):
        p = vec_df.vec.plot(show=False)
        assert p is not None

    def test_plot_custom_kind(self, vec_df):
        p = vec_df.vec.plot(kind="vector", show=False)
        assert p is not None


# ---------------------------------------------------------------------------
# Vec2Accessor (.vec2)
# ---------------------------------------------------------------------------


class TestVec2Accessor:
    def test_default_call(self, vec2_df):
        fs = vec2_df.vec2()
        assert isinstance(fs, vec2set)
        assert len(fs) == 2

    def test_R(self, vec2_df):
        r = vec2_df.vec2().R()
        assert isinstance(r, vec2)

    def test_var(self, vec2_df):
        v = vec2_df.vec2().var()
        assert isinstance(v, float)

    def test_ortensor(self, vec2_df):
        ot = vec2_df.vec2().ortensor()
        assert ot is not None

    def test_plot_default(self, vec2_df):
        p = vec2_df.vec2.plot(show=False)
        assert p is not None

    def test_plot_kind(self, vec2_df):
        p = vec2_df.vec2.plot(kind="pdf", show=False)
        assert p is not None

    def test_plot_kws(self, vec2_df):
        p = vec2_df.vec2.plot(plot_kws={"bins": 12}, kind="bar", show=False)
        assert p is not None


# ---------------------------------------------------------------------------
# DirAccessor (.dir)
# ---------------------------------------------------------------------------


class TestDirAccessor:
    def test_default_call(self, dir_df):
        fs = dir_df.dir()
        assert isinstance(fs, dir2set)
        assert len(fs) == 2

    def test_R(self, dir_df):
        r = dir_df.dir().R()
        assert isinstance(r, dir2)

    def test_var(self, dir_df):
        v = dir_df.dir().var()
        assert isinstance(v, float)

    def test_plot_default(self, dir_df):
        p = dir_df.dir.plot(show=False)
        assert p is not None

    def test_plot_kind(self, dir_df):
        p = dir_df.dir.plot(kind="bar", show=False)
        assert p is not None

    def test_plot_kws(self, dir_df):
        p = dir_df.dir.plot(plot_kws={"bins": 12}, show=False)
        assert p is not None


# ---------------------------------------------------------------------------
# FolAccessor (.fol)
# ---------------------------------------------------------------------------


class TestFolAccessor:
    def test_default_call(self, geo_df):
        fs = geo_df.fol()
        assert isinstance(fs, folset)
        assert len(fs) == 2

    def test_R(self, geo_df):
        r = geo_df.fol().R()
        assert isinstance(r, fol)

    def test_stat_methods(self, geo_df):
        assert isinstance(geo_df.fol().R(), fol)
        assert isinstance(geo_df.fol().fisher_statistics()["k"], float)

    def test_set_columns_and_wide_format(self):
        df = pd.DataFrame(
            {"S1_azi": [10], "S1_inc": [20], "S2_azi": [30], "S2_inc": [40]}
        )
        df.fol.set_columns(azi="S1_azi", inc="S1_inc")
        s1 = df.fol()
        df.fol.set_columns(azi="S2_azi", inc="S2_inc")
        s2 = df.fol()
        assert list(s1) == [fol(10, 20)]
        assert list(s2) == [fol(30, 40)]

    def test_plot_default(self, geo_df):
        p = geo_df.fol.plot(show=False)
        assert p is not None

    def test_plot_kind(self, geo_df):
        p = geo_df.fol.plot(kind="pole", show=False)
        assert p is not None


# ---------------------------------------------------------------------------
# LinAccessor (.lin)
# ---------------------------------------------------------------------------


class TestLinAccessor:
    def test_default_call(self, geo_df):
        fs = geo_df.lin()
        assert isinstance(fs, linset)
        assert len(fs) == 2

    def test_name_defaults_to_accessor_name(self, geo_df):
        assert geo_df.lin().name == "lin"

    def test_name_override(self, geo_df):
        assert geo_df.lin(name="custom").name == "custom"

    def test_R(self, geo_df):
        r = geo_df.lin().R()
        assert isinstance(r, lin)

    def test_stat_methods(self, geo_df):
        assert isinstance(geo_df.lin().R(), lin)
        assert isinstance(geo_df.lin().fisher_statistics()["k"], float)
        assert isinstance(geo_df.lin().fisher_statistics(level=0.99)["alpha"], float)
        assert isinstance(geo_df.lin().var(), float)
        assert isinstance(geo_df.lin().delta(), float)
        assert isinstance(geo_df.lin().rdegree(), float)
        assert geo_df.lin().ortensor() is not None

    def test_set_columns_custom(self):
        df = pd.DataFrame({"trend": [110, 200], "plunge": [26, 45]})
        df.lin.set_columns(azi="trend", inc="plunge")
        fs = df.lin()
        assert isinstance(fs, linset)
        assert len(fs) == 2

    def test_set_columns_invalid_kwarg_raises(self, geo_df):
        with pytest.raises(TypeError, match="Unknown column parameter"):
            geo_df.lin.set_columns(bogus="azi")

    def test_set_columns_sticks_across_accesses(self, geo_df):
        geo_df.lin.set_columns(azi="inc")
        assert geo_df.lin._columns == {"azi": "inc", "inc": "inc"}

    def test_nan_row_skipped(self):
        df = pd.DataFrame({"azi": [110, np.nan], "inc": [26, 45]})
        assert len(df.lin()) == 1

    def test_plot_default(self, geo_df):
        p = geo_df.lin.plot(show=False)
        assert p is not None

    def test_plot_label_kwarg(self, geo_df):
        p = geo_df.lin.plot(label="L3", show=False)
        assert p is not None


# ---------------------------------------------------------------------------
# FaultAccessor (.fault)
# ---------------------------------------------------------------------------


class TestFaultAccessor:
    def test_default_call(self, fault_df):
        from apsg import faultset

        fs = fault_df.fault()
        assert isinstance(fs, faultset)
        assert len(fs) == 2

    def test_ortensor(self, fault_df):
        ot = fault_df.fault().ortensor()
        assert ot is not None

    def test_plot_default(self, fault_df):
        p = fault_df.fault.plot(show=False)
        assert p is not None

    def test_plot_kind(self, fault_df):
        p = fault_df.fault.plot(kind="pair", show=False)
        assert p is not None


# ---------------------------------------------------------------------------
# GroupBy — apply / transform / aggregate
# ---------------------------------------------------------------------------


class TestLinGroupBy:
    def test_apply_lin(self, structure_df):
        def first_lin(fs):
            return fs.ortensor().eigenlins(which=0)

        result = structure_df.lin.groupby("structure").apply(first_lin)
        assert result.dtype.name == "lin"

    def test_apply_non_apsg_result(self, structure_df):
        result = structure_df.lin.groupby("structure").apply(len)
        assert result.dtype.name == "int64"

    def test_apply_empty_group(self, structure_df):
        sub = structure_df[structure_df["structure"] == "L3"]
        result = sub.lin.groupby("structure").apply(lambda fs: fs.R())
        assert isinstance(result, pd.Series)

    def test_apply_axis_label_preserved(self, structure_df):
        result = structure_df.lin.groupby("structure").apply(lambda fs: fs.R())
        assert list(result.index) == ["L3", "S1"]

    def test_apply_with_args(self, structure_df):
        def pick(fs, which):
            return fs.ortensor().eigenlins(which=which)

        result = structure_df.lin.groupby("structure").apply(pick, which=0)
        assert result.dtype.name == "lin"

    def test_apply_multi_column_by(self, structure_df):
        df = structure_df.copy()
        df["g2"] = "x"
        result = df.lin.groupby(["structure", "g2"]).apply(lambda fs: fs.R())
        assert result.dtype.name == "lin"
        assert isinstance(result.index, pd.MultiIndex)

    # ------------------------------------------------------------------ #
    # aggregate / agg
    # ------------------------------------------------------------------ #

    def test_aggregate_lin(self, structure_df):
        def first_lin(fs):
            return fs.ortensor().eigenlins(which=0)

        result = structure_df.lin.groupby("structure").aggregate(first_lin)
        assert result.dtype.name == "lin"

    def test_aggregate_non_apsg_result(self, structure_df):
        result = structure_df.lin.groupby("structure").aggregate(len)
        assert result.dtype.name == "int64"

    def test_agg_lin(self, structure_df):
        def first_lin(fs):
            return fs.ortensor().eigenlins(which=0)

        result = structure_df.lin.groupby("structure").agg(first_lin)
        assert result.dtype.name == "lin"

    def test_agg_list_of_functions(self, structure_df):
        result = structure_df.lin.groupby("structure").agg([len, "count"])
        assert isinstance(result, pd.DataFrame)
        assert list(result.columns) == ["len", "count"]

    def test_agg_list_of_tuples(self, structure_df):
        result = structure_df.lin.groupby("structure").aggregate(
            [
                ("major", lambda fs: gbf.eigenlin(fs, which=0)),
                ("minor", lambda fs: gbf.eigenlin(fs, which=2)),
            ]
        )
        assert isinstance(result, pd.DataFrame)
        assert list(result.columns) == ["major", "minor"]
        assert result["major"].dtype.name == "lin"

    # ------------------------------------------------------------------ #
    # transform
    # ------------------------------------------------------------------ #

    def test_transform_scalar_result(self, structure_df):
        def my_transform_scalar(fs):
            return fs.angle(fs.R())

        result = structure_df.lin.groupby("structure").transform(my_transform_scalar)
        assert result.dtype.name == "float64"
        assert len(result) == len(structure_df)

    def test_transform_vector_result(self, structure_df):
        def my_transform_vector(fs):
            return fs.cross(fs.R())

        result = structure_df.lin.groupby("structure").transform(my_transform_vector)
        assert result.dtype.name == "fol"
        assert len(result) == len(structure_df)

    def test_transform_length_mismatch_raises(self, structure_df):
        def bad(fs):
            return [1]  # structure_df's groups have 3 rows each, so this never matches

        with pytest.raises(ValueError, match="valid"):
            structure_df.lin.groupby("structure").transform(bad)


class TestVec2GroupBy:
    def test_apply_vec2(self, vec2_df):
        df = vec2_df.copy()
        df["group"] = ["a", "b"]
        result = df.vec2.groupby("group").apply(lambda fs: fs.R())
        assert result.dtype.name == "vec2"
        assert len(result) == 2

    def test_aggregate_vec2(self, vec2_df):
        df = vec2_df.copy()
        df["group"] = ["a", "a"]
        result = df.vec2.groupby("group").aggregate(lambda fs: fs.R())
        assert result.dtype.name == "vec2"

    def test_aggregate_non_apsg(self, vec2_df):
        df = vec2_df.copy()
        df["group"] = ["a", "b"]
        result = df.vec2.groupby("group").aggregate(len)
        assert result.dtype.name == "int64"

    def test_transform(self, vec2_df):
        df = vec2_df.copy()
        df["group"] = ["a", "a"]

        def angle_to_r(fs):
            return fs.angle(fs.R())

        result = df.vec2.groupby("group").transform(angle_to_r)
        assert result.dtype.name == "float64"
        assert len(result) == 2


# ---------------------------------------------------------------------------
# Wide-format multiple sequential configs (single-stateful-config model)
# ---------------------------------------------------------------------------


class TestSequentialWideFormat:
    def test_two_lin_sets_from_one_wide_df(self):
        df = pd.DataFrame(
            {
                "L1_azi": [110, 120],
                "L1_inc": [26, 30],
                "L2_azi": [200, 210],
                "L2_inc": [45, 50],
            }
        )
        df.lin.set_columns(azi="L1_azi", inc="L1_inc")
        l1 = df.lin(name="L1")
        df.lin.set_columns(azi="L2_azi", inc="L2_inc")
        l2 = df.lin(name="L2")
        assert l1.name == "L1"
        assert l2.name == "L2"
        assert list(l1) == [lin(110, 26), lin(120, 30)]
        assert list(l2) == [lin(200, 45), lin(210, 50)]


# ---------------------------------------------------------------------------
# gbf helpers
# ---------------------------------------------------------------------------


class TestGBFMean:
    def test_mean_vec3(self, vec_df):
        df = vec_df.copy()
        df["g"] = ["a", "a"]
        result = df.vec.groupby("g").apply(gbf.mean)
        assert isinstance(result.iloc[0], vec)

    def test_mean_vec2(self, vec2_df):
        df = vec2_df.copy()
        df["g"] = ["a", "a"]
        result = df.vec2.groupby("g").apply(gbf.mean)
        assert isinstance(result.iloc[0], vec2)

    def test_mean_dir(self, dir_df):
        df = dir_df.copy()
        df["g"] = ["a", "a"]
        result = df.dir.groupby("g").apply(gbf.mean)
        assert isinstance(result.iloc[0], float)

    def test_mean_lin(self, geo_df):
        df = geo_df.copy()
        df["g"] = ["a", "a"]
        result = df.lin.groupby("g").apply(gbf.mean)
        assert isinstance(result.iloc[0], lin)

    def test_mean_fol(self, geo_df):
        df = geo_df.copy()
        df["g"] = ["a", "a"]
        result = df.fol.groupby("g").apply(gbf.mean)
        assert isinstance(result.iloc[0], fol)

    def test_mean_fault_raises(self, fault_df):
        df = fault_df.copy()
        df["g"] = ["a", "a"]
        with pytest.raises(TypeError, match="mean is not defined"):
            df.fault.groupby("g").apply(gbf.mean)

    def test_mean_custom_function(self, geo_df):
        df = geo_df.copy()
        df["g"] = ["a", "a"]
        result = df.lin.groupby("g").agg([gbf.mean, gbf.resultant_magnitude])
        assert isinstance(result, pd.DataFrame)
        assert "mean" in result.columns

    def test_angle_to_mean_vec3(self, vec_df):
        df = vec_df.copy()
        df["g"] = ["a", "a"]
        result = df.vec.groupby("g").transform(gbf.angle_to_mean)
        assert isinstance(result, pd.Series)
        assert result.dtype == float

    def test_angle_to_mean_vec2(self, vec2_df):
        df = vec2_df.copy()
        df["g"] = ["a", "a"]
        result = df.vec2.groupby("g").transform(gbf.angle_to_mean)
        assert isinstance(result, pd.Series)
        assert result.dtype == float

    def test_angle_to_mean_dir(self, dir_df):
        df = dir_df.copy()
        df["g"] = ["a", "a"]
        result = df.dir.groupby("g").transform(gbf.angle_to_mean)
        assert isinstance(result, pd.Series)
        assert result.dtype == float

    def test_angle_to_mean_lin(self, geo_df):
        df = geo_df.copy()
        df["g"] = ["a", "a"]
        result = df.lin.groupby("g").transform(gbf.angle_to_mean)
        assert isinstance(result, pd.Series)
        assert result.dtype == float

    def test_angle_to_mean_fol(self, geo_df):
        df = geo_df.copy()
        df["g"] = ["a", "a"]
        result = df.fol.groupby("g").transform(gbf.angle_to_mean)
        assert isinstance(result, pd.Series)
        assert result.dtype == float

    def test_angle_to_mean_fault_raises(self, fault_df):
        df = fault_df.copy()
        df["g"] = ["a", "a"]
        with pytest.raises(TypeError):
            df.fault.groupby("g").transform(gbf.angle_to_mean)
