import matplotlib

matplotlib.use("Agg")

import matplotlib.colors
import matplotlib.contour
import matplotlib.lines
import matplotlib.quiver
import numpy as np
import pytest

from apsg import (
    StereoGrid,
    StereoNet,
    arc,
    arcset,
    cone,
    coneset,
    ellipsoid,
    ellipsoidset,
    fault,
    faultset,
    fol,
    folset,
    lin,
    linset,
    ortensor,
    ortensorset,
    pair,
    quicknet,
    rotation_from_axis_angle,
    stereonet_styles,
    stress,
    stressset,
    vec,
    vec2set,
)
from apsg.config import apsg_conf_context
from apsg.feature._geodata import Lineation
from apsg.feature._tensor3 import Ellipsoid, Rotation3, Stress3
from apsg.plotting import FlinnPlot, RamsayPlot, RosePlot, VollmerPlot
from apsg.plotting._stereo_engine._axes import StereonetAxes


def _tensor_sets(n=12, seed=3):
    """Ellipsoidset and stressset of noisy tensors around a diagonal mean."""
    rng = np.random.default_rng(seed)
    noise = rng.normal(scale=0.05, size=(n, 3, 3))
    mats = np.diag([1.3, 1.0, 0.7]) + (noise + noise.transpose(0, 2, 1)) / 2
    return ellipsoidset([Ellipsoid(m) for m in mats]), stressset(
        [Stress3(m) for m in mats]
    )


# ---------------------------------------------------------------------------
# Artist creation / type validation
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "method,valid_args,invalid_args",
    [
        ("point", (lin(10, 20),), (pair(140, 30, 110, 26),)),
        ("vector", (vec(1, 1, 1),), (pair(140, 30, 110, 26),)),
        ("scatter", (linset.random_fisher(n=5),), (pair(140, 30, 110, 26),)),
        ("great_circle", (fol(10, 20),), (lin(10, 20),)),
        ("gc", (fol(10, 20),), (lin(10, 20),)),
        ("cone", (cone(lin(0, 90), lin(0, 0), 30),), (lin(10, 20),)),
        ("arc", (arc(lin(0, 0), lin(90, 0)),), (pair(140, 30, 110, 26),)),
        ("pair", (pair(140, 30, 110, 26),), (lin(10, 20),)),
        ("fault", (fault(140, 30, 110, 26, 1),), (lin(10, 20),)),
        ("hoeppner", (fault(140, 30, 110, 26, 1),), (lin(10, 20),)),
        ("arrow", (lin(10, 20), lin(30, 40)), ("bad", "bad")),
        (
            "tensor",
            (ortensor([[3, 0, 0], [0, 2, 0], [0, 0, 1]]),),
            (lin(10, 20),),
        ),
        ("stress", (stress([[3, 0, 0], [0, 2, 0], [0, 0, 1]]),), (lin(10, 20),)),
        ("confidence", (linset.random_fisher(n=10),), (lin(10, 20),)),
        ("confidence", (_tensor_sets()[0],), (lin(10, 20),)),
        ("confidence", (_tensor_sets()[1],), (lin(10, 20),)),
    ],
)
def test_artist_type_validation(capsys, method, valid_args, invalid_args):
    s = StereoNet()
    getattr(s, method)(*valid_args)
    assert len(s._artists) == 1
    capsys.readouterr()

    getattr(s, method)(*invalid_args)
    assert len(s._artists) == 1  # unchanged -- invalid call was rejected
    assert "Not valid arguments" in capsys.readouterr().out


def test_contour_does_not_swallow_type_errors():
    # unlike every other plotting method, contour() has no try/except
    # wrapper -- a mismatched type raises rather than printing
    with pytest.raises(TypeError):
        StereoNet().contour(pair(140, 30, 110, 26))


def test_stress_marker_edge_color_and_width_are_applied():
    # regression: _stress() used to drop mec, so edges always followed the face color
    s = StereoNet()
    s.stress(stress([[3, 0, 0], [0, 2, 0], [0, 0, 1]]), mec="k", mew=2)
    s.init_figure()
    s._render()
    markers = [
        c
        for c in s.ax.get_children()
        if isinstance(c, matplotlib.lines.Line2D) and c.get_marker() == "*"
    ]
    assert len(markers) == 3  # sigma1, sigma2, sigma3
    assert all(m.get_markeredgecolor() == "k" for m in markers)
    assert all(m.get_markeredgewidth() == 2 for m in markers)
    # face colors still distinguish the three axes
    assert len({m.get_markerfacecolor() for m in markers}) == 3


def test_tensor_marker_edge_color_and_width_are_applied():
    # principal directions of a tensor (planes=False) are drawn as markers
    s = StereoNet()
    s.tensor(ortensor([[3, 0, 0], [0, 2, 0], [0, 0, 1]]), planes=False, mec="k", mew=2)
    s.init_figure()
    s._render()
    markers = [
        c
        for c in s.ax.get_children()
        if isinstance(c, matplotlib.lines.Line2D) and c.get_marker() == "o"
    ]
    assert len(markers) == 3
    assert all(m.get_markeredgecolor() == "k" for m in markers)
    assert all(m.get_markeredgewidth() == 2 for m in markers)
    assert len({m.get_markerfacecolor() for m in markers}) == 3


# ---------------------------------------------------------------------------
# Rendering smoke tests
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("kind", ["equal-area", "equal-angle"])
@pytest.mark.parametrize("hemisphere", ["lower", "upper"])
def test_full_render_smoke(tmp_path, kind, hemisphere):
    l = linset.random_fisher(position=lin(120, 40), n=20)
    f = folset.random_fisher(position=lin(300, 40), n=20)
    p = pair(120, 40, 120, 40)
    flt = fault(120, 40, 120, 40, 1)

    s = StereoNet(
        title="smoke",
        kind=kind,
        hemisphere=hemisphere,
        rotate_data=True,
        rotation=Rotation3.from_pair(p),
    )
    s.point(l)
    s.vector(vec(1, 1, 1))
    s.scatter(l, s=np.arange(len(l)), c=np.arange(len(l)))
    s.great_circle(f)
    s.arc(lin(0, 0), lin(90, 0), lin(0, 90))
    s.cone(cone(lin(0, 90), lin(0, 0), 30))
    s.pair(p)
    s.fault(flt)
    s.hoeppner(flt)
    s.arrow(lin(30, 10), lin(60, 20))
    s.tensor(ortensor([[3, 0, 0], [0, 2, 0], [0, 0, 1]]))
    s.stress(stress([[3, 0, 0], [0, 2, 0], [0, 0, 1]]))
    s.confidence(l, method="fisher")
    s.confidence(l, method="bootstrap", n_resamples=20)
    s.confidence(f, method="bingham")
    es, ss = _tensor_sets()
    s.confidence(es)
    s.confidence(ss, which=2, level=0.9, anisoft=True)
    s.contour(l, method="kamb")
    s.set_rotation(rotation_from_axis_angle([0, 0, 1], 30))

    out = tmp_path / "render.png"
    s.savefig(str(out))
    assert out.exists()
    assert out.stat().st_size > 0


# ---------------------------------------------------------------------------
# Arc-specific plotting
# ---------------------------------------------------------------------------


def test_arc_line_mode():
    s = StereoNet()
    s.arc(arc(lin(0, 0), lin(90, 0), curvature=0.4, positive=False, short=False))
    assert len(s._artists) == 1


def test_arc_points_mode():
    s = StereoNet()
    s.arc(arc(lin(0, 0), lin(90, 0)), kind="points")
    s.init_figure()
    s._render()

    lines = [c for c in s.ax.get_children() if isinstance(c, matplotlib.lines.Line2D)]
    line = lines[-1]
    assert line.get_linestyle() == "None"
    assert line.get_marker() != "None"


def test_arc_default_kind_is_line():
    s = StereoNet()
    s.arc(arc(lin(0, 0), lin(90, 0)))
    s.init_figure()
    s._render()

    lines = [c for c in s.ax.get_children() if isinstance(c, matplotlib.lines.Line2D)]
    line = lines[-1]
    assert line.get_marker() == "None"


def test_arcset_multi_arc():
    s = StereoNet()
    s.arc(
        arcset([arc(lin(0, 0), lin(90, 0)), arc(lin(10, 0), lin(80, 0), curvature=0.5)])
    )
    assert len(s._artists) == 1


def test_arc_legacy_raw_vector_chain_unchanged():
    # backward compatibility: raw Vector3-like features connected pairwise
    # in sequence, normalized internally into an implicit chain of Arc segments
    s = StereoNet()
    s.arc(lin(0, 0), lin(90, 0), lin(0, 90))
    assert len(s._artists) == 1


def test_arc_mixed_type_rejected(capsys):
    s = StereoNet()
    s.arc(lin(0, 0), arc(lin(0, 0), lin(90, 0)))
    assert len(s._artists) == 0
    assert "Not valid arguments" in capsys.readouterr().out


# ---------------------------------------------------------------------------
# Confidence ellipses of tensor sets (method "jelinek")
# ---------------------------------------------------------------------------


def _jelinek_line(s):
    """Render the net and return (x, y) of the last confidence line."""
    s.init_figure()
    s._render()
    line = [c for c in s.ax.get_children() if isinstance(c, matplotlib.lines.Line2D)][
        -1
    ]
    return np.asarray(line.get_xdata(), float), np.asarray(line.get_ydata(), float)


def _confidence_vectors(s, monkeypatch):
    """Render the net and return the (N, 3) vectors the last ``ax.path`` call got."""
    captured = []
    monkeypatch.setattr(
        StereonetAxes, "path", lambda self, v, **kw: captured.append(np.array(v))
    )
    s.init_figure()
    s._render()
    return captured[-1]


def test_confidence_jelinek_is_default_for_tensor_sets():
    es, ss = _tensor_sets()
    s = StereoNet()
    s.confidence(es)
    s.confidence(ss, normalize=True)
    assert [a.kwargs["method"] for a in s._artists] == ["jelinek", "jelinek"]
    assert s._artists[1].kwargs["normalize"] is True


@pytest.mark.parametrize(
    "args,kwargs",
    [
        ("es", {"method": "fisher"}),
        ("es", {"method": "bingham"}),
        ("l", {"method": "jelinek"}),
        ("both", {}),
    ],
)
def test_confidence_jelinek_rejects_mismatched_input(capsys, args, kwargs):
    es, _ = _tensor_sets()
    l = linset.random_fisher(n=10)
    s = StereoNet()
    s.confidence(*{"es": (es,), "l": (l,), "both": (es, l)}[args], **kwargs)
    assert len(s._artists) == 0
    assert "Not valid arguments" in capsys.readouterr().out


def test_confidence_jelinek_draws_all_three_ellipses(monkeypatch):
    es, ss = _tensor_sets()
    per_set = 2 * 3 * 181  # boundary and its antipode, 3 principal axes

    s = StereoNet()
    s.confidence(es)
    assert np.isfinite(_confidence_vectors(s, monkeypatch)[:, 0]).sum() == per_set

    s = StereoNet()
    s.confidence(es, ss)
    assert np.isfinite(_confidence_vectors(s, monkeypatch)[:, 0]).sum() == 2 * per_set


def test_confidence_jelinek_which_selects_ellipse(monkeypatch):
    es, _ = _tensor_sets()

    def vectors(**kwargs):
        s = StereoNet()
        s.confidence(es, **kwargs)
        return _confidence_vectors(s, monkeypatch)

    everything = vectors()
    # rows: k1 ellipse (181), gap, k2 ellipse (181), gap, k3 ellipse (181), ...
    starts = {0: 0, 1: 182, 2: 364}
    np.testing.assert_array_equal(vectors(which=None), everything)
    for which, start in starts.items():
        single = vectors(which=which)
        assert np.isfinite(single[:, 0]).sum() == 2 * 181  # boundary and its antipode
        np.testing.assert_array_equal(single[:181], everything[start : start + 181])


def test_confidence_jelinek_invalid_which():
    es, _ = _tensor_sets()
    s = StereoNet()
    s.confidence(es, which=3)
    s.init_figure()
    with pytest.raises(ValueError):
        s._render()


def test_confidence_jelinek_anisoft(monkeypatch):
    es, _ = _tensor_sets()

    def k3_ellipse_size(**kwargs):
        s = StereoNet()
        s.confidence(es, **kwargs)
        k3 = _confidence_vectors(s, monkeypatch)[364:545]
        centre = k3.mean(axis=0)
        centre /= np.linalg.norm(centre)
        return np.degrees(np.arccos(np.clip(k3 @ centre, -1, 1))).mean()

    s = StereoNet()
    s.confidence(es, anisoft=True)
    assert s._artists[0].kwargs["anisoft"] is True
    assert k3_ellipse_size(anisoft=True) < k3_ellipse_size()


def test_confidence_bingham_default_which_is_major_axis(monkeypatch):
    _, f = linset.random_fisher(n=30), folset.random_fisher(n=30)
    vectors = []
    for kwargs in ({}, {"which": 0}):
        s = StereoNet()
        s.confidence(f, method="bingham", **kwargs)
        vectors.append(_confidence_vectors(s, monkeypatch))
    np.testing.assert_array_equal(vectors[0], vectors[1])


def test_confidence_jelinek_draws_line():
    es, _ = _tensor_sets()
    s = StereoNet()
    s.confidence(es)
    x, y = _jelinek_line(s)
    assert np.isfinite(x).sum() > 3 * 181
    assert np.isfinite(y).sum() > 3 * 181


def test_confidence_jelinek_level_changes_the_curve(monkeypatch):
    es, _ = _tensor_sets()

    def k3_ellipse_size(level):
        s = StereoNet()
        s.confidence(es, level=level)
        # rows: k1 ellipse (181), gap, k2 ellipse (181), gap, k3 ellipse (181), ...
        k3 = _confidence_vectors(s, monkeypatch)[364:545]
        centre = k3.mean(axis=0)
        centre /= np.linalg.norm(centre)
        return np.degrees(np.arccos(np.clip(k3 @ centre, -1, 1))).mean()

    assert k3_ellipse_size(0.99) > k3_ellipse_size(0.5)


def test_confidence_jelinek_styled_and_json_roundtrip(tmp_path):
    es, ss = _tensor_sets()
    s = StereoNet()
    s.plot(stereonet_styles.confidence(method="jelinek"), es, ss)
    s.confidence(es, color="r")
    assert len(s._artists) == 2
    assert len(s._artists[0].args) == 2

    s2 = StereoNet.from_json(s.to_json())
    assert [a.kwargs["method"] for a in s2._artists] == ["jelinek", "jelinek"]
    out = tmp_path / "jelinek.png"
    s2.savefig(str(out))
    assert out.exists()


def test_styled_plotting_smoke(tmp_path):
    l = linset.random_fisher(n=10)
    f = folset.random_fisher(n=10)
    s = StereoNet()
    s.plot(stereonet_styles.point(), l)
    s.plot(stereonet_styles.vector(), vec(1, 1, 1))
    s.plot(stereonet_styles.scatter(), l)
    s.plot(stereonet_styles.great_circle(), f)
    s.plot(stereonet_styles.arc(), arc(lin(0, 0), lin(90, 0)))
    s.plot(stereonet_styles.arrow(), lin(30, 10), lin(60, 20))
    out = tmp_path / "styled.png"
    s.savefig(str(out))
    assert out.exists()


def test_plot_does_not_crash_on_style_type_error(capsys):
    # style.create_artist() can raise TypeError (e.g. scatter's s/c length
    # check) -- plot() must catch it like _add_artist() does for direct
    # calls, not propagate it to the caller
    l = linset.random_fisher(n=5)
    s = StereoNet()
    s.plot(stereonet_styles.scatter(s=[1, 2, 3]), l)
    assert len(s._artists) == 0
    assert "do not match" in capsys.readouterr().out


def test_quicknet_dispatch_branches(tmp_path):
    l = linset.random_fisher(n=5)
    f = folset.random_fisher(n=5)
    fs = faultset([fault(120, 40, 120, 40, 1)])
    ps = pair(120, 40, 120, 40)
    out = tmp_path / "quicknet.png"
    quicknet(
        vec(1, 1, 1),
        lin(10, 20),
        fol(10, 20),
        pair(120, 40, 120, 40),
        cone(lin(0, 90), lin(0, 0), 30),
        l,
        f,
        fs,
        ps,
        stress([[3, 0, 0], [0, 2, 0], [0, 0, 1]]),
        savefig=True,
        filename=str(out),
    )
    assert out.exists()


def test_quicknet_fol_as_pole_kwarg_does_not_crash(tmp_path):
    # regression: fol_as_pole was read with kwargs.get() (not popped), so it
    # stayed in kwargs and got passed to _quicknet_plot_one both positionally
    # and via **kwargs, raising "got multiple values for argument"
    out = tmp_path / "quicknet.png"
    quicknet(fol(10, 20), fol_as_pole=True, savefig=True, filename=str(out))
    assert out.exists()


def test_quicknet_dispatch_branches_with_artist_representation():
    from apsg.plotting._stereonet import _quicknet_plot_one

    e = ellipsoid([[3, 0, 0], [0, 2, 0], [0, 0, 1]])
    ot = ortensor([[3, 0, 0], [0, 2, 0], [0, 0, 1]])
    st = stress([[3, 0, 0], [0, 2, 0], [0, 0, 1]])
    g = StereoGrid(n=50)
    g.calculate_density(linset.random_fisher(n=5))

    cases = [
        (
            coneset(
                [cone(lin(0, 90), lin(0, 0), 20), cone(lin(0, 90), lin(90, 0), 30)]
            ),
            "_cone",
            1,
        ),
        (arc(vec(1, 0, 0), vec(0, 1, 0)), "_arc", 1),
        (
            arcset([arc(vec(1, 0, 0), vec(0, 1, 0)), arc(vec(0, 1, 0), vec(0, 0, 1))]),
            "_arc",
            1,
        ),
        (e, "_tensor", 1),
        (ellipsoidset([e, e, e]), "_tensor", 3),
        (ortensorset([ot, ot]), "_tensor", 2),
        (stressset([st, st, st]), "_stress", 3),
        (g, "_contour", 1),
    ]
    for arg, expected_method, expected_count in cases:
        s = StereoNet()
        _quicknet_plot_one(s, arg, fol_as_pole=False)
        methods = [a.stereonet_method for a in s._artists]
        assert methods == [expected_method] * expected_count


def test_apply_func_passes_vector3_not_raw_array():
    # regression: apply_func's grid points must be real Vector3 objects (with
    # apsg vector methods available), not the vendored engine's raw ndarray
    # rows -- e.g. Stress3.shear_stress calls n.normalized() on its argument
    g = StereoGrid(n=50)
    seen = []

    def probe(v):
        seen.append(v)
        return float(v.normalized().z)

    g.apply_func(probe)
    assert all(hasattr(v, "normalized") for v in seen)

    sig = stress([[3, 0, 0], [0, 2, 0], [0, 0, 1]])
    g2 = StereoGrid(n=50)
    g2.apply_func(sig.shear_stress)  # would raise AttributeError if regressed
    assert g2.calculated


def test_apply_func_and_angmech_contour_smoke(tmp_path):
    g = StereoGrid(n=200)
    g.apply_func(lambda v: float(v[0]))
    s = StereoNet()
    s.contour(g, levels=5)
    out = tmp_path / "applyfunc.png"
    s.savefig(str(out))
    assert out.exists()

    fs = faultset([fault(120, 40, 120, 40, 1), fault(200, 30, 300, 10, -1)])
    g2 = StereoGrid(n=200)
    g2.angmech(fs)
    s2 = StereoNet()
    s2.contour(g2)
    out2 = tmp_path / "angmech.png"
    s2.savefig(str(out2))
    assert out2.exists()


# ---------------------------------------------------------------------------
# Per-call StereoGrid ownership: multiple independent contour layers
# ---------------------------------------------------------------------------


def test_contour_requires_one_arg():
    with pytest.raises(TypeError):
        StereoNet().contour()


def test_contour_with_stereogrid_skips_calculation():
    g = StereoGrid(n=100)
    g.apply_func(lambda v: float(v[0]))
    before = g.values.copy()
    s = StereoNet()
    s.contour(g)
    np.testing.assert_array_equal(g.values, before)


def test_multiple_contour_calls_own_independent_grids(tmp_path):
    l1 = linset.random_fisher(position=lin(60, 40), n=40)
    l2 = linset.random_fisher(position=lin(250, 30), n=40)
    s = StereoNet()
    s.contour(l1, sigma=2)
    s.contour(l2, sigma=4, clip=True)
    assert len(s._artists) == 2
    grid1 = s._artists[0].args[0]
    grid2 = s._artists[1].args[0]
    assert grid1 is not grid2
    assert not np.array_equal(grid1.values, grid2.values)

    out = tmp_path / "multi.png"
    s.savefig(str(out))
    assert out.exists()
    contour_sets = [
        c for c in s.ax.get_children() if isinstance(c, matplotlib.contour.ContourSet)
    ]
    # 2 filled + 2 black-line-overlay ContourSets, one pair per layer
    assert len(contour_sets) == 4


def test_contour_default_method_is_kamb():
    l = linset.random_fisher(n=20)
    s = StereoNet()
    s.contour(l)
    grid = s._artists[-1].args[0]
    g2 = StereoGrid()
    g2.calculate_density(l, method="kamb")
    np.testing.assert_array_equal(grid.values, g2.values)


def test_contour_clip_defaults_true():
    l = linset.random_fisher(n=20)
    s = StereoNet()
    s.contour(l)
    assert s._artists[-1].kwargs["clip"] is True


def test_contour_default_cmap_is_greys_when_clipped_else_rdbu():
    l = linset.random_fisher(position=lin(60, 40), n=40)
    s = StereoNet()
    s.contour(l)  # clip defaults to True
    s.init_figure()
    s._render()
    filled = [
        c
        for c in s.ax.get_children()
        if isinstance(c, matplotlib.contour.ContourSet) and c.filled
    ]
    assert filled[0].get_cmap().name == "Greys"
    assert min(filled[0].levels) >= 0

    s2 = StereoNet()
    s2.contour(l, clip=False)
    s2.init_figure()
    s2._render()
    filled2 = [
        c
        for c in s2.ax.get_children()
        if isinstance(c, matplotlib.contour.ContourSet) and c.filled
    ]
    assert filled2[0].get_cmap().name == "RdBu"
    assert min(filled2[0].levels) < 0


# ---------------------------------------------------------------------------
# Contour numeric assertions -- pins the redesigned unified statistic
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("method", ["kamb", "sph"])
def test_contour_density_uniform_null_stats(method):
    np.random.seed(42)
    # a genuinely random uniform sample (not an evenly-spaced grid, which
    # has none of the natural clustering fluctuation the null model assumes)
    uniform = linset.random(n=1000)
    g = StereoGrid(n=1000)
    g.calculate_density(uniform, method=method, sigma=3, trimzero=False)
    # under a uniform/null distribution the unified statistic has mean~=0,
    # sd~=1 by construction (see _stereo_engine/_stereogrid.py)
    assert abs(g.values.mean()) < 0.3
    assert abs(g.values.std() - 1.0) < 0.3


@pytest.mark.parametrize("method", ["kamb", "sph"])
def test_contour_density_not_clipped_to_zero(method):
    # a concentrated cluster leaves the rest of the sphere emptier than
    # uniform would predict -- negative values must survive (breaking
    # change vs. the old apsg engine, which clipped negatives to 0)
    concentrated = linset.random_fisher(position=lin(0, 90), kappa=200, n=200)
    g = StereoGrid(n=1000)
    g.calculate_density(concentrated, method=method, sigma=3, trimzero=False)
    assert (g.values < 0).any()


def test_contour_default_sigma_is_three():
    l = linset.random_fisher(n=30)
    g = StereoGrid(n=300)
    g.calculate_density(l, method="kamb")  # sigma=None -> engine default 3
    g2 = StereoGrid(n=300)
    g2.calculate_density(l, method="kamb", sigma=3)
    np.testing.assert_array_equal(g.values, g2.values)


# ---------------------------------------------------------------------------
# contour() kwarg surface: filled/linewidth+lw/clip, removed kwargs
# ---------------------------------------------------------------------------


def test_contour_filled_defaults_true_and_lw_aliases_linewidth():
    l = linset.random_fisher(n=20)
    s = StereoNet()
    s.contour(l)
    assert s._artists[-1].kwargs["filled"] is True

    s2 = StereoNet()
    s2.contour(l, filled=False, lw=3)
    assert s2._artists[-1].kwargs["filled"] is False
    assert s2._artists[-1].kwargs["linewidth"] == 3
    assert "lw" not in s2._artists[-1].kwargs


def test_contour_clip_kwarg_survives_merge():
    # clip previously had no matching config field, so the artist-kwargs
    # intersection-merge silently dropped it -- it must now be honored
    l = linset.random_fisher(n=20)
    s = StereoNet()
    s.contour(l, clip=True)
    assert s._artists[-1].kwargs["clip"] is True


def test_contour_removed_kwargs_are_silently_ignored_not_errors():
    l = linset.random_fisher(n=10)
    s = StereoNet()
    s.contour(
        l,
        trimzero=False,
        clines=False,
        linewidths=5,
        show_data=True,
        data_kws={},
    )
    kwargs = s._artists[-1].kwargs
    assert "trimzero" not in kwargs
    assert "clines" not in kwargs
    assert "linewidths" not in kwargs
    assert "show_data" not in kwargs
    assert "data_kws" not in kwargs


def test_contour_filled_false_and_clip_render(tmp_path):
    l = linset.random_fisher(position=lin(120, 40), n=50)
    s = StereoNet()
    s.contour(l, filled=False, clip=True, lw=2)
    out = tmp_path / "lines_clip.png"
    s.savefig(str(out))
    assert out.exists()


# ---------------------------------------------------------------------------
# min_at/max_at return type
# ---------------------------------------------------------------------------


def test_min_max_at_return_lineation():
    l = linset.random_fisher(position=lin(120, 40), n=30)
    g = StereoGrid(n=300)
    g.calculate_density(l, method="kamb")
    assert isinstance(g.min_at(), Lineation)
    assert isinstance(g.max_at(), Lineation)


# ---------------------------------------------------------------------------
# Serialization round-trips
# ---------------------------------------------------------------------------


def test_stereonet_to_json_from_json_roundtrip(tmp_path):
    l = linset.random_fisher(n=10)
    s = StereoNet(title="rt")
    s.point(l)
    s.great_circle(folset.random_fisher(n=10))
    s.contour(l, method="kamb")

    j = s.to_json()
    s2 = StereoNet.from_json(j)
    assert len(s2._artists) == len(s._artists)
    out = tmp_path / "rt.png"
    s2.savefig(str(out))
    assert out.exists()


def test_stereonet_save_load_roundtrip(tmp_path):
    s = StereoNet()
    s.point(linset.random_fisher(n=5))
    path = tmp_path / "s.pkl"
    s.save(str(path))
    s2 = StereoNet.load(str(path))
    assert len(s2._artists) == 1


def test_stereogrid_to_json_from_json_roundtrip():
    l = linset.random_fisher(position=lin(120, 40), n=20)
    g = StereoGrid(n=200)
    g.calculate_density(l, method="kamb")
    g2 = StereoGrid.from_json(g.to_json())
    assert g2.calculated
    np.testing.assert_allclose(g2.values, g.values)


def test_stereogrid_save_load_roundtrip(tmp_path):
    fs = faultset([fault(120, 40, 120, 40, 1)])
    g = StereoGrid(n=200)
    g.angmech(fs)
    path = tmp_path / "g.pkl"
    g.save(str(path))
    g2 = StereoGrid.load(str(path))
    np.testing.assert_allclose(g2.values, g.values)


def test_stereogrid_apply_func_warns_and_does_not_roundtrip():
    g = StereoGrid(n=100)
    g.apply_func(lambda v: float(v[0]))
    with pytest.warns(UserWarning, match="apply_func"):
        j = g.to_json()
    assert j["calculation"] is None
    g2 = StereoGrid.from_json(j)
    assert not g2.calculated


def test_stereonet_from_json_backward_compat_stale_keys():
    # a hand-built old-shape dict with stray, now-removed config keys
    # (overlay_position, overlay, overlay_step, grid_type, grid_n, sigmanorm,
    # trimzero, clines, linewidths, show_data, data_kws) and no "grid" key
    # must still load without error, falling back to current defaults for
    # now-missing keys (e.g. rotation, grid, grid_step, grid_color, filled,
    # linewidth)
    old_dict = {
        "kwargs": {
            "kind": "equal-area",
            "hemisphere": "lower",
            "overlay_position": (0, 0, 0, 0),
            "rotate_data": False,
            "overlay": True,
            "overlay_step": 15,
            "clip_pole": 15,
            "grid_type": "gss",
            "grid_n": 3000,
            "tight_layout": False,
            "title_kws": {},
            "title": None,
        },
        "artists": [
            {
                "factory": "create_contour",
                "stereonet_method": "_contour",
                "args": (linset.random_fisher(n=5).to_json(),),
                "kwargs": {
                    "method": "sph",
                    "n_max": 10,
                    "sigma": None,
                    "sigmanorm": True,  # stale, no longer a config field
                    "trimzero": True,
                    "colorbar": False,
                    "label": "_contour",
                    "clines": True,
                    "linewidths": 1,
                    "linestyles": None,
                    "show_data": False,
                    "data_kws": {},
                },
            }
        ],
    }
    s = StereoNet.from_json(old_dict)
    assert len(s._artists) == 1


# ---------------------------------------------------------------------------
# Config field application
# ---------------------------------------------------------------------------


def test_config_fields_land_on_axes_and_grid(tmp_path):
    s = StereoNet(grid_step=20, clip_pole=10, hemisphere="upper")
    out = tmp_path / "cfg.png"
    s.savefig(str(out))
    assert s.ax.hemisphere == "upper"
    assert s.ax.clip_pole == 10


def test_rotate_data_defaults_true():
    s = StereoNet()
    assert s._kwargs["rotate_data"] is True


def test_rotation_accepts_rotation3_instance(tmp_path):
    R = Rotation3.from_pair(pair(120, 30, 120, 30))
    s = StereoNet(rotation=R)
    np.testing.assert_allclose(s.rotation, np.asarray(R))
    out = tmp_path / "rot.png"
    s.point(lin(10, 20))
    s.savefig(str(out))
    assert out.exists()


def test_rotation_default_is_identity():
    s = StereoNet()
    np.testing.assert_allclose(s.rotation, np.eye(3))


def test_grid_color_reaches_gridlines():
    s = StereoNet(grid_color="blue")
    s.init_figure()
    s._render()
    assert s.ax.xaxis.get_gridlines()[0].get_color() == "blue"
    assert s.ax.yaxis.get_gridlines()[0].get_color() == "blue"


def test_stereogrid_default_n_and_type_from_config():
    g = StereoGrid()
    assert g.grid_n == 3000
    assert g.grid_type == "gss"

    g2 = StereoGrid(n=500, type="sfs")
    assert g2.grid_n == 500
    assert g2.grid_type == "sfs"


# ---------------------------------------------------------------------------
# Grid z-order: must stay behind every plotted artist, including contours
# ---------------------------------------------------------------------------


def test_grid_zorder_below_points_and_contours():
    l = linset.random_fisher(position=lin(120, 40), n=30)
    s = StereoNet()
    s.point(l)
    s.contour(l, method="kamb")
    s.init_figure()
    s._render()

    grid_zorder = max(s.ax.xaxis.get_zorder(), s.ax.yaxis.get_zorder())
    artist_zorders = [
        c.get_zorder()
        for c in s.ax.get_children()
        if isinstance(c, (matplotlib.lines.Line2D, matplotlib.contour.ContourSet))
    ]
    assert artist_zorders  # sanity: something was actually plotted
    assert all(grid_zorder < z for z in artist_zorders)
    assert s.ax.patch.get_zorder() < grid_zorder


# ---------------------------------------------------------------------------
# Aesthetic tuning: dotted grid, center cross tick, thicker rim, filled lines
# ---------------------------------------------------------------------------


def test_grid_lines_are_dotted():
    s = StereoNet()
    s.init_figure()
    s._render()
    assert s.ax.xaxis.get_gridlines()[0].get_linestyle() == ":"
    assert s.ax.yaxis.get_gridlines()[0].get_linestyle() == ":"


def test_primitive_circle_is_thicker():
    s = StereoNet()
    s.init_figure()
    s._render()
    assert s.ax.spines["geo"].get_linewidth() == 1.5


def _cross_center(stereonet):
    # the center-cross is always the last 2 Line2D segments appended to
    # _azimuth_tick_artists, regardless of how many compass ticks survived
    # (a steep rotation can tip a compass reference out of the horizontal
    # plane, skipping that one tick -- see set_azimuth_ticks)
    ticklines = stereonet.ax.get_azimuth_ticklines()
    assert len(ticklines) >= 2
    cross = [(tuple(ln.get_xdata()), tuple(ln.get_ydata())) for ln in ticklines[-2:]]
    cx = np.mean([np.mean(xs) for xs, _ in cross])
    cy = np.mean([np.mean(ys) for _, ys in cross])
    return cx, cy


def test_center_cross_at_center_when_unrotated():
    s = StereoNet()
    s.init_figure()
    s._render()
    cx, cy = _cross_center(s)
    assert cx == pytest.approx(0.5)
    assert cy == pytest.approx(0.5)


def test_center_cross_moves_with_rotation_about_horizontal_axis():
    # marks the fixed physical vertical direction (0,0,1) -- a rotation
    # about the *vertical* axis leaves that direction unchanged (a
    # degenerate case), so use a horizontal rotation axis to actually
    # exercise the "the cross moves" behavior
    s = StereoNet()
    s.init_figure()
    s._render()
    cx0, cy0 = _cross_center(s)

    s.set_rotation(rotation_from_axis_angle([0, 1, 0], 30))
    cx1, cy1 = _cross_center(s)
    assert (cx1, cy1) != pytest.approx((cx0, cy0))

    # matches the same core-transform pipeline the compass ticks use
    core_transform = s.ax.transProjection + s.ax.transAffine
    expected = core_transform.transform((np.pi / 2, 0.0))
    np.testing.assert_allclose([cx1, cy1], expected, atol=1e-9)


def test_center_cross_unchanged_by_vertical_axis_rotation():
    s = StereoNet()
    s.init_figure()
    s._render()
    cx0, cy0 = _cross_center(s)
    s.set_rotation(rotation_from_axis_angle([0, 0, 1], 37))
    cx1, cy1 = _cross_center(s)
    assert (cx1, cy1) == pytest.approx((cx0, cy0))


def test_clip_pole_stays_fixed_regardless_of_rotation():
    # clip_pole is exactly what the user configured -- no auto-expansion
    s = StereoNet(clip_pole=15)
    s.init_figure()
    s._render()
    assert s.ax.clip_pole == pytest.approx(15.0)

    s.set_rotation(rotation_from_axis_angle([0, 1, 0], 40))
    assert s.ax.clip_pole == pytest.approx(15.0)


def test_transform_masks_points_near_antipodal_singularity():
    # regression for the reported repro: near the projection's own
    # antipodal singularity (rotated z -> -1), the azimuthal angle is
    # extremely sensitive to position, so two adjacent low-resolution
    # graticule samples can land far apart despite a bounded radius --
    # such points must be masked to NaN rather than projected to a
    # deceptively "valid" but wildly wrong position
    import apsg.plotting._stereo_engine._transforms as t

    R = np.asarray(rotation_from_axis_angle([0, 1, 0], 30))
    for cls in (t.EqualAreaTransform, t.EqualAngleTransform):
        transform = cls(60, rotation=R)
        glon = np.deg2rad(-89.77443609022556)
        glat = np.deg2rad(30.301507537688444)  # closest approach to z=-1
        out = transform.transform(np.array([[glon, glat]]))
        assert not np.isfinite(out).all(), f"{cls.__name__} did not mask the point"

        # a point comfortably far from the singularity is unaffected
        out_ok = transform.transform(np.array([[0.0, 0.0]]))
        assert np.isfinite(out_ok).all()


def test_rotated_gridlines_no_large_jumps(tmp_path):
    # end-to-end regression for the reported repro: with clip_pole left at
    # its default (no auto-expansion -- see test_clip_pole_stays_fixed...),
    # a horizontal-axis rotation must not leave any adjacent pair of
    # rendered gridline points connected by a spurious long chord
    s = StereoNet(rotation=rotation_from_axis_angle([0, 1, 0], 30))
    s.init_figure()
    s._render()
    assert s.ax.clip_pole == pytest.approx(15.0)

    for axis in (s.ax.xaxis, s.ax.yaxis):
        for gl in axis.get_gridlines():
            tpath = gl.get_transform().transform_path(gl.get_path())
            frac = s.ax.transAxes.inverted().transform(tpath.vertices)
            ok = np.isfinite(frac).all(axis=1)
            both_ok = ok[:-1] & ok[1:]
            d = np.hypot(np.diff(frac[:, 0]), np.diff(frac[:, 1]))
            # the primitive circle has radius 0.5 in axes-fraction -- a
            # jump larger than its diameter would have to cut across it
            assert np.all(d[both_ok] < 1.0)

    out = tmp_path / "no_fan.png"
    s.savefig(str(out))
    assert out.exists()


def test_contour_filled_true_adds_black_line_overlay():
    l = linset.random_fisher(position=lin(120, 40), n=40)
    s = StereoNet()
    s.contour(l, method="kamb")
    s.init_figure()
    s._render()
    contour_sets = [
        c for c in s.ax.get_children() if isinstance(c, matplotlib.contour.ContourSet)
    ]
    assert sorted(cs.filled for cs in contour_sets) == [False, True]


def test_contour_filled_false_has_no_black_overlay():
    l = linset.random_fisher(position=lin(120, 40), n=40)
    s = StereoNet()
    s.contour(l, method="kamb", filled=False)
    s.init_figure()
    s._render()
    contour_sets = [
        c for c in s.ax.get_children() if isinstance(c, matplotlib.contour.ContourSet)
    ]
    assert len(contour_sets) == 1
    assert contour_sets[0].filled is False


# ---------------------------------------------------------------------------
# Contour holes near the maximum: plateaued (e.g. integer angmech) statistics
# put many grid points exactly on the auto-computed top level edge, which
# left triangles unfilled there (a visible hole) unless that edge is nudged
# strictly above the true maximum.
# ---------------------------------------------------------------------------


def test_contour_top_level_strictly_exceeds_plateaued_max():
    g = StereoGrid()
    g.apply_func(lambda v: 90.0 if v[2] > 0 else 10.0)
    s = StereoNet()
    s.contour(g)
    s.init_figure()
    s._render()

    filled = [
        c
        for c in s.ax.get_children()
        if isinstance(c, matplotlib.contour.ContourSet) and c.filled
    ]
    assert len(filled) == 1
    assert filled[0].levels[-1] > 90.0


@pytest.mark.parametrize("hemisphere", ["lower", "upper"])
def test_contour_folds_whole_sphere_grid_before_projecting(hemisphere):
    g = StereoGrid(n=200)
    s = StereoNet(hemisphere=hemisphere)
    s.init_figure()
    s._render()

    X, Y = s.ax._graticule_to_axes_fraction(*s.ax._fold_axial_to_data(g._engine.grid))
    finite = np.isfinite(X) & np.isfinite(Y)
    r = np.hypot(X[finite] - 0.5, Y[finite] - 0.5)
    assert r.max() <= 0.5 + 1e-9


def test_contour_pads_rim_so_fill_reaches_the_boundary():
    g = StereoGrid(n=500)
    g.apply_func(lambda v: 1.0)
    s = StereoNet()
    s.contour(g, clip=False, levels=[0.5, 1.5])
    s.init_figure()
    s._render()

    filled = [
        c
        for c in s.ax.get_children()
        if isinstance(c, matplotlib.contour.ContourSet) and c.filled
    ]
    assert len(filled) == 1
    verts = np.concatenate(filled[0].get_paths()[0].to_polygons())
    r = np.hypot(verts[:, 0] - 0.5, verts[:, 1] - 0.5)
    assert r.max() >= 0.5 - 1e-6


def test_extend_antipodal_reprojects_true_antipode_not_a_fixed_ring():
    from apsg.plotting._stereo_engine._axes import _extend_antipodal
    from apsg.plotting._stereo_engine._transforms import graticule_from_ned

    s = StereoNet()
    s.init_figure()
    s._render()

    # a horizon point (z=0, on the primitive circle itself) and a
    # near-vertical point (z close to 1, near the disk center -- not
    # exactly 1, which would put its antipode exactly on the projection's
    # own far singularity and get masked to NaN)
    v = np.array(
        [
            [np.cos(np.radians(60)), np.sin(np.radians(60)), 0.0],  # lin(60, 0)
            [np.cos(np.radians(80)), 0.0, np.sin(np.radians(80))],  # lin(0, 80)
        ]
    )
    glon, glat = graticule_from_ned(v[:, 0], v[:, 1], v[:, 2])
    X, Y = s.ax._graticule_to_axes_fraction(glon, glat)
    values = np.array([10.0, 20.0])

    Xa, Ya, Va = _extend_antipodal(glon, glat, values, s.ax._graticule_to_axes_fraction)

    # exact values carried over -- no lookup/approximation involved
    assert np.array_equal(Va, values)

    r = np.hypot(X - 0.5, Y - 0.5)
    ra = np.hypot(Xa - 0.5, Ya - 0.5)

    # a horizon point's true antipode (trend+180, plunge 0) is also on the
    # horizon, i.e. also lands exactly on the primitive circle
    assert r[0] == pytest.approx(0.5, abs=1e-9)
    assert ra[0] == pytest.approx(0.5, abs=1e-6)

    # a near-vertical point's true antipode sits far out near the
    # projection's far singularity (r approaching sqrt(2)/2) -- unlike a
    # fixed-radius ring, the antipodal distance depends on the original
    # point's own position
    assert r[1] < 0.1
    assert ra[1] > 0.6


# ---------------------------------------------------------------------------
# Hoeppner arrow: pivots on its point (arrow's midpoint overlaps the marker)
# ---------------------------------------------------------------------------


def test_hoeppner_arrow_pivots_on_point():
    f = faultset([fault(170, 60, 182, 59, -1)])
    s = StereoNet()
    s.hoeppner(f)
    s.init_figure()
    s._render()

    quivers = [
        c for c in s.ax.get_children() if isinstance(c, matplotlib.quiver.Quiver)
    ]
    assert len(quivers) == 1
    q = quivers[0]

    assert q.pivot == "middle"
    px, py = s.ax.project(np.asarray(f[0].fol), clip_inside=True, fold=True)
    assert q.X[0] == pytest.approx(px[0], abs=1e-9)
    assert q.Y[0] == pytest.approx(py[0], abs=1e-9)


def test_hoeppner_pivot_is_configurable():
    f = faultset([fault(170, 60, 182, 59, -1)])
    with apsg_conf_context(stereonet_hoeppner={"pivot": "tail"}):
        s = StereoNet()
        s.hoeppner(f)
        s.init_figure()
        s._render()

    quivers = [
        c for c in s.ax.get_children() if isinstance(c, matplotlib.quiver.Quiver)
    ]
    assert len(quivers) == 1
    assert quivers[0].pivot == "tail"


@pytest.mark.parametrize("sense,expected_sign", [(-1, -1.0), (1, 1.0)])
@pytest.mark.parametrize(
    "stereonet_kwargs",
    [
        {},
        {"rotation": Rotation3.from_pair(pair(180, 45, 240, 27))},
    ],
    ids=["unrotated", "rotated"],
)
def test_fault_arrow_points_toward_or_away_from_vertical(
    sense, expected_sign, stereonet_kwargs
):
    f = fault(0, 30, 0, 30, sense)
    s = StereoNet(**stereonet_kwargs)
    s.fault(f)
    s.init_figure()
    s._render()

    quivers = [
        c for c in s.ax.get_children() if isinstance(c, matplotlib.quiver.Quiver)
    ]
    assert len(quivers) == 1
    q = quivers[0]

    # The arrow's tail sits off the net's true-vertical reference point
    # (axes-fraction (0.5, 0.5) only when unrotated -- see
    # ``vertical_axes_fraction``), so its direction relative to that point
    # tells us whether it points toward (sense=-1) or away from (sense=1)
    # the "+" cross mark.
    cx, cy = s.ax.vertical_axes_fraction()
    radial = np.array([q.X[0] - cx, q.Y[0] - cy])
    direction = np.array([q.U[0], q.V[0]])
    assert np.dot(radial, direction) == pytest.approx(
        expected_sign * np.linalg.norm(radial) * np.linalg.norm(direction), rel=1e-6
    )


# ---------------------------------------------------------------------------
# great_circle: one color per call (a whole FoliationSet), not per plane
# ---------------------------------------------------------------------------


def test_great_circle_set_is_one_artist_with_one_color_per_call():
    f1 = folset.random_fisher(position=fol(130, 60), n=5)
    f2 = folset.random_fisher(position=fol(210, 40), n=5)
    s = StereoNet()
    s.great_circle(f1, label="Set 1")
    s.great_circle(f2, label="Set 2")
    s.init_figure()
    s._render()

    lines = [
        c
        for c in s.ax.get_children()
        if isinstance(c, matplotlib.lines.Line2D) and c.get_color() != "black"
    ]
    # one Line2D artist per call, regardless of how many planes it draws
    assert len(lines) == 2
    assert lines[0].get_color() != lines[1].get_color()

    # ... and therefore exactly one legend entry per call, not one per plane
    _, labels = s.ax.get_legend_handles_labels()
    assert labels == ["Set 1", "Set 2"]


def test_great_circle_explicit_color_overrides_cycle():
    f = folset.random_fisher(position=fol(130, 60), n=4)
    s = StereoNet()
    s.great_circle(f, color="red")
    s.init_figure()
    s._render()

    lines = [
        c
        for c in s.ax.get_children()
        if isinstance(c, matplotlib.lines.Line2D) and c.get_color() != "black"
    ]
    assert all(line.get_color() == "red" for line in lines)


# ---------------------------------------------------------------------------
# vector()/pair(): the antipodal/paired marker must not silently burn an
# extra slot in the color cycle (previously threw off any later call's color)
# ---------------------------------------------------------------------------


def test_vector_open_marker_matches_filled_and_does_not_skip_next_color():
    s = StereoNet()
    s.vector(vec(1, 1, 1))
    s.point(lin(30, 40))
    s.init_figure()
    s._render()

    lines = [
        c
        for c in s.ax.get_children()
        if isinstance(c, matplotlib.lines.Line2D) and c.get_color() != "black"
    ]
    filled, open_, point = lines
    assert open_.get_color() == filled.get_color()
    assert open_.get_markeredgecolor() == filled.get_color()
    assert point.get_color() != filled.get_color()


def test_pair_lineation_marker_does_not_skip_next_color():
    s = StereoNet()
    s.pair(pair(120, 30, 110, 26))
    s.point(lin(30, 40))
    s.init_figure()
    s._render()

    lines = [
        c
        for c in s.ax.get_children()
        if isinstance(c, matplotlib.lines.Line2D) and c.get_color() != "black"
    ]
    great_circle, lin_marker, point = lines
    assert lin_marker.get_color() == great_circle.get_color()
    assert point.get_color() != great_circle.get_color()


# ---------------------------------------------------------------------------
# hemisphere: axial data (point/contour) must mirror through the center
# between "lower" and "upper", at the same radius (same plunge magnitude);
# directional data (vector) must keep the same two on-screen positions and
# only swap which one is filled vs open
# ---------------------------------------------------------------------------


def test_point_axial_data_mirrors_between_hemispheres():
    l = lin(50, 20)
    s_lower = StereoNet(hemisphere="lower")
    s_upper = StereoNet(hemisphere="upper")
    s_lower.init_figure()
    s_lower._render()
    s_upper.init_figure()
    s_upper._render()

    x_lower, y_lower = s_lower.ax.project(np.asarray(l), clip_inside=True, fold=True)
    x_upper, y_upper = s_upper.ax.project(np.asarray(l), clip_inside=True, fold=True)

    assert np.isfinite(x_lower).all() and np.isfinite(x_upper).all()
    r_lower = np.hypot(x_lower - 0.5, y_lower - 0.5)
    r_upper = np.hypot(x_upper - 0.5, y_upper - 0.5)
    assert r_upper == pytest.approx(r_lower, abs=1e-9)
    # mirrored through the center (opposite quadrant), not the same spot
    assert x_upper == pytest.approx(1.0 - x_lower, abs=1e-9)
    assert y_upper == pytest.approx(1.0 - y_lower, abs=1e-9)


def test_hemisphere_rotate_is_literally_the_180_degree_rotation():
    # StereonetAxes._hemisphere_rotate must be exactly the same 180-degree
    # rotation about the vertical axis that rotation_from_axis_angle
    # produces -- the whole point of the refactor is that it's a real
    # rotation, not a hand-rolled reflection that merely looks like one.
    R = rotation_from_axis_angle([0, 0, 1], 180)
    raw = np.array([[0.3, 0.4, 0.5], [-0.2, 0.6, -0.1]])
    v = raw / np.linalg.norm(raw, axis=-1, keepdims=True)

    s_lower = StereoNet(hemisphere="lower")
    s_lower.init_figure()
    s_lower._render()
    np.testing.assert_allclose(s_lower.ax._hemisphere_rotate(v), v)

    s_upper = StereoNet(hemisphere="upper")
    s_upper.init_figure()
    s_upper._render()
    np.testing.assert_allclose(s_upper.ax._hemisphere_rotate(v), v @ R.T)


def test_cone_mirrors_between_hemispheres_like_point():
    # cone() (via StereoNet._cone -> ax.path) previously showed literally
    # no difference between hemispheres (the antipodal-doubling made the
    # old, directional _vec_to_data reflection cancel out); after routing
    # through the shared _hemisphere_rotate it must mirror through the
    # center exactly like point() does.
    c = cone(lin(30, 40), lin(120, 10), 20)
    s_lower = StereoNet(hemisphere="lower")
    s_upper = StereoNet(hemisphere="upper")
    s_lower.cone(c)
    s_upper.cone(c)
    s_lower.init_figure()
    s_lower._render()
    s_upper.init_figure()
    s_upper._render()

    def axes_fraction(ax):
        (ln,) = [line for line in ax.lines if len(line.get_xdata()) > 10]
        xy = np.column_stack([ln.get_xdata(), ln.get_ydata()])
        disp = ln.get_transform().transform(xy)
        return ax.transAxes.inverted().transform(disp)

    XY_lower = axes_fraction(s_lower.ax)
    XY_upper = axes_fraction(s_upper.ax)
    finite = np.isfinite(XY_lower).all(axis=1) & np.isfinite(XY_upper).all(axis=1)
    assert finite.sum() > 0
    assert XY_upper[finite, 0] == pytest.approx(1.0 - XY_lower[finite, 0], abs=1e-9)
    assert XY_upper[finite, 1] == pytest.approx(1.0 - XY_lower[finite, 1], abs=1e-9)


def test_vector_marker_positions_fixed_only_filled_open_swap():
    v = vec(50, 20)
    s_lower = StereoNet(hemisphere="lower")
    s_upper = StereoNet(hemisphere="upper")
    s_lower.vector(v)
    s_upper.vector(v)
    s_lower.init_figure()
    s_lower._render()
    s_upper.init_figure()
    s_upper._render()

    lines_lower = [
        c
        for c in s_lower.ax.get_children()
        if isinstance(c, matplotlib.lines.Line2D) and c.get_color() != "black"
    ]
    lines_upper = [
        c
        for c in s_upper.ax.get_children()
        if isinstance(c, matplotlib.lines.Line2D) and c.get_color() != "black"
    ]
    filled_lower, open_lower = lines_lower
    filled_upper, open_upper = lines_upper

    # same two on-screen positions regardless of hemisphere ...
    assert filled_lower.get_xdata() == pytest.approx(open_upper.get_xdata())
    assert filled_lower.get_ydata() == pytest.approx(open_upper.get_ydata())
    assert open_lower.get_xdata() == pytest.approx(filled_upper.get_xdata())
    assert open_lower.get_ydata() == pytest.approx(filled_upper.get_ydata())
    # ... only which one is filled vs open swaps
    assert filled_lower.get_markerfacecolor() != "none"
    assert open_lower.get_markerfacecolor() == "none"
    assert filled_upper.get_markerfacecolor() != "none"
    assert open_upper.get_markerfacecolor() == "none"


def test_contour_density_mirrors_between_hemispheres():
    l = linset.random_fisher(position=lin(50, 20), kappa=50, n=200)
    s_lower = StereoNet(hemisphere="lower")
    s_lower.contour(l, clip=False)
    s_lower.init_figure()
    s_lower._render()

    s_upper = StereoNet(hemisphere="upper")
    s_upper.contour(l, clip=False)
    s_upper.init_figure()
    s_upper._render()

    def peak_xy(s):
        eng = s.ax.stereogrids[-1]
        X, Y = s.ax._graticule_to_axes_fraction(*s.ax._fold_axial_to_data(eng.grid))
        i = np.argmax(eng.values)
        return X[i], Y[i]

    x_lower, y_lower = peak_xy(s_lower)
    x_upper, y_upper = peak_xy(s_upper)
    assert x_upper == pytest.approx(1.0 - x_lower, abs=1e-9)
    assert y_upper == pytest.approx(1.0 - y_lower, abs=1e-9)


def test_great_circle_trace_mirrors_between_hemispheres():
    f = fol(130, 60)
    s_lower = StereoNet(hemisphere="lower")
    s_lower.great_circle(f)
    s_lower.init_figure()
    s_lower._render()

    s_upper = StereoNet(hemisphere="upper")
    s_upper.great_circle(f)
    s_upper.init_figure()
    s_upper._render()

    def axes_fraction_xy(s):
        line = [
            c
            for c in s.ax.get_children()
            if isinstance(c, matplotlib.lines.Line2D) and c.get_color() != "black"
        ][0]
        glon, glat = line.get_data()
        return s.ax._graticule_to_axes_fraction(glon, glat)

    x_lower, y_lower = axes_fraction_xy(s_lower)
    x_upper, y_upper = axes_fraction_xy(s_upper)
    finite = np.isfinite(x_lower)
    assert x_upper[finite] == pytest.approx(1.0 - x_lower[finite], abs=1e-9)
    assert y_upper[finite] == pytest.approx(1.0 - y_lower[finite], abs=1e-9)


# ---------------------------------------------------------------------------
# axial data on a rotated net: rotating the net (rotate_data=True, the
# default) must show the same picture as pre-rotating the data on an
# unrotated net -- the fold/reflect decision has to be made in the rotated
# (effective) frame, not the raw input frame
# ---------------------------------------------------------------------------


def test_point_on_rotated_net_matches_pre_rotated_data():
    l = linset.random_fisher(position=lin(0, 30), kappa=50, n=50)
    R = rotation_from_axis_angle(lin(90, 0), 50)

    s_manual = StereoNet()
    s_manual.point(l.transform(R))
    s_manual.init_figure()
    s_manual._render()

    s_net = StereoNet(rotation=R)
    s_net.point(l)
    s_net.init_figure()
    s_net._render()

    x_manual, y_manual = s_manual.ax.project(
        np.asarray(l.transform(R)), clip_inside=True, fold=True
    )
    x_net, y_net = s_net.ax.project(np.asarray(l), clip_inside=True, fold=True)

    assert np.isfinite(x_net).all()
    assert x_net == pytest.approx(x_manual, abs=1e-9)
    assert y_net == pytest.approx(y_manual, abs=1e-9)


def test_point_on_rotated_net_without_rotate_data_uses_raw_frame():
    l = linset.random_fisher(position=lin(0, 30), kappa=50, n=50)
    R = rotation_from_axis_angle(lin(90, 0), 50)

    s_unrotated = StereoNet()
    s_unrotated.point(l)
    s_unrotated.init_figure()
    s_unrotated._render()

    s_net = StereoNet(rotation=R, rotate_data=False)
    s_net.point(l)
    s_net.init_figure()
    s_net._render()

    x_plain, y_plain = s_unrotated.ax.project(
        np.asarray(l), clip_inside=True, fold=True
    )
    x_net, y_net = s_net.ax.project(np.asarray(l), clip_inside=True, fold=True)

    assert np.isfinite(x_net).all()
    assert x_net == pytest.approx(x_plain, abs=1e-9)
    assert y_net == pytest.approx(y_plain, abs=1e-9)


# ---------------------------------------------------------------------------
# Newly exposed visual-appearance config keys
# ---------------------------------------------------------------------------


def test_primitive_lw_and_color_are_configurable():
    s = StereoNet(primitive_lw=4, primitive_color="red")
    s.init_figure()
    s._render()
    assert s.ax.spines["geo"].get_linewidth() == 4
    assert s.ax.spines["geo"].get_edgecolor() == matplotlib.colors.to_rgba("red")


def test_grid_style_is_configurable():
    s = StereoNet(grid_style="--")
    s.init_figure()
    s._render()
    assert s.ax.xaxis.get_gridlines()[0].get_linestyle() == "--"


def test_azimuth_ticks_on_by_default_and_removable():
    s_on = StereoNet()
    s_on.init_figure()
    s_on._render()
    assert [t.get_text() for t in s_on.ax.get_azimuth_ticklabels()] == [
        "N",
        "E",
        "S",
        "W",
    ]

    s_off = StereoNet(azimuth_ticks=False)
    s_off.init_figure()
    s_off._render()
    assert s_off.ax.get_azimuth_ticklabels() == []
    assert s_off.ax.get_azimuth_ticklines() == []


def test_azimuth_ticks_kws_overrides_angles_and_labels():
    s = StereoNet(
        azimuth_ticks_kws={"angles": [0, 120, 240], "labels": ["A", "B", "C"]}
    )
    s.init_figure()
    s._render()
    assert [t.get_text() for t in s.ax.get_azimuth_ticklabels()] == ["A", "B", "C"]


def test_stereonet_legend_kws_overrides_defaults(tmp_path):
    s = StereoNet(legend_kws={"loc": "lower right", "prop": {"size": 20}})
    s.point(lin(10, 20), label="L")
    s.init_figure()
    s._render()
    assert s.ax.get_legend()._loc == 4  # matplotlib code for "lower right"
    assert s.ax.get_legend().get_texts()[0].get_fontsize() == 20


def test_roseplot_legend_kws_overrides_defaults():
    v = vec2set.random_vonmises(position=120, n=10)
    r = RosePlot(legend_kws={"prop": {"size": 20}})
    r.bar(v, legend=True, label="bar")
    r.init_figure()
    r._render()
    assert r.ax.get_legend().get_texts()[0].get_fontsize() == 20


def test_fabricplot_legend_kws_overrides_defaults():
    ot = linset.random_fisher(position=lin(120, 40), n=10).ortensor()
    vp = VollmerPlot(legend_kws={"prop": {"size": 20}})
    vp.point(ot, label="pt")
    vp.init_figure()
    vp._render()
    assert vp.ax.get_legend().get_texts()[0].get_fontsize() == 20


def test_contour_line_color_is_configurable():
    l = linset.random_fisher(position=lin(120, 40), n=30)
    s = StereoNet()
    s.contour(l, line_color="blue")
    s.init_figure()
    s._render()
    lines = [
        c
        for c in s.ax.get_children()
        if isinstance(c, matplotlib.contour.ContourSet) and not c.filled
    ]
    assert lines
    assert lines[0].get_edgecolor()[0] == pytest.approx(
        matplotlib.colors.to_rgba("blue")
    )


def test_contour_colorbar_kws_reach_figure_colorbar(tmp_path):
    l = linset.random_fisher(position=lin(120, 40), n=30)
    s = StereoNet()
    s.contour(l, colorbar=True, colorbar_kws={"shrink": 0.8, "anchor": (0.0, 0.5)})
    out = tmp_path / "colorbar.png"
    s.savefig(str(out))
    assert out.exists()


@pytest.mark.parametrize(
    "plot_cls,kwargs,expected_attr,expected_value",
    [
        (VollmerPlot, {"border_color": "red"}, None, None),
        (VollmerPlot, {"tick_color": "green"}, None, None),
        (VollmerPlot, {"label_fontsize": 20}, None, None),
        (VollmerPlot, {"background_color": "yellow"}, None, None),
        (RamsayPlot, {"refline_color": "purple", "refline_lw": 3}, None, None),
        (FlinnPlot, {"refline_color": "purple", "refline_lw": 3}, None, None),
    ],
)
def test_fabricplot_styling_keys_render_without_error(
    tmp_path, plot_cls, kwargs, expected_attr, expected_value
):
    ot = linset.random_fisher(position=lin(120, 40), n=10).ortensor()
    p = plot_cls(**kwargs)
    p.point(ot)
    out = tmp_path / f"{plot_cls.__name__}.png"
    p.savefig(str(out))
    assert out.exists()


def test_vollmer_border_color_reaches_border_line():
    ot = linset.random_fisher(position=lin(120, 40), n=10).ortensor()
    vp = VollmerPlot(border_color="red", border_lw=4)
    vp.point(ot)
    vp.init_figure()
    vp._render()
    border_line = vp.ax.lines[0]
    assert border_line.get_color() == "red"
    assert border_line.get_linewidth() == 4


def test_vollmer_tick_color_reaches_tick_lines():
    ot = linset.random_fisher(position=lin(120, 40), n=10).ortensor()
    vp = VollmerPlot(tick_color="green", tick_lw=3)
    vp.point(ot)
    vp.init_figure()
    vp._render()
    tick_lines = [ln for ln in vp.ax.lines if ln.get_color() == "green"]
    assert tick_lines
    assert all(ln.get_linewidth() == 3 for ln in tick_lines)


def test_ramsay_refline_color_and_width_configurable():
    ot = linset.random_fisher(position=lin(120, 40), n=10).ortensor()
    rp = RamsayPlot(refline_color="purple", refline_lw=3)
    rp.point(ot)
    rp.init_figure()
    rp._render()
    refline = [ln for ln in rp.ax.lines if ln.get_color() == "purple"]
    assert refline
    assert refline[0].get_linewidth() == 3


# ---------------------------------------------------------------------------
# format_coord: cursor readout must match what point() actually plotted,
# on both hemispheres and independent of net rotation/rotate_data
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("hemisphere", ["lower", "upper"])
def test_format_coord_matches_plotted_point(hemisphere):
    s = StereoNet(hemisphere=hemisphere)
    l = lin(50, 20)
    s.point(l)
    s.init_figure()
    s._render()
    glon, glat = s.ax._fold_axial_to_data(np.asarray(l))
    assert s.format_coord(float(glon[0]), float(glat[0])) == "L:50/20 S:230/70"


@pytest.mark.parametrize("hemisphere", ["lower", "upper"])
@pytest.mark.parametrize("rotate_data", [True, False])
def test_format_coord_matches_plotted_point_on_rotated_net(hemisphere, rotate_data):
    R = rotation_from_axis_angle(lin(90, 0), 40)
    s = StereoNet(hemisphere=hemisphere, rotation=R, rotate_data=rotate_data)
    l = lin(20, 70)
    s.point(l)
    s.init_figure()
    s._render()
    glon, glat = s.ax._fold_axial_to_data(np.asarray(l))
    assert s.format_coord(float(glon[0]), float(glat[0])) == "L:20/70 S:200/20"


def test_format_coord_not_blank_inside_circle_on_upper_hemisphere():
    # regression: format_coord used to route through the directional
    # (vector-style) hemisphere reflection, which flips which raw vectors
    # land inside the primitive circle -- making this cursor readout blank
    # across the *entire* net whenever hemisphere="upper"
    s = StereoNet(hemisphere="upper")
    s.init_figure()
    s._render()
    rng = np.random.default_rng(0)
    checked = 0
    for _ in range(200):
        glon = rng.uniform(-np.pi, np.pi)
        glat = rng.uniform(-np.pi / 2, np.pi / 2)
        X, Y = s.ax._graticule_to_axes_fraction(np.array([glon]), np.array([glat]))
        if (X[0] - 0.5) ** 2 + (Y[0] - 0.5) ** 2 <= 0.25:
            checked += 1
            assert s.format_coord(glon, glat) != ""
    assert checked > 0  # sanity: the sampling actually hit the circle
