import matplotlib.pyplot as plt
import numpy as np
import pytest

import apsg
from apsg import FisherQQPlot, linset, vec, vecset
from apsg.feature._distributions import VonMisesFisher
from apsg.feature._hypothesis import fisher_goodness_of_fit, fisher_qq_data
from apsg.plotting import FigurePlot


@pytest.fixture
def fisher_sample():
    return vecset.random_fisher(n=80, kappa=30, mean=vec(120, 40), rng=0)


def _panel_data(plot):
    plot.fig = plt.figure()
    plot._draw()
    out = []
    for ax in plot.fig.axes:
        line = ax.get_lines()[1]
        out.append((np.asarray(line.get_xdata()), np.asarray(line.get_ydata())))
    return out


def test_qq_data_matches_goodness_of_fit(fisher_sample):
    X = VonMisesFisher.as_array(fisher_sample)
    q = fisher_qq_data(X)
    res = fisher_goodness_of_fit(X)
    n = q["n"]
    assert res["colatitude"].extra["k"] == q["k"]
    assert res["colatitude"].statistic == pytest.approx(
        (q["D"] - 0.2 / n) * (np.sqrt(n) + 0.26 + 0.5 / np.sqrt(n))
    )
    assert np.all(np.diff(q["x"]) >= 0) and np.all(np.diff(q["u"]) >= 0)


def test_fisher_data_lie_on_the_diagonal(fisher_sample):
    plot = FisherQQPlot(fisher_sample)
    for expected, observed in _panel_data(plot):
        assert np.corrcoef(expected, observed)[0, 1] > 0.98
    assert not plot._tests["colatitude"].reject()
    assert not plot._tests["longitude"].reject()
    plot.render2fig(plt.figure())
    assert "M_E" in plot.fig.axes[0].get_title()
    assert "consistent" in plot.fig.axes[0].get_title()


def test_non_fisher_data_are_rejected():
    # a girdle is far from a Fisher distribution
    rng = np.random.default_rng(1)
    phi = rng.uniform(0, 2 * np.pi, 80)
    X = np.c_[np.cos(phi), np.sin(phi), rng.normal(0, 0.05, 80)]
    plot = FisherQQPlot(X)
    assert plot._tests["colatitude"].reject() or plot._tests["longitude"].reject()
    plot.render2fig(plt.figure())
    assert any("rejected" in ax.get_title() for ax in plot.fig.axes)


def test_axial_sets_and_arrays_are_accepted():
    ls = linset.random_fisher(n=30, kappa=20, mean=vec(10, 30), rng=0)
    FisherQQPlot(ls).render2fig(plt.figure())
    FisherQQPlot(np.asarray(ls)).render2fig(plt.figure())
    FisherQQPlot(ls, which="colatitude").render2fig(plt.figure())
    assert len(plt.gcf().axes) == 1


def test_protocol(fisher_sample, tmp_path):
    plot = FisherQQPlot(fisher_sample, title="mine")
    assert isinstance(plot, FigurePlot)
    plot.savefig(tmp_path / "qq.png")
    assert (tmp_path / "qq.png").stat().st_size > 0
    fig = plt.figure(figsize=(7, 3))
    plot.render2fig(fig)
    assert tuple(fig.get_size_inches()) == (7, 3)
    assert fig._suptitle.get_text() == "mine"
    sub = plt.figure(figsize=(10, 5)).subfigures(1, 2)[0]
    plot.render2fig(sub)


def test_options_are_validated(fisher_sample):
    with pytest.raises(ValueError):
        FisherQQPlot(fisher_sample, which="other")
    with pytest.raises(ValueError):
        FisherQQPlot(fisher_sample, level=0.5)
    with pytest.raises(TypeError, match="unexpected"):
        FisherQQPlot(fisher_sample, nonsense=1)
    with pytest.raises(ValueError):
        FisherQQPlot(fisher_sample[:2])
    assert apsg.apsg_conf.qqplot["level"] == 0.95
