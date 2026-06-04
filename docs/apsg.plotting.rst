===============
plotting module
===============

The :mod:`apsg.plotting` module provides plotting classes for structural geology data. It includes
``StereoNet`` for stereographic projection, ``RosePlot`` for rose diagrams, and fabric plot types
(``VollmerPlot``, ``RamsayPlot``, ``FlinnPlot``, ``HsuPlot``) for strain and fabric analysis.

Usage
-----

Stereonet plots::

    >>> from apsg import folset, linset
    >>> from apsg.plotting import StereoNet
    >>> fols = folset.random_fisher(kappa=50, n=20)
    >>> lins = linset.random_fisher(kappa=100, n=20)
    >>> f = fols.data[0]
    >>> s = StereoNet(title="My data")
    >>> s.point(fols)
    >>> s.point(lins)
    >>> s.great_circle(f)
    >>> s.show()

Customize plot appearance::

    >>> s = StereoNet(title="Custom", kind="equal-angle", hemisphere="upper")
    >>> s.point(lins, marker="s", mfc="red", ms=8)
    >>> s.contour(fols, levels=4, cmap="Blues", colorbar=True)
    >>> s.show()

Quick plot one-liner::

    >>> from apsg import quicknet
    >>> quicknet(fols, lins, title="Quick net")

Rose diagrams::

    >>> from apsg import vec2set
    >>> from apsg.plotting import RosePlot
    >>> v = vec2set.random_vonmises(position=120, kappa=100, n=50)
    >>> p = RosePlot(grid=False)
    >>> p.bar(v, fc="none", ec="k")
    >>> p.pdf(v)
    >>> p.muci(v)
    >>> p.show()

Fabric plots::

    >>> from apsg.feature import Ellipsoid, EllipsoidSet
    >>> from apsg.plotting import VollmerPlot, FlinnPlot, RamsayPlot, HsuPlot
    >>> e1 = Ellipsoid.from_stretch(2, 1, 0.5)
    >>> e2 = Ellipsoid.from_stretch(1.5, 1.2, 0.8)
    >>> es = EllipsoidSet([e1, e2])
    >>> vp = VollmerPlot()
    >>> vp.point(es)
    >>> vp.show()
    >>>
    >>> fp = FlinnPlot()
    >>> fp.point(es)
    >>> fp.show()
    >>>
    >>> rp = RamsayPlot()
    >>> rp.point(es)
    >>> rp.show()
    >>>
    >>> hp = HsuPlot()
    >>> hp.point(es)
    >>> hp.show()

Save and load plots::

    >>> s.save('stereonet.pkl')
    >>> s2 = StereoNet.load('stereonet.pkl')
    >>> s2.show()

.. automodule:: apsg.plotting
    :autosummary:
    :members:
    :show-inheritance:
    :autosummary-no-nesting:
    :exclude-members: StereoNetArtistFactory, RosePlotArtistFactory, FabricPlotArtistFactory
