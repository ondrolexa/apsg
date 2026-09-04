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

``kind`` selects equal-area (``"equal-area"``/``"schmidt"``, the default -- also known as a
Schmidt net) or equal-angle (``"equal-angle"``/``"wulff"``, a Wulff net) projection.
``hemisphere`` selects which hemisphere axial data (``point``, ``great_circle``, ``cone``,
``contour``, ...) is projected onto -- ``"lower"`` (default) or ``"upper"``: a line plunging
into the lower hemisphere at a given azimuth plots at that azimuth on a lower net, and at the
antipodal azimuth (same distance from center) on an upper net. Genuinely directional data
(``vector``) behaves differently: since it already draws a filled marker and an open one at
its antipode, the two markers' positions stay the same between the nets and only which one is
filled vs open swaps::

    >>> s = StereoNet(kind="equal-angle", hemisphere="upper")
    >>> s.point(fols)
    >>> s.show()

N/E/S/W compass tick labels are shown around the rim by default; pass ``azimuth_ticks=False``
to remove them, or ``azimuth_ticks_kws`` to customize their angles/labels/position. Legend
placement and styling for any of ``StereoNet``, ``RosePlot`` or the fabric plots can be
overridden via ``legend_kws``, without disturbing apsg's own defaults for the options you
don't pass::

    >>> s = StereoNet(azimuth_ticks=False, legend_kws={"loc": "lower left"})
    >>> s.point(fols, label="Poles")
    >>> s.show()

``rotation`` rotates the whole net (grid, ticks and, when ``rotate_data`` is True, the
default, plotted data too, so a fabric's appearance relative to the grid is unchanged).
Pass ``rotate_data=False`` to rotate only the grid, leaving already-plotted data anchored to
the true, unrotated frame::

    >>> from apsg import lin, rotation_from_axis_angle
    >>> R = rotation_from_axis_angle(lin(90, 0), 30)  # 30 degrees about a horizontal E-W axis
    >>> s = StereoNet(rotation=R, rotate_data=False)
    >>> s.point(fols)
    >>> s.show()

Arcs::

    >>> from apsg import arc, arcset, lin
    >>> a = arc(lin(0, 0), lin(90, 0), curvature=0.4)
    >>> path = arcset.from_vectors(lin(0, 0), lin(45, 20), lin(90, 0))
    >>> s = StereoNet()
    >>> s.arc(a, path)
    >>> s.show()

Quick plot one-liner -- dispatches each argument to whichever plotting method matches its
type (lines/poles, planes, pairs, faults, cones, arcs, tensors, stress tensors, or a
``StereoGrid`` contour)::

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
