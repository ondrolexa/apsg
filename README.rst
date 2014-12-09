==============================================
APSG - python module for structural geologists
==============================================

.. image:: https://badge.fury.io/py/apsg.svg
    :target: http://badge.fury.io/py/apsg

.. image:: https://badge.fury.io/gh/ondrolexa%2Fapsg.svg
    :target: http://badge.fury.io/gh/ondrolexa%2Fapsg

APSG defines several new python classes to easily manage, analyze and visualize orientational structural geology data.

* Free software: BSD license
* Documentation: https://apsg.readthedocs.org.

**APSG** defines several new python classes to easily manage, analyze
and visualize orientation structural geology data. Base class ``Vec3``
is derived from ``numpy.array`` class and offers several new method
which will be explained on following examples.

Basic usage
-----------

APSG module could be imported either into own name space or into
active one for easier interactive work::

    from apsg import *

Basic operations with vectors
-----------------------------

Instance of vector object ``Vec3`` could be created from any iterable
object as list, tuple or array::

    >>> u = Vec3([1, -2, 3])
    >>> v = Vec3([-2, 1, 1])

For common vector operation we can use standard mathematical operators
or special methods using dot notation::

    >>> u + v
    V(-1.000, -1.000, 4.000)
    >>> u - v
    V(3.000, -3.000, 2.000)
    >>> 3*u - 2*v
    V(7.000, -8.000, 7.000)

Its magnitude or length is most commonly defined as its Euclidean norm
and could be calculated using ``abs``::

    >>> abs(v)
    2.4494897427831779
    >>> abs(u+v)
    4.2426406871192848

For *dot product* we can use multiplication operator ``*``
or ``dot`` method::

    >>> u*v
    -1
    >>> u.dot(v)
    -1

For *cross product* we can use operator ``**`` or method ``cross``::

    >>> u**v
    V(-5.000, -7.000, -3.000)
    >>> u.cross(v)
    V(-5.000, -7.000, -3.000)

To project vector ``u`` onto vector ``v`` we can use
method ``proj``::

    >>> u.proj(v)
    V(0.333, -0.167, -0.167)

To find angle (in degrees) between to vectors we use method ``angle``::

    >>> u.angle(v)
    96.263952719927218

Method ``rotate`` provide possibility to rotate vector around
another vector. For example, to rotate vector ``u`` around
vector ``v`` for 45°::

    >>> u.rotate(v, 45)
    V(2.248, 0.558, 2.939)

Classes Lin and Fol
-------------------

To work with orientation data in structural geology, APSG
provide two classes derived from ``Vec3`` class. There is ``Fol``
class to represent planar features by planes and ``Lin`` class
to represent linear feature by lines. Both classes provide all
``Vec3`` methods, but they differ in way how instance is created
and how some operations are calculated, as structural geology
data are commonly axial in nature.

To create instance of ``Lin`` or ``Fol`` class, we have to provide
dip direction and dip, both in degrees::

    >>> Lin(120, 60)
    L:120/60
    >>> Fol(216, 62)
    S:216/62

or we can create instance from ``Vec3`` object using ``aslin``
and ``asfol`` properties::

    >>> u.aslin
    L:297/53
    >>> u.asfol
    S:117/37

Vec3 methods for Lin and Fol
----------------------------

To find angle between two linear or planar features::

    >>> l1 = Lin(110, 40)
    >>> l2 = Lin(160, 30)
    >>> l1.angle(l2)
    41.597412680035468
    >>> p1 = Fol(330, 50)
    >>> p2 = Fol(250, 40)
    >>> p1.angle(p2)
    54.696399321975328

To construct planar feature defined by two linear features::

    >>> l1**l2
    S:113/40

To construct linear feature defined as intersection of two planar features::

    >>> p1**p2
    L:278/36

**Cross product** of planar and linear features could be used to construct
plane defined by linear feature and normal of planar feature::

    >>> l2**p2
    S:96/53

or to find perpendicular linear feature on given plane::

    >>> p2**l2
    L:276/37

To rotate structural features we can use method ``rotate``::

    >>> p2.rotate(l2, 45)
    S:269/78

Classes Pair and Fault
----------------------

To work with paired orientation data like foliations and lineations
or fault data in structural geology, APSG provide two base ``Pair``
class and derived ``Fault`` class. Both classes are instantiated
providing dip direction and dip of planar and linear measurements,
which are automatically orthogonalized. If misfit is too high,
warning is raised. The ``Fault`` class expects one more argument
providing sense of movement information, either 1 or -1. 

To create instance of ``Pair`` class, we have to provide
dip direction and dip of planar and linear feature, both in degrees::

    >>> p = Pair(120, 40, 162, 28)
    >>> p
    P:118/39-163/30
    >>> p.misfit
    3.5623168411508175
    >>> type(p)
    <class 'apsg.core.Pair'>

Planar and linear features are accessible using ``fol`` and ``lin``
properties::

    >>> p.fol
    S:118/39
    >>> p.lin
    L:163/30
    >>> type(p.fol)
    <class 'apsg.core.Fol'>
    >>> type(p.lin)
    <class 'apsg.core.Lin'>

To rotate ``Pair`` instance we can use ``rotate`` method::

    >>> p.rotate(Lin(45, 10), 60)
    P:314/83-237/61

Instantiation of ``Fault`` class is similar, we just have to provide argument
to define sense of movement::

    >>> f = Fault(120, 60, 110, 58, -1)  # -1 for normal fault
    >>> f
    F:120/59-110/59 -

Note the change in sense of movement after ``Fault`` rotation::

    >>> f.rotate(Lin(45, 10), 60)
    F:312/62-340/59 +

``Fault`` class also provide ``p``, ``t`` and ``m`` properties to get PT-axes
and kinematic plane::

    >>> f.p
    L:315/75
    >>> f.t
    L:116/14
    >>> f.m
    S:27/85

Group class
-----------

``Group`` class serve as a homogeneous container for ``Lin`` or ``Fol`` objects.
It allows grouping of features either for visualization or batch analysis::

    >>> g = Group([Lin(120,60), Lin(116,50), Lin(132,45), Lin(90,60), Lin(84,52)],
                  name='L1')
    >>> g
    L1: 5 Lin

Method ``len`` returns number of features in group::

    >>> len(g)
    5

To measure angles between all features in group and another feature,
we can use method ``angle``::

    >>> g.angle(Lin(110,50))
    array([ 11.49989817,   3.85569115,  15.61367789,  15.11039885,  16.3947936 ])

To rotate all features in group around another feature,
we can use method ``rotate``::

    >>> gr = g.rotate(Lin(150, 30), 45)

To show data in list you can use ``data`` method::

    >>> gr.data
    [L:107/35, L:113/26, L:126/30, L:93/26, L:94/18]

Property ``R`` gives mean or resultant of all features in group::

    >>> g = Group.randn_lin(mean=Lin(40, 20))
    >>> g.R
    L:39/21

``Group`` class offers several methods to infer spherical statistics as
spherical variance, Fisher's statistics, confidence cones on
data etc.::

    >>> g.var
    0.063710393842001833
    >>> g.fisher_stats
    {'csd': 20.548142386914282, 'a95': 3.7054501701829596, 'k': 15.539065767748088}
    >>> g.delta
    20.562501451172906

To calculate orientation tensor of all features in group,
we can use method ``ortensor``::

    >>> g.ortensor
    Ortensor:
    (E1:88.4,E2:7.338,E3:4.266)
    [[ 49.53851     35.19161279  22.15886785]
     [ 35.19161279  34.90101673  16.01083238]
     [ 22.15886785  16.01083238  15.56047326]]

Ortensor class
--------------

``Ortensor`` class represents orientation tensor of set of planar
or linear features. Eigenvalues and eigenvectors could be obtained
by methods ``eigenvals`` and ``eigenvects``. Eigenvectors could be also
represented by linear or planar features using properties ``eigenlins``
and ``eigenfols``::

    >>> ot = Ortensor(g)
    >>> ot.eigenvals
    (0.88395980871958957, 0.073383662044666884, 0.042656529235744325)
    >>> ot.eigenvects.data
    [V(-0.731, -0.586, -0.351), V(0.345, -0.760, 0.550), V(-0.589, 0.280, 0.758)]
    >>> ot.eigenlins.data
    [L:39/21, L:294/33, L:155/49]
    >>> ot.eigenfols.data
    [S:219/69, S:114/57, S:335/41]

StereoNet class
---------------

Any ``Fol``, ``Lin`` or ``Group`` object could be visualized as plane,
line or pole in stereographic projection using StereoNet class::

    >>> s = StereoNet()
    >>> s.plane(Fol(150, 40))
    >>> s.pole(Fol(150, 40))
    >>> s.line(Lin(112, 30))
    >>> s.show()

.. image:: http://ondrolexa.github.io/apsg/images/plane-line-pole_020.png
    :alt: A basic stereonet with a plane, line and pole
    :align: center

A cones (or small circles) could be plotted as well::

    >>> s = StereoNet()
    >>> g = Group.randn_lin(mean=Lin(40, 15))
    >>> s.line(g, 'k.')
    >>> s.cone(g.R, g.fisher_stats['a95'], 'r')  # confidence cone on resultant
    >>> s.cone(g.R, g.fisher_stats['csd'], 'g')  # confidence cone on 63% of data
    >>> s.show()

.. image:: http://ondrolexa.github.io/apsg/images/group_020.png
    :alt: A basic stereonet group of linear features
    :align: center

To make density contours plots, a ``contour`` and ``contourf``
methods are available::

    >>> s = StereoNet()
    >>> g = Group.randn_lin(mean=Lin(40, 20))
    >>> s.contourf(g, 8, legend=True)
    >>> s.contour(g, 8, colors='k')
    >>> s.line(g, 'wo')
    >>> s.show()

.. image:: http://ondrolexa.github.io/apsg/images/density_020.png
    :alt: A density contour plot
    :align: center

Some tricks
-----------

Double cross product is allowed (note quick plot feature)::

    >>> p = Fol(250,40)
    >>> l = Lin(160,25)
    >>> StereoNet(p, l, l**p, p**l, l**p**l, p**l**p)

.. image:: http://ondrolexa.github.io/apsg/images/cross_020.png
    :alt: A cross product tricks
    :align: center

Correct measurements of planar linear pairs by instantiation
of Pair class::

    >>> pl = Pair(250, 40, 160, 25)
    >>> pl.misfit
    18.889520432245405
    >>> s = StereoNet()
    >>> s.plane(Fol(250, 40), 'b')
    >>> s.line(Lin(160, 25), 'bo')
    >>> s.plane(pl.fol, 'g')
    >>> s.line(pl.lin, 'go')
    >>> s.show()

.. image:: http://ondrolexa.github.io/apsg/images/fixpair_020.png
    :alt: Fix pair of plane and line
    :align: center
