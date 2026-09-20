==============
feature module
==============

The :mod:`apsg.feature` module provides classes to store and analyze structural geology data. It includes
representations of planar and linear features (``Foliation``, ``Lineation``, ``Fault``, ``Cone``, ``Arc``, ``Pair``),
their corresponding sets (``FoliationSet``, ``LineationSet``, etc.), tensor classes (``Stress3``,
``Ellipsoid``, ``OrientationTensor3``), and higher-level analysis tools like ``ClusterSet`` for
clustering of orientation data.

The main APSG namespace provides lowercase aliases for commonly used classes (e.g. ``fol`` for
``Foliation``, ``lin`` for ``Lineation``, ``vec`` for ``Vector3``).
See :doc:`index` for the full list.

Usage
-----

Create individual features::

    >>> from apsg import fol, lin, fault, pair, cone, arc, vec
    >>> f = fol(120, 30)                 # dip direction=120, dip=30
    >>> l = lin(210, 45)                 # trend=210, plunge=45
    >>> f2 = fault(120, 30, 80, 20, 1)   # dip direction, dip, trend, plunge, sense
    >>> p = pair(300, 20, 200, 60)       # dip direction, dip, trend, plunge
    >>> c = cone(vec(45, 30), 20)        # axis + half-angle
    >>> a = arc(lin(0, 0), lin(90, 0))   # curved path between two vectors

Create sets of features for batch analysis::

    >>> from apsg import folset, linset, vecset
    >>> fols = folset.random_fisher(position=f, kappa=50, n=30)
    >>> lins = linset.random_fisher(position=l, kappa=100, n=30)
    >>> vecs = vecset.uniform_gss(n=50)

Analyze orientation data::

    >>> ot = fols.ortensor()
    >>> ot.eigenvalues()
    >>> ot.eigenvectors()
    >>> fols.fisher_statistics()

Cluster analysis::

    >>> from apsg import cluster
    >>> cs = cluster(fols)
    >>> cs.cluster(k=3)

Tensor analysis::

    >>> from apsg import ellipsoid, stress, ortensor
    >>> e = ellipsoid.from_stretch(2, 1, 0.5)
    >>> s = stress.from_ratio(r=0.5, mag=1)
    >>> ot = ortensor.from_features(fols)

Mean tensor of a group of tensors (``EllipsoidSet`` or ``Stress3Set``) with confidence ellipses
of its principal axes, following the linear perturbation method of Jelínek (1978)::

    >>> from apsg import ellipsoidset
    >>> es = ellipsoidset([ellipsoid.from_stretch(2 + 0.1 * i, 1, 0.5) for i in range(10)])
    >>> res = es.mean_tensor()                     # normalize=True divides tensors by trace/3
    >>> res["mean"], res["eigenvalues"]
    >>> res["ellipses"][0]["mu"]                   # principal axis of the mean tensor
    >>> res["ellipses"][0]["gamma"]                # semi-angles of its confidence ellipse (deg)
    >>> es.mean_tensor(anisoft=True)               # extra (n-1)/n factor as in jelinekstat/TomoFab

Serialization::

    >>> from apsg.feature import feature_from_json
    >>> data = f.to_json()
    >>> f2 = feature_from_json(data)
    >>> fols.to_csv('foliations.csv')

.. automodule:: apsg.feature
    :autosummary:
    :members:
    :show-inheritance:
    :autosummary-no-nesting:
    :exclude-members: Vector2, Vector3, Axial2, Axial3, Matrix2, Matrix3
