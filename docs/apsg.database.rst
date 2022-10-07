===============
database module
===============

SqlAlchemy interface to PySDB database

Usage
-----

    >>> # Create database
    >>> from apsg.database import SDBSession
    >>> db = SDBSession('database.sdb', create=True)
    >>> # Create unit
    >>> unit = db.unit(name='DMU', description='Deamonic Magmatic Unit')
    >>> # Create site
    >>> site = db.site(unit=unit, name='LX001', x_coord=25934.36, y_coord=564122.5, description='diorite dyke')
    >>> # Create structural types
    >>> S2 = db.structype(structure='S2', description='Solid-state foliation', planar=1)
    >>> L2 = db.structype(structure='L2', description='Solid-state lineation', planar=0)
    >>> # Add measurement
    >>> fol = db.add_structdata(site=site, structype=S2, azimuth=150, inclination=36)
    >>> # Close database
    >>> db.close()

Tags example
------------

    >>> db = SDBSession('database.sdb')
    >>> unit = db.unit(name='DMU')
    >>> site = db.site(name='LX001')
    >>> struct = db.structype(structure='S2')
    >>> tag_plot = db.tag(name='plot')
    >>> tag_ap = db.tag(name='AP')
    >>> fol = db.add_structdata(site=site, structype=struct, azimuth=324, inclination=78, tags=[tag_plot, tag_ap])
    >>> db.close()

Attach example
--------------

    >>> db = SDBSession('database.sdb')
    >>> unit = db.unit(name='DMU')
    >>> site = db.site(name='LX001')
    >>> S = db.structype(structure='S')
    >>> L = db.structype(structure='L')
    >>> fol = db.add_structdata(site=site, structype=S, azimuth=220, inclination=28)
    >>> lin = db.add_structdata(site=site, structype=L, azimuth=212, inclination=26)
    >>> pair = db.attach(fol, lin)
    >>> db.close()

Using apsg interface
--------------------

    >>> db = SDBSession('database.sdb')
    >>> unit = db.unit(name='DMU')
    >>> site = db.site(name='LX003')
    >>> S2 = db.structype(structure='S2')
    >>> L2 = db.structype(structure='L2')

You can directly insert ``Foliation``, ``Lineation`` or ``Pair``

    >>> f = fol(196, 39)
    >>> l = lin(210, 37)
    >>> db.add_fol(f, site=site, structype=S2)
    >>> db.add_lin(l, site=site, structype=L2)
    >>> p = Pair(258, 42, 220, 30)
    >>> db.add_pair(p, S2, L2, site=site)
    >>> db.close()

To retrieve data as ``FeatureSet`` you can use ``getset`` method:

    >>> db = SDBSession('database.sdb')
    >>> S2 = db.structype(structure='S2')
    >>> g = db.getset(structype=S2)

or directly

    >>> g = db.getset('S2')

.. automodule:: apsg.database
    :members:
    :undoc-members:
    :show-inheritance:

