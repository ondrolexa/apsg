===============
database module
===============

The :mod:`apsg.database` module provides an SQLAlchemy interface to the PySDB database format.
PySDB is a simple sqlite3-based relational database for storing structural geology field data.
The module supports creating and querying databases with units, sites, structural types, and
measurements. You can also use the standalone GUI application `pysdb <https://github.com/ondrolexa/pysdb>`_
or the QGIS plugin `readsdb <https://github.com/ondrolexa/readsdb>`_ for map-based visualization.

Usage
-----

Create a new database and add data::

    >>> from apsg.database import SDBSession
    >>> db = SDBSession('database.sdb', create=True)
    >>> unit = db.unit('DMU', description='Deamonic Magmatic Unit')
    >>> site = db.site('LX001', unit=unit, x_coord=25934.36, y_coord=564122.5, description='diorite dyke')
    >>> S2 = db.structype('S2', description='Solid-state foliation', planar=1)
    >>> L2 = db.structype('L2', description='Solid-state lineation', planar=0)
    >>> fol = db.add_structdata(site, S2, 150, 36)
    >>> lin = db.add_structdata(site, L2, 83, 16)
    >>> db.commit()
    >>> db.close()

Add tags and attach linear to planar data::

    >>> db = SDBSession('database.sdb')
    >>> site = db.site('LX001')
    >>> tag = db.tag('plot', description='to be plotted')
    >>> fol = db.add_structdata(site, S2, 324, 78, tags=[tag])
    >>> lin = db.add_structdata(site, L2, 212, 26)
    >>> pair = db.attach(fol, lin)
    >>> db.commit()
    >>> db.close()

Insert ``Foliation``, ``Lineation`` or ``Pair`` objects directly::

    >>> from apsg.feature import Foliation, Lineation, Pair
    >>> db = SDBSession('database.sdb')
    >>> site = db.site('LX001')
    >>> S2 = db.structype('S2')
    >>> L2 = db.structype('L2')
    >>> f = Foliation(196, 39)
    >>> l = Lineation(210, 37)
    >>> db.add_fol(site, S2, f)
    >>> db.add_lin(site, L2, l)
    >>> p = Pair(258, 42, 220, 30)           # dip direction, dip, trend, plunge
    >>> db.add_pair(site, S2, L2, p)
    >>> db.commit()
    >>> db.close()

Retrieve data as APSG feature sets::

    >>> db = SDBSession('database.sdb')
    >>> g = db.getset('S2')
    >>> type(g).__name__
    'FoliationSet'

.. automodule:: apsg.database
    :autosummary:
    :members:
    :show-inheritance:
    :autosummary-no-nesting:
