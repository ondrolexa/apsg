===============
database module
===============

The :mod:`apsg.database` module provides two independent session classes for structural
geology field data, both of which convert to/from APSG ``Foliation``/``Lineation``/``Pair``/
``Fault`` (and their ``*Set`` counterparts):

* :class:`~apsg.database.SDBSession` - a SQLAlchemy interface to a local PySDB ``.sdb``
  sqlite3 file. PySDB is a simple sqlite3-based relational database for storing structural
  geology field data. You can also use the standalone GUI application
  `pysdb <https://github.com/ondrolexa/pysdb>`_ or the QGIS plugin
  `readsdb <https://github.com/ondrolexa/readsdb>`_ for map-based visualization.
* :class:`~apsg.database.WebSDBSession` - a dependency-free client (standard library
  ``urllib`` only) for a remote `websdb <https://github.com/ondrolexa/websdb>`_ REST API
  project, the actively-developed hosted successor to PySDB.

SDBSession usage
----------------

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

WebSDBSession usage
-------------------

A :class:`~apsg.database.WebSDBSession` binds to a single websdb project for its whole
lifetime (given by id or by name) and opens read-only (``mode='r'``, the default) unless
``mode='rw'`` is requested explicitly - the server independently enforces the caller's
actual project role regardless of ``mode``. Connection details fall back to the
``WEBSDB_URL``/``WEBSDB_USERNAME``/``WEBSDB_PASSWORD``/``WEBSDB_TOKEN`` environment
variables when not passed explicitly::

    >>> from apsg.database import WebSDBSession
    >>> db = WebSDBSession('Erzgebirge', mode='rw')
    >>> print(db.info())

Filtered reads return regular APSG feature sets, with ``site``/``rock``/``unit``/``tags``
(plus the websdb row ids) stashed in each feature's ``_attrs`` so it can be handed straight
back to the matching write method::

    >>> fols = db.folset(unit='Gneiss', tags=['banding'])
    >>> lins = db.linset(site_contains='erz', structype='S2')
    >>> faults = db.faultset(rock_contains='dyke')
    >>> pairs = db.pairset(ptags=['fol-tag'], ltags=['lin-tag'])

Writing (``mode='rw'`` only) uses the same get-or-create convention as ``SDBSession``::

    >>> from apsg.feature import Fault, Pair
    >>> site = db.site('LX001', lon=13.1, lat=50.6, description='diorite dyke')
    >>> rock = db.rock('Default', site, unit='Gneiss')
    >>> fol = db.add_fol(rock, 200, 30, structype='S2', tags=['banding'])
    >>> pair = db.add_pair(rock, Pair(258, 42, 220, 30))
    >>> fault = db.add_fault(rock, Fault(280, 60, 210, 35, 'n'))
    >>> db.update_fol(fol, description='resampled')
    >>> db.delete_fol(fol)

.. automodule:: apsg.database
    :autosummary:
    :members:
    :show-inheritance:
    :autosummary-no-nesting:
