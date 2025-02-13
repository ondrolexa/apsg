===============
database module
===============

sqlalchemy interface to PySDB database

PySDB database is a simple sqlite3-based relational database to store structural data from the field. You can use
**apsg** to manipulate the data or you can use the GUI application `pysdb <https://github.com/ondrolexa/pysdb>`_. There
is also the QGIS plugin `readsdb <https://github.com/ondrolexa/readsdb>`_ to plot data on a map or use map-based
select to plot stereonets.

The following snippet demonstrate how to create database programmatically

    >>> # Create database
    >>> from apsg.database import SDBSession
    >>> db = SDBSession('database.sdb', create=True)
    >>> # Create unit
    >>> unit = db.unit('DMU', description='Deamonic Magmatic Unit')
    >>> # Create site
    >>> site = db.site('LX001', unit=unit, x_coord=25934.36, y_coord=564122.5, description='diorite dyke')
    >>> # Create structural types
    >>> S2 = db.structype('S2', description='Solid-state foliation', planar=1)
    >>> L2 = db.structype('L2', description='Solid-state lineation', planar=0)
    >>> # Add measurement
    >>> fol = db.add_structdata(site, S2, 150, 36)
    >>> lin = db.add_structdata(site, L2, 83, 16)
    >>> # Commit changes
    >>> db.commit()
    >>> # Close database
    >>> db.close()

You can tag individual data

    >>> db = SDBSession('database.sdb')
    >>> site = db.site('LX001')
    >>> struct = db.structype('S2')
    >>> tag_plot = db.tag('plot', description='to be plotted')
    >>> tag_ap = db.tag('AP', description='axial plane')
    >>> fol = db.add_structdata(site, struct, 324, 78, tags=[tag_plot, tag_ap])
    >>> db.commit()
    >>> db.close()

or you can attach linear and planar features (e.g. fault data)

    >>> db = SDBSession('database.sdb')
    >>> site = db.site(name='LX001')
    >>> S = db.structype(structure='S')
    >>> L = db.structype(structure='L')
    >>> fol = db.add_structdata(site, S, 220, 28)
    >>> lin = db.add_structdata(site, L, 212, 26)
    >>> pair = db.attach(fol, lin)
    >>> db.commit()
    >>> db.close()

You can open existing database and select existing site and type of structure

    >>> db = SDBSession('database.sdb')
    >>> site = db.site('LX001')
    >>> S2 = db.structype('S2')
    >>> L2 = db.structype('L2')

and insert ``Foliation``, ``Lineation`` or ``Pair`` directly

    >>> f = fol(196, 39)
    >>> l = lin(210, 37)
    >>> db.add_fol(site, S2, f)
    >>> db.add_lin(site, L2, l)
    >>> p = pair(258, 42, 220, 30)
    >>> db.add_pair(site, S2, L2, pair)
    >>> db.commit()
    >>> db.close()

To retrieve data as ``FeatureSet`` you can use ``getset`` method:

    >>> db = SDBSession('database.sdb')
    >>> S2 = db.structype('S2')
    >>> g = db.getset(S2)

or directly

    >>> g = db.getset('S2')

.. automodule:: apsg.database
    :autosummary:
    :members:
    :show-inheritance:
    :autosummary-no-nesting:
