# -*- coding: utf-8 -*-
"""
API to read data from PySDB database

"""

import sqlite3
from .core import Fol, Lin, Group

class Datasource(object):
    """PySDB database access class"""
    TESTSEL = "SELECT sites.id, sites.name, sites.x_coord, sites.y_coord, \
    sites.description, structdata.id, structdata.id_sites, \
    structdata.id_structype, structdata.azimuth, structdata.inclination, \
    structype.id, structype.structure, structype.description, \
    structype.structcode, structype.groupcode  \
    FROM sites \
    INNER JOIN structdata ON sites.id = structdata.id_sites \
    INNER JOIN structype ON structype.id = structdata.id_structype \
    LIMIT 1"
    STRUCTSEL = "SELECT structype.structure  \
    FROM sites  \
    INNER JOIN structdata ON sites.id = structdata.id_sites  \
    INNER JOIN structype ON structype.id = structdata.id_structype  \
    INNER JOIN units ON units.id = sites.id_units  \
    GROUP BY structype.structure  \
    ORDER BY structype.structure ASC"
    SELECT = "SELECT structdata.azimuth, structdata.inclination   \
    FROM sites   \
    INNER JOIN structdata ON sites.id = structdata.id_sites   \
    INNER JOIN structype ON structype.id = structdata.id_structype   \
    INNER JOIN units ON units.id = sites.id_units"

    def __new__(cls, db=None):
        try:
            cls.con = sqlite3.connect(db)
            cls.con.execute("pragma encoding='UTF-8'")
            cls.con.execute(Datasource.TESTSEL)
            print("Connected. PySDB version: %s" % cls.con.execute("SELECT value FROM meta WHERE name='version'").fetchall()[0][0])
            return super(Datasource, cls).__new__(cls)
        except sqlite3.Error as e:
            print("Error %s:" % e.args[0])
            raise sqlite3.Error

    def execsql(self, sql):
        return self.con.execute(sql).fetchall()

    @property
    def structures(self):
        return [el[0] for el in self.execsql(Datasource.STRUCTSEL)]

    def select(self, struct=None):
        tpsel = "SELECT planar FROM structype WHERE structure='%s'" % struct
        dtsel = Datasource.SELECT + " WHERE structype.structure='%s'" % struct
        tp = self.execsql(tpsel)[0][0]
        if tp:
            res = Group([Fol(el[0], el[1]) for el in self.execsql(dtsel)], name=struct)
        else:
            res = Group([Lin(el[0], el[1]) for el in self.execsql(dtsel)], name=struct)
        return res

