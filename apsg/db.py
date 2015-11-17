# -*- coding: utf-8 -*-
"""
API to read data from PySDB database

"""

import sqlite3
from .core import Fol, Lin, Group


class SDB(object):
    """PySDB database access class"""
    _SELECT = "SELECT sites.name as name, sites.x_coord as x, \
    sites.y_coord as y, units.name as unit, structdata.azimuth as azimuth, \
    structdata.inclination as inclination, structype.structure as structure, \
    structype.planar as planar, structdata.description as description, \
    GROUP_CONCAT(tags.name) AS tags \
    FROM structdata \
    INNER JOIN sites ON structdata.id_sites=sites.id \
    INNER JOIN structype ON structype.id = structdata.id_structype \
    INNER JOIN units ON units.id = sites.id_units \
    LEFT OUTER JOIN tagged ON structdata.id = tagged.id_structdata \
    LEFT OUTER JOIN tags ON tags.id = tagged.id_tags"
    _TYPSEL = "SELECT planar FROM structype"
    _VERSEL = "SELECT value FROM meta WHERE name='version'"

    def __new__(cls, db=None):
        try:
            cls.conn = sqlite3.connect(db)
            cls.conn.row_factory = sqlite3.Row
            cls.conn.execute("pragma encoding='UTF-8'")
            cls.conn.execute(SDB._SELECT + " LIMIT 1")
            ver = cls.conn.execute(SDB._VERSEL).fetchall()[0][0]
            print("Connected. PySDB version: %s" % ver)
            return super(SDB, cls).__new__(cls)
        except sqlite3.Error as e:
            print("Error %s:" % e.args[0])
            raise sqlite3.Error

    def execsql(self, sql):
        return self.conn.execute(sql).fetchall()

    @property
    def structures(self):
        dtsel = SDB._SELECT + " GROUP BY structdata.id"
        return list(set([el['structure'] for el in self.execsql(dtsel)]))

    @property
    def sites(self):
        dtsel = SDB._SELECT + " GROUP BY structdata.id"
        return list(set([el['name'] for el in self.execsql(dtsel)]))

    def select(self, struct, sites=None, units=None):
        where = ["structype.structure='%s'" % struct]
        if sites:
            if isinstance(sites, str):
                where.append("sites.name='%s'" % sites)
            elif isinstance(sites, (list, tuple)):
                unf = ' OR '.join(["sites.name='%s'" % site for site in sites])
                where.append("(" + unf + ")")
            else:
                raise ValueError('Keyword sites must be list or string.')
        if units:
            if isinstance(units, str):
                where.append("unit='%s'" % units)
            elif isinstance(units, (list, tuple)):
                unf = ' OR '.join(["unit='%s'" % unit for unit in units])
                where.append("(" + unf + ")")
            else:
                raise ValueError('Keyword units must be list or string.')

        tpsel = SDB._TYPSEL + " WHERE structure='%s'" % struct
        dtsel = SDB._SELECT + " WHERE " + ' AND '.join(where) + " GROUP BY structdata.id"
        tp = self.execsql(tpsel)[0][0]
        if tp:
            res = Group([Fol(el['azimuth'], el['inclination'])
                         for el in self.execsql(dtsel)], name=struct)
        else:
            res = Group([Lin(el['azimuth'], el['inclination'])
                         for el in self.execsql(dtsel)], name=struct)
        return res
