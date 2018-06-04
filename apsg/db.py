# -*- coding: utf-8 -*-

"""
API to read data from PySDB database

"""

import sqlite3

from .core import Fol, Lin, Group


__all__ = ("SDB",)


class SDB(object):
    """PySDB database access class"""
    _SELECT = """SELECT sites.name as name, sites.x_coord as x,
    sites.y_coord as y, units.name as unit, structdata.azimuth as azimuth,
    structdata.inclination as inclination, structype.structure as structure,
    structype.planar as planar, structdata.description as description,
    GROUP_CONCAT(tags.name) AS tags
    FROM structdata
    INNER JOIN sites ON structdata.id_sites=sites.id
    INNER JOIN structype ON structype.id = structdata.id_structype
    INNER JOIN units ON units.id = sites.id_units
    LEFT OUTER JOIN tagged ON structdata.id = tagged.id_structdata
    LEFT OUTER JOIN tags ON tags.id = tagged.id_tags"""

    def __new__(cls, db=None):
        cls.conn = sqlite3.connect(db)
        cls.conn.row_factory = sqlite3.Row
        cls.conn.execute("pragma encoding='UTF-8'")
        cls.conn.execute(SDB._SELECT + " LIMIT 1")
        return super(SDB, cls).__new__(cls)

    def __repr__(self):
        return "PySDB database version: {}".format(self.meta("version"))

    def meta(self, name):
        if name == "crs":
            val = self.conn.execute(
                "SELECT value FROM meta WHERE name='crs'"
            ).fetchall()
            if not val:
                name = "proj4"
        return self.conn.execute(
            "SELECT value FROM meta WHERE name='{}'".format(name)
        ).fetchall()[0][0]

    def info(self, verbose=False):
        print("PySDB database version: {}".format(self.meta("version")))
        print("PySDB database crs: {}".format(self.meta("crs")))
        print("PySDB database version: {}".format(self.meta("created")))
        print("PySDB database version: {}".format(self.meta("updated")))
        print("Number of sites: {}".format(len(self.sites())))
        print("Number of units: {}".format(len(self.units())))
        print("Number of structures: {}".format(len(self.structures())))
        r = self.execsql(self._make_select())
        print("Number of measurements: {}".format(len(r)))
        if verbose:
            for s in self.structures():
                r = self.execsql(self._make_select(structs=s))
                print("   Number of {} measurements: {}".format(s, len(r)))

    def _make_select(self, structs=None, sites=None, units=None, tags=None):
        w = []
        if structs:
            if isinstance(structs, str):
                w.append("structype.structure='%s'" % structs)
            elif isinstance(structs, (list, tuple)):
                u = " OR ".join(
                    ["structype.structure='%s'" % struct for struct in structs]
                )
                w.append("(" + u + ")")
            else:
                raise ValueError("Keyword structs must be list or string.")
        if sites:
            if isinstance(sites, str):
                w.append("sites.name='%s'" % sites)
            elif isinstance(sites, (list, tuple)):
                u = " OR ".join(["sites.name='%s'" % site for site in sites])
                w.append("(" + u + ")")
            else:
                raise ValueError("Keyword sites must be list or string.")
        if units:
            if isinstance(units, str):
                w.append("unit='%s'" % units)
            elif isinstance(units, (list, tuple)):
                u = " OR ".join(["unit='%s'" % unit for unit in units])
                w.append("(" + u + ")")
            else:
                raise ValueError("Keyword units must be list or string.")
        if tags:
            if isinstance(tags, str):
                w.append("tags.name like '%%%s%%'" % tags)
            elif isinstance(tags, (list, tuple)):
                u = " AND ".join(["tags.name like '%%%s%%'" % tag for tag in tags])
                w.append("(" + u + ")")
            else:
                raise ValueError("Keyword tags must be list or string.")
        sel = SDB._SELECT
        if w:
            sel += " WHERE " + " AND ".join(w) + " GROUP BY structdata.id"
        else:
            sel += " GROUP BY structdata.id"
        return sel

    def execsql(self, sql):
        return self.conn.execute(sql).fetchall()

    def structures(self, **kwargs):
        """Return list of structures in data. For kwargs see group method."""
        dtsel = self._make_select(**kwargs)
        return list(set([el["structure"] for el in self.execsql(dtsel)]))

    def sites(self, **kwargs):
        """Return list of sites in data. For kwargs see group method."""
        dtsel = self._make_select(**kwargs)
        return list(set([el["name"] for el in self.execsql(dtsel)]))

    def units(self, **kwargs):
        """Return list of units in data. For kwargs see group method."""
        dtsel = self._make_select(**kwargs)
        return list(set([el["unit"] for el in self.execsql(dtsel)]))

    def tags(self, **kwargs):
        """Return list of tags in data. For kwargs see group method."""
        dtsel = self._make_select(**kwargs)
        tags = [el["tags"] for el in self.execsql(dtsel) if el["tags"] is not None]
        return list(set(",".join(tags).split(",")))

    def group(self, struct, **kwargs):
        """Method to retrieve data from SDB database to apsg.Group

        Args:
          struct:  name of structure to retrieve
        Kwargs:
          sites: name or list of names of sites to retrieve from
          units: name or list of names of units to retrieve from
          tags:  tag or list of tags to retrieve

        Example:
          >>> g = db.group('L2', units=['HG', 'MG'], tags='bos')

        """
        dtsel = self._make_select(structs=struct, **kwargs)
        tpsel = "SELECT planar FROM structype WHERE structure='{}'".format(struct)
        tp = self.execsql(tpsel)[0][0]
        if tp:
            res = Group(
                [Fol(el["azimuth"], el["inclination"]) for el in self.execsql(dtsel)],
                name=struct,
            )
        else:
            res = Group(
                [Lin(el["azimuth"], el["inclination"]) for el in self.execsql(dtsel)],
                name=struct,
            )
        return res
