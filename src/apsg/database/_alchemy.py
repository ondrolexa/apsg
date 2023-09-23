# -*- coding: utf-8 -*-

"""
SQLAlchemy API to access PySDB database
"""

import os
from datetime import datetime
import contextlib

from apsg.feature._geodata import Lineation, Foliation, Pair
from apsg.feature._container import LineationSet, FoliationSet

from sqlalchemy import create_engine, event
from sqlalchemy.orm import sessionmaker
from sqlalchemy import Column, Float, ForeignKey, Integer, String, Table, Text, text
from sqlalchemy.orm import relationship
from sqlalchemy.ext.declarative import declarative_base

Base = declarative_base()
metadata = Base.metadata


class Meta(Base):
    __tablename__ = "meta"

    id = Column(Integer, primary_key=True)
    name = Column(String(16), nullable=False)
    value = Column(Text)


class Site(Base):
    __tablename__ = "sites"

    id = Column(Integer, primary_key=True)
    id_units = Column(ForeignKey("units.id"), nullable=False, index=True)
    name = Column(String(16), nullable=False, server_default=text("''"))
    x_coord = Column(Float, server_default=text("NULL"))
    y_coord = Column(Float, server_default=text("NULL"))
    description = Column(Text)

    unit = relationship("Unit", back_populates="sites")

    structdata = relationship("Structdata", back_populates="site")

    def __repr__(self):
        return "Site:{} ({})".format(self.name, self.unit.name)


tagged = Table(
    "tagged",
    metadata,
    Column("id", Integer, autoincrement=True),
    Column("id_tags", Integer, ForeignKey("tags.id"), primary_key=True),
    Column("id_structdata", Integer, ForeignKey("structdata.id"), primary_key=True),
)


class Attached(Base):
    __tablename__ = "attach"

    id = Column(Integer, primary_key=True)
    id_structdata_planar = Column(
        ForeignKey("structdata.id"), nullable=False, index=True
    )
    id_structdata_linear = Column(
        ForeignKey("structdata.id"), nullable=False, index=True
    )

    def __repr__(self):
        return "{} - {}".format(self.planar, self.linear)


class Structdata(Base):
    __tablename__ = "structdata"

    id = Column(Integer, primary_key=True)
    id_sites = Column(ForeignKey("sites.id"), nullable=False, index=True)
    id_structype = Column(ForeignKey("structype.id"), nullable=False, index=True)
    azimuth = Column(Float, nullable=False, server_default=text("0"))
    inclination = Column(Float, nullable=False, server_default=text("0"))
    description = Column(Text)

    site = relationship("Site", back_populates="structdata")
    structype = relationship("Structype", back_populates="structdata")

    tags = relationship("Tag", secondary=tagged, back_populates="structdata")

    attach_planar = relationship(
        "Attached", backref="planar", primaryjoin=id == Attached.id_structdata_planar
    )
    attach_linear = relationship(
        "Attached", backref="linear", primaryjoin=id == Attached.id_structdata_linear
    )

    def __repr__(self):
        return "{}:{:g}/{:g}".format(
            self.structype.structure, self.azimuth, self.inclination
        )


class Structype(Base):
    __tablename__ = "structype"

    id = Column(Integer, primary_key=True)
    pos = Column(Integer, nullable=False, server_default=text("0"))
    structure = Column(String(16), nullable=False)
    description = Column(Text)
    structcode = Column(Integer, server_default=text("0"))
    groupcode = Column(Integer, server_default=text("0"))
    planar = Column(Integer, server_default=text("1"))

    structdata = relationship("Structdata", back_populates="structype")

    def __repr__(self):
        return "Type:{}".format(self.structure)


class Tag(Base):
    __tablename__ = "tags"

    id = Column(Integer, primary_key=True)
    pos = Column(Integer, nullable=False, server_default=text("0"))
    name = Column(String(16), nullable=False)
    description = Column(Text)

    structdata = relationship("Structdata", secondary=tagged, back_populates="tags")

    def __repr__(self):
        return "Tag:{}".format(self.name)


class Unit(Base):
    __tablename__ = "units"

    id = Column(Integer, primary_key=True)
    pos = Column(Integer, nullable=False, server_default=text("0"))
    name = Column(String(60), nullable=False)
    description = Column(Text)

    sites = relationship("Site", back_populates="unit")

    def __repr__(self):
        return "Unit:{}".format(self.name)


def default_meta():
    return dict(
        version="3.1.0",
        crs="EPSG:4326",
        created=datetime.now().strftime("%d.%m.%Y %H:%M"),
        updated=datetime.now().strftime("%d.%m.%Y %H:%M"),
        accessed=datetime.now().strftime("%d.%m.%Y %H:%M"),
    )


def default_initial_values():
    return [
        Structype(
            pos=1,
            structure="S",
            description="Default planar feature",
            structcode=35,
            groupcode=13,
            planar=1,
        ),
        Structype(
            pos=2,
            structure="L",
            description="Default linear feature",
            structcode=78,
            groupcode=13,
            planar=0,
        ),
        Unit(pos=1, name="Default", description="Default unit"),
    ]


def before_commit_meta_update(session):
    u = session.query(Meta).filter_by(name="updated").first()
    u.value = datetime.now().strftime("%d.%m.%Y %H:%M")


def before_insert_pos_update(mapper, connection, target):
    if target.pos is None:
        t = str(mapper.persist_selectable)
        query = "SELECT max({}.pos) FROM {}".format(t, t)
        maxpos = connection.scalar(text(query))
        if maxpos is None:
            maxpos = 1
        else:
            maxpos += 1
        target.pos = maxpos


class SDBSession:
    """
    SqlAlchemy interface to PySDB database

    Args:
        sdbfile (str): filename of PySDB database

    Keyword Args:
        create (bool): if True existing sdbfile will be deleted
            and new database will be created
        autocommit(bool): if True, each operation is autocommitted

    Example:
        >>> db = SDBSession('database.sdb', create=True)

    """

    def __init__(self, sdb_file, **kwargs):
        if kwargs.get("create", False):
            with contextlib.suppress(FileNotFoundError):
                os.remove(sdb_file)
        self.sdb_engine = create_engine("sqlite:///{}".format(sdb_file))
        if kwargs.get("create", False):
            metadata.create_all(self.sdb_engine)
        sdb_Session = sessionmaker(bind=self.sdb_engine)
        self.session = sdb_Session()
        if kwargs.get("create", False):
            meta = default_meta()
            meta.update(kwargs.get("meta", {}))
            self.session.add_all(default_initial_values())
            self.session.add_all([Meta(name=n, value=v) for n, v in meta.items()])
            self.session.commit()
        self.autocommit = kwargs.get("autocommit", False)
        # add listeners
        event.listen(self.session, "before_commit", before_commit_meta_update)
        event.listen(Unit, "before_insert", before_insert_pos_update)
        event.listen(Structype, "before_insert", before_insert_pos_update)
        event.listen(Tag, "before_insert", before_insert_pos_update)

    def __repr__(self):
        return self.info()

    def info(self, data=False):
        lines = []
        lines.append(f"PySDB database version: {self.meta('version').value}")
        lines.append(f"PySDB database CRS: {self.meta('crs').value}")
        lines.append(f"PySDB database created: {self.meta('created').value}")
        lines.append(f"PySDB database updated: {self.meta('created').value}")
        lines.append(f"Number of sites: {self.session.query(Site).count()}")
        lines.append(f"Number of units: {self.session.query(Unit).count()}")
        lines.append(f"Number of structures: {self.session.query(Structype).count()}")
        lines.append(
            f"Number of measurements: {self.session.query(Structdata).count()}"
        )
        if data:
            for s in self.structypes():
                n = self.session.query(Structdata).filter_by(structype=s).count()
                if n > 0:
                    lines.append(f"Number of {s.structure} measurements: {n}")
        return "\n".join(lines)

    def close(self):
        """
        Close session

        """
        self.session.close()

    def commit(self):
        """
        commit session

        """
        self.session.commit()

    def rollback(self):
        """
        rollback session

        """
        self.session.rollback()

    def meta(self, name, **kwargs):
        """
        Insert, update or retrieve (when kwargs empty) Meta

        Args:
            name (str): meta name

        Keyword Args:
            value (str): meta value

        Returns:
            Meta
        """
        meta = self.session.query(Meta).filter_by(name=name).first()
        if kwargs:
            if meta is None:
                meta = Meta(name=name, **kwargs)
                self.session.add(meta)
            else:
                self.session.query(Meta).filter_by(name=name).update(kwargs)
            if self.autocommit:
                self.commit()
        return meta

    def site(self, name, **kwargs):
        """
        Insert, update or retrieve (when kwargs empty) Site

        Args:
            name (str): site name

        Keyword Args:
            x_coord (float): x coord or longitude
            y_coord (float): y coord or latitude
            description (str): site description
            unit (Unit): unit instance (mus be provided)

        Returns:
            Site
        """
        site = self.session.query(Site).filter_by(name=name).first()
        if kwargs:
            if site is None:
                site = Site(name=name, **kwargs)
                self.session.add(site)
            else:
                self.session.query(Site).filter_by(name=name).update(kwargs)
            if self.autocommit:
                self.commit()
        return site

    def unit(self, name, **kwargs):
        """
        Insert, update or retrieve (when kwargs empty) Unit

        Args:
            name (str): unit name

        Keyword Args:
            description (str): unit description

        Returns:
            Unit
        """
        unit = self.session.query(Unit).filter_by(name=name).first()
        if kwargs:
            if unit is None:
                unit = Unit(name=name, **kwargs)
                self.session.add(unit)
            else:
                self.session.query(Unit).filter_by(name=name).update(kwargs)
            if self.autocommit:
                self.commit()
        return unit

    def tag(self, name, **kwargs):
        """
        Insert, update or retrieve (when kwargs empty) Tag

        Args:
            name (str): tag name

        Keyword Args:
            description (str): tag description

        Returns:
            Tag
        """
        tag = self.session.query(Tag).filter_by(name=name).first()
        if kwargs:
            if tag is None:
                tag = Tag(name=name, **kwargs)
                self.session.add(tag)
            else:
                self.session.query(Tag).filter_by(name=name).update(kwargs)
            if self.autocommit:
                self.commit()
        return tag

    def structype(self, structure, **kwargs):
        """
        Insert, update or retrieve (when kwargs empty) Structype

        Args:
            structure (str): label for structure

        Keyword Args:
            description (str): structype description
            planar (int): 1 for planar 0 for linear
            structcode (int): structcode (optional)
            groupcode (int): groupcode (optional)

        Returns:
            Structype
        """
        structype = self.session.query(Structype).filter_by(structure=structure).first()
        if kwargs:
            if structype is None:
                structype = Structype(structure=structure, **kwargs)
                self.session.add(structype)
            else:
                self.session.query(Structype).filter_by(structure=structure).update(
                    kwargs
                )
            if self.autocommit:
                self.commit()
        return structype

    def add_structdata(self, site, structype, azimuth, inclination, **kwargs):
        """
        Add structdata to site

        Args:
            site (Site): site instance
            structype (Structype): structype instance
            azimuth (float): dip direction or plunge direction
            inclination (float): dip or plunge

        Keyword Args:
            description (str): structdata description

        Returns:
            Structdata

        """
        data = Structdata(
            site=site,
            structype=structype,
            azimuth=azimuth,
            inclination=inclination,
            **kwargs,
        )
        self.session.add(data)
        if self.autocommit:
            self.commit()
        return data

    def add_fol(self, site, structype, fol, **kwargs):
        """
        Add Foliation to site

        Args:
            site (Site): site instance
            structype (Structype): structype instance
            fol (Foliation): foliation instance

        Keyword Args:
            description (str): structdata description

        Returns:
            Structdata

        """
        assert isinstance(
            fol, Foliation
        ), "fol argument must be instance of Foliation class"
        assert structype.planar, "structype must be planar"
        azimuth, inclination = fol.geo
        return self.add_structdata(site, structype, azimuth, inclination, **kwargs)

    def add_lin(self, site, structype, lin, **kwargs):
        """
        Add Lineation to site

        Args:
            site (Site): site instance
            structype (Structype): structype instance
            lin (Lineation): lineation instance

        Keyword Args:
            description (str): structdata description

        Returns:
            Structdata

        """
        assert isinstance(
            lin, Lineation
        ), "lin argument must be instance of Lineation class"
        assert not structype.planar, "structype must be linear"
        azimuth, inclination = lin.geo
        return self.add_structdata(site, structype, azimuth, inclination, **kwargs)

    def attach(self, fol, lin):
        """
        Add Lineation to site

        Args:
            fol (Foliation): foliation instance
            lin (Lineation): lineation instance

        Returns:
            Attached

        """
        pair = Attached(planar=fol, linear=lin)
        self.session.add(pair)
        if self.autocommit:
            self.commit()
        return pair

    def add_pair(self, pair, foltype, lintype, **kwargs):
        """
        Add attached foliation and lineation to database

        Args:
            pair (Pair): pair instance
            foltype (Structype): structype instance
            lintype (Structype): structype instance

        Returns:
            Attached

        """
        assert isinstance(pair, Pair), "pair argument must be instance of Pair class"
        kwargs["structype"] = foltype
        fol = self.add_fol(pair.fol, **kwargs)
        kwargs["structype"] = lintype
        lin = self.add_lin(pair.lin, **kwargs)
        return self.attach(fol, lin)

    def sites(self, **kwargs):
        """
        Retrieve Site or list of Sites based on criteria in kwargs

        Keyword arguments are passed to sqlalchemy filter_by method
        """
        if kwargs:
            sites = self.session.query(Site).filter_by(**kwargs).all()
        else:
            sites = self.session.query(Site).all()
        if len(sites) == 1:
            return sites[0]
        else:
            return sites

    def units(self, **kwargs):
        """
        Retrieve Unit or list of Units based on criteria in kwargs

        Keyword arguments are passed to sqlalchemy filter_by method
        """
        if kwargs:
            units = self.session.query(Unit).filter_by(**kwargs).all()
        else:
            units = self.session.query(Unit).all()
        if len(units) == 1:
            return units[0]
        else:
            return units

    def structypes(self, **kwargs):
        """
        Retrieve Structype or list of Structypes based on criteria in kwargs

        Keyword arguments are passed to sqlalchemy filter_by method
        """
        if kwargs:
            structypes = self.session.query(Structype).filter_by(**kwargs).all()
        else:
            structypes = self.session.query(Structype).all()
        if len(structypes) == 1:
            return structypes[0]
        else:
            return structypes

    def tags(self, **kwargs):
        """
        Retrieve Tag or list of Tags based on criteria in kwargs

        Keyword arguments are passed to sqlalchemy filter_by method
        """
        if kwargs:
            tags = self.session.query(Tag).filter_by(**kwargs).all()
        else:
            tags = self.session.query(Tag).all()
        if len(tags) == 1:
            return tags[0]
        else:
            return tags

    def getset(self, structype, **kwargs):
        """Method to retrieve data from SDB database to ``FeatureSet``.

        Args:
          structype (str, Structype):  structure or list of structures to retrieve

        Keyword arguments are passed to sqlalchemy filter_by method

        """
        if isinstance(structype, str):
            structypes = (
                self.session.query(Structype).filter_by(structure=structype).all()
            )
            assert len(structypes) == 1, f"There is no structure {structype} in db"
            structype = structypes[0]
        data = (
            self.session.query(Structdata)
            .filter_by(structype=structype, **kwargs)
            .all()
        )
        if structype.planar:
            res = FoliationSet(
                [Foliation(v.azimuth, v.inclination) for v in data],
                name=structype.structure,
            )
        else:
            res = LineationSet(
                [Lineation(v.azimuth, v.inclination) for v in data],
                name=structype.structure,
            )
        return res
