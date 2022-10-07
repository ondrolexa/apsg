# -*- coding: utf-8 -*-

"""
SQLAlchemy API to access PySDB database
"""

import os
from datetime import datetime
import contextlib

from apsg.feature._geodata import Lineation, Foliation, Pair, Fault
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


def initial_meta():
    return [
        Meta(name="version", value="3.0.4"),
        Meta(name="crs", value="EPSG:4326"),
        Meta(name="created", value=datetime.now().strftime("%d.%m.%Y %H:%M")),
        Meta(name="updated", value=datetime.now().strftime("%d.%m.%Y %H:%M")),
        Meta(name="accessed", value=datetime.now().strftime("%d.%m.%Y %H:%M")),
    ]


def initial_values():
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
        maxpos = connection.execute("SELECT max({}.pos) FROM {}".format(t, t)).scalar()
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
            self.session.add_all(initial_values())
            self.session.add_all(initial_meta())
            self.session.commit()
        # add listeners
        event.listen(self.session, "before_commit", before_commit_meta_update)
        event.listen(Unit, "before_insert", before_insert_pos_update)
        event.listen(Structype, "before_insert", before_insert_pos_update)
        event.listen(Tag, "before_insert", before_insert_pos_update)

    def close(self):
        self.session.close()

    def commit(self):
        self.session.commit()

    def site(self, **kwargs):
        """
        Insert or retrieve Site
        """
        assert "name" in kwargs, "name must be provided for site"
        site = self.session.query(Site).filter_by(name=kwargs["name"]).first()
        if site is None:
            assert "unit" in kwargs, "unit must be provided to create site"
            site = Site(**kwargs)
            self.session.add(site)
            self.commit()
        return site

    def unit(self, **kwargs):
        """
        Insert or retrieve Unit
        """
        assert "name" in kwargs, "name must be provided for unit"
        unit = self.session.query(Unit).filter_by(name=kwargs["name"]).first()
        if unit is None:
            unit = Unit(**kwargs)
            self.session.add(unit)
            self.commit()
        return unit

    def tag(self, **kwargs):
        """
        Insert or retrieve Tag
        """
        assert "name" in kwargs, "name must be provided for tag"
        tag = self.session.query(Tag).filter_by(name=kwargs["name"]).first()
        if tag is None:
            tag = Tag(**kwargs)
            self.session.add(tag)
            self.commit()
        return tag

    def structype(self, **kwargs):
        """
        Insert or retrieve Structype
        """
        assert "structure" in kwargs, "structure must be provided for structype"
        structype = (
            self.session.query(Structype)
            .filter_by(structure=kwargs["structure"])
            .first()
        )
        if structype is None:
            structype = Structype(**kwargs)
            self.session.add(structype)
            self.commit()
        return structype

    def add_structdata(self, **kwargs):
        """
        Add measurement to site

        Keyword Args:
            site(Site): site instance
            structype(Structype): structype instance
            azimuth(float): dip direction or plunge direction
            inclination(float): dip or plunge

        Returns:
            Structdata

        """
        assert "site" in kwargs, "site must be provided for structdata"
        assert "structype" in kwargs, "structype must be provided for structdata"
        data = Structdata(**kwargs)
        self.session.add(data)
        self.commit()
        return data

    def add_fol(self, fol, **kwargs):
        """
        Add Foliation to site

        Args:
            fol (Foliation): foliation instance

        Keyword Args:
            site(Site): site instance
            structype(Structype): structype instance

        Returns:
            Structdata

        """
        assert isinstance(
            fol, Foliation
        ), "fol argument must be instance of Foliation class"
        assert "site" in kwargs, "site must be provided for structdata"
        assert "structype" in kwargs, "structype must be provided for structdata"
        assert kwargs["structype"].planar, "structype must be planar"
        azi, inc = fol.geo
        kwargs["azimuth"] = azi
        kwargs["inclination"] = inc
        return self.add_structdata(**kwargs)

    def add_lin(self, lin, **kwargs):
        """
        Add Lineation to site

        Args:
            lin (Lineation): lineation instance

        Keyword Args:
            site(Site): site instance
            structype(Structype): structype instance

        Returns:
            Structdata

        """
        assert isinstance(
            lin, Lineation
        ), "lin argument must be instance of Lineation class"
        assert "site" in kwargs, "site must be provided for structdata"
        assert "structype" in kwargs, "structype must be provided for structdata"
        assert not kwargs["structype"].planar, "structype must be linear"
        azi, inc = lin.geo
        kwargs["azimuth"] = azi
        kwargs["inclination"] = inc
        return self.add_structdata(**kwargs)

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
