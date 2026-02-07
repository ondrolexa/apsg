# -*- coding: utf-8 -*-

"""
SQLAlchemy API to access PySDB database
"""

import contextlib
import os
import warnings
from datetime import datetime

from sqlalchemy import (
    Column,
    Float,
    ForeignKey,
    Integer,
    String,
    Table,
    Text,
    create_engine,
    event,
    text,
)
from sqlalchemy.orm import (
    DeclarativeBase,
    Mapped,
    aliased,
    mapped_column,
    relationship,
    sessionmaker,
)

from apsg.feature._container import FaultSet, FoliationSet, LineationSet, PairSet
from apsg.feature._geodata import Fault, Foliation, Lineation, Pair


class Base(DeclarativeBase):
    pass


metadata = Base.metadata


class Meta(Base):
    __tablename__ = "meta"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    name: Mapped[str] = mapped_column(String(16), nullable=False, unique=True)
    value: Mapped[str] = mapped_column(Text)

    def __repr__(self):
        return "Meta:{}={}".format(self.name, self.value)


class Site(Base):
    __tablename__ = "sites"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    id_units: Mapped[int] = mapped_column(
        ForeignKey("units.id"), nullable=False, index=True
    )
    name: Mapped[str] = mapped_column(String(16), nullable=False, unique=True)
    x_coord: Mapped[float | None] = mapped_column(Float, server_default=text("NULL"))
    y_coord: Mapped[float | None] = mapped_column(Float, server_default=text("NULL"))
    description: Mapped[str | None] = mapped_column(Text)

    unit = relationship("Unit", back_populates="sites", cascade="save-update")

    structdata = relationship(
        "Structdata", back_populates="site", cascade="all, delete-orphan"
    )

    def __repr__(self):
        return "Site:{} ({})".format(self.name, self.unit.name)


tagged = Table(
    "tagged",
    metadata,
    Column("id", Integer, primary_key=True, autoincrement=True),
    Column("id_tags", Integer, ForeignKey("tags.id"), nullable=False, index=True),
    Column(
        "id_structdata",
        Integer,
        ForeignKey("structdata.id"),
        nullable=False,
        index=True,
    ),
)


class Attached(Base):
    __tablename__ = "attach"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    id_structdata_planar: Mapped[int] = mapped_column(
        ForeignKey("structdata.id", ondelete="CASCADE"), nullable=False, index=True
    )
    id_structdata_linear: Mapped[int] = mapped_column(
        ForeignKey("structdata.id", ondelete="CASCADE"), nullable=False, index=True
    )

    planar = relationship(
        "Structdata",
        back_populates="attach_planar",
        foreign_keys=[id_structdata_planar],
    )
    linear = relationship(
        "Structdata",
        back_populates="attach_linear",
        foreign_keys=[id_structdata_linear],
    )

    def __repr__(self):
        return "{} - {}".format(self.planar, self.linear)


class Structdata(Base):
    __tablename__ = "structdata"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    id_sites: Mapped[int] = mapped_column(
        ForeignKey("sites.id"), nullable=False, index=True
    )
    id_structype: Mapped[int] = mapped_column(
        ForeignKey("structype.id"), nullable=False, index=True
    )
    azimuth: Mapped[float] = mapped_column(
        Float, nullable=False, server_default=text("0")
    )
    inclination: Mapped[float] = mapped_column(
        Float, nullable=False, server_default=text("0")
    )
    description: Mapped[str | None] = mapped_column(Text)

    site = relationship("Site", back_populates="structdata")
    structype = relationship(
        "Structype", back_populates="structdata", cascade="save-update"
    )

    tags = relationship(
        "Tag", secondary=tagged, back_populates="structdata", cascade="save-update"
    )

    attach_planar = relationship(
        "Attached",
        back_populates="planar",
        primaryjoin=id == Attached.id_structdata_planar,
        cascade="all, delete",
    )
    attach_linear = relationship(
        "Attached",
        back_populates="linear",
        primaryjoin=id == Attached.id_structdata_linear,
        cascade="all, delete",
    )

    def __repr__(self):
        return "{}:{:g}/{:g}".format(
            self.structype.structure, self.azimuth, self.inclination
        )


class Structype(Base):
    __tablename__ = "structype"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    pos: Mapped[int] = mapped_column(Integer, nullable=False, server_default=text("0"))
    structure: Mapped[str] = mapped_column(String(16), nullable=False, unique=True)
    description: Mapped[str | None] = mapped_column(Text)
    structcode: Mapped[int] = mapped_column(Integer, server_default=text("0"))
    groupcode: Mapped[int] = mapped_column(Integer, server_default=text("0"))
    planar: Mapped[int] = mapped_column(Integer, server_default=text("1"))

    structdata = relationship(
        "Structdata", back_populates="structype", cascade="all, delete"
    )

    def __repr__(self):
        return "Type:{}".format(self.structure)


class Tag(Base):
    __tablename__ = "tags"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    pos: Mapped[int] = mapped_column(Integer, nullable=False, server_default=text("0"))
    name: Mapped[str] = mapped_column(String(16), nullable=False, unique=True)
    description: Mapped[str | None] = mapped_column(Text)

    structdata = relationship("Structdata", secondary=tagged, back_populates="tags")

    def __repr__(self):
        return "Tag:{}".format(self.name)


class Unit(Base):
    __tablename__ = "units"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    pos: Mapped[int] = mapped_column(Integer, nullable=False, server_default=text("0"))
    name: Mapped[str] = mapped_column(String(60), nullable=False, unique=True)
    description: Mapped[str | None] = mapped_column(Text)

    sites = relationship("Site", back_populates="unit", cascade="all, delete")

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
        autocommit(bool): if True, each operation is autocommitted.
            Default False

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
        self.session.execute(text("PRAGMA foreign_keys=ON"))
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
        lines.append(f"PySDB database updated: {self.meta('updated').value}")
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

    def add(self, obj):
        """
        add to session

        """
        self.session.add(obj)

    def add_all(self, list_obj):
        """
        add to session

        """
        self.session.add_all(list_obj)

    def meta(self, name, **kwargs):
        """
        Query Meta when no kwargs or insert/update when kwargs provided

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
                for key, value in kwargs.items():
                    setattr(meta, key, value)
            if self.autocommit:
                self.commit()
        return meta

    def site(self, name, **kwargs):
        """
        Query Site when no kwargs or insert/update when kwargs provided

        Args:
            name (str): site name

        Keyword Args:
            x_coord (float): x coord or longitude
            y_coord (float): y coord or latitude
            description (str): site description
            unit (Unit): unit instance (must be provided)

        Returns:
            Site
        """
        site = self.session.query(Site).filter_by(name=name).first()
        if kwargs:
            if site is None:
                site = Site(name=name, **kwargs)
                if site.unit is None:
                    site.unit = self.session.query(Unit).filter_by(id=1).first()
                self.session.add(site)
            else:
                for key, value in kwargs.items():
                    setattr(site, key, value)
            if self.autocommit:
                self.commit()
        if site is None:
            raise ValueError(f"Site {name} not found")
        else:
            return site

    def unit(self, name, **kwargs):
        """
        Query Unit when no kwargs or insert/update when kwargs provided

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
                for key, value in kwargs.items():
                    setattr(unit, key, value)
            if self.autocommit:
                self.commit()
        if unit is None:
            raise ValueError(f"Unit {name} not found")
        else:
            return unit

    def remove_unit(self, name, **kwargs):
        """
        Remove Unit

        Note: When assign is None associated sites are removed

        Args:
            name (str): name of unit to be removed

        Keyword Args:
            assign (str): Structype to be assigned to data

        """
        unit = self.session.query(Unit).filter_by(name=name).first()
        if unit is not None:
            assign = kwargs.get("assign", None)
            if assign is not None:
                new_unit = self.session.query(Unit).filter_by(name=assign).first()
                if new_unit is None:
                    new_unit = self.unit(assign)
                for data in self.session.query(Site).filter_by(unit=unit).all():
                    data.unit = new_unit

            self.session.delete(unit)
            if self.autocommit:
                self.commit()
        else:
            raise ValueError(f"Unit {name} not found")

    def tag(self, name, **kwargs):
        """
        Query Tag when no kwargs or insert/update when kwargs provided

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
                for key, value in kwargs.items():
                    setattr(tag, key, value)
            if self.autocommit:
                self.commit()
        if tag is None:
            raise ValueError(f"Tag {name} not found")
        else:
            return tag

    def structype(self, structure, **kwargs):
        """
        Query Structype when no kwargs or insert/update when kwargs provided

        Args:
            structure (str): label used for structure

        Keyword Args:
            description (str): Structype description
            planar (int): 1 for planar 0 for linear Structype
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
                for key, value in kwargs.items():
                    setattr(structype, key, value)
            if self.autocommit:
                self.commit()
        if structype is None:
            raise ValueError(f"Structype {structure} not found")
        else:
            return structype

    def remove_structype(self, structure, **kwargs):
        """
        Remove Structype

        Note: When assign is None associated data are removed

        Args:
            structure (str): structure of structype to be removed

        Keyword Args:
            assign (str): Structype to be assigned to data

        """
        structype = self.session.query(Structype).filter_by(structure=structure).first()
        if structype is not None:
            assign = kwargs.get("assign", None)
            if assign is not None:
                new_structype = (
                    self.session.query(Structype).filter_by(structure=assign).first()
                )
                if new_structype is None:
                    new_structype = self.structype(assign, planar=structype.planar)
                if structype.planar == new_structype.planar:
                    for data in (
                        self.session.query(Structdata)
                        .filter_by(structype=structype)
                        .all()
                    ):
                        data.structype = new_structype
                else:
                    if structype.planar:
                        raise ValueError("Structure to be assigned must be planar")
                    else:
                        raise ValueError("Structure to be assigned must be linear")

            self.session.delete(structype)
            if self.autocommit:
                self.commit()
        else:
            raise ValueError(f"Structype {structure} not found")

    def add_structdata(self, site, structype, azimuth, inclination, **kwargs):
        """
        Add structural measurement to site

        Args:
            site (Site): Site instance
            structype (Structype): Structype instance
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
            site (Site): Site instance
            structype (Structype): Structype instance
            fol (Foliation): Foliation instance

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
            site (Site): Site instance
            structype (Structype): Structype instance
            lin (Lineation): Lineation instance

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

    def attach(self, planar, linear):
        """
        Attach Foliation to Lineation

        Args:
            planar (Structdata): planar Structdata
            linear (Structdata): linear Structdata

        Returns:
            Attached

        """
        assert planar.structype.planar == 1, "First argument must be planar Structdata"
        assert linear.structype.planar == 0, "Second argument must be linear Structdata"
        attach = Attached(planar=planar, linear=linear)
        self.session.add(attach)
        if self.autocommit:
            self.commit()
        return attach

    def add_pair(self, site, foltype, lintype, pair, **kwargs):
        """
        Add attached foliation and lineation to database. Note that
        measurements of foliation and lineation are corrected.

        Args:
            site (Site): site instance
            foltype (Structype): structype instance
            lintype (Structype): structype instance
            pair (Pair): Pair instance

        Returns:
            Attached

        """
        assert isinstance(pair, Pair), "pair argument must be instance of Pair class"
        assert foltype.planar, "foltype must be planar"
        assert not lintype.planar, "lintype must be linear"
        fol = self.add_fol(site, foltype, pair.fol, **kwargs)
        lin = self.add_lin(site, lintype, pair.lin, **kwargs)
        return self.attach(fol, lin)

    def sites(self):
        """
        Returns list of all Site instances

        """
        return self.session.query(Site).all()

    def units(self, **kwargs):
        """
        Returns list of all Unit instances

        """
        return self.session.query(Unit).all()

    def structypes(self, planar=None):
        """
        Returns list of Structype instances

        Keyword Args:
            planar (int): 0 for linear, 1 for planar, None for all. Default None

        """
        if planar is not None:
            structypes = self.session.query(Structype).filter_by(planar=planar).all()
        else:
            structypes = self.session.query(Structype).all()
        return structypes

    def tags(self):
        """
        Returns list of all Tag instances

        """
        return self.session.query(Tag).all()

    def getset(self, structype, **kwargs):
        """Method to retrieve data from SDB database to ``FeatureSet``.

        Args:
          structype (str | Structype):  structure to retrieve
          site (dict): keyword args passed to filter site
          unit (dict): keyword args passed to filter unit
          tag (dict): keyword args passed to filter tag
          store (list): list of properties to be stored in feature as _attrs
              Available values:"id", "x_coord", "y_coord", "site", "unit", "tags"
              or "geo". Default []

        Example:
          >> g = db.getset("S2")
          >> g = db.getset("S2", unit=dict(name="Main unit"))
          >> g = db.getset("S2", tag=dict(name="AP"))

        """
        site = kwargs.get("site", {})
        unit = kwargs.get("unit", {})
        tag = kwargs.get("tag", {})
        store = kwargs.get("store", [])

        def get_attrs(v):
            res = {}
            if "id" in store:
                res["id"] = v.id
            if "x_coord" in store:
                res["x_coord"] = v.site.x_coord
            if "y_coord" in store:
                res["y_coord"] = v.site.y_coord
            if "site" in store:
                res["site"] = v.site.name
            if "unit" in store:
                res["unit"] = v.site.unit.name
            if "tags" in store:
                res["tags"] = [t.name for t in v.tags]
            if "geo" in store:
                res["geo"] = (v.site.x_coord, v.site.y_coord)
            return res

        if isinstance(structype, str):
            dbstruct = (
                self.session.query(Structype).filter_by(structure=structype).first()
            )
            assert dbstruct is not None, f"There is no structure {structype} in db."
            structype = dbstruct
        if isinstance(structype, Structype):
            data = (
                self.session.query(Structdata)
                .filter_by(structype=structype)
                .join(Structdata.site)
                .filter_by(**site)
                .join(Site.unit)
                .filter_by(**unit)
                .outerjoin(Structdata.tags)
                .filter_by(**tag)
                .all()
            )
            if structype.planar == 1:
                res = FoliationSet(
                    [
                        Foliation(
                            v.azimuth,
                            v.inclination,
                            **get_attrs(v),
                        )
                        for v in data
                    ],
                    name=structype.structure,
                )
            else:
                res = LineationSet(
                    [
                        Lineation(
                            v.azimuth,
                            v.inclination,
                            **get_attrs(v),
                        )
                        for v in data
                    ],
                    name=structype.structure,
                )
            return res
        else:
            raise ValueError("structype argument must be string or Structype")

    def getpairs(self, ptype, ltype, site={}, unit={}, ptag={}, ltag={}):
        """Method to retrieve data from SDB database to ``PairSet``.

        Args:
          ptype (str | Structype):  planar structure to retrieve
          ltype (str | Structype):  linear structure to retrieve

        Keyword Args:
          site (dict): keyword args passed to filter site
          unit (dict): keyword args passed to filter unit
          ptag (dict): keyword args passed to filter planar tag
          ltag (dict): keyword args passed to filter linear tag

        """
        if isinstance(ptype, str):
            dbstruct = self.session.query(Structype).filter_by(structure=ptype).first()
            assert dbstruct is not None, f"There is no structure {ptype} in db."
            ptype = dbstruct
        if isinstance(ltype, str):
            dbstruct = self.session.query(Structype).filter_by(structure=ltype).first()
            assert dbstruct is not None, f"There is no structure {ltype} in db."
            ltype = dbstruct
        if isinstance(ptype, Structype) and isinstance(ltype, Structype):
            AttachPlanar = aliased(Structdata)
            AttachLinear = aliased(Structdata)
            TagPlanar = aliased(Tag)
            TagLinear = aliased(Tag)
            data = (
                self.session.query(Attached)
                .join(AttachPlanar, Attached.planar)
                .filter_by(structype=ptype)
                .join(AttachLinear, Attached.linear)
                .filter_by(structype=ltype)
                .join(Structdata.site)
                .filter_by(**site)
                .join(Site.unit)
                .filter_by(**unit)
                .outerjoin(TagPlanar, AttachPlanar.tags)
                .filter_by(**ptag)
                .outerjoin(TagLinear, AttachLinear.tags)
                .filter_by(**ltag)
                .all()
            )
            pairs = []
            warnings.filterwarnings("error")
            for v in data:
                try:
                    pair = Pair(
                        v.planar.azimuth,
                        v.planar.inclination,
                        v.linear.azimuth,
                        v.linear.inclination,
                    )
                    pairs.append(pair)
                except UserWarning:
                    print(
                        f"Too big misfit for pair {v.planar}-{v.linear} on {v.planar.site}"
                    )
            warnings.resetwarnings()
            res = PairSet(pairs, name=f"{ptype.structure}-{ltype.structure}")
            return res
        else:
            raise ValueError("structype argument must be string or Structype")

    def getfaults(self, ptype, ltype, sense, site={}, unit={}, ptag={}, ltag={}):
        """Method to retrieve data from SDB database to ``FaultSet``.

        Args:
          ptype (str | Structype):  planar structure to retrieve
          ltype (str | Structype):  linear structure to retrieve
          sense (float or str): sense of movement +/-1 hanging-wall down/up. When str,
              must be one of 's', 'd', 'n', 'r'.

        Keyword Args:
          site (dict): keyword args passed to filter site
          unit (dict): keyword args passed to filter unit
          ptag (dict): keyword args passed to filter planar tag
          ltag (dict): keyword args passed to filter linear tag

        """
        if isinstance(ptype, str):
            dbstruct = self.session.query(Structype).filter_by(structure=ptype).first()
            assert dbstruct is not None, f"There is no structure {ptype} in db."
            ptype = dbstruct
        if isinstance(ltype, str):
            dbstruct = self.session.query(Structype).filter_by(structure=ltype).first()
            assert dbstruct is not None, f"There is no structure {ltype} in db."
            ltype = dbstruct
        if isinstance(ptype, Structype) and isinstance(ltype, Structype):
            AttachPlanar = aliased(Structdata)
            AttachLinear = aliased(Structdata)
            TagPlanar = aliased(Tag)
            TagLinear = aliased(Tag)
            data = (
                self.session.query(Attached)
                .join(AttachPlanar, Attached.planar)
                .filter_by(structype=ptype)
                .join(AttachLinear, Attached.linear)
                .filter_by(structype=ltype)
                .join(Structdata.site)
                .filter_by(**site)
                .join(Site.unit)
                .filter_by(**unit)
                .outerjoin(TagPlanar, AttachPlanar.tags)
                .filter_by(**ptag)
                .outerjoin(TagLinear, AttachLinear.tags)
                .filter_by(**ltag)
                .all()
            )
            faults = []
            warnings.filterwarnings("error")
            for v in data:
                try:
                    fault = Fault(
                        v.planar.azimuth,
                        v.planar.inclination,
                        v.linear.azimuth,
                        v.linear.inclination,
                        sense,
                    )
                    faults.append(fault)
                except UserWarning:
                    print(
                        f"Too big misfit for pair {v.planar}-{v.linear} on {v.planar.site}"
                    )
            warnings.resetwarnings()
            res = FaultSet(faults, name=f"{ptype.structure}-{ltype.structure}")
            return res
        else:
            raise ValueError("structype argument must be string or Structype")
