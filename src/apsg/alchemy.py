
"""
SqlAlchemy interface to PySDB database

Create example:

    >>> from apsg.alchemy import AlchemySession

    >>> db = AlchemySession('database.sdb', create=True)
    >>> unit = db.unit(name='DMU', description='Deamonic Magmatic Unit')
    >>> site1 = db.site(unit=unit, name='LX001', x_coord=25487.54, y_coord=563788.2, description='granite sheet')
    >>> site2 = db.site(unit=unit, name='LX002', x_coord=25934.36, y_coord=564122.5, description='diorite dyke')
    >>> S2 = db.structype(structure='S2', description='Solid-state foliation', planar=1)
    >>> L2 = db.structype(structure='L2', description='Solid-state lineation', planar=0)
    >>> fol = db.add_structdata(site=site2, structype=S2, azimuth=150, inclination=36)
    >>> db.close()

Update example:

    >>> db = AlchemySession('database.sdb')
    >>> unit = db.unit(name='DMU')
    >>> site3 = db.site(unit=unit, name='LX003', x_coord=25713.7, y_coord=563977.1, description='massive gabbro')
    >>> db.close()

Tags example:

    >>> db = AlchemySession('database.sdb')
    >>> unit = db.unit(name='DMU')
    >>> site1 = db.site(name='LX001')
    >>> struct = db.structype(structure='S2')
    >>> tag_plot = db.tag(name='plot')
    >>> tag_ap = db.tag(name='AP')
    >>> fol = db.add_structdata(site=site1, structype=struct, azimuth=324, inclination=78, tags=[tag_plot, tag_ap])
    >>> db.close()

Attach example:

    >>> db = AlchemySession('database.sdb')
    >>> unit = db.unit(name='DMU')
    >>> site = db.site(name='LX001')
    >>> S = db.structype(structure='S')
    >>> L = db.structype(structure='L')
    >>> fol = db.add_structdata(site=site, structype=S, azimuth=220, inclination=28)
    >>> lin = db.add_structdata(site=site, structype=L, azimuth=212, inclination=26)
    >>> pair = db.attach(fol, lin)
    >>> db.close()

APSG classes example:

    >>> db = AlchemySession('database.sdb')
    >>> unit = db.unit(name='DMU')
    >>> site = db.site(name='LX003')

    >>> S2 = db.structype(structure='S2')
    >>> L2 = db.structype(structure='L2')

    >>> f = Fol(196, 39)
    >>> l = Lin(210, 37)
    >>> db.add_fol(f, site=site, structype=S2)
    >>> db.add_lin(l, site=site, structype=L2)

    >>> p = Pair(258, 42, 220, 30)
    >>> db.add_pair(p, S2, L2, site=site)
    >>> db.close()

"""

import os
from datetime import datetime
import contextlib

from apsg.core import Fol, Lin, Pair

from sqlalchemy import create_engine, event
from sqlalchemy.orm import sessionmaker
from sqlalchemy import Column, Float, ForeignKey, Integer, String, Table, Text, text
from sqlalchemy.orm import relationship
from sqlalchemy.ext.declarative import declarative_base

Base = declarative_base()
metadata = Base.metadata


class Meta(Base):
    __tablename__ = 'meta'

    id = Column(Integer, primary_key=True)
    name = Column(String(16), nullable=False)
    value = Column(Text)


class Site(Base):
    __tablename__ = 'sites'

    id = Column(Integer, primary_key=True)
    id_units = Column(ForeignKey(u'units.id'), nullable=False, index=True)
    name = Column(String(16), nullable=False, server_default=text("''"))
    x_coord = Column(Float, server_default=text("NULL"))
    y_coord = Column(Float, server_default=text("NULL"))
    description = Column(Text)

    unit = relationship(u'Unit', back_populates='sites')

    structdata = relationship(u'Structdata', back_populates='site')

    def __repr__(self):
        return 'Site:{} ({})'.format(self.name, self.unit.name)


tagged = Table(u'tagged', metadata,
               Column(u'id', Integer, autoincrement=True),
               Column(u'id_tags', Integer, ForeignKey(u'tags.id'), primary_key=True),
               Column(u'id_structdata', Integer, ForeignKey(u'structdata.id'), primary_key=True)
               )


class Attached(Base):
    __tablename__ = 'attach'

    id = Column(Integer, primary_key=True)
    id_structdata_planar = Column(ForeignKey(u'structdata.id'), nullable=False, index=True)
    id_structdata_linear = Column(ForeignKey(u'structdata.id'), nullable=False, index=True)

    def __repr__(self):
        return '{} - {}'.format(self.planar, self.linear)


class Structdata(Base):
    __tablename__ = 'structdata'

    id = Column(Integer, primary_key=True)
    id_sites = Column(ForeignKey(u'sites.id'), nullable=False, index=True)
    id_structype = Column(ForeignKey(u'structype.id'), nullable=False, index=True)
    azimuth = Column(Float, nullable=False, server_default=text("0"))
    inclination = Column(Float, nullable=False, server_default=text("0"))
    description = Column(Text)

    site = relationship(u'Site', back_populates='structdata')
    structype = relationship(u'Structype', back_populates='structdata')

    tags = relationship(u'Tag', secondary=tagged, back_populates='structdata')

    attach_planar = relationship(u'Attached', backref='planar', primaryjoin=id == Attached.id_structdata_planar)
    attach_linear = relationship(u'Attached', backref='linear', primaryjoin=id == Attached.id_structdata_linear)

    def __repr__(self):
        return '{}:{:g}/{:g}'.format(self.structype.structure, self.azimuth, self.inclination)


class Structype(Base):
    __tablename__ = 'structype'

    id = Column(Integer, primary_key=True)
    pos = Column(Integer, nullable=False, server_default=text("0"))
    structure = Column(String(16), nullable=False)
    description = Column(Text)
    structcode = Column(Integer, server_default=text("0"))
    groupcode = Column(Integer, server_default=text("0"))
    planar = Column(Integer, server_default=text("1"))

    structdata = relationship(u'Structdata', back_populates='structype')

    def __repr__(self):
        return 'Type:{}'.format(self.structure)


class Tag(Base):
    __tablename__ = 'tags'

    id = Column(Integer, primary_key=True)
    pos = Column(Integer, nullable=False, server_default=text("0"))
    name = Column(String(16), nullable=False)
    description = Column(Text)

    structdata = relationship(u'Structdata', secondary=tagged, back_populates='tags')

    def __repr__(self):
        return 'Tag:{}'.format(self.name)


class Unit(Base):
    __tablename__ = 'units'

    id = Column(Integer, primary_key=True)
    pos = Column(Integer, nullable=False, server_default=text("0"))
    name = Column(String(60), nullable=False)
    description = Column(Text)

    sites = relationship(u'Site', back_populates='unit')

    def __repr__(self):
        return 'Unit:{}'.format(self.name)


def initial_meta():
    return [Meta(name='version', value='3.0.4'),
            Meta(name='crs', value='EPSG:4326'),
            Meta(name='created', value=datetime.now().strftime("%d.%m.%Y %H:%M")),
            Meta(name='updated', value=datetime.now().strftime("%d.%m.%Y %H:%M")),
            Meta(name='accessed', value=datetime.now().strftime("%d.%m.%Y %H:%M"))]


def initial_values():
    return [Structype(pos=1, structure='S', description='Default planar feature', structcode=35, groupcode=13, planar=1),
            Structype(pos=2, structure='L', description='Default linear feature', structcode=78, groupcode=13, planar=0),
            Unit(pos=1, name='Default', description='Default unit')]


def before_commit_meta_update(session):
    u = session.query(Meta).filter_by(name='updated').first()
    u.value = datetime.now().strftime("%d.%m.%Y %H:%M")


def before_insert_pos_update(mapper, connection, target):
    if target.pos is None:
        t = str(mapper.persist_selectable)
        maxpos = connection.execute('SELECT max({}.pos) FROM {}'.format(t, t)).scalar()
        if maxpos is None:
            maxpos = 1
        else:
            maxpos += 1
        target.pos = maxpos


class AlchemySession():
    def __init__(self, sdb_file, **kwargs):
        if kwargs.get('create', False):
            with contextlib.suppress(FileNotFoundError):
                os.remove(sdb_file)
        self.sdb_engine = create_engine('sqlite:///{}'.format(sdb_file))
        if kwargs.get('create', False):
            metadata.create_all(self.sdb_engine)
        sdb_Session = sessionmaker(bind=self.sdb_engine)
        self.session = sdb_Session()
        if kwargs.get('create', False):
            self.session.add_all(initial_values())
            self.session.add_all(initial_meta())
            self.session.commit()
        # add listeners
        event.listen(self.session, 'before_commit', before_commit_meta_update)
        event.listen(Unit, 'before_insert', before_insert_pos_update)
        event.listen(Structype, 'before_insert', before_insert_pos_update)
        event.listen(Tag, 'before_insert', before_insert_pos_update)

    def close(self):
        self.session.close()

    def commit(self):
        self.session.commit()

    def site(self, **kwargs):
        assert 'name' in kwargs, 'name must be provided for site'
        site = self.session.query(Site).filter_by(name=kwargs['name']).first()
        if site is None:
            assert 'unit' in kwargs, 'unit must be provided to create site'
            site = Site(**kwargs)
            self.session.add(site)
            self.commit()
        return site

    def unit(self, **kwargs):
        assert 'name' in kwargs, 'name must be provided for unit'
        unit = self.session.query(Unit).filter_by(name=kwargs['name']).first()
        if unit is None:
            unit = Unit(**kwargs)
            self.session.add(unit)
            self.commit()
        return unit

    def tag(self, **kwargs):
        assert 'name' in kwargs, 'name must be provided for tag'
        tag = self.session.query(Tag).filter_by(name=kwargs['name']).first()
        if tag is None:
            tag = Tag(**kwargs)
            self.session.add(tag)
            self.commit()
        return tag

    def structype(self, **kwargs):
        assert 'structure' in kwargs, 'structure must be provided for structype'
        structype = self.session.query(Structype).filter_by(structure=kwargs['structure']).first()
        if structype is None:
            structype = Structype(**kwargs)
            self.session.add(structype)
            self.commit()
        return structype

    def add_structdata(self, **kwargs):
        assert 'site' in kwargs, 'site must be provided for structdata'
        assert 'structype' in kwargs, 'structype must be provided for structdata'
        data = Structdata(**kwargs)
        self.session.add(data)
        self.commit()
        return data

    def add_fol(self, fol, **kwargs):
        assert isinstance(fol, Fol), 'fol argument must be instance of Fol class'
        assert 'site' in kwargs, 'site must be provided for structdata'
        assert 'structype' in kwargs, 'structype must be provided for structdata'
        assert kwargs['structype'].planar, 'structype must be planar'
        kwargs['azimuth'] = fol.dd[0]
        kwargs['inclination'] = fol.dd[1]
        return self.add_structdata(**kwargs)

    def add_lin(self, lin, **kwargs):
        assert isinstance(lin, Lin), 'lin argument must be instance of Lin class'
        assert 'site' in kwargs, 'site must be provided for structdata'
        assert 'structype' in kwargs, 'structype must be provided for structdata'
        assert not kwargs['structype'].planar, 'structype must be linear'
        kwargs['azimuth'] = lin.dd[0]
        kwargs['inclination'] = lin.dd[1]
        return self.add_structdata(**kwargs)

    def attach(self, fol, lin):
        pair = Attached(planar=fol, linear=lin)
        self.session.add(pair)
        self.commit()
        return pair

    def add_pair(self, pair, foltype, lintype, **kwargs):
        assert isinstance(pair, Pair), 'pair argument must be instance of Pair class'
        kwargs['structype'] = foltype
        fol = self.add_fol(pair.fol, **kwargs)
        kwargs['structype'] = lintype
        lin = self.add_lin(pair.lin, **kwargs)
        return self.attach(fol, lin)

    def sites(self, **kwargs):
        if kwargs:
            sites = self.session.query(Site).filter_by(**kwargs).all()
        else:
            sites = self.session.query(Site).all()
        return sites

    def units(self, **kwargs):
        if kwargs:
            units = self.session.query(Unit).filter_by(**kwargs).all()
        else:
            units = self.session.query(Unit).all()
        return units

    def structypes(self, **kwargs):
        if kwargs:
            structypes = self.session.query(Structype).filter_by(**kwargs).all()
        else:
            structypes = self.session.query(Structype).all()
        return structypes

    def tags(self, **kwargs):
        if kwargs:
            tags = self.session.query(Tag).filter_by(**kwargs).all()
        else:
            tags = self.session.query(Tag).all()
        return tags
