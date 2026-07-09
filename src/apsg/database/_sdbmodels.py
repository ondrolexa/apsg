"""
SQLAlchemy model definitions for PySDB database
"""

from sqlalchemy import (
    Column,
    Float,
    ForeignKey,
    Integer,
    String,
    Table,
    Text,
    text,
)
from sqlalchemy.orm import (
    DeclarativeBase,
    Mapped,
    mapped_column,
    relationship,
)


class Base(DeclarativeBase):
    """SQLAlchemy declarative base for all models."""

    pass


metadata = Base.metadata


class Meta(Base):
    """SQLAlchemy model for database metadata key-value pairs."""

    __tablename__ = "meta"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    name: Mapped[str] = mapped_column(String(16), nullable=False, unique=True)
    value: Mapped[str] = mapped_column(Text)

    def __repr__(self):
        """Return string representation of the meta entry."""
        return "Meta:{}={}".format(self.name, self.value)


class Site(Base):
    """SQLAlchemy model for geological sampling sites."""

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
        """Return string representation of the site."""
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
    """SQLAlchemy model linking planar and linear structural data."""

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
        """Return string representation of the attachment."""
        return "{} - {}".format(self.planar, self.linear)


class Structdata(Base):
    """SQLAlchemy model for structural attitude measurements."""

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
        """Return string representation of the measurement."""
        return "{}:{:g}/{:g}".format(
            self.structype.structure, self.azimuth, self.inclination
        )


class Structype(Base):
    """SQLAlchemy model for structure type definitions."""

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
        """Return string representation of the structure type."""
        return "Type:{}".format(self.structure)


class Tag(Base):
    """SQLAlchemy model for tagging structural data."""

    __tablename__ = "tags"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    pos: Mapped[int] = mapped_column(Integer, nullable=False, server_default=text("0"))
    name: Mapped[str] = mapped_column(String(16), nullable=False, unique=True)
    description: Mapped[str | None] = mapped_column(Text)

    structdata = relationship("Structdata", secondary=tagged, back_populates="tags")

    def __repr__(self):
        """Return string representation of the tag."""
        return "Tag:{}".format(self.name)


class Unit(Base):
    """SQLAlchemy model for geological units."""

    __tablename__ = "units"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    pos: Mapped[int] = mapped_column(Integer, nullable=False, server_default=text("0"))
    name: Mapped[str] = mapped_column(String(60), nullable=False, unique=True)
    description: Mapped[str | None] = mapped_column(Text)

    sites = relationship("Site", back_populates="unit", cascade="all, delete")

    def __repr__(self):
        """Return string representation of the unit."""
        return "Unit:{}".format(self.name)
