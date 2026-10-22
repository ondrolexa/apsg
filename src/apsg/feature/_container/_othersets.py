"""Sets of cones and arcs."""

from itertools import pairwise

import numpy as np

from apsg.feature._container._base import FeatureSet, _per_item
from apsg.feature._container._vectorsets import Vector3Set
from apsg.feature._geodata import (
    Arc,
    Cone,
)
from apsg.math._vector import Vector3


class ConeSet(FeatureSet):
    """
    Class to store set of ``Cone`` features.
    """

    __feature_class__ = Cone

    def __init__(self, data, name="Default"):
        super().__init__(data, name=name)
        if not all(isinstance(obj, Cone) for obj in data):
            raise TypeError("Data must be instances of Cone")

    def __repr__(self):
        return f"C({len(self)}) {self.name}"

    revangle = _per_item("revangle", "Return array of revolution angles.")

    @property
    def apical_angle(self):
        """Return array of apical angles."""

        return np.array([c.apical_angle() for c in self])


class ArcSet(FeatureSet):
    """
    Class to store set of ``Arc`` features.
    """

    __feature_class__ = Arc

    def __init__(self, data, name="Default"):
        super().__init__(data, name=name)
        if not all(isinstance(obj, Arc) for obj in data):
            raise TypeError("Data must be instances of Arc")

    def __repr__(self):
        return f"A({len(self)}) {self.name}"

    curvature = _per_item("curvature", "Return array of curvatures.")
    positive = _per_item("positive", "Return array of curvature directions.")
    short = _per_item("short", "Return array of short/reflex path flags.")

    @classmethod
    def from_vectors(
        cls, *args, curvature=0, positive=True, short=True, name="Default"
    ):
        """
        Create ``ArcSet`` connecting consecutive pairs of vectors.

        Accepts either a single ``Vector3Set``-like object (a ``Vector3Set``,
        ``LineationSet``, ``FoliationSet``, ...) or 2 or more individual
        ``Vector3``-like features as separate positional arguments. Consecutive
        pairs are connected into one ``Arc`` each, so N vectors produce N-1 arcs.

        Args:
            *args: a single ``Vector3Set``-like object, or 2 or more individual
                ``Vector3``-like features.

        Keyword Args:
            curvature (float): passed to each ``Arc``. Default 0.
            positive (bool): passed to each ``Arc``. Default True.
            short (bool): passed to each ``Arc``. Default True.
            name: name of dataset. Default is 'Default'.

        Examples:
            >>> a = arcset.from_vectors(lin(0, 0), lin(45, 20), lin(90, 0))
            >>> a = arcset.from_vectors(linset.random_fisher(n=5))

        Returns:
            ArcSet: The created feature set.
        """
        if len(args) == 1 and isinstance(args[0], Vector3Set):
            points = args[0].data
        else:
            points = args
        if len(points) < 2 or not all(isinstance(p, Vector3) for p in points):
            raise TypeError(
                "ArcSet.from_vectors needs a Vector3Set or 2 or more Vector3-like features"
            )
        arcs = [
            Arc(p1, p2, curvature=curvature, positive=positive, short=short)
            for p1, p2 in pairwise(points)
        ]
        return cls(arcs, name=name)
