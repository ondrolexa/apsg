"""Feature sets: containers of features sharing a common ``.data`` tuple.

Split into submodules by kind; everything is re-exported here, so
``apsg.feature._container`` keeps working as a single import path.
"""

from apsg.feature._container._base import FeatureSet, G
from apsg.feature._container._cluster import ClusterSet
from apsg.feature._container._othersets import ArcSet, ConeSet
from apsg.feature._container._pairsets import FaultSet, PairSet
from apsg.feature._container._tensorsets import (
    EllipseSet,
    EllipsoidSet,
    OrientationTensor2Set,
    OrientationTensor3Set,
    Stress2Set,
    Stress3Set,
)
from apsg.feature._container._vectorsets import (
    Direction2Set,
    FoliationSet,
    LineationSet,
    Vector2Set,
    Vector3Set,
)

__all__ = (
    "ArcSet",
    "ClusterSet",
    "ConeSet",
    "Direction2Set",
    "EllipseSet",
    "EllipsoidSet",
    "FaultSet",
    "FeatureSet",
    "FoliationSet",
    "G",
    "LineationSet",
    "OrientationTensor2Set",
    "OrientationTensor3Set",
    "PairSet",
    "Stress2Set",
    "Stress3Set",
    "Vector2Set",
    "Vector3Set",
)
