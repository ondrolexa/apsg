# -*- coding: utf-8 -*-

from apsg.feature._geodata import Lineation, Foliation, Pair, Fault
from apsg.feature._container import (
    FeatureSet,
    Vector2Set,
    Vector3Set,
    LineationSet,
    FoliationSet,
    PairSet,
    FaultSet,
    EllipsoidSet,
    OrientationTensor3Set,
    G,
)
from apsg.feature._tensor3 import (
    DeformationGradient3,
    VelocityGradient3,
    Stress3,
    Ellipsoid,
    OrientationTensor3,
)
from apsg.feature._tensor2 import (
    DeformationGradient2,
    VelocityGradient2,
    Stress2,
    Ellipse,
    OrientationTensor2,
)
from apsg.feature._paleomag import Core

__all__ = (
    "Lineation",
    "Foliation",
    "Pair",
    "Fault",
    "FeatureSet",
    "Vector3Set",
    "LineationSet",
    "FoliationSet",
    "PairSet",
    "FaultSet",
    "EllipsoidSet",
    "G",
    "DeformationGradient3",
    "VelocityGradient3",
    "Stress3",
    "Ellipsoid",
    "OrientationTensor3",
    "DeformationGradient2",
    "VelocityGradient2",
    "Stress2",
    "Ellipse",
    "OrientationTensor2",
    "Core",
)
