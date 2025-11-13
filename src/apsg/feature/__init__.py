# -*- coding: utf-8 -*-
import sys

from apsg.feature._container import (
    ClusterSet,
    ConeSet,
    Direction2Set,
    EllipseSet,
    EllipsoidSet,
    FaultSet,
    FeatureSet,
    FoliationSet,
    G,
    LineationSet,
    OrientationTensor2Set,
    OrientationTensor3Set,
    PairSet,
    Vector2Set,
    Vector3Set,
)
from apsg.feature._geodata import Cone, Direction, Fault, Foliation, Lineation, Pair
from apsg.feature._paleomag import Core
from apsg.feature._tensor2 import (
    DeformationGradient2,
    Ellipse,
    OrientationTensor2,
    Stress2,
    VelocityGradient2,
)
from apsg.feature._tensor3 import (
    DeformationGradient3,
    Ellipsoid,
    OrientationTensor3,
    Stress3,
    VelocityGradient3,
)

__all__ = (
    "Direction",
    "Lineation",
    "Foliation",
    "Pair",
    "Fault",
    "Cone",
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
    "Vector2Set",
    "Direction2Set",
    "FeatureSet",
    "Vector3Set",
    "LineationSet",
    "FoliationSet",
    "PairSet",
    "FaultSet",
    "ConeSet",
    "EllipseSet",
    "EllipsoidSet",
    "OrientationTensor2Set",
    "OrientationTensor3Set",
    "G",
    "ClusterSet",
    "Core",
)


def feature_from_json(obj_json):
    dtype_cls = getattr(sys.modules[__name__], obj_json["datatype"])
    args = []
    for arg in obj_json["args"]:
        if isinstance(arg, dict):
            args.append([feature_from_json(jd) for jd in arg["collection"]])
        else:
            args.append(arg)
    kwargs = obj_json.get("kwargs", {})
    return dtype_cls(*args, **kwargs)
