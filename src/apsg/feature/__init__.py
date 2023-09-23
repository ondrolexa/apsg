# -*- coding: utf-8 -*-
import sys
from apsg.feature._geodata import Lineation, Foliation, Pair, Fault, Cone
from apsg.feature._container import (
    FeatureSet,
    Vector2Set,
    Vector3Set,
    LineationSet,
    FoliationSet,
    PairSet,
    FaultSet,
    ConeSet,
    EllipseSet,
    EllipsoidSet,
    OrientationTensor2Set,
    OrientationTensor3Set,
    G,
    ClusterSet,
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
