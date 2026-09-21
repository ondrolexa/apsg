import sys

from apsg.feature._container import (
    ArcSet,
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
    Stress2Set,
    Stress3Set,
    Vector2Set,
    Vector3Set,
)
from apsg.feature._geodata import (
    Arc,
    Cone,
    Direction,
    Fault,
    Foliation,
    Lineation,
    Pair,
)
from apsg.feature._paleomag import Core
from apsg.feature._tensor2 import (
    DeformationGradient2,
    Ellipse,
    OrientationTensor2,
    Rotation2,
    Stress2,
    VelocityGradient2,
)
from apsg.feature._tensor3 import (
    DeformationGradient3,
    Ellipsoid,
    OrientationTensor3,
    Rotation3,
    Stress3,
    VelocityGradient3,
)
from apsg.math._matrix import Matrix2, Matrix3
from apsg.math._vector import Axial2, Axial3, Vector2, Vector3

__all__ = (
    "Arc",
    "ArcSet",
    "Axial2",
    "Axial3",
    "ClusterSet",
    "Cone",
    "ConeSet",
    "Core",
    "DeformationGradient2",
    "DeformationGradient3",
    "Direction",
    "Direction2Set",
    "Ellipse",
    "EllipseSet",
    "Ellipsoid",
    "EllipsoidSet",
    "Fault",
    "FaultSet",
    "FeatureSet",
    "Foliation",
    "FoliationSet",
    "G",
    "Lineation",
    "LineationSet",
    "Matrix2",
    "Matrix3",
    "OrientationTensor2",
    "OrientationTensor2Set",
    "OrientationTensor3",
    "OrientationTensor3Set",
    "Pair",
    "PairSet",
    "Rotation2",
    "Rotation3",
    "Stress2",
    "Stress2Set",
    "Stress3",
    "Stress3Set",
    "Vector2",
    "Vector2Set",
    "Vector3",
    "Vector3Set",
    "VelocityGradient2",
    "VelocityGradient3",
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
