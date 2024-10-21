# -*- coding: utf-8 -*-

from apsg.math import (
    Vector3 as vec,
    Vector2 as vec2,
    Matrix2 as matrix2,
    Matrix3 as matrix,
)
from apsg.config import apsg_conf
from apsg.feature import (
    Lineation as lin,
    Foliation as fol,
    Pair as pair,
    Fault as fault,
    Cone as cone,
)
from apsg.feature import (
    Vector3Set as vecset,
    Vector2Set as vec2set,
    LineationSet as linset,
    FoliationSet as folset,
    PairSet as pairset,
    FaultSet as faultset,
    ConeSet as coneset,
    EllipseSet as ellipseset,
    EllipsoidSet as ellipsoidset,
    OrientationTensor2Set as ortensor2set,
    OrientationTensor3Set as ortensorset,
    ClusterSet as cluster,
)
from apsg.feature import (
    DeformationGradient3 as defgrad,
    VelocityGradient3 as velgrad,
    Stress3 as stress,
    Ellipsoid as ellipsoid,
    OrientationTensor3 as ortensor,
)
from apsg.feature import (
    DeformationGradient2 as defgrad2,
    VelocityGradient2 as velgrad2,
    Stress2 as stress2,
    Ellipse as ellipse,
    OrientationTensor2 as ortensor2,
)
from apsg.feature import G
from apsg.plotting import (
    StereoGrid,
    StereoNet,
    RosePlot,
    VollmerPlot,
    RamsayPlot,
    FlinnPlot,
    HsuPlot,
    quicknet,
)

__all__ = (
    "apsg_conf",
    "vec",
    "vec2",
    "matrix",
    "matrix2",
    "lin",
    "fol",
    "pair",
    "fault",
    "cone",
    "vecset",
    "vec2set",
    "linset",
    "folset",
    "pairset",
    "faultset",
    "coneset",
    "ellipseset",
    "ellipsoidset",
    "ortensor2set",
    "ortensorset",
    "cluster",
    "G",
    "defgrad",
    "velgrad",
    "stress",
    "ellipsoid",
    "ortensor",
    "defgrad2",
    "velgrad2",
    "stress2",
    "ellipse",
    "ortensor2",
    "StereoGrid",
    "StereoNet",
    "RosePlot",
    "VollmerPlot",
    "RamsayPlot",
    "FlinnPlot",
    "HsuPlot",
    "quicknet",
)

__version__ = "1.2.2"
__author__ = "Ondrej Lexa"
__email__ = "lexa.ondrej@gmail.com"
