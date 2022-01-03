# -*- coding: utf-8 -*-

from apsg.math import Vector3 as vec3, Vector2 as vec2
from apsg.config import apsg_conf
from apsg.feature import (
    Lineation as lin,
    Foliation as fol,
    Pair as pair,
    Fault as fault,
)
from apsg.feature import (
    Vector3Set as vec3set,
    Vector2Set as vec2set,
    LineationSet as linset,
    FoliationSet as folset,
    PairSet as pairset,
    FaultSet as faultset,
    EllipsoidSet as ellipsoidset,
    OrientationTensor3Set as ortensorset,
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
    quicknet
)


__all__ = (
    "apsg_conf",
    "vec3",
    "vec2",
    "lin",
    "fol",
    "pair",
    "fault",
    "vec3set",
    "vec2set",
    "linset",
    "folset",
    "pairset",
    "faultset",
    "ellipsoidset",
    "ortensorset",
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
    "quicknet"
)

__version__ = "1.0.0"
__author__ = "Ondrej Lexa"
__email__ = "lexa.ondrej@gmail.com"
