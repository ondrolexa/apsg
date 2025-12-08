# -*- coding: utf-8 -*-

from apsg.config import apsg_conf
from apsg.feature import (
    ClusterSet as cluster,
)
from apsg.feature import (
    Cone as cone,
)
from apsg.feature import (
    ConeSet as coneset,
)
from apsg.feature import (
    DeformationGradient2 as defgrad2,
)
from apsg.feature import (
    DeformationGradient3 as defgrad,
)
from apsg.feature import (
    Direction as dir2,
)
from apsg.feature import (
    Direction2Set as dir2set,
)
from apsg.feature import (
    Ellipse as ellipse,
)
from apsg.feature import (
    EllipseSet as ellipseset,
)
from apsg.feature import (
    Ellipsoid as ellipsoid,
)
from apsg.feature import (
    EllipsoidSet as ellipsoidset,
)
from apsg.feature import (
    Fault as fault,
)
from apsg.feature import (
    FaultSet as faultset,
)
from apsg.feature import (
    Foliation as fol,
)
from apsg.feature import (
    FoliationSet as folset,
)
from apsg.feature import G
from apsg.feature import (
    Lineation as lin,
)
from apsg.feature import (
    LineationSet as linset,
)
from apsg.feature import (
    OrientationTensor2 as ortensor2,
)
from apsg.feature import (
    OrientationTensor2Set as ortensor2set,
)
from apsg.feature import (
    OrientationTensor3 as ortensor,
)
from apsg.feature import (
    OrientationTensor3Set as ortensorset,
)
from apsg.feature import (
    Pair as pair,
)
from apsg.feature import (
    PairSet as pairset,
)
from apsg.feature import (
    Stress2 as stress2,
)
from apsg.feature import (
    Stress3 as stress,
)
from apsg.feature import (
    Vector2Set as vec2set,
)
from apsg.feature import (
    Vector3Set as vecset,
)
from apsg.feature import (
    VelocityGradient2 as velgrad2,
)
from apsg.feature import (
    VelocityGradient3 as velgrad,
)
from apsg.math import (
    Matrix2 as matrix2,
)
from apsg.math import (
    Matrix3 as matrix,
)
from apsg.math import (
    Vector2 as vec2,
)
from apsg.math import (
    Vector3 as vec,
)
from apsg.plotting import (
    FlinnPlot,
    HsuPlot,
    RamsayPlot,
    RosePlot,
    StereoGrid,
    StereoNet,
    VollmerPlot,
    quicknet,
)

__all__ = (
    "apsg_conf",
    "vec",
    "vec2",
    "matrix",
    "matrix2",
    "dir2",
    "lin",
    "fol",
    "pair",
    "fault",
    "cone",
    "vecset",
    "vec2set",
    "dir2set",
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

__version__ = "1.3.7"
__author__ = "Ondrej Lexa"
__email__ = "lexa.ondrej@gmail.com"
