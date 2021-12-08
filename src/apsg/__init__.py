# -*- coding: utf-8 -*-

from apsg.math import Vector3 as vec3
from apsg.config import apsg_conf
from apsg.feature import (
    Lineation as lin,
    Foliation as fol,
    Pair as pair,
    Fault as fault,
)
from apsg.feature import (
    Vector3Set as vecset,
    LineationSet as linset,
    FoliationSet as folset,
    PairSet as pairset,
    FaultSet as faultset,
)
from apsg.feature import (
    DeformationGradient3 as defgrad,
    VelocityGradient3 as velgrad,
    Stress3 as stress,
    Ellipsoid as ellipsoid,
    OrientationTensor3 as ortensor,
)
from apsg.feature import G
from apsg.plotting import StereoGrid, StereoNet


__all__ = (
    "apsg_conf",
    "vec3",
    "lin",
    "fol",
    "pair",
    "fault",
    "vecset",
    "linset",
    "folset",
    "pairset",
    "faultset",
    "G",
    "defgrad",
    "velgrad",
    "stress",
    "ellipsoid",
    "ortensor",
    "StereoGrid",
    "StereoNet",
)

__version__ = "1.0.0"
__author__ = "Ondrej Lexa"
__email__ = "lexa.ondrej@gmail.com"
