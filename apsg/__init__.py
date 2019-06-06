# -*- coding: utf-8 -*-


"""
Easily manage, analyze and visualize orientational structural geology data.
"""


from apsg.db import SDB
from apsg.core import (
    Vec3,
    Fol,
    Lin,
    Pair,
    Fault,
    Group,
    PairSet,
    FaultSet,
    Cluster,
    StereoGrid,
    G,
)
from apsg.setting import settings
from apsg.tensor import DefGrad, VelGrad, Stress, Tensor, Ortensor, Ellipsoid
from apsg.algebra.linear.helper import sind, cosd, tand, acosd, asind, atand, atan2d
from apsg.plotting import StereoNet, VollmerPlot, RamsayPlot, FlinnPlot, HsuPlot


__all__ = (
    "Vec3",
    "Fol",
    "Lin",
    "Pair",
    "Fault",
    "Group",
    "PairSet",
    "FaultSet",
    "Cluster",
    "StereoGrid",
    "G",
    "settings",
    "SDB",
    "DefGrad",
    "VelGrad",
    "Stress",
    "Tensor",
    "Ortensor",
    "Ellipsoid",
    "sind",
    "cosd",
    "tand",
    "acosd",
    "asind",
    "atand",
    "atan2d",
    "StereoNet",
    "VollmerPlot",
    "RamsayPlot",
    "FlinnPlot",
    "HsuPlot"
)


__version__ = "0.6.1"
__author__ = "Ondrej Lexa"
__email__ = "lexa.ondrej@gmail.com"
