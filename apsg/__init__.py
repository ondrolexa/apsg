# -*- coding: utf-8 -*-


from .core import (
    Vec3,
    Fol,
    Lin,
    Pair,
    Fault,
    Group,
    PairSet,
    FaultSet,
    Ortensor,
    Cluster,
    StereoGrid,
    G,
    settings,
)

from .db import SDB
from .tensors import DefGrad, VelGrad, Stress
from .helpers import sind, cosd, tand, acosd, asind, atand, atan2d
from .plotting import StereoNet, FabricPlot


__all__ = (
    "Vec3",
    "Fol",
    "Lin",
    "Pair",
    "Fault",
    "Group",
    "PairSet",
    "FaultSet",
    "Ortensor",
    "Cluster",
    "StereoGrid",
    "G",
    "settings",
    "SDB",
    "DefGrad",
    "VelGrad",
    "Stress",
    "sind",
    "cosd",
    "tand",
    "acosd",
    "asind",
    "atand",
    "atan2d",
    "StereoNet",
    "FabricPlot",
)

__version__ = "0.5.4"
__author__ = "Ondrej Lexa"
__email__ = "lexa.ondrej@gmail.com"
