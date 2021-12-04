# -*- coding: utf-8 -*-

from apsg.math import Vector3
from apsg.config import apsg_conf
from apsg.feature import Lineation, Foliation, Pair, Fault
from apsg.feature import Vector3Set, LineationSet, FoliationSet, G, PairSet, FaultSet
from apsg.feature import DefGrad3, VelGrad3, Stress3, Ellipsoid, Ortensor3, StereoGrid
from apsg.plotting import StereoNet


def vec3(*args):
    return Vector3(*args)


def lin(*args):
    return Lineation(*args)


def fol(*args):
    return Foliation(*args)


def pair(*args):
    return Pair(*args)


def fault(*args):
    return Fault(*args)


__all__ = (
    "apsg_conf",
    "vec3",
    "lin",
    "fol",
    "pair",
    "fault",
    "Vector3Set",
    "LineationSet",
    "FoliationSet",
    "G",
    "DefGrad3",
    "VelGrad3",
    "Stress3",
    "Ellipsoid",
    "Ortensor3",
    "StereoGrid",
    "StereoNet",
)

__version__ = "1.0.0"
__author__ = "Ondrej Lexa"
__email__ = "lexa.ondrej@gmail.com"
