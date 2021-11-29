# -*- coding: utf-8 -*-

from apsg.math import Vector3
from apsg.config import apsg_conf
from apsg.feature import Lineation, Foliation, Pair, Vector3Set, LineationSet, FoliationSet, G

def vec3(*args):
    return Vector3(*args)

def lin(*args):
    return Lineation(*args)

def fol(*args):
    return Foliation(*args)

def pair(*args):
    return Pair(*args)

__all__ = ('apsg_conf', 'vec3', 'lin', 'fol', 'pair', 'Vector3Set', 'LineationSet', 'FoliationSet', 'G')

__version__ = "1.0.0"
__author__ = "Ondrej Lexa"
__email__ = "lexa.ondrej@gmail.com"
