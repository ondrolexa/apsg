# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt

from .core import (Vec3, Fol, Lin, Pair, Fault,
                   Group, PairSet, FaultSet,
                   Ortensor, Cluster, StereoGrid, G)
from .plotting import StereoNet, FabricPlot
from .tensors import DefGrad, VelGrad, Stress
from .db import SDB
from .helpers import sind, cosd, tand, acosd, asind, atand, atan2d

__author__ = 'Ondrej Lexa'
__email__ = 'lexa.ondrej@gmail.com'
