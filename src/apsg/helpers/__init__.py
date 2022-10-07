# -*- coding: utf-8 -*-

from apsg.helpers._math import sind, cosd, tand, acosd, asind, atand, atan2d, sqrt2
from apsg.helpers._helper import eformat
from apsg.helpers._notation import (
    geo2vec_planar,
    geo2vec_linear,
    vec2geo_planar,
    vec2geo_linear,
)

__all__ = (
    "sind",
    "cosd",
    "tand",
    "acosd",
    "asind",
    "atand",
    "atan2d",
    "sqrt2",
    "is_like_vec3",
    "is_like_matrix3",
    "eformat",
    "geo2vec_planar",
    "geo2vec_linear",
    "vec2geo_planar",
    "vec2geo_linear",
)
