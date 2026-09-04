"""Stereonet rendering engine (projection math, registered matplotlib
projections, density grid) -- an internal implementation detail, not a
public API of this subpackage.

``apsg.plotting.StereoNet``/``apsg.plotting.StereoGrid`` are the public,
config-driven, feature-type-aware entry points built on top of this engine;
nothing here understands ``apsg_conf`` or apsg feature types directly.
"""

from ._axes import SchmidtNetAxes, WulffNetAxes
from ._stereogrid import StereoGrid as _EngineStereoGrid
from ._transforms import rotation_from_axis_angle

__all__ = [
    "SchmidtNetAxes",
    "WulffNetAxes",
    "_EngineStereoGrid",
    "rotation_from_axis_angle",
]
