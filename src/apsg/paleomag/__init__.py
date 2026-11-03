"""Paleomagnetic demagnetization data: specimens, samples and their plots.

Not imported by ``from apsg import *``; use ``from apsg.paleomag import *``.
"""

from apsg.paleomag._plots import demag_plot, stereo_plot, zijderveld_plot
from apsg.paleomag._specimen import (
    Component,
    DemagStep,
    Sample,
    Specimen,
    bedding_vector,
    core_rotation,
    lines_planes_mean,
    untilt_rotation,
    vgp,
)

__all__ = (
    "Component",
    "DemagStep",
    "Sample",
    "Specimen",
    "bedding_vector",
    "core_rotation",
    "demag_plot",
    "lines_planes_mean",
    "stereo_plot",
    "untilt_rotation",
    "vgp",
    "zijderveld_plot",
)
