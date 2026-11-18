"""Paleomagnetic demagnetization data: specimens, samples and their plots.

Not imported by ``from apsg import *``; use ``from apsg.paleomag import *``.
"""

from apsg.paleomag._inclination import inclination_only_mean
from apsg.paleomag._plots import demag_plot, stereo_plot, zijderveld_plot
from apsg.paleomag._specimen import (
    Component,
    DemagStep,
    Sample,
    Specimen,
    bailey_halls_mean,
    bedding_vector,
    core_rotation,
    kirschvink_mean,
    lines_planes_mean,
    mcfadden_fold_test,
    untilt_rotation,
    vgp,
)

__all__ = (
    "Component",
    "DemagStep",
    "Sample",
    "Specimen",
    "bailey_halls_mean",
    "bedding_vector",
    "core_rotation",
    "demag_plot",
    "inclination_only_mean",
    "kirschvink_mean",
    "lines_planes_mean",
    "mcfadden_fold_test",
    "stereo_plot",
    "untilt_rotation",
    "vgp",
    "zijderveld_plot",
)
