"""Anisotropy of magnetic susceptibility (AMS): grain models and statistics.

Not imported by ``from apsg import *``; use ``from apsg.ams import *``.

Grain models (kinematic coefficient ``grain_K``, ``VelocityGradient3.pseudo``,
the March/Willis distribution ``ACG.from_displacement`` and the susceptibility
tensor ``ams_from_orientation``/``ams_from_grains``) predict the fabric of a
population of non-interacting grains in a homogeneous flow.
"""

from apsg.ams._model import (
    AMSEvolution,
    AMSModel,
    GrainPopulation,
    jeffery_evolution,
)
from apsg.ams._plots import (
    AMSPlot,
    AMSStereoPlot,
    BootstrapPlot,
    LFPlot,
    ModelPlot,
    PjTPlot,
    PTPlot,
)
from apsg.ams._readers import (
    from_dataframe,
    read_ams,
    read_anisoft_table,
    read_csv,
    read_sufar,
    to_dataframe,
)
from apsg.ams._specimen import AMSSite, AMSSpecimen
from apsg.feature import (
    TensorBootstrap,
    compare_tensor_means,
    tensor_bootstrap,
    tensor_error_bootstrap,
)
from apsg.feature._distributions import ACG
from apsg.feature._grainmodels import (
    ams_from_grains,
    ams_from_orientation,
    ams_from_rotations,
    grain_anisotropy,
    grain_K,
    grain_susceptibilities,
    jeffery_rotations,
    kinematic_K,
    orientation_tensor_from_ams,
    osborn_factors,
)

__all__ = (
    "ACG",
    "AMSEvolution",
    "AMSModel",
    "AMSPlot",
    "AMSSite",
    "AMSSpecimen",
    "AMSStereoPlot",
    "BootstrapPlot",
    "GrainPopulation",
    "LFPlot",
    "ModelPlot",
    "PTPlot",
    "PjTPlot",
    "TensorBootstrap",
    "ams_from_grains",
    "ams_from_orientation",
    "ams_from_rotations",
    "compare_tensor_means",
    "from_dataframe",
    "grain_K",
    "grain_anisotropy",
    "grain_susceptibilities",
    "jeffery_evolution",
    "jeffery_rotations",
    "kinematic_K",
    "orientation_tensor_from_ams",
    "osborn_factors",
    "read_ams",
    "read_anisoft_table",
    "read_csv",
    "read_sufar",
    "tensor_bootstrap",
    "tensor_error_bootstrap",
    "to_dataframe",
)
