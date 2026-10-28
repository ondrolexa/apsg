"""
Probability distributions of orientations, separated by dimension (circle,
sphere, pairs, rotations) and symmetry (vectorial, axial), with a common
interface for density, fitting and sampling (``OrientationDistribution``).
"""

from apsg.feature._distributions._axial import (
    ACG,
    AxialUniform3,
    Bingham,
    Watson,
    axial_cross_moments,
    bingham_ellipse,
)
from apsg.feature._distributions._base import (
    ConfidenceRegion,
    FitInfo,
    OrientationDistribution,
    as_rng,
    get_distribution,
    legacy_rng,
)
from apsg.feature._distributions._circular import (
    AxialUniform2,
    AxialVonMises,
    Cardioid,
    JonesPewsey,
    Uniform2,
    VonMises,
    WrappedCauchy,
    WrappedNormal,
    dispersion_interval,
    upton_interval,
    vonmises_kde,
)
from apsg.feature._distributions._frames import (
    Lisle,
    lisle_ellipses,
    lisle_moments,
    pair_arrays,
)
from apsg.feature._distributions._rotations import (
    MatrixFisher,
    matrix_fisher_log_norm,
    proper_svd,
    rotation_arrays,
)
from apsg.feature._distributions._spherical import (
    ESAG,
    Kent,
    Uniform3,
    VonMisesFisher,
    fisher_cone,
    mean_direction_ellipse,
)

__all__ = (
    "ACG",
    "ESAG",
    "AxialUniform2",
    "AxialUniform3",
    "AxialVonMises",
    "Bingham",
    "Cardioid",
    "ConfidenceRegion",
    "FitInfo",
    "JonesPewsey",
    "Kent",
    "Lisle",
    "MatrixFisher",
    "OrientationDistribution",
    "Uniform2",
    "Uniform3",
    "VonMises",
    "VonMisesFisher",
    "Watson",
    "WrappedCauchy",
    "WrappedNormal",
    "as_rng",
    "axial_cross_moments",
    "bingham_ellipse",
    "dispersion_interval",
    "fisher_cone",
    "get_distribution",
    "legacy_rng",
    "lisle_ellipses",
    "lisle_moments",
    "matrix_fisher_log_norm",
    "mean_direction_ellipse",
    "pair_arrays",
    "proper_svd",
    "rotation_arrays",
    "upton_interval",
    "vonmises_kde",
)
