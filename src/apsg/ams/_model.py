"""Forward models of the AMS of grains in a flow."""

from dataclasses import dataclass

import numpy as np

from apsg.feature import Ellipsoid, EllipsoidSet, VelocityGradient3
from apsg.feature._distributions import ACG
from apsg.feature._grainmodels import (
    ams_from_grains,
    ams_from_rotations,
    grain_K,
    grain_susceptibilities,
    jeffery_rotations,
)
from apsg.feature._statistics import jelinek_shape

__all__ = ("AMSEvolution", "AMSModel", "GrainPopulation", "jeffery_evolution")


@dataclass(frozen=True)
class GrainPopulation:
    """Population of uniaxial grains of one shape.

    Attributes:
        kind: ``"prolate"`` (long axis is the unique axis) or ``"oblate"``
            (the pole of the flat grain).
        ratio: axial ratio of the spheroid, ``>= 1``.
        K: susceptibility perpendicular to the unique axis.
        delta: ``K_axis - K_transverse``; positive for prolate (default 0.3),
            negative for oblate (default -0.3) grains when None.
        weight: weight of the population in the rock.
    """

    kind: str
    ratio: float
    K: float = 1.0
    delta: float | None = None
    weight: float = 1.0

    def __post_init__(self):
        if self.kind not in ("prolate", "oblate"):
            raise ValueError(f"kind must be 'prolate' or 'oblate', got {self.kind!r}")
        grain_K(self.kind, self.ratio)
        if self.delta is None:
            object.__setattr__(self, "delta", 0.3 if self.kind == "prolate" else -0.3)
        if (self.delta > 0) != (self.kind == "prolate") and self.delta != 0:
            raise ValueError("delta must be positive for prolate, negative for oblate")
        if self.weight < 0 or self.K <= 0:
            raise ValueError("weight must be non-negative and K positive")

    @classmethod
    def from_susceptibility(cls, kind, ratio, k, weight=1.0):
        """Population of grains of intrinsic susceptibility `k` and shape anisotropy.

        The grain susceptibilities follow from the demagnetising factors
        (``grain_susceptibilities``), so the grain anisotropy is controlled by
        its shape, as for magnetite.
        """
        Ka, Kb = grain_susceptibilities(k, ratio, kind)
        return cls(kind, ratio, K=Kb, delta=Ka - Kb, weight=weight)

    @property
    def kinematic_K(self):
        """float: kinematic coefficient of the grain shape (`grain_K`)."""
        return grain_K(self.kind, self.ratio)


@dataclass(frozen=True, eq=False)
class AMSEvolution:
    """AMS tensors of a model at a sequence of times.

    Attributes:
        times: ``(n,)`` times (or strains).
        tensors: ``(n, 3, 3)`` susceptibility tensors.
    """

    times: np.ndarray
    tensors: np.ndarray

    def __len__(self):
        return len(self.times)

    def __repr__(self):
        return f"AMSEvolution({len(self)} steps, t = {self.times[0]:g} to {self.times[-1]:g})"

    def ellipsoids(self):
        """Return the tensors as `EllipsoidSet`."""
        return EllipsoidSet([Ellipsoid(t) for t in self.tensors], name="model")

    def table(self):
        """Return a `DataFrame` of the time, the parameters of Jelinek (1981) and
        the plunge and azimuth of ``k1`` and ``k3``."""
        import pandas as pd

        rows = []
        for t, T in zip(self.times, self.tensors, strict=True):
            e = Ellipsoid(T)
            lins = e.eigenlins()
            row = {"time": float(t), **jelinek_shape(np.linalg.eigvalsh(T))}
            iso = (
                row["P"] - 1 < 1e-9
            )  # axes and shape of an isotropic tensor are undefined
            if iso:
                row["T"] = row["U"] = np.nan
            for i in (0, 2):
                azi, inc = lins[i].geo
                row[f"k{i + 1}_azi"] = np.nan if iso else float(azi)
                row[f"k{i + 1}_inc"] = np.nan if iso else float(inc)
            rows.append(row)
        return pd.DataFrame(rows)

    def switch(self):
        """Return the first time at which a horizontal ``k1`` becomes steeper than
        45 degrees (the switch of the lineation), or None."""
        inc = self.table()["k1_inc"].abs().to_numpy()
        flat = np.nonzero(inc <= 45)[0]
        if len(flat) == 0:
            return None
        steep = np.nonzero(inc[flat[0] :] > 45)[0]
        return float(self.times[flat[0] + steep[0]]) if len(steep) else None


class AMSModel:
    """AMS of non-interacting grains carried by a homogeneous flow.

    Each population turns like a material line in the pseudo-velocity-gradient
    ``K E + W`` of its shape (Willis, 1977), and starting from a uniform
    distribution of axes its orientation is an angular central Gaussian
    (``ACG.from_displacement``); the susceptibility tensor of the rock is the
    weighted sum of the populations (Jezek and Hrouda, 2002).

    Args:
        flow: ``VelocityGradient3`` or a 3x3 array, the velocity gradient with
            time in the units of the model (``defgrad(time)`` integrates it).
        populations: ``GrainPopulation`` or a sequence of them.

    Examples:
        >>> L = VelocityGradient3([[0, 1, 0], [0, 0, 0], [0, 0, 0]])  # doctest: +SKIP
        >>> model = AMSModel(L, GrainPopulation("prolate", 3))  # doctest: +SKIP
        >>> model.evolve(np.linspace(0.1, 10, 50)).table()  # doctest: +SKIP
    """

    def __init__(self, flow, populations):
        self.flow = (
            flow if isinstance(flow, VelocityGradient3) else VelocityGradient3(flow)
        )
        if isinstance(populations, GrainPopulation):
            populations = [populations]
        self.populations = list(populations)
        if not self.populations or not all(
            isinstance(p, GrainPopulation) for p in self.populations
        ):
            raise TypeError("populations must be GrainPopulation objects")
        if sum(p.weight for p in self.populations) <= 0:
            raise ValueError("weights must have a positive sum")

    def fabric(self, time, population=0):
        """Return the `ACG` distribution of the grain axes of a population at `time`."""
        p = self.populations[population]
        D = np.asarray(self.flow.pseudo(p.kinematic_K).defgrad(time))
        return ACG.from_displacement(D, "lines" if p.kind == "prolate" else "poles")

    def tensor(self, time):
        """Return the susceptibility tensor of the rock at `time` as `Ellipsoid`."""
        return ams_from_grains(
            (p.weight, self.fabric(time, i), p.K, p.delta)
            for i, p in enumerate(self.populations)
        )

    def evolve(self, times):
        """Return the `AMSEvolution` at the given times."""
        t = np.asarray(times, dtype=float)
        if t.ndim != 1 or len(t) == 0 or not np.all(np.isfinite(t)) or np.any(t < 0):
            raise ValueError(
                "times must be a non-empty 1-D array of non-negative values"
            )
        return AMSEvolution(t, np.array([np.asarray(self.tensor(x)) for x in t]))


def jeffery_evolution(
    flow, semi_axes, susceptibilities, times, n=500, rng=None, rtol=1e-9
):
    """AMS of triaxial ellipsoids rotating in a flow according to Jeffery (1922).

    `n` grains start with uniformly random orientations and are carried by
    the flow (``jeffery_rotations``); the susceptibility tensor is the mean of
    the grains', each having the principal susceptibilities
    `susceptibilities` along its axes. Unlike `AMSModel` it works for any
    three semi-axes, but the result is a sample, so use a few hundred grains
    and a seed. Triaxial grains in shear do not return to their orientation
    after a fixed strain (Hrouda and Jezek, 1999).

    Args:
        flow: ``VelocityGradient3`` or 3x3 array.
        semi_axes: three positive semi-axes (axis 1 longest, by convention).
        susceptibilities: principal susceptibilities along the three axes.
        times: increasing times, starting at 0 (undeformed).
        n: number of grains.
        rng: None, an integer seed or a ``numpy.random.Generator``.
        rtol: tolerance of the integrator.

    Returns:
        AMSEvolution: tensors at `times`.
    """
    from scipy.spatial.transform import Rotation

    from apsg.feature._distributions._base import as_rng

    t = np.asarray(times, dtype=float)
    rng = as_rng(rng)
    seed = int(rng.integers(0, 2**31 - 1))
    R0 = Rotation.random(int(n), random_state=seed).as_matrix()
    R = jeffery_rotations(np.asarray(flow), semi_axes, R0, t, rtol=rtol)
    return AMSEvolution(
        t, np.array([np.asarray(ams_from_rotations(Rt, susceptibilities)) for Rt in R])
    )
