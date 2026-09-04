import pickle
import warnings

import numpy as np

from apsg.config import apsg_conf
from apsg.feature import feature_from_json
from apsg.feature._geodata import Lineation
from apsg.math._vector import Vector3
from apsg.plotting._stereo_engine._stereogrid import StereoGrid as _EngineStereoGrid


def _forward(engine_attr, doc=None):
    """A read-only property reading ``self._engine.<engine_attr>``."""
    return property(lambda self: getattr(self._engine, engine_attr), doc=doc)


def _forward_rw(engine_attr, doc=None):
    """A read/write property proxying ``self._engine.<engine_attr>``."""
    return property(
        lambda self: getattr(self._engine, engine_attr),
        lambda self, value: setattr(self._engine, engine_attr, value),
        doc=doc,
    )


class StereoGrid:
    """
    The class to store values with associated uniformly positions.

    ``StereoGrid`` is used to calculate continous functions on sphere e.g. density
    distribution. Rendering (contouring) is done by ``StereoNet.contour``, which can
    draw an existing, already-populated grid (e.g. built via ``apply_func``/
    ``angmech``) directly.

    Keyword Args:
        type (str): Type of contouring grid "gss" or "sfs". Default from
            ``apsg_conf.stereogrid.type`` ("gss")
        n (int): Number of counting points in grid. Default from
            ``apsg_conf.stereogrid.n`` (3000)

    Note: Euclidean norms are used as weights. Normalize data if you dont want to use
    weigths.

    """

    def __init__(self, **kwargs):
        self._kwargs = apsg_conf.stereogrid.copy()
        self._kwargs.update((k, kwargs[k]) for k in self._kwargs.keys() & kwargs.keys())
        self.grid_n = self._kwargs["n"]
        self.grid_type = self._kwargs["type"]
        self._engine = _EngineStereoGrid(grid_n=self.grid_n, grid_type=self.grid_type)
        # remembers the inputs+params of the last calculate_density/angmech call,
        # so to_json can serialize and recompute them on from_json rather than
        # persisting the computed `values` array (see to_json/from_json below)
        self._last_calculation = None

    grid = _forward("grid", doc="The (grid_n, 3) NED unit-vector counting grid.")
    values = _forward_rw("values", doc="The (grid_n,) computed values.")
    calculated = _forward_rw("calculated")
    features = _forward("features")

    def __repr__(self):
        if self.calculated:
            info = (
                f"\nMaximum: {self.max():.4f} at {self.max_at()}"
                + f"\nMinimum: {self.min():.4f} at {self.min_at()}"
            )
        else:
            info = ""
        return f"StereoGrid ({self.grid_type}, {self.grid_n} points)" + info

    def min(self):
        """Returns minimum value of the grid."""
        return float(self._engine.min())

    def max(self):
        """Returns maximum value of the grid."""
        return float(self._engine.max())

    def min_at(self):
        """Returns position of minimum value of the grid as ``Lineation``."""
        return Lineation(self._engine.min_at())

    def max_at(self):
        """Returns position of maximum value of the grid as ``Lineation``."""
        return Lineation(self._engine.max_at())

    def calculate_density(self, features, **kwargs):
        """Calculate density distribution of vectors from ``FeatureSet`` object.

        Both "kamb" and "sph" report the *same* statistic: standard deviations
        above the value expected under a uniform (random) distribution -- i.e.
        "level=3" means "3 sigma", regardless of which method computed it.

        Args:
            method (str): "kamb" for modified Kamb contouring technique with
                exponential smoothing or "sph" for spherical harmonics method.
                Default "sph"
            n_max (int): maximum harmonic degree i.e. the angular resolution. Must
                be even number (for "sph" method). Default is derived from `sigma`
                and the sample size.
            sigma (float): controls how much to smooth, for either method
                (Kamb 1959, Vollmer 1995). Default 3.
            trimzero (bool): If True, zero contour is not drawn. Default True

        """
        method = kwargs.get("method", "sph")
        sigma = kwargs.get("sigma", None)
        n_max = kwargs.get("n_max", None)
        trimzero = kwargs.get("trimzero", True)
        self._engine.calculate_density(
            features, method=method, sigma=sigma, n_max=n_max, trimzero=trimzero
        )
        self._last_calculation = {
            "op": "calculate_density",
            "data": features,
            "kwargs": {
                "method": method,
                "sigma": sigma,
                "n_max": n_max,
                "trimzero": trimzero,
            },
        }

    def apply_func(self, func, *args, **kwargs):
        """Calculate values of user-defined function on sphere.

        Function must accept Vector3 like (or 3 elements array)
        as first argument and return scalar value.

        Note: values computed via ``apply_func`` cannot be serialized (the
        callable itself isn't saved) -- ``to_json``/``save`` will warn and skip
        persisting them; call ``apply_func`` again after ``from_json``/``load``.

        Args:
            func (function): function used to calculate values
            *args: passed to function func as args
            **kwargs: passed to function func as kwargs

        """

        def wrapped(v, *a, **kw):
            return func(Vector3(v), *a, **kw)

        self._engine.apply_func(wrapped, *args, **kwargs)
        self._last_calculation = "apply_func"

    def angmech(self, faults, **kwargs):
        """Implementation of Angelier-Mechler dihedra method

        Args:
            faults (FaultSet): ``FaultSet`` of data.

        Keyword Args:
            method (str): 'probability' or 'classic'. Classic method assigns +/-1
                to individual positions, while 'probability' returns maximum
                likelihood estimate.

        """
        method = kwargs.get("method", "classic")
        self._engine.angmech(
            np.asarray(faults.fvec), np.asarray(faults.lvec), method=method
        )
        self._last_calculation = {
            "op": "angmech",
            "data": faults,
            "kwargs": {"method": method},
        }

    def _data_to_json(self, data):
        if hasattr(data, "to_json"):
            return {"kind": "feature", "data": data.to_json()}
        return {"kind": "array", "data": np.asarray(data).tolist()}

    def _data_from_json(self, obj):
        if obj["kind"] == "feature":
            return feature_from_json(obj["data"])
        return np.asarray(obj["data"])

    def to_json(self):
        """Return ``StereoGrid`` as a JSON-compatible dict.

        Serializes constructor kwargs plus the *inputs and parameters* of the
        last ``calculate_density``/``angmech`` call (not the computed ``values``
        array) -- ``from_json`` rebuilds the grid and recomputes them, matching
        how ``StereoNet`` itself never persists computed contour data.
        """
        calculation = None
        if isinstance(self._last_calculation, dict):
            calculation = {
                "op": self._last_calculation["op"],
                "data": self._data_to_json(self._last_calculation["data"]),
                "kwargs": self._last_calculation["kwargs"],
            }
        elif self._last_calculation == "apply_func":
            warnings.warn(
                "StereoGrid values were computed via apply_func(), which uses "
                "an arbitrary Python callable that cannot be serialized -- the "
                "computed values are not saved. Call apply_func() again after "
                "from_json()/load().",
                UserWarning,
                stacklevel=2,
            )
        return dict(
            kwargs=dict(n=self.grid_n, type=self.grid_type),
            calculation=calculation,
        )

    @classmethod
    def from_json(cls, json_dict):
        """Create ``StereoGrid`` from a JSON-compatible dict (see ``to_json``)."""
        grid = cls(**json_dict["kwargs"])
        calculation = json_dict.get("calculation")
        if calculation is not None:
            data = grid._data_from_json(calculation["data"])
            if calculation["op"] == "calculate_density":
                grid.calculate_density(data, **calculation["kwargs"])
            elif calculation["op"] == "angmech":
                grid.angmech(data, **calculation["kwargs"])
        return grid

    def save(self, filename):
        """
        Save StereoGrid to pickle file

        Args:
            filename (str): name of pickle file
        Returns:
            None: The grid is serialized and written to a pickle file.
        """
        with open(filename, "wb") as f:
            pickle.dump(self.to_json(), f, pickle.HIGHEST_PROTOCOL)

    @classmethod
    def load(cls, filename):
        """
        Load StereoGrid from pickle file

        Args:
            filename (str): name of pickle file
        Returns:
            StereoGrid: Loaded grid instance from pickle file.
        """
        with open(filename, "rb") as f:
            data = pickle.load(f)
        return cls.from_json(data)
