import numpy as np

from apsg.config import apsg_conf
from apsg.math._vector import Vector3
from apsg.feature._geodata import Lineation, Foliation, Pair, Fault
from apsg.feature._container import (
    FeatureSet,
    Vector3Set,
    LineationSet,
    FoliationSet,
    PairSet,
    FaultSet,
)


class StereoNet_Artists:
    def update_kwargs(self, style):
        self.kwargs = apsg_conf[style].copy()
        self.kwargs["label"] = self.stereonet_method


class StereoNet_Point(StereoNet_Artists):
    def __init__(self, *args, **kwargs):
        self.stereonet_method = "_line"
        self.args = args
        self.parse_kwargs(kwargs)

    def parse_kwargs(self, kwargs):
        super().update_kwargs("default_point_kwargs")
        self.kwargs.update((k, kwargs[k]) for k in self.kwargs.keys() & kwargs.keys())
        if not isinstance(self.kwargs["label"], str):
            if len(self.args) == 1:
                self.kwargs["label"] = self.args[0].label()
            else:
                self.kwargs["label"] = f"Linear ({len(self.args)})"


class StereoNet_Pole(StereoNet_Artists):
    def __init__(self, *args, **kwargs):
        self.stereonet_method = "_line"
        self.args = args
        self.parse_kwargs(kwargs)

    def parse_kwargs(self, kwargs):
        super().update_kwargs("default_pole_kwargs")
        self.kwargs.update((k, kwargs[k]) for k in self.kwargs.keys() & kwargs.keys())
        if not isinstance(self.kwargs["label"], str):
            if len(self.args) == 1:
                self.kwargs["label"] = self.args[0].label()
            else:
                self.kwargs["label"] = f"Pole ({len(self.args)})"


class StereoNet_Vector(StereoNet_Artists):
    def __init__(self, *args, **kwargs):
        self.stereonet_method = "_vector"
        self.args = args
        self.parse_kwargs(kwargs)

    def parse_kwargs(self, kwargs):
        super().update_kwargs("default_vector_kwargs")
        self.kwargs.update((k, kwargs[k]) for k in self.kwargs.keys() & kwargs.keys())
        if not isinstance(self.kwargs["label"], str):
            if len(self.args) == 1:
                self.kwargs["label"] = self.args[0].label()
            else:
                self.kwargs["label"] = f"Vector ({len(self.args)})"


class StereoNet_Scatter(StereoNet_Artists):
    def __init__(self, *args, **kwargs):
        self.stereonet_method = "_scatter"
        self.args = args
        self.parse_kwargs(kwargs)

    def parse_kwargs(self, kwargs):
        super().update_kwargs("default_scatter_kwargs")
        self.kwargs.update((k, kwargs[k]) for k in self.kwargs.keys() & kwargs.keys())
        if not isinstance(self.kwargs["label"], str):
            if len(self.args) == 1:
                self.kwargs["label"] = self.args[0].label()
            else:
                self.kwargs["label"] = f"Scatter ({len(self.args)})"
        # parse size or color arguments to kwargs as list
        if self.kwargs["s"] is not None:
            self.kwargs["s"] = np.atleast_1d(self.kwargs["s"]).tolist()
            nof = np.vstack(self.args).shape[0]
            nos = len(self.kwargs["s"])
            if nof != nos:
                raise TypeError(
                    f"Number of sizes ({nos}) do not match number of features ({nof})"
                )
        if self.kwargs["c"] is not None:
            self.kwargs["c"] = np.atleast_1d(self.kwargs["c"]).tolist()
            nof = np.vstack(self.args).shape[0]
            noc = len(self.kwargs["c"])
            if np.vstack(self.args).shape[0] != len(self.kwargs["c"]):
                raise TypeError(
                    f"Number of colors ({noc}) do not match number of features ({nof})"
                )


class StereoNet_Great_Circle(StereoNet_Artists):
    def __init__(self, *args, **kwargs):
        self.stereonet_method = "_great_circle"
        self.args = args
        self.parse_kwargs(kwargs)

    def parse_kwargs(self, kwargs):
        super().update_kwargs("default_great_circle_kwargs")
        self.kwargs.update((k, kwargs[k]) for k in self.kwargs.keys() & kwargs.keys())
        if not isinstance(self.kwargs["label"], str):
            if len(self.args) == 1:
                self.kwargs["label"] = self.args[0].label()
            else:
                self.kwargs["label"] = f"Planar ({len(self.args)})"


class StereoNet_Cone(StereoNet_Artists):
    def __init__(self, *args, **kwargs):
        self.stereonet_method = "_cone"
        self.args = args
        self.parse_kwargs(kwargs)

    def parse_kwargs(self, kwargs):
        super().update_kwargs("default_cone_kwargs")
        self.kwargs.update((k, kwargs[k]) for k in self.kwargs.keys() & kwargs.keys())
        self.kwargs["angles"] = np.atleast_1d(kwargs["angles"]).tolist()
        nof = np.vstack(self.args).shape[0]
        noa = len(self.kwargs["angles"])
        if np.vstack(self.args).shape[0] != len(self.kwargs["angles"]):
            raise TypeError(
                f"Number of angles ({noa}) do not match number of features ({nof})"
            )
        if not isinstance(self.kwargs["label"], str):
            if len(self.args) == 1:
                if issubclass(type(self.args[0]), Vector3):
                    self.kwargs[
                        "label"
                    ] = f"Cone {self.args[0].label()} ({self.kwargs['angles'][0]})"
                else:
                    self.kwargs["label"] = f"Cones ({len(self.args[0])})"
            else:
                self.kwargs["label"] = f"Cones ({len(self.args)})"


class StereoNet_Pair(StereoNet_Artists):
    def __init__(self, *args, **kwargs):
        self.stereonet_method = "_pair"
        self.args = args
        self.parse_kwargs(kwargs)

    def parse_kwargs(self, kwargs):
        super().update_kwargs("default_pair_kwargs")
        self.kwargs.update((k, kwargs[k]) for k in self.kwargs.keys() & kwargs.keys())
        if not isinstance(self.kwargs["label"], str):
            if len(self.args) == 1:
                self.kwargs["label"] = self.args[0].label()
            else:
                self.kwargs["label"] = f"Pair ({len(self.args)})"


class StereoNet_Fault(StereoNet_Artists):
    def __init__(self, *args, **kwargs):
        self.stereonet_method = "_fault"
        self.args = args
        self.parse_kwargs(kwargs)

    def parse_kwargs(self, kwargs):
        super().update_kwargs("default_fault_kwargs")
        self.kwargs.update((k, kwargs[k]) for k in self.kwargs.keys() & kwargs.keys())
        if not isinstance(self.kwargs["label"], str):
            if len(self.args) == 1:
                self.kwargs["label"] = self.args[0].label()
            else:
                self.kwargs["label"] = f"Fault ({len(self.args)})"


class StereoNet_Arrow(StereoNet_Artists):
    def __init__(self, *args, **kwargs):
        self.stereonet_method = "_arrow"
        self.args = args[:2]  # take max 2 args
        self.parse_kwargs(kwargs)

    def parse_kwargs(self, kwargs):
        super().update_kwargs("default_quiver_kwargs")
        self.kwargs.update((k, kwargs[k]) for k in self.kwargs.keys() & kwargs.keys())
        if not isinstance(self.kwargs["label"], str):
            if len(self.args) == 1:
                self.kwargs["label"] = self.args[0].label()
            else:
                self.kwargs["label"] = f"Fault ({len(self.args)})"
        self.kwargs = (
            np.copysign(1, np.atleast_1d(kwargs.get("sense", 1))).astype(int).tolist()
        )


class StereoNet_Contourf(StereoNet_Artists):
    def __init__(self, *args, **kwargs):
        self.stereonet_method = "_contourf"
        self.args = args[:1]  # take only first arg
        self.parse_kwargs(kwargs)

    def parse_kwargs(self, kwargs):
        super().update_kwargs("default_contourf_kwargs")
        self.kwargs.update((k, kwargs[k]) for k in self.kwargs.keys() & kwargs.keys())
        if not isinstance(self.kwargs["label"], str):
            self.kwargs["label"] = self.args[0].label()


class ArtistFactory:
    @staticmethod
    def create_point(*args, **kwargs):
        if all([issubclass(type(arg), (Vector3, Vector3Set)) for arg in args]):
            return StereoNet_Point(*args, **kwargs)
        else:
            raise TypeError("Not valid arguments for Stereonet point")

    @staticmethod
    def create_pole(*args, **kwargs):
        if all([issubclass(type(arg), (Foliation, FoliationSet)) for arg in args]):
            return StereoNet_Pole(*args, **kwargs)
        else:
            raise TypeError("Not valid arguments for Stereonet pole")

    @staticmethod
    def create_scatter(*args, **kwargs):
        if all([issubclass(type(arg), (Vector3, Vector3Set)) for arg in args]):
            return StereoNet_Scatter(*args, **kwargs)
        else:
            raise TypeError("Not valid arguments for Stereonet scatter")

    @staticmethod
    def create_vector(*args, **kwargs):
        if all([issubclass(type(arg), (Vector3, Vector3Set)) for arg in args]):
            return StereoNet_Vector(*args, **kwargs)
        else:
            raise TypeError("Not valid arguments for Stereonet vector")

    @staticmethod
    def create_great_circle(*args, **kwargs):
        if all([issubclass(type(arg), (Foliation, FoliationSet)) for arg in args]):
            return StereoNet_Great_Circle(*args, **kwargs)
        else:
            raise TypeError("Not valid arguments for Stereonet great circle")

    @staticmethod
    def create_cone(*args, **kwargs):
        if all([issubclass(type(arg), (Vector3, Vector3Set)) for arg in args]):
            if "angles" in kwargs:
                return StereoNet_Cone(*args, **kwargs)
            else:
                raise TypeError("Keyword argument angles must be provided.")
        else:
            raise TypeError("Not valid arguments for Stereonet cone")

    @staticmethod
    def create_pair(*args, **kwargs):
        if all([issubclass(type(arg), (Pair, PairSet)) for arg in args]):
            return StereoNet_Pair(*args, **kwargs)
        else:
            raise TypeError("Not valid arguments for Stereonet pair")

    @staticmethod
    def create_fault(*args, **kwargs):
        if all([issubclass(type(arg), (Fault, FaultSet)) for arg in args]):
            return StereoNet_Fault(*args, **kwargs)
        else:
            raise TypeError("Not valid arguments for Stereonet fault")

    @staticmethod
    def create_arrow(*args, **kwargs):
        if all([issubclass(type(arg), (Vector3, Vector3Set)) for arg in args[:2]]):
            return StereoNet_Arrow(*args, **kwargs)
        else:
            raise TypeError("Not valid arguments for Stereonet arrow")

    @staticmethod
    def create_contourf(*args, **kwargs):
        if issubclass(type(args[0]), Vector3Set):
            return StereoNet_Contourf(*args, **kwargs)
        else:
            raise TypeError("Not valid arguments for Stereonet contourf")
