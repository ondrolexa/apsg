import numpy as np

from apsg.config import apsg_conf
from apsg.math._vector import Vector3
from apsg.feature._geodata import Foliation, Pair, Fault, Cone
from apsg.feature._tensor3 import Tensor3, Ellipsoid
from apsg.feature._container import (
    Vector3Set,
    Vector2Set,
    FoliationSet,
    PairSet,
    FaultSet,
    ConeSet,
    EllipsoidSet,
)

# StereoNet


class StereoNet_Artists:
    def __init__(self, factory, *args, **kwargs):
        self.factory = factory

    def update_kwargs(self, style):
        self.kwargs = apsg_conf[style].copy()
        self.kwargs["label"] = self.stereonet_method

    def to_json(self):
        return dict(
            factory=self.factory,
            stereonet_method=self.stereonet_method,
            args=tuple([obj.to_json() for obj in self.args]),
            kwargs=self.kwargs.copy(),
        )


class StereoNet_Point(StereoNet_Artists):
    def __init__(self, factory, *args, **kwargs):
        super().__init__(factory, *args, **kwargs)
        self.stereonet_method = "_line"
        self.args = args
        self.parse_kwargs(kwargs)

    def parse_kwargs(self, kwargs):
        super().update_kwargs("stereonet_default_point_kwargs")
        self.kwargs.update((k, kwargs[k]) for k in self.kwargs.keys() & kwargs.keys())
        if not isinstance(self.kwargs["label"], str):
            if len(self.args) == 1:
                self.kwargs["label"] = self.args[0].label()
            else:
                self.kwargs["label"] = f"Linear ({len(self.args)})"


class StereoNet_Pole(StereoNet_Artists):
    def __init__(self, factory, *args, **kwargs):
        super().__init__(factory, *args, **kwargs)
        self.stereonet_method = "_line"
        self.args = args
        self.parse_kwargs(kwargs)

    def parse_kwargs(self, kwargs):
        super().update_kwargs("stereonet_default_pole_kwargs")
        self.kwargs.update((k, kwargs[k]) for k in self.kwargs.keys() & kwargs.keys())
        if not isinstance(self.kwargs["label"], str):
            if len(self.args) == 1:
                self.kwargs["label"] = self.args[0].label()
            else:
                self.kwargs["label"] = f"Pole ({len(self.args)})"


class StereoNet_Vector(StereoNet_Artists):
    def __init__(self, factory, *args, **kwargs):
        super().__init__(factory, *args, **kwargs)
        self.stereonet_method = "_vector"
        self.args = args
        self.parse_kwargs(kwargs)

    def parse_kwargs(self, kwargs):
        super().update_kwargs("stereonet_default_vector_kwargs")
        self.kwargs.update((k, kwargs[k]) for k in self.kwargs.keys() & kwargs.keys())
        if not isinstance(self.kwargs["label"], str):
            if len(self.args) == 1:
                self.kwargs["label"] = self.args[0].label()
            else:
                self.kwargs["label"] = f"Vector ({len(self.args)})"


class StereoNet_Scatter(StereoNet_Artists):
    def __init__(self, factory, *args, **kwargs):
        super().__init__(factory, *args, **kwargs)
        self.stereonet_method = "_scatter"
        self.args = args
        self.parse_kwargs(kwargs)

    def parse_kwargs(self, kwargs):
        super().update_kwargs("stereonet_default_scatter_kwargs")
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
    def __init__(self, factory, *args, **kwargs):
        super().__init__(factory, *args, **kwargs)
        self.stereonet_method = "_great_circle"
        self.args = args
        self.parse_kwargs(kwargs)

    def parse_kwargs(self, kwargs):
        super().update_kwargs("stereonet_default_great_circle_kwargs")
        self.kwargs.update((k, kwargs[k]) for k in self.kwargs.keys() & kwargs.keys())
        if not isinstance(self.kwargs["label"], str):
            if len(self.args) == 1:
                self.kwargs["label"] = self.args[0].label()
            else:
                self.kwargs["label"] = f"Planar ({len(self.args)})"


class StereoNet_Arc(StereoNet_Artists):
    def __init__(self, factory, *args, **kwargs):
        super().__init__(factory, *args, **kwargs)
        self.stereonet_method = "_arc"
        self.args = args
        self.parse_kwargs(kwargs)

    def parse_kwargs(self, kwargs):
        super().update_kwargs("stereonet_default_arc_kwargs")
        self.kwargs.update((k, kwargs[k]) for k in self.kwargs.keys() & kwargs.keys())
        if not isinstance(self.kwargs["label"], str):
            if len(self.args) == 1:
                self.kwargs["label"] = self.args[0].label()
            else:
                self.kwargs["label"] = f"Planar ({len(self.args)})"


# class StereoNet_Cone(StereoNet_Artists):
#     def __init__(self, factory, *args, **kwargs):
#         super().__init__(factory, *args, **kwargs)
#         self.stereonet_method = "_cone"
#         self.args = args
#         self.parse_kwargs(kwargs)

#     def parse_kwargs(self, kwargs):
#         super().update_kwargs("stereonet_default_cone_kwargs")
#         self.kwargs.update((k, kwargs[k]) for k in self.kwargs.keys() & kwargs.keys())
#         self.kwargs["angle"] = np.atleast_1d(kwargs["angle"]).tolist()
#         nof = np.vstack(self.args).shape[0]
#         noa = len(self.kwargs["angle"])
#         if np.vstack(self.args).shape[0] != len(self.kwargs["angle"]):
#             raise TypeError(
#                 f"Number of angles ({noa}) do not match number of features ({nof})"
#             )
#         if not isinstance(self.kwargs["label"], str):
#             if len(self.args) == 1:
#                 if issubclass(type(self.args[0]), Vector3):
#                     self.kwargs[
#                         "label"
#                     ] = f"Cone {self.args[0].label()} ({self.kwargs['angle'][0]})"
#                 else:
#                     self.kwargs["label"] = f"Cones ({len(self.args[0])})"
#             else:
#                 self.kwargs["label"] = f"Cones ({len(self.args)})"


class StereoNet_Cone(StereoNet_Artists):
    def __init__(self, factory, *args, **kwargs):
        super().__init__(factory, *args, **kwargs)
        self.stereonet_method = "_cone"
        self.args = args
        self.parse_kwargs(kwargs)

    def parse_kwargs(self, kwargs):
        super().update_kwargs("stereonet_default_cone_kwargs")
        self.kwargs.update((k, kwargs[k]) for k in self.kwargs.keys() & kwargs.keys())
        if not isinstance(self.kwargs["label"], str):
            if len(self.args) == 1:
                self.kwargs["label"] = self.args[0].label()
            else:
                self.kwargs["label"] = f"Cones ({len(self.args)})"


class StereoNet_Pair(StereoNet_Artists):
    def __init__(self, factory, *args, **kwargs):
        super().__init__(factory, *args, **kwargs)
        self.stereonet_method = "_pair"
        self.args = args
        self.parse_kwargs(kwargs)

    def parse_kwargs(self, kwargs):
        super().update_kwargs("stereonet_default_pair_kwargs")
        self.kwargs.update((k, kwargs[k]) for k in self.kwargs.keys() & kwargs.keys())
        if not isinstance(self.kwargs["label"], str):
            if len(self.args) == 1:
                self.kwargs["label"] = self.args[0].label()
            else:
                self.kwargs["label"] = f"Pair ({len(self.args)})"


class StereoNet_Fault(StereoNet_Artists):
    def __init__(self, factory, *args, **kwargs):
        super().__init__(factory, *args, **kwargs)
        self.stereonet_method = "_fault"
        self.args = args
        self.parse_kwargs(kwargs)

    def parse_kwargs(self, kwargs):
        super().update_kwargs("stereonet_default_fault_kwargs")
        self.kwargs.update((k, kwargs[k]) for k in self.kwargs.keys() & kwargs.keys())
        if not isinstance(self.kwargs["label"], str):
            if len(self.args) == 1:
                self.kwargs["label"] = self.args[0].label()
            else:
                self.kwargs["label"] = f"Fault ({len(self.args)})"


class StereoNet_Hoeppner(StereoNet_Artists):
    def __init__(self, factory, *args, **kwargs):
        super().__init__(factory, *args, **kwargs)
        self.stereonet_method = "_hoeppner"
        self.args = args
        self.parse_kwargs(kwargs)

    def parse_kwargs(self, kwargs):
        super().update_kwargs("stereonet_default_hoeppner_kwargs")
        self.kwargs.update((k, kwargs[k]) for k in self.kwargs.keys() & kwargs.keys())
        if not isinstance(self.kwargs["label"], str):
            if len(self.args) == 1:
                self.kwargs["label"] = self.args[0].label()
            else:
                self.kwargs["label"] = f"Fault ({len(self.args)})"


class StereoNet_Arrow(StereoNet_Artists):
    def __init__(self, factory, *args, **kwargs):
        super().__init__(factory, *args, **kwargs)
        self.stereonet_method = "_arrow"
        self.args = args[:2]  # take max 2 args
        self.parse_kwargs(kwargs)

    def parse_kwargs(self, kwargs):
        super().update_kwargs("stereonet_default_arrow_kwargs")
        self.kwargs.update((k, kwargs[k]) for k in self.kwargs.keys() & kwargs.keys())
        if not isinstance(self.kwargs["label"], str):
            if len(self.args) == 1:
                self.kwargs["label"] = self.args[0].label()
            else:
                self.kwargs["label"] = f"Fault ({len(self.args)})"
        self.kwargs["sense"] = (
            np.copysign(1, np.atleast_1d(kwargs.get("sense", 1))).astype(int).tolist()
        )


class StereoNet_Tensor(StereoNet_Artists):
    def __init__(self, factory, *args, **kwargs):
        super().__init__(factory, *args, **kwargs)
        self.stereonet_method = "_tensor"
        self.args = args[:1]  # take max 1 args
        self.parse_kwargs(kwargs)

    def parse_kwargs(self, kwargs):
        super().update_kwargs("stereonet_default_tensor_kwargs")
        self.kwargs.update((k, kwargs[k]) for k in self.kwargs.keys() & kwargs.keys())
        if not isinstance(self.kwargs["label"], str):
            self.kwargs["label"] = self.args[0].label()


class StereoNet_Contour(StereoNet_Artists):
    def __init__(self, factory, *args, **kwargs):
        super().__init__(factory, *args, **kwargs)
        self.stereonet_method = "_contour"
        if len(args) > 0:
            self.args = args[:1]  # take only first arg
        else:
            self.args = ()
        self.parse_kwargs(kwargs)

    def parse_kwargs(self, kwargs):
        super().update_kwargs("stereonet_default_contour_kwargs")
        self.kwargs.update((k, kwargs[k]) for k in self.kwargs.keys() & kwargs.keys())
        if not isinstance(self.kwargs["label"], str):
            self.kwargs["label"] = self.args[0].label()


class StereoNetArtistFactory:
    @staticmethod
    def create_point(*args, **kwargs):
        if all([issubclass(type(arg), (Vector3, Vector3Set)) for arg in args]):
            return StereoNet_Point("create_point", *args, **kwargs)
        else:
            raise TypeError("Not valid arguments for Stereonet point")

    @staticmethod
    def create_pole(*args, **kwargs):
        if all([issubclass(type(arg), (Foliation, FoliationSet)) for arg in args]):
            return StereoNet_Pole("create_pole", *args, **kwargs)
        else:
            raise TypeError("Not valid arguments for Stereonet pole")

    @staticmethod
    def create_scatter(*args, **kwargs):
        if all([issubclass(type(arg), (Vector3, Vector3Set)) for arg in args]):
            return StereoNet_Scatter("create_scatter", *args, **kwargs)
        else:
            raise TypeError("Not valid arguments for Stereonet scatter")

    @staticmethod
    def create_vector(*args, **kwargs):
        if all([issubclass(type(arg), (Vector3, Vector3Set)) for arg in args]):
            return StereoNet_Vector("create_vector", *args, **kwargs)
        else:
            raise TypeError("Not valid arguments for Stereonet vector")

    @staticmethod
    def create_great_circle(*args, **kwargs):
        if all([issubclass(type(arg), (Foliation, FoliationSet)) for arg in args]):
            return StereoNet_Great_Circle("create_great_circle", *args, **kwargs)
        else:
            raise TypeError("Not valid arguments for Stereonet great circle")

    @staticmethod
    def create_arc(*args, **kwargs):
        if issubclass(type(args[0]), Vector3Set):
            args = args[0].data
        if all([issubclass(type(arg), Vector3) for arg in args]):
            return StereoNet_Arc("create_arc", *args, **kwargs)
        else:
            raise TypeError("Not valid arguments for Stereonet arc")

    # @staticmethod
    # def create_cone(*args, **kwargs):
    #     if all([issubclass(type(arg), (Cone, ConeSet)) for arg in args]):
    #         if "angle" in kwargs:
    #             return StereoNet_Cone("create_cone", *args, **kwargs)
    #         else:
    #             raise TypeError("Keyword argument angle must be provided.")
    #     else:
    #         raise TypeError("Not valid arguments for Stereonet cone")

    @staticmethod
    def create_cone(*args, **kwargs):
        if all([issubclass(type(arg), (Cone, ConeSet)) for arg in args]):
            return StereoNet_Cone("create_cone", *args, **kwargs)
        else:
            raise TypeError("Not valid arguments for Stereonet cone")

    @staticmethod
    def create_pair(*args, **kwargs):
        if all([issubclass(type(arg), (Pair, PairSet)) for arg in args]):
            return StereoNet_Pair("create_pair", *args, **kwargs)
        else:
            raise TypeError("Not valid arguments for Stereonet pair")

    @staticmethod
    def create_fault(*args, **kwargs):
        if all([issubclass(type(arg), (Fault, FaultSet)) for arg in args]):
            return StereoNet_Fault("create_fault", *args, **kwargs)
        else:
            raise TypeError("Not valid arguments for Stereonet fault")

    @staticmethod
    def create_hoeppner(*args, **kwargs):
        if all([issubclass(type(arg), (Fault, FaultSet)) for arg in args]):
            return StereoNet_Hoeppner("create_hoeppner", *args, **kwargs)
        else:
            raise TypeError("Not valid arguments for Stereonet heoppner")

    @staticmethod
    def create_arrow(*args, **kwargs):
        if all([issubclass(type(arg), (Vector3, Vector3Set)) for arg in args[:2]]):
            return StereoNet_Arrow("create_arrow", *args, **kwargs)
        else:
            raise TypeError("Not valid arguments for Stereonet arrow")

    @staticmethod
    def create_tensor(*args, **kwargs):
        if all([issubclass(type(arg), Tensor3) for arg in args[:1]]):
            return StereoNet_Tensor("create_tensor", *args, **kwargs)
        else:
            raise TypeError("Not valid arguments for Stereonet arrow")

    @staticmethod
    def create_contour(*args, **kwargs):
        if len(args) == 0:
            return StereoNet_Contour("create_contour", **kwargs)
        elif issubclass(type(args[0]), Vector3Set):
            return StereoNet_Contour("create_contour", *args, **kwargs)
        else:
            raise TypeError("Not valid arguments for Stereonet contour")


# RosePlot


class RosePlot_Artists:
    def __init__(self, factory, *args, **kwargs):
        self.factory = factory

    def update_kwargs(self, style):
        self.kwargs = apsg_conf[style].copy()
        self.kwargs["label"] = self.roseplot_method


class RosePlot_Bar(RosePlot_Artists):
    def __init__(self, factory, *args, **kwargs):
        super().__init__(factory, *args, **kwargs)
        self.roseplot_method = "_bar"
        self.args = args
        self.parse_kwargs(kwargs)

    def parse_kwargs(self, kwargs):
        super().update_kwargs("roseplot_default_bar_kwargs")
        self.kwargs.update((k, kwargs[k]) for k in self.kwargs.keys() & kwargs.keys())


class RosePlot_Pdf(RosePlot_Artists):
    def __init__(self, factory, *args, **kwargs):
        super().__init__(factory, *args, **kwargs)
        self.roseplot_method = "_pdf"
        self.args = args
        self.parse_kwargs(kwargs)

    def parse_kwargs(self, kwargs):
        super().update_kwargs("roseplot_default_pdf_kwargs")
        self.kwargs.update((k, kwargs[k]) for k in self.kwargs.keys() & kwargs.keys())
        if self.kwargs["color"] is None:
            del self.kwargs["color"]


class RosePlot_Muci(RosePlot_Artists):
    def __init__(self, factory, *args, **kwargs):
        super().__init__(factory, *args, **kwargs)
        self.roseplot_method = "_muci"
        self.args = args
        self.parse_kwargs(kwargs)

    def parse_kwargs(self, kwargs):
        super().update_kwargs("roseplot_default_muci_kwargs")
        self.kwargs.update((k, kwargs[k]) for k in self.kwargs.keys() & kwargs.keys())


class RosePlotArtistFactory:
    @staticmethod
    def create_bar(*args, **kwargs):
        if all([issubclass(type(arg), Vector2Set) for arg in args]):
            return RosePlot_Bar("create_bar", *args, **kwargs)
        else:
            raise TypeError("Not valid arguments for Roseplot bar")

    @staticmethod
    def create_pdf(*args, **kwargs):
        if all([issubclass(type(arg), Vector2Set) for arg in args]):
            return RosePlot_Pdf("create_pdf", *args, **kwargs)
        else:
            raise TypeError("Not valid arguments for Roseplot pdf")

    @staticmethod
    def create_muci(*args, **kwargs):
        if all([issubclass(type(arg), Vector2Set) for arg in args]):
            return RosePlot_Muci("create_muci", *args, **kwargs)
        else:
            raise TypeError("Not valid arguments for Roseplot muci")


# FabricPlots


class FabricPlot_Artists:
    def __init__(self, factory, *args, **kwargs):
        self.factory = factory

    def update_kwargs(self, style):
        self.kwargs = apsg_conf[style].copy()
        self.kwargs["label"] = self.fabricplot_method

    def to_json(self):
        return dict(
            factory=self.factory,
            fabricplot_method=self.fabricplot_method,
            args=tuple([obj.to_json() for obj in self.args]),
            kwargs=self.kwargs.copy(),
        )


class FabricPlot_Point(FabricPlot_Artists):
    def __init__(self, factory, *args, **kwargs):
        super().__init__(factory, *args, **kwargs)
        self.fabricplot_method = "_point"
        self.args = args
        self.parse_kwargs(kwargs)

    def parse_kwargs(self, kwargs):
        super().update_kwargs("fabricplot_default_point_kwargs")
        self.kwargs.update((k, kwargs[k]) for k in self.kwargs.keys() & kwargs.keys())
        if not isinstance(self.kwargs["label"], str):
            if len(self.args) == 1:
                self.kwargs["label"] = self.args[0].label()
            else:
                self.kwargs["label"] = f"Tensors ({len(self.args)})"


class FabricPlot_Path(FabricPlot_Artists):
    def __init__(self, factory, *args, **kwargs):
        super().__init__(factory, *args, **kwargs)
        self.fabricplot_method = "_path"
        self.args = args
        self.parse_kwargs(kwargs)

    def parse_kwargs(self, kwargs):
        super().update_kwargs("fabricplot_default_path_kwargs")
        self.kwargs.update((k, kwargs[k]) for k in self.kwargs.keys() & kwargs.keys())
        if not isinstance(self.kwargs["label"], str):
            if len(self.args) == 1:
                self.kwargs["label"] = self.args[0].label()
            else:
                self.kwargs["label"] = f"Paths ({len(self.args)})"


class FabricPlotArtistFactory:
    @staticmethod
    def create_point(*args, **kwargs):
        if all([issubclass(type(arg), (Ellipsoid, EllipsoidSet)) for arg in args]):
            return FabricPlot_Point("create_point", *args, **kwargs)
        else:
            raise TypeError("Not valid arguments for Fabric plot point")

    @staticmethod
    def create_path(*args, **kwargs):
        if all([issubclass(type(arg), EllipsoidSet) for arg in args]):
            return FabricPlot_Path("create_path", *args, **kwargs)
        else:
            raise TypeError("Not valid arguments for Fabric plot path")
