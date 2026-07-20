import numpy as np

from apsg.config import apsg_conf
from apsg.feature._container import (
    ConeSet,
    EllipsoidSet,
    FaultSet,
    FoliationSet,
    PairSet,
    Vector2Set,
    Vector3Set,
)
from apsg.feature._geodata import Cone, Fault, Foliation, Pair
from apsg.feature._tensor3 import Ellipsoid, Stress3, Tensor3
from apsg.math._vector import Vector3

# StereoNet


class StereoNet_Artists:
    def __init__(self, factory, *args, **kwargs):
        """Initialize stereonet artist with factory reference."""
        self.factory = factory

    def update_kwargs(self, style):
        """Update kwargs from global style configuration."""
        self.kwargs = getattr(apsg_conf, style).copy()
        self.kwargs["label"] = self.stereonet_method  # ty: ignore

    def to_json(self):
        """Serialize artist to JSON-compatible dict."""
        return dict(
            factory=self.factory,
            stereonet_method=self.stereonet_method,  # ty: ignore
            args=tuple(obj.to_json() for obj in self.args),  # ty: ignore
            kwargs=self.kwargs.copy(),
        )


class StereoNet_Point(StereoNet_Artists):
    def __init__(self, factory, *args, **kwargs):
        """Initialize stereonet point artist."""
        super().__init__(factory, *args, **kwargs)
        self.stereonet_method = "_point"
        self.args = args
        self.parse_kwargs(kwargs)

    def parse_kwargs(self, kwargs):
        """Parse and apply point style kwargs."""
        super().update_kwargs("stereonet_point")
        self.kwargs.update((k, kwargs[k]) for k in self.kwargs.keys() & kwargs.keys())
        if not isinstance(self.kwargs["label"], str):
            if len(self.args) == 1:
                self.kwargs["label"] = self.args[0].label()
            else:
                self.kwargs["label"] = f"Linear ({len(self.args)})"


class StereoNet_Vector(StereoNet_Artists):
    def __init__(self, factory, *args, **kwargs):
        """Initialize stereonet vector artist."""
        super().__init__(factory, *args, **kwargs)
        self.stereonet_method = "_vector"
        self.args = args
        self.parse_kwargs(kwargs)

    def parse_kwargs(self, kwargs):
        """Parse and apply vector style kwargs."""
        super().update_kwargs("stereonet_vector")
        self.kwargs.update((k, kwargs[k]) for k in self.kwargs.keys() & kwargs.keys())
        if not isinstance(self.kwargs["label"], str):
            if len(self.args) == 1:
                self.kwargs["label"] = self.args[0].label()
            else:
                self.kwargs["label"] = f"Vector ({len(self.args)})"


class StereoNet_Scatter(StereoNet_Artists):
    def __init__(self, factory, *args, **kwargs):
        """Initialize stereonet scatter artist."""
        super().__init__(factory, *args, **kwargs)
        self.stereonet_method = "_scatter"
        self.args = args
        self.parse_kwargs(kwargs)

    def parse_kwargs(self, kwargs):
        """Parse scatter style kwargs and validate size/color arrays."""
        super().update_kwargs("stereonet_scatter")
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
        """Initialize stereonet great circle artist."""
        super().__init__(factory, *args, **kwargs)
        self.stereonet_method = "_great_circle"
        self.args = args
        self.parse_kwargs(kwargs)

    def parse_kwargs(self, kwargs):
        """Parse and apply great circle style kwargs."""
        super().update_kwargs("stereonet_great_circle")
        self.kwargs.update((k, kwargs[k]) for k in self.kwargs.keys() & kwargs.keys())
        if not isinstance(self.kwargs["label"], str):
            if len(self.args) == 1:
                self.kwargs["label"] = self.args[0].label()
            else:
                self.kwargs["label"] = f"Planar ({len(self.args)})"


class StereoNet_Arc(StereoNet_Artists):
    def __init__(self, factory, *args, **kwargs):
        """Initialize stereonet arc artist."""
        super().__init__(factory, *args, **kwargs)
        self.stereonet_method = "_arc"
        self.args = args
        self.parse_kwargs(kwargs)

    def parse_kwargs(self, kwargs):
        """Parse and apply arc style kwargs."""
        super().update_kwargs("stereonet_arc")
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
#         super().update_kwargs("stereonet_cone")
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
        """Initialize stereonet cone artist."""
        super().__init__(factory, *args, **kwargs)
        self.stereonet_method = "_cone"
        self.args = args
        self.parse_kwargs(kwargs)

    def parse_kwargs(self, kwargs):
        """Parse and apply cone style kwargs."""
        super().update_kwargs("stereonet_cone")
        self.kwargs.update((k, kwargs[k]) for k in self.kwargs.keys() & kwargs.keys())
        if not isinstance(self.kwargs["label"], str):
            if len(self.args) == 1:
                self.kwargs["label"] = self.args[0].label()
            else:
                self.kwargs["label"] = f"Cones ({len(self.args)})"


class StereoNet_Bingham(StereoNet_Artists):
    def __init__(self, factory, *args, **kwargs):
        """Initialize stereonet Bingham confidence ellipse artist."""
        super().__init__(factory, *args, **kwargs)
        self.stereonet_method = "_bingham"
        self.args = args
        self.parse_kwargs(kwargs)

    def parse_kwargs(self, kwargs):
        """Parse and apply Bingham confidence ellipse style kwargs."""
        super().update_kwargs("stereonet_bingham")
        self.kwargs.update((k, kwargs[k]) for k in self.kwargs.keys() & kwargs.keys())
        if not isinstance(self.kwargs["label"], str):
            if len(self.args) == 1:
                self.kwargs["label"] = self.args[0].label()
            else:
                self.kwargs["label"] = f"Bingham ({len(self.args)})"


class StereoNet_Pair(StereoNet_Artists):
    def __init__(self, factory, *args, **kwargs):
        """Initialize stereonet pair artist."""
        super().__init__(factory, *args, **kwargs)
        self.stereonet_method = "_pair"
        self.args = args
        self.parse_kwargs(kwargs)

    def parse_kwargs(self, kwargs):
        """Parse and apply pair style kwargs."""
        super().update_kwargs("stereonet_pair")
        self.kwargs.update((k, kwargs[k]) for k in self.kwargs.keys() & kwargs.keys())
        if not isinstance(self.kwargs["label"], str):
            if len(self.args) == 1:
                self.kwargs["label"] = self.args[0].label()
            else:
                self.kwargs["label"] = f"Pair ({len(self.args)})"


class StereoNet_Fault(StereoNet_Artists):
    def __init__(self, factory, *args, **kwargs):
        """Initialize stereonet fault artist."""
        super().__init__(factory, *args, **kwargs)
        self.stereonet_method = "_fault"
        self.args = args
        self.parse_kwargs(kwargs)

    def parse_kwargs(self, kwargs):
        """Parse and apply fault style kwargs."""
        super().update_kwargs("stereonet_fault")
        self.kwargs.update((k, kwargs[k]) for k in self.kwargs.keys() & kwargs.keys())
        if not isinstance(self.kwargs["label"], str):
            if len(self.args) == 1:
                self.kwargs["label"] = self.args[0].label()
            else:
                self.kwargs["label"] = f"Fault ({len(self.args)})"


class StereoNet_Hoeppner(StereoNet_Artists):
    def __init__(self, factory, *args, **kwargs):
        """Initialize stereonet Hoeppner plot artist."""
        super().__init__(factory, *args, **kwargs)
        self.stereonet_method = "_hoeppner"
        self.args = args
        self.parse_kwargs(kwargs)

    def parse_kwargs(self, kwargs):
        """Parse and apply Hoeppner style kwargs."""
        super().update_kwargs("stereonet_hoeppner")
        self.kwargs.update((k, kwargs[k]) for k in self.kwargs.keys() & kwargs.keys())
        if not isinstance(self.kwargs["label"], str):
            if len(self.args) == 1:
                self.kwargs["label"] = self.args[0].label()
            else:
                self.kwargs["label"] = f"Fault ({len(self.args)})"


class StereoNet_Arrow(StereoNet_Artists):
    def __init__(self, factory, *args, **kwargs):
        """Initialize stereonet arrow artist."""
        super().__init__(factory, *args, **kwargs)
        self.stereonet_method = "_arrow"
        self.args = args[:2]  # take max 2 args
        self.parse_kwargs(kwargs)

    def parse_kwargs(self, kwargs):
        """Parse arrow style kwargs and validate sense."""
        super().update_kwargs("stereonet_arrow")
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
        """Initialize stereonet tensor artist."""
        super().__init__(factory, *args, **kwargs)
        self.stereonet_method = "_tensor"
        self.args = args[:1]  # take max 1 args
        self.parse_kwargs(kwargs)

    def parse_kwargs(self, kwargs):
        """Parse and apply tensor style kwargs."""
        super().update_kwargs("stereonet_tensor")
        self.kwargs.update((k, kwargs[k]) for k in self.kwargs.keys() & kwargs.keys())
        if not isinstance(self.kwargs["label"], str):
            self.kwargs["label"] = self.args[0].label()


class StereoNet_Stress(StereoNet_Artists):
    def __init__(self, factory, *args, **kwargs):
        """Initialize stereonet stress artist."""
        super().__init__(factory, *args, **kwargs)
        self.stereonet_method = "_stress"
        self.args = args[:1]  # take max 1 args
        self.parse_kwargs(kwargs)

    def parse_kwargs(self, kwargs):
        """Parse and apply stress style kwargs."""
        super().update_kwargs("stereonet_stress")
        self.kwargs.update((k, kwargs[k]) for k in self.kwargs.keys() & kwargs.keys())
        if not isinstance(self.kwargs["label"], str):
            self.kwargs["label"] = self.args[0].label()


class StereoNet_Contour(StereoNet_Artists):
    def __init__(self, factory, *args, **kwargs):
        """Initialize stereonet contour artist."""
        super().__init__(factory, *args, **kwargs)
        self.stereonet_method = "_contour"
        if len(args) > 0:
            self.args = args[:1]  # take only first arg
        else:
            self.args = ()
        self.parse_kwargs(kwargs)

    def parse_kwargs(self, kwargs):
        """Parse and apply contour style kwargs."""
        super().update_kwargs("stereonet_contour")
        self.kwargs.update((k, kwargs[k]) for k in self.kwargs.keys() & kwargs.keys())
        if not isinstance(self.kwargs["label"], str):
            self.kwargs["label"] = self.args[0].label()


class StereoNetArtistFactory:
    @staticmethod
    def create_point(*args, **kwargs):
        """Create stereonet point artist from Vector3 data."""
        if all([isinstance(arg, (Vector3, Vector3Set)) for arg in args]):
            return StereoNet_Point("create_point", *args, **kwargs)
        else:
            raise TypeError("Not valid arguments for Stereonet point")

    @staticmethod
    def create_scatter(*args, **kwargs):
        """Create stereonet scatter artist from Vector3 data."""
        if all([isinstance(arg, (Vector3, Vector3Set)) for arg in args]):
            return StereoNet_Scatter("create_scatter", *args, **kwargs)
        else:
            raise TypeError("Not valid arguments for Stereonet scatter")

    @staticmethod
    def create_vector(*args, **kwargs):
        """Create stereonet vector artist from Vector3 data."""
        if all([isinstance(arg, (Vector3, Vector3Set)) for arg in args]):
            return StereoNet_Vector("create_vector", *args, **kwargs)
        else:
            raise TypeError("Not valid arguments for Stereonet vector")

    @staticmethod
    def create_great_circle(*args, **kwargs):
        """Create stereonet great circle artist from Foliation data."""
        if all([isinstance(arg, (Foliation, FoliationSet)) for arg in args]):
            return StereoNet_Great_Circle("create_great_circle", *args, **kwargs)
        else:
            raise TypeError("Not valid arguments for Stereonet great circle")

    @staticmethod
    def create_arc(*args, **kwargs):
        """Create stereonet arc artist from Vector3 data."""
        if isinstance(args[0], Vector3Set):
            args = args[0].data
        if all([isinstance(arg, Vector3) for arg in args]):
            return StereoNet_Arc("create_arc", *args, **kwargs)
        else:
            raise TypeError("Not valid arguments for Stereonet arc")

    # @staticmethod
    # def create_cone(*args, **kwargs):
    #     if all([isinstance(arg, (Cone, ConeSet)) for arg in args]):
    #         if "angle" in kwargs:
    #             return StereoNet_Cone("create_cone", *args, **kwargs)
    #         else:
    #             raise TypeError("Keyword argument angle must be provided.")
    #     else:
    #         raise TypeError("Not valid arguments for Stereonet cone")

    @staticmethod
    def create_cone(*args, **kwargs):
        """Create stereonet cone artist from Cone data."""
        if all([isinstance(arg, (Cone, ConeSet)) for arg in args]):
            return StereoNet_Cone("create_cone", *args, **kwargs)
        else:
            raise TypeError("Not valid arguments for Stereonet cone")

    @staticmethod
    def create_bingham(*args, **kwargs):
        """Create stereonet Bingham confidence ellipse artist from Vector3Set data."""
        if all([isinstance(arg, Vector3Set) for arg in args]):
            return StereoNet_Bingham("create_bingham", *args, **kwargs)
        else:
            raise TypeError("Not valid arguments for Stereonet bingham")

    @staticmethod
    def create_pair(*args, **kwargs):
        """Create stereonet pair artist from Pair data."""
        if all([isinstance(arg, (Pair, PairSet)) for arg in args]):
            return StereoNet_Pair("create_pair", *args, **kwargs)
        else:
            raise TypeError("Not valid arguments for Stereonet pair")

    @staticmethod
    def create_fault(*args, **kwargs):
        """Create stereonet fault artist from Fault data."""
        if all([isinstance(arg, (Fault, FaultSet)) for arg in args]):
            return StereoNet_Fault("create_fault", *args, **kwargs)
        else:
            raise TypeError("Not valid arguments for Stereonet fault")

    @staticmethod
    def create_hoeppner(*args, **kwargs):
        """Create stereonet Hoeppner plot artist from Fault data."""
        if all([isinstance(arg, (Fault, FaultSet)) for arg in args]):
            return StereoNet_Hoeppner("create_hoeppner", *args, **kwargs)
        else:
            raise TypeError("Not valid arguments for Stereonet heoppner")

    @staticmethod
    def create_arrow(*args, **kwargs):
        """Create stereonet arrow artist from Vector3 data."""
        if all([isinstance(arg, (Vector3, Vector3Set)) for arg in args[:2]]):
            return StereoNet_Arrow("create_arrow", *args, **kwargs)
        else:
            raise TypeError("Not valid arguments for Stereonet arrow")

    @staticmethod
    def create_tensor(*args, **kwargs):
        """Create stereonet tensor artist from Tensor3 data."""
        if all([isinstance(arg, Tensor3) for arg in args[:1]]):
            return StereoNet_Tensor("create_tensor", *args, **kwargs)
        else:
            raise TypeError("Not valid arguments for Stereonet arrow")

    @staticmethod
    def create_stress(*args, **kwargs):
        """Create stereonet stress artist from Stress3 data."""
        if all([isinstance(arg, Stress3) for arg in args[:1]]):
            return StereoNet_Stress("create_stress", *args, **kwargs)
        else:
            raise TypeError("Not valid arguments for Stereonet arrow")

    @staticmethod
    def create_contour(*args, **kwargs):
        """Create stereonet contour artist from Vector3Set data."""
        if len(args) == 0:
            return StereoNet_Contour("create_contour", **kwargs)
        elif isinstance(args[0], Vector3Set):
            return StereoNet_Contour("create_contour", *args, **kwargs)
        else:
            raise TypeError("Not valid arguments for Stereonet contour")


# RosePlot


class RosePlot_Artists:
    def __init__(self, factory, *args, **kwargs):
        """Initialize rose plot artist with factory reference."""
        self.factory = factory

    def update_kwargs(self, style):
        """Update kwargs from global rose plot style configuration."""
        self.kwargs = getattr(apsg_conf, style).copy()
        self.kwargs["label"] = self.roseplot_method  # ty: ignore


class RosePlot_Bar(RosePlot_Artists):
    def __init__(self, factory, *args, **kwargs):
        """Initialize rose plot bar artist."""
        super().__init__(factory, *args, **kwargs)
        self.roseplot_method = "_bar"
        self.args = args
        self.parse_kwargs(kwargs)

    def parse_kwargs(self, kwargs):
        """Parse bar style kwargs."""
        super().update_kwargs("roseplot_bar")
        self.kwargs.update((k, kwargs[k]) for k in self.kwargs.keys() & kwargs.keys())


class RosePlot_Pdf(RosePlot_Artists):
    def __init__(self, factory, *args, **kwargs):
        """Initialize rose plot PDF artist."""
        super().__init__(factory, *args, **kwargs)
        self.roseplot_method = "_pdf"
        self.args = args
        self.parse_kwargs(kwargs)

    def parse_kwargs(self, kwargs):
        """Parse PDF style kwargs."""
        super().update_kwargs("roseplot_pdf")
        self.kwargs.update((k, kwargs[k]) for k in self.kwargs.keys() & kwargs.keys())
        if self.kwargs["color"] is None:
            del self.kwargs["color"]


class RosePlot_Muci(RosePlot_Artists):
    def __init__(self, factory, *args, **kwargs):
        """Initialize rose plot muci artist."""
        super().__init__(factory, *args, **kwargs)
        self.roseplot_method = "_muci"
        self.args = args
        self.parse_kwargs(kwargs)

    def parse_kwargs(self, kwargs):
        """Parse muci style kwargs."""
        super().update_kwargs("roseplot_muci")
        self.kwargs.update((k, kwargs[k]) for k in self.kwargs.keys() & kwargs.keys())


class RosePlotArtistFactory:
    @staticmethod
    def create_bar(*args, **kwargs):
        """Create rose plot bar artist from Vector2Set data."""
        if all([isinstance(arg, Vector2Set) for arg in args]):
            return RosePlot_Bar("create_bar", *args, **kwargs)
        else:
            raise TypeError("Not valid arguments for Roseplot bar")

    @staticmethod
    def create_pdf(*args, **kwargs):
        """Create rose plot PDF artist from Vector2Set data."""
        if all([isinstance(arg, Vector2Set) for arg in args]):
            return RosePlot_Pdf("create_pdf", *args, **kwargs)
        else:
            raise TypeError("Not valid arguments for Roseplot pdf")

    @staticmethod
    def create_muci(*args, **kwargs):
        """Create rose plot muci artist from Vector2Set data."""
        if all([isinstance(arg, Vector2Set) for arg in args]):
            return RosePlot_Muci("create_muci", *args, **kwargs)
        else:
            raise TypeError("Not valid arguments for Roseplot muci")


# FabricPlots


class FabricPlot_Artists:
    def __init__(self, factory, *args, **kwargs):
        """Initialize fabric plot artist with factory reference."""
        self.factory = factory

    def update_kwargs(self, style):
        """Update kwargs from global fabric plot style configuration."""
        self.kwargs = getattr(apsg_conf, style).copy()
        self.kwargs["label"] = self.fabricplot_method  # ty: ignore

    def to_json(self):
        """Serialize fabric plot artist to JSON-compatible dict."""
        return dict(
            factory=self.factory,
            fabricplot_method=self.fabricplot_method,  # ty: ignore
            args=(obj.to_json() for obj in self.args),  # ty: ignore
            kwargs=self.kwargs.copy(),
        )


class FabricPlot_Point(FabricPlot_Artists):
    def __init__(self, factory, *args, **kwargs):
        """Initialize fabric plot point artist."""
        super().__init__(factory, *args, **kwargs)
        self.fabricplot_method = "_point"
        self.args = args
        self.parse_kwargs(kwargs)

    def parse_kwargs(self, kwargs):
        """Parse fabric point style kwargs."""
        super().update_kwargs("fabricplot_point")
        self.kwargs.update((k, kwargs[k]) for k in self.kwargs.keys() & kwargs.keys())
        if not isinstance(self.kwargs["label"], str):
            if len(self.args) == 1:
                self.kwargs["label"] = self.args[0].label()
            else:
                self.kwargs["label"] = f"Tensors ({len(self.args)})"


class FabricPlot_Path(FabricPlot_Artists):
    def __init__(self, factory, *args, **kwargs):
        """Initialize fabric plot path artist."""
        super().__init__(factory, *args, **kwargs)
        self.fabricplot_method = "_path"
        self.args = args
        self.parse_kwargs(kwargs)

    def parse_kwargs(self, kwargs):
        """Parse fabric path style kwargs."""
        super().update_kwargs("fabricplot_path")
        self.kwargs.update((k, kwargs[k]) for k in self.kwargs.keys() & kwargs.keys())
        if not isinstance(self.kwargs["label"], str):
            if len(self.args) == 1:
                self.kwargs["label"] = self.args[0].label()
            else:
                self.kwargs["label"] = f"Paths ({len(self.args)})"


class FabricPlotArtistFactory:
    @staticmethod
    def create_point(*args, **kwargs):
        """Create fabric plot point artist from Ellipsoid data."""
        if all([isinstance(arg, (Ellipsoid, EllipsoidSet)) for arg in args]):
            return FabricPlot_Point("create_point", *args, **kwargs)
        else:
            raise TypeError("Not valid arguments for Fabric plot point")

    @staticmethod
    def create_path(*args, **kwargs):
        """Create fabric plot path artist from EllipsoidSet data."""
        if all([isinstance(arg, EllipsoidSet) for arg in args]):
            return FabricPlot_Path("create_path", *args, **kwargs)
        else:
            raise TypeError("Not valid arguments for Fabric plot path")
