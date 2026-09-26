import numpy as np

from apsg.config import apsg_conf
from apsg.feature._container import (
    ArcSet,
    ConeSet,
    EllipsoidSet,
    FaultSet,
    FoliationSet,
    PairSet,
    Stress3Set,
    Vector2Set,
    Vector3Set,
)
from apsg.feature._geodata import Arc, Cone, Fault, Foliation, Pair
from apsg.feature._tensor3 import Ellipsoid, Stress3, Tensor3
from apsg.math._vector import Vector3
from apsg.plotting._stereogrid import StereoGrid

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
        return {
            "factory": self.factory,
            "stereonet_method": self.stereonet_method,  # ty: ignore
            "args": tuple(obj.to_json() for obj in self.args),  # ty: ignore
            "kwargs": self.kwargs.copy(),
        }


class _SimpleStereoNetArtist(StereoNet_Artists):
    """Generic artist for stereonet plot types that only need a config key
    and a fallback multi-feature label -- see ``StereoNetArtistFactory._create``.
    """

    def __init__(self, factory, method, config_key, label_template, /, *args, **kwargs):
        """Initialize a simple stereonet artist."""
        super().__init__(factory, *args, **kwargs)
        self.stereonet_method = method
        self.args = args
        self._config_key = config_key
        self._label_template = label_template
        self.parse_kwargs(kwargs)

    def parse_kwargs(self, kwargs):
        """Parse and apply style kwargs."""
        super().update_kwargs(self._config_key)
        self.kwargs.update((k, kwargs[k]) for k in self.kwargs.keys() & kwargs.keys())
        if not isinstance(self.kwargs["label"], str):
            if len(self.args) == 1:
                self.kwargs["label"] = self.args[0].label()
            else:
                self.kwargs["label"] = f"{self._label_template} ({len(self.args)})"


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
                self.kwargs["label"] = f"Arc ({len(self.args)})"


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
        """Initialize stereonet contour artist.

        ``args[0]`` is either a ``Vector3Set`` (a new ``StereoGrid`` is
        created and its density calculated immediately) or an already
        populated ``StereoGrid`` (e.g. built via ``apply_func``/``angmech``,
        used as-is -- no calculation is triggered). Each artist owns its own
        grid, so a ``StereoNet`` can carry multiple independent contour
        layers.
        """
        super().__init__(factory, *args, **kwargs)
        self.stereonet_method = "_contour"
        self.parse_kwargs(args[0], kwargs)

    def parse_kwargs(self, source, kwargs):
        """Parse contour style kwargs and resolve/calculate the grid."""
        super().update_kwargs("stereonet_contour")
        self.kwargs.update((k, kwargs[k]) for k in self.kwargs.keys() & kwargs.keys())
        method = self.kwargs.pop("method")
        sigma = self.kwargs.pop("sigma")
        n_max = self.kwargs.pop("n_max")
        if isinstance(source, StereoGrid):
            grid = source
        else:
            grid = StereoGrid()
            grid.calculate_density(source, method=method, sigma=sigma, n_max=n_max)
        self.args = (grid,)
        if not isinstance(self.kwargs["label"], str):
            self.kwargs["label"] = (
                "Contour" if isinstance(source, StereoGrid) else source.label()
            )

    def to_json(self):
        """Serialize contour artist to JSON-compatible dict.

        Overrides the base ``StereoNet_Artists.to_json`` since ``self.args``
        holds a ``StereoGrid`` (its own ``to_json`` shape), not a feature
        object using the ``datatype``-based convention.
        """
        return {
            "factory": self.factory,
            "stereonet_method": self.stereonet_method,
            "args": (self.args[0].to_json(),),
            "kwargs": self.kwargs.copy(),
        }


class StereoNetArtistFactory:
    @staticmethod
    def _create(
        name,
        valid_types,
        stereonet_method,
        config_key,
        label_template,
        /,
        *args,
        **kwargs,
    ):
        """Build a ``_SimpleStereoNetArtist``, validating arg types first.

        The control parameters are positional-only (before the ``/``) so
        that no real kwarg forwarded by a caller -- e.g. ``confidence()``'s
        own ``method=``, or anyone's ``label=`` -- can ever collide with
        one of them by name.
        """
        if not all(isinstance(arg, valid_types) for arg in args):
            what = stereonet_method.lstrip("_").replace("_", " ")
            raise TypeError(f"Not valid arguments for Stereonet {what}")
        return _SimpleStereoNetArtist(
            name, stereonet_method, config_key, label_template, *args, **kwargs
        )

    @staticmethod
    def create_point(*args, **kwargs):
        """Create stereonet point artist from Vector3 data."""
        return StereoNetArtistFactory._create(
            "create_point",
            (Vector3, Vector3Set),
            "_point",
            "stereonet_point",
            "Linear",
            *args,
            **kwargs,
        )

    @staticmethod
    def create_scatter(*args, **kwargs):
        """Create stereonet scatter artist from Vector3 data."""
        if all(isinstance(arg, (Vector3, Vector3Set)) for arg in args):
            return StereoNet_Scatter("create_scatter", *args, **kwargs)
        else:
            raise TypeError("Not valid arguments for Stereonet scatter")

    @staticmethod
    def create_vector(*args, **kwargs):
        """Create stereonet vector artist from Vector3 data."""
        return StereoNetArtistFactory._create(
            "create_vector",
            (Vector3, Vector3Set),
            "_vector",
            "stereonet_vector",
            "Vector",
            *args,
            **kwargs,
        )

    @staticmethod
    def create_great_circle(*args, **kwargs):
        """Create stereonet great circle artist from Foliation data."""
        return StereoNetArtistFactory._create(
            "create_great_circle",
            (Foliation, FoliationSet),
            "_great_circle",
            "stereonet_great_circle",
            "Planar",
            *args,
            **kwargs,
        )

    @staticmethod
    def create_arc(*args, **kwargs):
        """Create stereonet arc artist from Arc/ArcSet data, or from a raw
        Vector3-like sequence connected pairwise in order via
        ArcSet.from_vectors (legacy convention).
        """
        if len(args) == 1 and isinstance(args[0], ArcSet):
            args = tuple(args[0])
        elif not (args and all(isinstance(arg, Arc) for arg in args)):
            try:
                args = tuple(ArcSet.from_vectors(*args))
            except TypeError:
                raise TypeError("Not valid arguments for Stereonet arc")

        return StereoNet_Arc("create_arc", *args, **kwargs)

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
        return StereoNetArtistFactory._create(
            "create_cone",
            (Cone, ConeSet),
            "_cone",
            "stereonet_cone",
            "Cones",
            *args,
            **kwargs,
        )

    @staticmethod
    def create_confidence(*args, **kwargs):
        """Create stereonet confidence cone/ellipse artist from Vector3Set data,
        or (method "jelinek") from EllipsoidSet/Stress3Set data.
        """
        tensor_sets = (EllipsoidSet, Stress3Set)
        is_tensor = [isinstance(arg, tensor_sets) for arg in args]
        if any(is_tensor) and not all(is_tensor):
            raise TypeError(
                "Not valid arguments for Stereonet confidence: "
                "vector sets and tensor sets cannot be combined"
            )
        tensors = bool(args) and all(is_tensor)
        if tensors:
            kwargs.setdefault("method", "jelinek")
        artist = StereoNetArtistFactory._create(
            "create_confidence",
            (Vector3Set, EllipsoidSet, Stress3Set),
            "_confidence",
            "stereonet_confidence",
            "Confidence",
            *args,
            **kwargs,
        )
        method = artist.kwargs["method"]
        if tensors and method != "jelinek":
            raise TypeError(
                "Not valid arguments for Stereonet confidence: "
                f"method {method!r} is not valid for EllipsoidSet and Stress3Set "
                "(use method 'jelinek')"
            )
        if not tensors and method == "jelinek":
            raise TypeError(
                "Not valid arguments for Stereonet confidence: "
                "method 'jelinek' requires EllipsoidSet or Stress3Set"
            )
        return artist

    @staticmethod
    def create_pair(*args, **kwargs):
        """Create stereonet pair artist from Pair data."""
        return StereoNetArtistFactory._create(
            "create_pair",
            (Pair, PairSet),
            "_pair",
            "stereonet_pair",
            "Pair",
            *args,
            **kwargs,
        )

    @staticmethod
    def create_fault(*args, **kwargs):
        """Create stereonet fault artist from Fault data."""
        return StereoNetArtistFactory._create(
            "create_fault",
            (Fault, FaultSet),
            "_fault",
            "stereonet_fault",
            "Fault",
            *args,
            **kwargs,
        )

    @staticmethod
    def create_hoeppner(*args, **kwargs):
        """Create stereonet Hoeppner plot artist from Fault/Pair data."""
        return StereoNetArtistFactory._create(
            "create_hoeppner",
            (Fault, FaultSet, Pair, PairSet),
            "_hoeppner",
            "stereonet_hoeppner",
            "Fault",
            *args,
            **kwargs,
        )

    @staticmethod
    def create_dihedra(*args, **kwargs):
        """Create stereonet fault dihedra artist from Fault data."""
        return StereoNetArtistFactory._create(
            "create_dihedra",
            (Fault, FaultSet),
            "_dihedra",
            "stereonet_dihedra",
            "Dihedra",
            *args,
            **kwargs,
        )

    @staticmethod
    def create_beachball(*args, **kwargs):
        """Create stereonet stress beach ball artist from Stress3 data."""
        return StereoNetArtistFactory._create(
            "create_beachball",
            (Stress3, Stress3Set),
            "_beachball",
            "stereonet_beachball",
            "Beachball",
            *args,
            **kwargs,
        )

    @staticmethod
    def create_arrow(*args, **kwargs):
        """Create stereonet arrow artist from Vector3 data."""
        if all(isinstance(arg, (Vector3, Vector3Set)) for arg in args[:2]):
            return StereoNet_Arrow("create_arrow", *args, **kwargs)
        else:
            raise TypeError("Not valid arguments for Stereonet arrow")

    @staticmethod
    def create_tensor(*args, **kwargs):
        """Create stereonet tensor artist from Tensor3 data."""
        if all(isinstance(arg, Tensor3) for arg in args[:1]):
            return StereoNet_Tensor("create_tensor", *args, **kwargs)
        else:
            raise TypeError("Not valid arguments for Stereonet tensor")

    @staticmethod
    def create_stress(*args, **kwargs):
        """Create stereonet stress artist from Stress3 data."""
        if all(isinstance(arg, Stress3) for arg in args[:1]):
            return StereoNet_Stress("create_stress", *args, **kwargs)
        else:
            raise TypeError("Not valid arguments for Stereonet stress")

    @staticmethod
    def create_contour(*args, **kwargs):
        """Create stereonet contour artist from a Vector3Set or a StereoGrid."""
        if len(args) >= 1 and isinstance(args[0], (Vector3Set, StereoGrid)):
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
        if all(isinstance(arg, Vector2Set) for arg in args):
            return RosePlot_Bar("create_bar", *args, **kwargs)
        else:
            raise TypeError("Not valid arguments for Roseplot bar")

    @staticmethod
    def create_pdf(*args, **kwargs):
        """Create rose plot PDF artist from Vector2Set data."""
        if all(isinstance(arg, Vector2Set) for arg in args):
            return RosePlot_Pdf("create_pdf", *args, **kwargs)
        else:
            raise TypeError("Not valid arguments for Roseplot pdf")

    @staticmethod
    def create_muci(*args, **kwargs):
        """Create rose plot muci artist from Vector2Set data."""
        if all(isinstance(arg, Vector2Set) for arg in args):
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
        return {
            "factory": self.factory,
            "fabricplot_method": self.fabricplot_method,  # ty: ignore
            "args": (obj.to_json() for obj in self.args),  # ty: ignore
            "kwargs": self.kwargs.copy(),
        }


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
        if all(isinstance(arg, (Ellipsoid, EllipsoidSet)) for arg in args):
            return FabricPlot_Point("create_point", *args, **kwargs)
        else:
            raise TypeError("Not valid arguments for Fabric plot point")

    @staticmethod
    def create_path(*args, **kwargs):
        """Create fabric plot path artist from EllipsoidSet data."""
        if all(isinstance(arg, EllipsoidSet) for arg in args):
            return FabricPlot_Path("create_path", *args, **kwargs)
        else:
            raise TypeError("Not valid arguments for Fabric plot path")
