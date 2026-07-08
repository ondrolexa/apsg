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
from apsg.plotting._plot_artists import (
    FabricPlotArtistFactory,
    RosePlotArtistFactory,
    StereoNetArtistFactory,
)


class StereoNetStyle:
    def __init__(self, **kwargs):
        self.kwargs.update((k, kwargs[k]) for k in self.kwargs.keys() & kwargs.keys())
        label = kwargs.get("label", None)
        if label is not None:
            if isinstance(label, str):
                self.kwargs["label"] = label

    def __repr__(self):
        return f"{self.__class__.__name__}: {self.kwargs}"


class StereoNetPointStyle(StereoNetStyle):
    def __init__(self, **kwargs):
        self._valid = (Vector3, Vector3Set)
        self.kwargs = getattr(apsg_conf, "stereonet_point").copy()
        super().__init__(**kwargs)

    def create_artist(self, *args):
        filtered = (arg for arg in args if isinstance(arg, self._valid))
        return StereoNetArtistFactory.create_point(*filtered, **self.kwargs)


class StereoNetScatterStyle(StereoNetStyle):
    def __init__(self, **kwargs):
        self._valid = (Vector3, Vector3Set)
        self.kwargs = getattr(apsg_conf, "stereonet_scatter").copy()
        super().__init__(**kwargs)

    def create_artist(self, *args):
        filtered = (arg for arg in args if isinstance(arg, self._valid))
        return StereoNetArtistFactory.create_pole(*filtered, **self.kwargs)


class StereoNetVectorStyle(StereoNetStyle):
    def __init__(self, **kwargs):
        self._valid = (Vector3, Vector3Set)
        self.kwargs = getattr(apsg_conf, "stereonet_vector").copy()
        super().__init__(**kwargs)

    def create_artist(self, *args):
        filtered = (arg for arg in args if isinstance(arg, self._valid))
        return StereoNetArtistFactory.create_pole(*filtered, **self.kwargs)


class StereoNetGreatCircleStyle(StereoNetStyle):
    def __init__(self, **kwargs):
        self._valid = (Foliation, FoliationSet)
        self.kwargs = getattr(apsg_conf, "stereonet_great_circle").copy()
        super().__init__(**kwargs)

    def create_artist(self, *args):
        filtered = (arg for arg in args if isinstance(arg, self._valid))
        return StereoNetArtistFactory.create_great_circle(*filtered, **self.kwargs)


class StereoNetArcStyle(StereoNetStyle):
    def __init__(self, **kwargs):
        self._valid = (Foliation, FoliationSet)
        self.kwargs = getattr(apsg_conf, "stereonet_arc").copy()
        super().__init__(**kwargs)

    def create_artist(self, *args):
        filtered = (arg for arg in args if isinstance(arg, self._valid))
        return StereoNetArtistFactory.create_arc(*filtered, **self.kwargs)


class StereoNetConeStyle(StereoNetStyle):
    def __init__(self, **kwargs):
        self._valid = (Cone, ConeSet)
        self.kwargs = getattr(apsg_conf, "stereonet_cone").copy()
        super().__init__(**kwargs)

    def create_artist(self, *args):
        filtered = (arg for arg in args if isinstance(arg, self._valid))
        return StereoNetArtistFactory.create_cone(*filtered, **self.kwargs)


class StereoNetPairStyle(StereoNetStyle):
    def __init__(self, **kwargs):
        self._valid = (Pair, PairSet)
        self.kwargs = getattr(apsg_conf, "stereonet_pair").copy()
        super().__init__(**kwargs)

    def create_artist(self, *args):
        filtered = (arg for arg in args if isinstance(arg, self._valid))
        return StereoNetArtistFactory.create_pair(*filtered, **self.kwargs)


class StereoNetFaultStyle(StereoNetStyle):
    def __init__(self, **kwargs):
        self._valid = (Fault, FaultSet)
        self.kwargs = getattr(apsg_conf, "stereonet_fault").copy()
        super().__init__(**kwargs)

    def create_artist(self, *args):
        filtered = (arg for arg in args if isinstance(arg, self._valid))
        return StereoNetArtistFactory.create_fault(*filtered, **self.kwargs)


class StereoNetHoeppnerStyle(StereoNetStyle):
    def __init__(self, **kwargs):
        self._valid = (Fault, FaultSet)
        self.kwargs = getattr(apsg_conf, "stereonet_hoeppner").copy()
        super().__init__(**kwargs)

    def create_artist(self, *args):
        filtered = (arg for arg in args if isinstance(arg, self._valid))
        return StereoNetArtistFactory.create_hoeppner(*filtered, **self.kwargs)


class StereoNetArrowStyle(StereoNetStyle):
    def __init__(self, **kwargs):
        self._valid = (Vector3, Vector3Set)
        self.kwargs = getattr(apsg_conf, "stereonet_arrow").copy()
        super().__init__(**kwargs)

    def create_artist(self, *args):
        filtered = (arg for arg in args if isinstance(arg, self._valid))
        return StereoNetArtistFactory.create_arrow(*filtered, **self.kwargs)


class StereoNetTensorStyle(StereoNetStyle):
    def __init__(self, **kwargs):
        self._valid = Tensor3
        self.kwargs = getattr(apsg_conf, "stereonet_tensor").copy()
        super().__init__(**kwargs)

    def create_artist(self, *args):
        filtered = (arg for arg in args if isinstance(arg, self._valid))
        return StereoNetArtistFactory.create_tensor(*filtered, **self.kwargs)


class StereoNetStressStyle(StereoNetStyle):
    def __init__(self, **kwargs):
        self._valid = Stress3
        self.kwargs = getattr(apsg_conf, "stereonet_stress").copy()
        super().__init__(**kwargs)

    def create_artist(self, *args):
        filtered = (arg for arg in args if isinstance(arg, self._valid))
        return StereoNetArtistFactory.create_stress(*filtered, **self.kwargs)


class StereoNetContourStyle(StereoNetStyle):
    def __init__(self, **kwargs):
        self._valid = (Vector3, Vector3Set)
        self.kwargs = getattr(apsg_conf, "stereonet_contour").copy()
        super().__init__(**kwargs)

    def create_artist(self, *args):
        filtered = (arg for arg in args if isinstance(arg, self._valid))
        return StereoNetArtistFactory.create_contour(*filtered, **self.kwargs)


class StereoNetStyleFactory:
    @staticmethod
    def point(**kwargs):
        return StereoNetPointStyle(**kwargs)

    @staticmethod
    def vector(**kwargs):
        return StereoNetVectorStyle(**kwargs)

    @staticmethod
    def scatter(**kwargs):
        return StereoNetScatterStyle(**kwargs)

    @staticmethod
    def great_circle(**kwargs):
        return StereoNetGreatCircleStyle(**kwargs)

    gc = great_circle

    @staticmethod
    def arc(**kwargs):
        return StereoNetArcStyle(**kwargs)

    @staticmethod
    def cone(**kwargs):
        return StereoNetConeStyle(**kwargs)

    @staticmethod
    def pair(**kwargs):
        return StereoNetPairStyle(**kwargs)

    @staticmethod
    def fault(**kwargs):
        return StereoNetFaultStyle(**kwargs)

    @staticmethod
    def hoeppner(**kwargs):
        return StereoNetHoeppnerStyle(**kwargs)

    @staticmethod
    def arrow(**kwargs):
        return StereoNetArrowStyle(**kwargs)

    @staticmethod
    def tensor(**kwargs):
        return StereoNetTensorStyle(**kwargs)

    @staticmethod
    def stress(**kwargs):
        return StereoNetStressStyle(**kwargs)

    @staticmethod
    def contour(**kwargs):
        return StereoNetContourStyle(**kwargs)


class RosePlotStyle:
    def __init__(self, **kwargs):
        self.kwargs.update((k, kwargs[k]) for k in self.kwargs.keys() & kwargs.keys())
        label = kwargs.get("label", None)
        if label is not None:
            if isinstance(label, str):
                self.kwargs["label"] = label

    def __repr__(self):
        return f"{self.__class__.__name__}: {self.kwargs}"


class RosePlotBarStyle(RosePlotStyle):
    def __init__(self, **kwargs):
        self._valid = Vector2Set
        self.kwargs = getattr(apsg_conf, "roseplot_bar").copy()
        super().__init__(**kwargs)

    def create_artist(self, *args):
        filtered = (arg for arg in args if isinstance(arg, self._valid))
        return RosePlotArtistFactory.create_bar(*filtered, **self.kwargs)


class RosePlotPdfStyle(RosePlotStyle):
    def __init__(self, **kwargs):
        self._valid = Vector2Set
        self.kwargs = getattr(apsg_conf, "roseplot_pdf").copy()
        super().__init__(**kwargs)

    def create_artist(self, *args):
        filtered = (arg for arg in args if isinstance(arg, self._valid))
        return RosePlotArtistFactory.create_pdf(*filtered, **self.kwargs)


class RosePlotMuciStyle(RosePlotStyle):
    def __init__(self, **kwargs):
        self._valid = Vector2Set
        self.kwargs = getattr(apsg_conf, "roseplot_muci").copy()
        super().__init__(**kwargs)

    def create_artist(self, *args):
        filtered = (arg for arg in args if isinstance(arg, self._valid))
        return RosePlotArtistFactory.create_muci(*filtered, **self.kwargs)


class RosePlotStyleFactory:
    @staticmethod
    def bar(**kwargs):
        return RosePlotBarStyle(**kwargs)

    @staticmethod
    def pdf(**kwargs):
        return RosePlotPdfStyle(**kwargs)

    @staticmethod
    def muci(**kwargs):
        return RosePlotMuciStyle(**kwargs)


class FabricPlotStyle:
    def __init__(self, **kwargs):
        self.kwargs.update((k, kwargs[k]) for k in self.kwargs.keys() & kwargs.keys())
        label = kwargs.get("label", None)
        if label is not None:
            if isinstance(label, str):
                self.kwargs["label"] = label

    def __repr__(self):
        return f"{self.__class__.__name__}: {self.kwargs}"


class FabricPlotPointStyle(FabricPlotStyle):
    def __init__(self, **kwargs):
        self._valid = (Ellipsoid, EllipsoidSet)
        self.kwargs = getattr(apsg_conf, "fabricplot_point").copy()
        super().__init__(**kwargs)

    def create_artist(self, *args):
        filtered = (arg for arg in args if isinstance(arg, self._valid))
        return FabricPlotArtistFactory.create_bar(*filtered, **self.kwargs)


class FabricPlotPathStyle(FabricPlotStyle):
    def __init__(self, **kwargs):
        self._valid = EllipsoidSet
        self.kwargs = getattr(apsg_conf, "fabricplot_path").copy()
        super().__init__(**kwargs)

    def create_artist(self, *args):
        filtered = (arg for arg in args if isinstance(arg, self._valid))
        return FabricPlotArtistFactory.create_pdf(*filtered, **self.kwargs)


class FabricPlotStyleFactory:
    @staticmethod
    def point(**kwargs):
        return FabricPlotPointStyle(**kwargs)

    @staticmethod
    def path(**kwargs):
        return FabricPlotPathStyle(**kwargs)
