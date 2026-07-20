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
        """Initialize stereonet style with kwargs."""
        self.kwargs.update((k, kwargs[k]) for k in self.kwargs.keys() & kwargs.keys())
        label = kwargs.get("label", None)
        if label is not None:
            if isinstance(label, str):
                self.kwargs["label"] = label

    def __repr__(self):
        """Return string representation of style."""
        return f"{self.__class__.__name__}: {self.kwargs}"


class StereoNetPointStyle(StereoNetStyle):
    def __init__(self, **kwargs):
        """Initialize stereonet point style."""
        self._valid = (Vector3, Vector3Set)
        self.kwargs = getattr(apsg_conf, "stereonet_point").copy()
        super().__init__(**kwargs)

    def create_artist(self, *args):
        """Create stereonet point artist with configured style."""
        filtered = (arg for arg in args if isinstance(arg, self._valid))
        return StereoNetArtistFactory.create_point(*filtered, **self.kwargs)


class StereoNetScatterStyle(StereoNetStyle):
    def __init__(self, **kwargs):
        """Initialize stereonet scatter style."""
        self._valid = (Vector3, Vector3Set)
        self.kwargs = getattr(apsg_conf, "stereonet_scatter").copy()
        super().__init__(**kwargs)

    def create_artist(self, *args):
        """Create stereonet scatter artist with configured style."""
        filtered = (arg for arg in args if isinstance(arg, self._valid))
        return StereoNetArtistFactory.create_pole(*filtered, **self.kwargs)


class StereoNetVectorStyle(StereoNetStyle):
    def __init__(self, **kwargs):
        """Initialize stereonet vector style."""
        self._valid = (Vector3, Vector3Set)
        self.kwargs = getattr(apsg_conf, "stereonet_vector").copy()
        super().__init__(**kwargs)

    def create_artist(self, *args):
        """Create stereonet vector artist with configured style."""
        filtered = (arg for arg in args if isinstance(arg, self._valid))
        return StereoNetArtistFactory.create_pole(*filtered, **self.kwargs)


class StereoNetGreatCircleStyle(StereoNetStyle):
    def __init__(self, **kwargs):
        """Initialize stereonet great circle style."""
        self._valid = (Foliation, FoliationSet)
        self.kwargs = getattr(apsg_conf, "stereonet_great_circle").copy()
        super().__init__(**kwargs)

    def create_artist(self, *args):
        """Create stereonet great circle artist with configured style."""
        filtered = (arg for arg in args if isinstance(arg, self._valid))
        return StereoNetArtistFactory.create_great_circle(*filtered, **self.kwargs)


class StereoNetArcStyle(StereoNetStyle):
    def __init__(self, **kwargs):
        """Initialize stereonet arc style."""
        self._valid = (Foliation, FoliationSet)
        self.kwargs = getattr(apsg_conf, "stereonet_arc").copy()
        super().__init__(**kwargs)

    def create_artist(self, *args):
        """Create stereonet arc artist with configured style."""
        filtered = (arg for arg in args if isinstance(arg, self._valid))
        return StereoNetArtistFactory.create_arc(*filtered, **self.kwargs)


class StereoNetConeStyle(StereoNetStyle):
    def __init__(self, **kwargs):
        """Initialize stereonet cone style."""
        self._valid = (Cone, ConeSet)
        self.kwargs = getattr(apsg_conf, "stereonet_cone").copy()
        super().__init__(**kwargs)

    def create_artist(self, *args):
        """Create stereonet cone artist with configured style."""
        filtered = (arg for arg in args if isinstance(arg, self._valid))
        return StereoNetArtistFactory.create_cone(*filtered, **self.kwargs)


class StereoNetBinghamStyle(StereoNetStyle):
    def __init__(self, **kwargs):
        """Initialize stereonet Bingham confidence ellipse style."""
        self._valid = Vector3Set
        self.kwargs = getattr(apsg_conf, "stereonet_bingham").copy()
        super().__init__(**kwargs)

    def create_artist(self, *args):
        """Create stereonet Bingham confidence ellipse artist with configured style."""
        filtered = (arg for arg in args if isinstance(arg, self._valid))
        return StereoNetArtistFactory.create_bingham(*filtered, **self.kwargs)


class StereoNetPairStyle(StereoNetStyle):
    def __init__(self, **kwargs):
        """Initialize stereonet pair style."""
        self._valid = (Pair, PairSet)
        self.kwargs = getattr(apsg_conf, "stereonet_pair").copy()
        super().__init__(**kwargs)

    def create_artist(self, *args):
        """Create stereonet pair artist with configured style."""
        filtered = (arg for arg in args if isinstance(arg, self._valid))
        return StereoNetArtistFactory.create_pair(*filtered, **self.kwargs)


class StereoNetFaultStyle(StereoNetStyle):
    def __init__(self, **kwargs):
        """Initialize stereonet fault style."""
        self._valid = (Fault, FaultSet)
        self.kwargs = getattr(apsg_conf, "stereonet_fault").copy()
        super().__init__(**kwargs)

    def create_artist(self, *args):
        """Create stereonet fault artist with configured style."""
        filtered = (arg for arg in args if isinstance(arg, self._valid))
        return StereoNetArtistFactory.create_fault(*filtered, **self.kwargs)


class StereoNetHoeppnerStyle(StereoNetStyle):
    def __init__(self, **kwargs):
        """Initialize stereonet Hoeppner plot style."""
        self._valid = (Fault, FaultSet)
        self.kwargs = getattr(apsg_conf, "stereonet_hoeppner").copy()
        super().__init__(**kwargs)

    def create_artist(self, *args):
        """Create stereonet Hoeppner plot artist with configured style."""
        filtered = (arg for arg in args if isinstance(arg, self._valid))
        return StereoNetArtistFactory.create_hoeppner(*filtered, **self.kwargs)


class StereoNetArrowStyle(StereoNetStyle):
    def __init__(self, **kwargs):
        """Initialize stereonet arrow style."""
        self._valid = (Vector3, Vector3Set)
        self.kwargs = getattr(apsg_conf, "stereonet_arrow").copy()
        super().__init__(**kwargs)

    def create_artist(self, *args):
        """Create stereonet arrow artist with configured style."""
        filtered = (arg for arg in args if isinstance(arg, self._valid))
        return StereoNetArtistFactory.create_arrow(*filtered, **self.kwargs)


class StereoNetTensorStyle(StereoNetStyle):
    def __init__(self, **kwargs):
        """Initialize stereonet tensor style."""
        self._valid = Tensor3
        self.kwargs = getattr(apsg_conf, "stereonet_tensor").copy()
        super().__init__(**kwargs)

    def create_artist(self, *args):
        """Create stereonet tensor artist with configured style."""
        filtered = (arg for arg in args if isinstance(arg, self._valid))
        return StereoNetArtistFactory.create_tensor(*filtered, **self.kwargs)


class StereoNetStressStyle(StereoNetStyle):
    def __init__(self, **kwargs):
        """Initialize stereonet stress style."""
        self._valid = Stress3
        self.kwargs = getattr(apsg_conf, "stereonet_stress").copy()
        super().__init__(**kwargs)

    def create_artist(self, *args):
        """Create stereonet stress artist with configured style."""
        filtered = (arg for arg in args if isinstance(arg, self._valid))
        return StereoNetArtistFactory.create_stress(*filtered, **self.kwargs)


class StereoNetContourStyle(StereoNetStyle):
    def __init__(self, **kwargs):
        """Initialize stereonet contour style."""
        self._valid = (Vector3, Vector3Set)
        self.kwargs = getattr(apsg_conf, "stereonet_contour").copy()
        super().__init__(**kwargs)

    def create_artist(self, *args):
        """Create stereonet contour artist with configured style."""
        filtered = (arg for arg in args if isinstance(arg, self._valid))
        return StereoNetArtistFactory.create_contour(*filtered, **self.kwargs)


class StereoNetStyleFactory:
    @staticmethod
    def point(**kwargs):
        """Return StereoNetPointStyle with given kwargs."""
        return StereoNetPointStyle(**kwargs)

    @staticmethod
    def vector(**kwargs):
        """Return StereoNetVectorStyle with given kwargs."""
        return StereoNetVectorStyle(**kwargs)

    @staticmethod
    def scatter(**kwargs):
        """Return StereoNetScatterStyle with given kwargs."""
        return StereoNetScatterStyle(**kwargs)

    @staticmethod
    def great_circle(**kwargs):
        """Return StereoNetGreatCircleStyle with given kwargs."""
        return StereoNetGreatCircleStyle(**kwargs)

    gc = great_circle

    @staticmethod
    def arc(**kwargs):
        """Return StereoNetArcStyle with given kwargs."""
        return StereoNetArcStyle(**kwargs)

    @staticmethod
    def cone(**kwargs):
        """Return StereoNetConeStyle with given kwargs."""
        return StereoNetConeStyle(**kwargs)

    @staticmethod
    def bingham(**kwargs):
        """Return StereoNetBinghamStyle with given kwargs."""
        return StereoNetBinghamStyle(**kwargs)

    @staticmethod
    def pair(**kwargs):
        """Return StereoNetPairStyle with given kwargs."""
        return StereoNetPairStyle(**kwargs)

    @staticmethod
    def fault(**kwargs):
        """Return StereoNetFaultStyle with given kwargs."""
        return StereoNetFaultStyle(**kwargs)

    @staticmethod
    def hoeppner(**kwargs):
        """Return StereoNetHoeppnerStyle with given kwargs."""
        return StereoNetHoeppnerStyle(**kwargs)

    @staticmethod
    def arrow(**kwargs):
        """Return StereoNetArrowStyle with given kwargs."""
        return StereoNetArrowStyle(**kwargs)

    @staticmethod
    def tensor(**kwargs):
        """Return StereoNetTensorStyle with given kwargs."""
        return StereoNetTensorStyle(**kwargs)

    @staticmethod
    def stress(**kwargs):
        """Return StereoNetStressStyle with given kwargs."""
        return StereoNetStressStyle(**kwargs)

    @staticmethod
    def contour(**kwargs):
        """Return StereoNetContourStyle with given kwargs."""
        return StereoNetContourStyle(**kwargs)


class RosePlotStyle:
    def __init__(self, **kwargs):
        """Initialize rose plot style with kwargs."""
        self.kwargs.update((k, kwargs[k]) for k in self.kwargs.keys() & kwargs.keys())
        label = kwargs.get("label", None)
        if label is not None:
            if isinstance(label, str):
                self.kwargs["label"] = label

    def __repr__(self):
        """Return string representation of style."""
        return f"{self.__class__.__name__}: {self.kwargs}"


class RosePlotBarStyle(RosePlotStyle):
    def __init__(self, **kwargs):
        """Initialize rose plot bar style."""
        self._valid = Vector2Set
        self.kwargs = getattr(apsg_conf, "roseplot_bar").copy()
        super().__init__(**kwargs)

    def create_artist(self, *args):
        """Create rose plot bar artist with configured style."""
        filtered = (arg for arg in args if isinstance(arg, self._valid))
        return RosePlotArtistFactory.create_bar(*filtered, **self.kwargs)


class RosePlotPdfStyle(RosePlotStyle):
    def __init__(self, **kwargs):
        """Initialize rose plot PDF style."""
        self._valid = Vector2Set
        self.kwargs = getattr(apsg_conf, "roseplot_pdf").copy()
        super().__init__(**kwargs)

    def create_artist(self, *args):
        """Create rose plot PDF artist with configured style."""
        filtered = (arg for arg in args if isinstance(arg, self._valid))
        return RosePlotArtistFactory.create_pdf(*filtered, **self.kwargs)


class RosePlotMuciStyle(RosePlotStyle):
    def __init__(self, **kwargs):
        """Initialize rose plot muci style."""
        self._valid = Vector2Set
        self.kwargs = getattr(apsg_conf, "roseplot_muci").copy()
        super().__init__(**kwargs)

    def create_artist(self, *args):
        """Create rose plot muci artist with configured style."""
        filtered = (arg for arg in args if isinstance(arg, self._valid))
        return RosePlotArtistFactory.create_muci(*filtered, **self.kwargs)


class RosePlotStyleFactory:
    @staticmethod
    def bar(**kwargs):
        """Return RosePlotBarStyle with given kwargs."""
        return RosePlotBarStyle(**kwargs)

    @staticmethod
    def pdf(**kwargs):
        """Return RosePlotPdfStyle with given kwargs."""
        return RosePlotPdfStyle(**kwargs)

    @staticmethod
    def muci(**kwargs):
        """Return RosePlotMuciStyle with given kwargs."""
        return RosePlotMuciStyle(**kwargs)


class FabricPlotStyle:
    def __init__(self, **kwargs):
        """Initialize fabric plot style with kwargs."""
        self.kwargs.update((k, kwargs[k]) for k in self.kwargs.keys() & kwargs.keys())
        label = kwargs.get("label", None)
        if label is not None:
            if isinstance(label, str):
                self.kwargs["label"] = label

    def __repr__(self):
        """Return string representation of style."""
        return f"{self.__class__.__name__}: {self.kwargs}"


class FabricPlotPointStyle(FabricPlotStyle):
    def __init__(self, **kwargs):
        """Initialize fabric plot point style."""
        self._valid = (Ellipsoid, EllipsoidSet)
        self.kwargs = getattr(apsg_conf, "fabricplot_point").copy()
        super().__init__(**kwargs)

    def create_artist(self, *args):
        """Create fabric plot point artist with configured style."""
        filtered = (arg for arg in args if isinstance(arg, self._valid))
        return FabricPlotArtistFactory.create_bar(*filtered, **self.kwargs)


class FabricPlotPathStyle(FabricPlotStyle):
    def __init__(self, **kwargs):
        """Initialize fabric plot path style."""
        self._valid = EllipsoidSet
        self.kwargs = getattr(apsg_conf, "fabricplot_path").copy()
        super().__init__(**kwargs)

    def create_artist(self, *args):
        """Create fabric plot path artist with configured style."""
        filtered = (arg for arg in args if isinstance(arg, self._valid))
        return FabricPlotArtistFactory.create_pdf(*filtered, **self.kwargs)


class FabricPlotStyleFactory:
    @staticmethod
    def point(**kwargs):
        """Return FabricPlotPointStyle with given kwargs."""
        return FabricPlotPointStyle(**kwargs)

    @staticmethod
    def path(**kwargs):
        """Return FabricPlotPathStyle with given kwargs."""
        return FabricPlotPathStyle(**kwargs)
