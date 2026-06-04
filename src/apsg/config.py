from collections.abc import Mapping
from dataclasses import asdict, dataclass, field
from typing import Any


class BaseConfig(Mapping):
    def __getitem__(self, key: str) -> Any:
        try:
            return getattr(self, key)
        except AttributeError:
            raise KeyError(key)

    def __iter__(self):
        return iter(asdict(self))

    def __len__(self):
        return len(asdict(self))

    def __repr__(self):
        items = ", ".join(f"{k}={v!r}" for k, v in asdict(self).items())
        return f"{self.__class__.__name__}({items})"

    def update(self, config_dict: dict[str, Any]):
        for key, value in config_dict.items():
            if not hasattr(self, key):
                raise KeyError(f"'{key}' is not a valid configuration key.")
            current_value = getattr(self, key)
            if isinstance(current_value, BaseConfig) and isinstance(value, dict):
                current_value.update(value)
            else:
                setattr(self, key, value)

    def copy(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class StereonetMarkerConfig(BaseConfig):
    alpha: Any = None
    color: Any = None
    mec: Any = None
    mfc: Any = None
    ls: str = "none"
    marker: str = "o"
    mew: int = 1
    ms: int = 6


@dataclass
class StereonetConfig(BaseConfig):
    kind: str = "equal-area"
    overlay_position: tuple[float, float, float, float] = (0, 0, 0, 0)
    rotate_data: bool = False
    minor_ticks: Any = None
    major_ticks: Any = None
    overlay: bool = True
    overlay_step: int = 15
    overlay_resolution: int = 181
    clip_pole: int = 15
    hemisphere: str = "lower"
    grid_type: str = "gss"
    grid_n: int = 3000
    tight_layout: bool = False
    title_kws: dict = field(default_factory=dict)


@dataclass
class StereonetPointConfig(StereonetMarkerConfig):
    pass


@dataclass
class StereonetVectorConfig(StereonetMarkerConfig):
    mew: int = 2


@dataclass
class StereonetGreatCircleConfig(BaseConfig):
    alpha: Any = None
    color: Any = None
    ls: str = "-"
    lw: float = 1.5


@dataclass
class StereonetArcConfig(BaseConfig):
    alpha: Any = None
    color: Any = None
    ls: str = "-"
    lw: float = 1.5


@dataclass
class StereonetScatterConfig(BaseConfig):
    alpha: Any = None
    s: Any = None
    c: Any = None
    linewidths: float = 1.5
    marker: str = "o"
    cmap: Any = None
    legend: bool = False
    num: str = "auto"


@dataclass
class StereonetConeConfig(BaseConfig):
    alpha: Any = None
    color: Any = None
    ls: str = "-"
    lw: float = 1.5


@dataclass
class StereonetPairConfig(BaseConfig):
    alpha: Any = None
    color: Any = None
    ls: str = "-"
    lw: float = 1.5
    line_marker: str = "o"


@dataclass
class StereonetFaultConfig(BaseConfig):
    alpha: Any = None
    color: Any = None
    ls: str = "-"
    lw: float = 1.5


@dataclass
class StereonetHoeppnerConfig(StereonetMarkerConfig):
    ms: int = 5


@dataclass
class StereonetArrowConfig(BaseConfig):
    color: Any = None
    width: int = 2
    headwidth: int = 5
    pivot: str = "mid"
    units: str = "dots"


@dataclass
class StereonetTensorConfig(BaseConfig):
    planes: bool = True
    alpha: Any = None
    color: Any = None
    ls: str = "-"
    lw: float = 1.5
    marker: str = "o"
    mew: int = 1
    ms: int = 9


@dataclass
class StereonetStressConfig(BaseConfig):
    alpha: Any = None
    color: Any = None
    ls: str = "none"
    marker: str = "*"
    mew: int = 1
    ms: int = 12


@dataclass
class StereonetContourConfig(BaseConfig):
    alpha: Any = None
    antialiased: bool = True
    method: str = "sph"
    n_max: int = 6
    cmap: str = "Greys"
    levels: int = 6
    clines: bool = True
    linewidths: float = 1
    linestyles: Any = None
    colorbar: bool = False
    trimzero: bool = True
    sigma: Any = None
    sigmanorm: bool = True
    show_data: bool = False
    data_kws: dict = field(default_factory=dict)


@dataclass
class RoseplotConfig(BaseConfig):
    bins: int = 36
    density: bool = True
    arrowness: float = 0.95
    rwidth: float = 1
    scaled: bool = False
    kappa: int = 250
    pdf_res: int = 901
    title: Any = None
    grid: bool = True
    grid_kws: dict = field(default_factory=dict)
    tight_layout: bool = False
    title_kws: dict = field(default_factory=dict)


@dataclass
class RoseplotBarConfig(BaseConfig):
    alpha: Any = None
    color: Any = None
    ec: Any = None
    fc: Any = None
    ls: str = "-"
    lw: float = 1.5
    legend: bool = False


@dataclass
class RoseplotPdfConfig(BaseConfig):
    alpha: Any = None
    color: Any = None
    ec: Any = None
    fc: Any = None
    ls: str = "-"
    lw: float = 1.5
    legend: bool = False


@dataclass
class RoseplotMuciConfig(BaseConfig):
    confidence_level: int = 95
    alpha: Any = None
    color: str = "r"
    ls: str = "-"
    lw: float = 1.5
    n_resamples: int = 9999


@dataclass
class FabricplotConfig(BaseConfig):
    ticks: bool = True
    n_ticks: int = 10
    tick_size: float = 0.2
    margin: float = 0.05
    grid: bool = True
    grid_color: str = "k"
    grid_style: str = ":"
    title: Any = None
    tight_layout: bool = False
    title_kws: dict = field(default_factory=dict)


@dataclass
class FabricplotPointConfig(BaseConfig):
    alpha: Any = None
    color: Any = None
    mec: Any = None
    mfc: Any = None
    ls: str = "none"
    marker: str = "o"
    mew: int = 1
    ms: int = 8


@dataclass
class FabricplotPathConfig(BaseConfig):
    alpha: Any = None
    color: Any = None
    ls: str = "-"
    lw: float = 1.5
    marker: Any = None
    mec: Any = None
    mew: int = 1
    mfc: Any = None
    ms: int = 6


@dataclass
class AppConfig(BaseConfig):
    notation: str = "dd"
    vec2geo: bool = False
    ndigits: int = 3
    figsize: tuple = (8, 6)
    dpi: int = 100
    facecolor: str = "white"
    stereonet: StereonetConfig = field(default_factory=StereonetConfig)
    stereonet_point: StereonetPointConfig = field(default_factory=StereonetPointConfig)
    stereonet_vector: StereonetVectorConfig = field(
        default_factory=StereonetVectorConfig
    )
    stereonet_great_circle: StereonetGreatCircleConfig = field(
        default_factory=StereonetGreatCircleConfig
    )
    stereonet_arc: StereonetArcConfig = field(default_factory=StereonetArcConfig)
    stereonet_scatter: StereonetScatterConfig = field(
        default_factory=StereonetScatterConfig
    )
    stereonet_cone: StereonetConeConfig = field(default_factory=StereonetConeConfig)
    stereonet_pair: StereonetPairConfig = field(default_factory=StereonetPairConfig)
    stereonet_fault: StereonetFaultConfig = field(default_factory=StereonetFaultConfig)
    stereonet_hoeppner: StereonetHoeppnerConfig = field(
        default_factory=StereonetHoeppnerConfig
    )
    stereonet_arrow: StereonetArrowConfig = field(default_factory=StereonetArrowConfig)
    stereonet_tensor: StereonetTensorConfig = field(
        default_factory=StereonetTensorConfig
    )
    stereonet_stress: StereonetStressConfig = field(
        default_factory=StereonetStressConfig
    )
    stereonet_contour: StereonetContourConfig = field(
        default_factory=StereonetContourConfig
    )
    roseplot: RoseplotConfig = field(default_factory=RoseplotConfig)
    roseplot_bar: RoseplotBarConfig = field(default_factory=RoseplotBarConfig)
    roseplot_pdf: RoseplotPdfConfig = field(default_factory=RoseplotPdfConfig)
    roseplot_muci: RoseplotMuciConfig = field(default_factory=RoseplotMuciConfig)
    fabricplot: FabricplotConfig = field(default_factory=FabricplotConfig)
    fabricplot_point: FabricplotPointConfig = field(
        default_factory=FabricplotPointConfig
    )
    fabricplot_path: FabricplotPathConfig = field(default_factory=FabricplotPathConfig)


apsg_conf = AppConfig()
