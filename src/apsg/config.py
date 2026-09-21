from collections.abc import Mapping
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from typing import Any


class BaseConfig(Mapping):
    """Base configuration class with dict-like access."""

    def __getitem__(self, key: str) -> Any:
        """Get item by key."""
        try:
            return getattr(self, key)
        except AttributeError:
            raise KeyError(key)

    def __iter__(self):
        """Iterate over configuration keys."""
        return iter(asdict(self))

    def __len__(self):
        """Return number of configuration items."""
        return len(asdict(self))

    def __repr__(self):
        """Return string representation."""
        items = ", ".join(f"{k}={v!r}" for k, v in asdict(self).items())
        return f"{self.__class__.__name__}({items})"

    def update(self, config_dict: dict[str, Any]):
        """Update configuration with given mapping."""
        for key, value in config_dict.items():
            if not hasattr(self, key):
                raise KeyError(f"'{key}' is not a valid configuration key.")
            current_value = getattr(self, key)
            if isinstance(current_value, BaseConfig) and isinstance(value, dict):
                current_value.update(value)
            else:
                setattr(self, key, value)

    def copy(self) -> dict[str, Any]:
        """Return a copy as a dictionary."""
        return asdict(self)


@dataclass
class StereonetMarkerConfig(BaseConfig):
    """Stereonet marker style configuration."""

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
    """Stereonet global configuration."""

    kind: str = "equal-area"
    rotation: Any = None
    rotate_data: bool = True
    grid: bool = True
    grid_step: int = 15
    grid_color: str = "grey"
    grid_style: str = ":"
    clip_pole: int = 15
    hemisphere: str = "lower"
    primitive_lw: float = 1.5
    primitive_color: Any = None
    azimuth_ticks: bool = True
    azimuth_ticks_kws: dict = field(default_factory=dict)
    tight_layout: bool = False
    title_kws: dict = field(default_factory=dict)
    legend_kws: dict = field(default_factory=dict)


@dataclass
class StereonetPointConfig(StereonetMarkerConfig):
    """Stereonet point marker style configuration."""

    pass


@dataclass
class StereonetVectorConfig(StereonetMarkerConfig):
    """Stereonet vector marker style configuration."""

    mew: int = 2


@dataclass
class StereonetGreatCircleConfig(BaseConfig):
    """Stereonet great circle line style configuration."""

    alpha: Any = None
    color: Any = None
    ls: str = "-"
    lw: float = 1.5


@dataclass
class StereonetArcConfig(BaseConfig):
    """Stereonet arc style configuration (line, points or filled)."""

    kind: str = "line"  # "line", "points" or "filled"
    alpha: Any = None
    color: Any = None
    ls: str = "-"
    lw: float = 1.5
    marker: str = "o"
    ms: int = 6
    mec: Any = None
    mfc: Any = None
    mew: int = 1


@dataclass
class StereonetScatterConfig(BaseConfig):
    """Stereonet scatter plot configuration."""

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
    """Stereonet cone boundary line style configuration."""

    alpha: Any = None
    color: Any = None
    ls: str = "-"
    lw: float = 1.5


@dataclass
class StereonetConfidenceConfig(BaseConfig):
    """Stereonet confidence cone/ellipse style configuration."""

    method: str = "fisher"
    which: Any = None
    level: float = 0.95
    n_resamples: int = 1000
    normalize: bool = False
    anisoft: bool = False
    alpha: Any = None
    color: Any = None
    ls: str = "--"
    lw: float = 1.5


@dataclass
class StereonetPairConfig(BaseConfig):
    """Stereonet pair line and marker style configuration."""

    alpha: Any = None
    color: Any = None
    ls: str = "-"
    lw: float = 1.5
    line_marker: str = "o"


@dataclass
class StereonetFaultConfig(BaseConfig):
    """Stereonet fault plane line style configuration."""

    alpha: Any = None
    color: Any = None
    ls: str = "-"
    lw: float = 1.5


@dataclass
class StereonetHoeppnerConfig(StereonetMarkerConfig):
    """Stereonet Hoeppner plot marker style configuration."""

    ms: int = 4
    pivot: str = "middle"


@dataclass
class StereonetArrowConfig(BaseConfig):
    """Stereonet arrow style configuration."""

    color: Any = None
    width: int = 2
    headwidth: int = 5
    pivot: str = "mid"
    units: str = "dots"


@dataclass
class StereonetTensorConfig(BaseConfig):
    """Stereonet tensor plot style configuration."""

    planes: bool = True
    alpha: Any = None
    color: Any = None
    ls: str = "-"
    lw: float = 1.5
    marker: str = "o"
    mec: Any = None
    mew: int = 1
    ms: int = 9


@dataclass
class StereonetStressConfig(BaseConfig):
    """Stereonet stress axes marker style configuration."""

    alpha: Any = None
    color: Any = None
    ls: str = "none"
    marker: str = "*"
    mec: Any = None
    mew: int = 1
    ms: int = 12


@dataclass
class StereonetContourConfig(BaseConfig):
    """Stereonet contour plot configuration."""

    alpha: Any = None
    antialiased: bool = True
    method: str = "kamb"
    n_max: Any = None
    cmap: Any = None
    levels: int = 6
    filled: bool = True
    linewidth: float = 1
    linestyles: Any = None
    colorbar: bool = False
    colorbar_kws: dict = field(
        default_factory=lambda: {"shrink": 0.5, "anchor": (0.0, 0.3)}
    )
    line_color: str = "k"
    sigma: Any = None
    clip: bool = True


@dataclass
class StereogridConfig(BaseConfig):
    """StereoGrid global configuration."""

    type: str = "gss"
    n: int = 3000


@dataclass
class RoseplotConfig(BaseConfig):
    """Rose plot global configuration."""

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
    legend_kws: dict = field(default_factory=dict)


@dataclass
class RoseplotBarConfig(BaseConfig):
    """Rose plot bar style configuration."""

    alpha: Any = None
    color: Any = None
    ec: Any = None
    fc: Any = None
    ls: str = "-"
    lw: float = 1.5
    legend: bool = False


@dataclass
class RoseplotPdfConfig(BaseConfig):
    """Rose plot PDF line style configuration."""

    alpha: Any = None
    color: Any = None
    ec: Any = None
    fc: Any = None
    ls: str = "-"
    lw: float = 1.5
    legend: bool = False


@dataclass
class RoseplotMuciConfig(BaseConfig):
    """Rose plot confidence interval line style configuration."""

    confidence_level: int = 95
    alpha: Any = None
    color: str = "r"
    ls: str = "-"
    lw: float = 1.5
    n_resamples: int = 9999


@dataclass
class FabricplotConfig(BaseConfig):
    """Fabric plot global configuration."""

    ticks: bool = True
    n_ticks: int = 10
    tick_size: float = 0.2
    margin: float = 0.05
    grid: bool = True
    grid_color: str = "k"
    grid_style: str = ":"
    border_color: str = "k"
    border_lw: float = 2
    tick_color: str = "k"
    tick_lw: float = 1
    label_fontsize: int = 14
    refline_color: str = "k"
    refline_lw: float = 0.5
    background_color: str = "w"
    title: Any = None
    tight_layout: bool = False
    title_kws: dict = field(default_factory=dict)
    legend_kws: dict = field(default_factory=dict)


@dataclass
class FabricplotPointConfig(BaseConfig):
    """Fabric plot point marker style configuration."""

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
    """Fabric plot path line style configuration."""

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
    """Top-level application configuration."""

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
    stereonet_confidence: StereonetConfidenceConfig = field(
        default_factory=StereonetConfidenceConfig
    )
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
    stereogrid: StereogridConfig = field(default_factory=StereogridConfig)
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


@contextmanager
def apsg_conf_context(**kwargs):
    """Context manager to temporarily override configuration."""
    saved = apsg_conf.copy()
    apsg_conf.update(kwargs)
    try:
        yield
    finally:
        apsg_conf.update(saved)
