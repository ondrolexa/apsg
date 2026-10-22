import numbers
from collections.abc import Mapping, Sequence
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from typing import Any

NOTATIONS = ("dd", "rhr", "quadrant")
STRESS_CONVENTIONS = ("geological", "mechanical")
NET_KINDS = ("equal-area", "schmidt", "earea", "equal-angle", "wulff", "eangle")
HEMISPHERES = ("lower", "upper")
LEGEND_POSITIONS = ("right", "left", "top", "bottom")
GRID_TYPES = ("gss", "sfs")
CONTOUR_METHODS = ("kamb", "sph")


# validated configuration fields: the check, and what it expects for the message


def _number(value):
    return isinstance(value, numbers.Real) and not isinstance(value, bool)


def _checked(default, check, expect, **kwargs):
    return field(default=default, metadata={"check": check, "expect": expect}, **kwargs)


def _choice(default, options):
    return _checked(default, lambda v: v in options, f"one of {options}")


def _positive(default, integer=False, minimum=0, strict=True):
    kind = int if integer else numbers.Real

    def check(v):
        if not isinstance(v, kind) or isinstance(v, bool):
            return False
        return v > minimum if strict else v >= minimum

    sign = ">" if strict else ">="
    expect = f"{'an integer' if integer else 'a number'} {sign} {minimum}"
    return _checked(default, check, expect)


def _figsize(default):
    def check(v):
        return (
            isinstance(v, Sequence)
            and len(v) == 2
            and all(_number(x) and x > 0 for x in v)
        )

    return _checked(default, check, "two positive numbers (width, height)")


def _levels(v):
    if isinstance(v, int) and not isinstance(v, bool):
        return v >= 1
    return isinstance(v, Sequence) and len(v) > 0 and all(_number(x) for x in v)


def _optional_positive(v):
    return v is None or (_number(v) and v > 0)


def _bounds(v):
    return isinstance(v, Sequence) and len(v) == 4 and all(_number(x) for x in v)


class BaseConfig(Mapping):
    """Base configuration class with dict-like access.

    Fields declared with a ``check`` in their metadata are validated whenever
    they are set (directly, by ``update`` or by ``apsg_conf_context``), and
    unknown keys can't be set.
    """

    def __setattr__(self, key, value):
        fields = type(self).__dataclass_fields__
        if key not in fields:
            raise AttributeError(f"'{key}' is not a valid configuration key.")
        self.validate(key, value)
        super().__setattr__(key, value)

    def validate(self, key, value):
        """Raise ``ValueError`` unless `value` is valid for the field `key`."""
        meta = type(self).__dataclass_fields__[key].metadata
        check = meta.get("check")
        if check is None:
            return
        try:
            valid = check(value)
        except (TypeError, ValueError):
            valid = False
        if not valid:
            raise ValueError(f"{key} must be {meta['expect']}, got {value!r}")

    def override(self, target, overrides):
        """Update the dict `target` (a copy of this configuration) with the
        values in `overrides` for its keys, validating those of this
        configuration's fields. Other keys of `overrides` are ignored."""
        fields = type(self).__dataclass_fields__
        for key in target.keys() & overrides.keys():
            if key in fields:
                self.validate(key, overrides[key])
            target[key] = overrides[key]

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

    kind: str = _choice("equal-area", NET_KINDS)
    rotation: Any = None
    rotate_data: bool = True
    grid: bool = True
    grid_step: int = _positive(15)
    grid_color: str = "grey"
    grid_style: str = ":"
    clip_pole: int = _checked(15, lambda v: _number(v) and 0 <= v < 90, "in [0, 90)")
    hemisphere: str = _choice("lower", HEMISPHERES)
    primitive_lw: float = _positive(1.5, strict=False)
    primitive_color: Any = None
    azimuth_ticks: bool = True
    azimuth_ticks_kws: dict = field(default_factory=dict)
    figsize: tuple = _figsize((5, 5))
    title_kws: dict = field(default_factory=dict)
    legend_position: str = _choice("right", LEGEND_POSITIONS)
    legend_kws: dict = field(default_factory=dict)


@dataclass
class StereonetPointConfig(StereonetMarkerConfig):
    """Stereonet point marker style configuration."""


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

    kind: str = _choice("line", ("line", "points", "filled"))
    region: str = _choice("inside", ("inside", "outside"))  # kind="filled" only
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

    # None: "fisher" for vectors, "watson" for axes, "jelinek" for tensors
    method: Any = None
    which: Any = None
    level: float = _checked(0.95, lambda v: _number(v) and 0 < v < 1, "in (0, 1)")
    n_resamples: int = _positive(1000, integer=True)
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
class StereonetDihedraConfig(BaseConfig):
    """Stereonet fault dihedra (filled extensional quadrants) configuration."""

    color: Any = None
    alpha: Any = None  # None: 0.3 for a few faults, lowered as 1.5 / n for larger sets


@dataclass
class StereonetBeachballConfig(BaseConfig):
    """Stereonet stress beach ball (filled compressive quadrants) configuration."""

    color: Any = "k"
    alpha: Any = None  # None: 1 for one stress, lowered as 1.5 / n for larger sets


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
    method: str = _choice("kamb", CONTOUR_METHODS)
    n_max: Any = None
    cmap: Any = None
    levels: int = _checked(6, _levels, "an integer >= 1 or a non-empty sequence")
    filled: bool = True
    linewidth: float = _positive(1, strict=False)
    linestyles: Any = None
    colorbar: bool = False
    colorbar_kws: dict = field(default_factory=dict)
    colorbar_bounds: tuple = _checked(
        (0.955, 0.02, 0.03, 0.17), _bounds, "four numbers (x0, y0, width, height)"
    )
    colorbar_fontsize: float = _positive(7)
    line_color: str = "k"
    sigma: float | None = _checked(None, _optional_positive, "None or a number > 0")
    clip: bool = True


@dataclass
class StereogridConfig(BaseConfig):
    """StereoGrid global configuration."""

    type: str = _choice("gss", GRID_TYPES)
    n: int = _positive(3000, integer=True, minimum=10, strict=False)


@dataclass
class RoseplotConfig(BaseConfig):
    """Rose plot global configuration."""

    bins: int = _positive(36, integer=True)
    density: bool = True
    arrowness: float = 0.95
    rwidth: float = 1
    scaled: bool = False
    kappa: int = _positive(250)
    pdf_res: int = _positive(901, integer=True, minimum=2, strict=False)
    title: Any = None
    grid: bool = True
    grid_kws: dict = field(default_factory=dict)
    figsize: tuple = _figsize((5, 5))
    title_kws: dict = field(default_factory=dict)
    legend_position: str = _choice("right", LEGEND_POSITIONS)
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

    confidence_level: int = _checked(
        95, lambda v: _number(v) and 0 < v < 100, "a percentage in (0, 100)"
    )
    alpha: Any = None
    color: str = "r"
    ls: str = "-"
    lw: float = 1.5
    n_resamples: int = _positive(9999, integer=True)


@dataclass
class FabricplotConfig(BaseConfig):
    """Fabric plot global configuration."""

    ticks: bool = True
    n_ticks: int = _positive(10, integer=True)
    tick_size: float = _positive(0.2, strict=False)
    margin: float = _positive(0.05, strict=False)
    grid: bool = True
    grid_color: str = "k"
    grid_style: str = ":"
    border_color: str = "k"
    border_lw: float = 2
    tick_color: str = "k"
    tick_lw: float = 1
    label_fontsize: int = _positive(14)
    refline_color: str = "k"
    refline_lw: float = 0.5
    background_color: str = "w"
    title: Any = None
    figsize: tuple = _figsize((5, 5))
    title_kws: dict = field(default_factory=dict)
    legend_position: str = _choice("right", LEGEND_POSITIONS)
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

    notation: str = _choice("dd", NOTATIONS)
    stress_convention: str = _choice("geological", STRESS_CONVENTIONS)
    vec2geo: bool = False
    ndigits: int = _positive(3, integer=True, strict=False)
    figsize: tuple = _figsize((8, 6))
    dpi: int = _positive(100)
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
    stereonet_dihedra: StereonetDihedraConfig = field(
        default_factory=StereonetDihedraConfig
    )
    stereonet_beachball: StereonetBeachballConfig = field(
        default_factory=StereonetBeachballConfig
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


def stress_sign():
    """Sign that maps the canonical (mechanical, tension-positive) stress matrix
    to the active ``apsg_conf.stress_convention``.

    Returns:
        float: -1.0 for ``"geological"`` (compression positive), +1.0 for
        ``"mechanical"`` (tension positive). The mapping is its own inverse.
    """
    convention = apsg_conf.stress_convention
    if convention == "geological":
        return -1.0
    if convention == "mechanical":
        return 1.0
    raise ValueError(
        f"Unknown stress_convention {convention!r}, expected one of {STRESS_CONVENTIONS}"
    )


@contextmanager
def apsg_conf_context(**kwargs):
    """Context manager to temporarily override configuration."""
    saved = apsg_conf.copy()
    try:
        # inside the try, so a value failing validation doesn't leave the
        # values set before it changed
        apsg_conf.update(kwargs)
        yield
    finally:
        apsg_conf.update(saved)
