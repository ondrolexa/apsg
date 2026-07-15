import re

from apsg.config import apsg_conf
from apsg.helpers._math import asind, atan2d, cosd, sind

NOTATIONS = ("dd", "rhr", "quadrant")

# NOTATION TRANSORMATIONS


def fol2vec_dd(azi, inc):
    """Convert dip direction/dip to normal vector components."""
    return -cosd(azi) * sind(inc), -sind(azi) * sind(inc), cosd(inc)


def fol2vec_rhr(strike, dip):
    """Convert strike/dip (RHR) to normal vector components."""
    return fol2vec_dd(strike + 90, dip)


def _notation_fn(table):
    """Look up the conversion function for the active `apsg_conf.notation`."""
    try:
        return table[apsg_conf.notation]
    except KeyError:
        raise ValueError(
            f"Unknown notation {apsg_conf.notation!r}, expected one of {NOTATIONS}"
        ) from None


def geo2vec_planar(*args):
    """Transform geological measurement of plane to normal vector.

    Conversion is done according to `notation` configuration. Quadrant notation
    is a textual convention (see `parse_quadrant_planar`) and is not accepted here.

    Args:
        azi (float): dip direction or strike
        inc (float): dip
    """
    return _notation_fn(
        {"dd": fol2vec_dd, "rhr": fol2vec_rhr, "quadrant": fol2vec_rhr}
    )(*args)


##############################


def lin2vec_dd(azi, inc):
    """Convert plunge direction/plunge to vector components."""
    return cosd(azi) * cosd(inc), sind(azi) * cosd(inc), sind(inc)


def geo2vec_linear(*args):
    """Transform geological measurement of line to vector.

    Args:
        azi (float): plunge direction
        inc (float): plunge
    """
    return lin2vec_dd(*args)


##############################


def vec2fol_dd(v):
    """Convert normal vector to dip direction/dip."""
    n = v.uv()
    if n.z < 0:
        n = -n
    return (atan2d(n.y, n.x) + 180) % 360, 90 - asind(n.z)


def vec2fol_dd_signed(v):
    """Convert normal vector to signed dip direction/dip."""
    n = v.uv()
    return (atan2d(n.y, n.x) + 180) % 360, 90 - asind(n.z)


def vec2fol_rhr(v):
    """Convert normal vector to strike/dip (RHR)."""
    n = v.uv()
    if n.z < 0:
        n = -n
    return (atan2d(n.y, n.x) + 90) % 360, 90 - asind(n.z)


def vec2fol_rhr_signed(v):
    """Convert normal vector to signed strike/dip (RHR)."""
    n = v.uv()
    return (atan2d(n.y, n.x) + 90) % 360, 90 - asind(n.z)


def vec2geo_planar_signed(arg):
    """Transform normal vector to signed planar measurement."""
    return _notation_fn(
        {
            "dd": vec2fol_dd_signed,
            "rhr": vec2fol_rhr_signed,
            "quadrant": vec2fol_rhr_signed,
        }
    )(arg)


def vec2geo_planar(arg):
    """Transform normal vector to geological measurement of plane.

    Conversion is done according to `notation` configuration. Under `"quadrant"`
    notation this still returns numeric RHR strike/dip - use `format_planar` to
    render it as quadrant text.

    Args:
        v (Vector3): ``Vector3`` like object
    """
    return _notation_fn(
        {"dd": vec2fol_dd, "rhr": vec2fol_rhr, "quadrant": vec2fol_rhr}
    )(arg)


##############################


def vec2lin_dd(v):
    """Convert vector to plunge direction/plunge."""
    n = v.uv()
    if n.z < 0:
        n = -n
    return atan2d(n.y, n.x) % 360, asind(n.z)


def vec2lin_dd_signed(v):
    """Convert vector to signed plunge direction/plunge."""
    n = v.uv()
    return atan2d(n.y, n.x) % 360, asind(n.z)


def vec2geo_linear_signed(arg):
    """Transform vector to signed linear measurement."""
    return _notation_fn(
        {
            "dd": vec2lin_dd_signed,
            "rhr": vec2lin_dd_signed,
            "quadrant": vec2lin_dd_signed,
        }
    )(arg)


def vec2geo_linear(arg):
    """Transform vector to geological measurement of line.

    Args:
        v (Vector3): ``Vector3`` like object
    """
    return vec2lin_dd(arg)


##############################
# QUADRANT NOTATION (textual strike/bearing convention)


_BEARING_RE = re.compile(
    r"^\s*([NS])\s*(\d{1,2}(?:\.\d+)?)\s*([EW])\s*$", re.IGNORECASE
)
_DIP_QUADRANTS = {"NE": 45, "SE": 135, "SW": 225, "NW": 315}  # bucket centers
_PLANAR_QUADRANT_RE = re.compile(
    r"^\s*([NSns]\s*\d{1,2}(?:\.\d+)?\s*[EWew])\s*,\s*(\d{1,2}(?:\.\d+)?)\s*(NE|SE|SW|NW)\s*$"
)
_LINEAR_QUADRANT_RE = re.compile(
    r"^\s*([NSns]\s*\d{1,2}(?:\.\d+)?\s*[EWew])\s*,\s*(\d{1,2}(?:\.\d+)?)\s*$"
)


def azi2bearing(azi):
    """Format azimuth (0-360 degrees) as quadrant bearing string, e.g. 45 -> 'N45E'."""
    azi = azi % 360
    if azi <= 90:
        return f"N{azi:.0f}E"
    if azi <= 180:
        return f"S{180 - azi:.0f}E"
    if azi <= 270:
        return f"S{azi - 180:.0f}W"
    return f"N{360 - azi:.0f}W"


def bearing2azi(s):
    """Parse quadrant bearing string, e.g. 'N45E', to azimuth in degrees 0-360."""
    m = _BEARING_RE.match(s)
    if not m:
        raise ValueError(f"Cannot parse quadrant bearing: {s!r}")
    ns, angle, ew = m.group(1).upper(), float(m.group(2)), m.group(3).upper()
    if not 0 <= angle <= 90:
        raise ValueError(f"Quadrant bearing angle must be 0-90: {s!r}")
    return {
        ("N", "E"): angle,
        ("N", "W"): 360 - angle,
        ("S", "E"): 180 - angle,
        ("S", "W"): 180 + angle,
    }[(ns, ew)]


def _dip_qualifier(azi):
    """Bucket a dip-direction azimuth into cardinal quadrant NE/SE/SW/NW."""
    return ("NE", "SE", "SW", "NW")[int(azi % 360 // 90)]


def parse_quadrant_planar(s):
    """Parse quadrant planar measurement, e.g. 'N30E,40NW', to RHR (strike, dip)."""
    m = _PLANAR_QUADRANT_RE.match(s)
    if not m:
        raise ValueError(f"Cannot parse quadrant planar measurement: {s!r}")
    bearing, dip, dip_quadrant = m.groups()
    strike, dip = bearing2azi(bearing), float(dip)
    target = _DIP_QUADRANTS[dip_quadrant.upper()]
    if abs((strike + 90 - target + 180) % 360 - 180) > 90:
        strike = (strike + 180) % 360
    return strike, dip


def format_quadrant_planar(strike, dip):
    """Format RHR (strike, dip) as quadrant planar measurement string.

    A strike is an axial (bidirectional) line, so `strike` and `strike + 180`
    are the same physical line; by convention it is reported as a bearing
    measured from North (e.g. N30E or N80W rather than its S-quadrant
    equivalent), independent of the dip qualifier, which is derived from the
    actual dip azimuth `strike + 90`.
    """
    dip_azi = (strike + 90) % 360
    display_strike = strike if (strike <= 90 or strike > 270) else strike - 180
    return f"{azi2bearing(display_strike)},{dip:.0f}{_dip_qualifier(dip_azi)}"


def parse_quadrant_linear(s):
    """Parse quadrant linear measurement, e.g. 'N45E,30', to (trend, plunge)."""
    m = _LINEAR_QUADRANT_RE.match(s)
    if not m:
        raise ValueError(f"Cannot parse quadrant linear measurement: {s!r}")
    bearing, plunge = m.groups()
    return bearing2azi(bearing), float(plunge)


def format_quadrant_linear(azi, inc):
    """Format (trend, plunge) as quadrant linear measurement string."""
    return f"{azi2bearing(azi)},{inc:.0f}"


def format_planar(azi, inc):
    """Format a planar geo-tuple according to the active `apsg_conf.notation`."""
    if apsg_conf.notation == "quadrant":
        return format_quadrant_planar(azi, inc)
    return f"{azi:.0f}/{inc:.0f}"


def format_linear(azi, inc):
    """Format a linear geo-tuple according to the active `apsg_conf.notation`."""
    if apsg_conf.notation == "quadrant":
        return format_quadrant_linear(azi, inc)
    return f"{azi:.0f}/{inc:.0f}"
