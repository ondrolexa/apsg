from apsg.config import apsg_conf
from apsg.helpers._math import asind, atan2d, cosd, sind

# NOTATION TRANSORMATIONS


def fol2vec_dd(azi, inc):
    """Convert dip direction/dip to normal vector components."""
    return -cosd(azi) * sind(inc), -sind(azi) * sind(inc), cosd(inc)


def fol2vec_rhr(strike, dip):
    """Convert strike/dip (RHR) to normal vector components."""
    return fol2vec_dd(strike + 90, dip)


def geo2vec_planar(*args):
    """Transform geological measurement of plane to normal vector.

    Conversion is done according to `notation` configuration.

    Args:
        azi (float): dip direction or strike
        inc (float): dip
    """
    return {"dd": fol2vec_dd, "rhr": fol2vec_rhr}[apsg_conf.notation](*args)


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
    return {"dd": vec2fol_dd_signed, "rhr": vec2fol_rhr_signed}[apsg_conf.notation](arg)


def vec2geo_planar(arg):
    """Transform normal vector to geological measurement of plane.

    Conversion is done according to `notation` configuration.

    Args:
        v (Vector3): ``Vector3`` like object
    """
    return {"dd": vec2fol_dd, "rhr": vec2fol_rhr}[apsg_conf.notation](arg)


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
    return {"dd": vec2lin_dd_signed, "rhr": vec2lin_dd_signed}[apsg_conf.notation](arg)


def vec2geo_linear(arg):
    """Transform vector to geological measurement of line.

    Args:
        v (Vector3): ``Vector3`` like object
    """
    return vec2lin_dd(arg)
