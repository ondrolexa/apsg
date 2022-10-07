from apsg.config import apsg_conf
from apsg.helpers._math import sind, cosd, asind, atan2d

# NOTATION TRANSORMATIONS


def fol2vec_dd(azi, inc):
    return -cosd(azi) * sind(inc), -sind(azi) * sind(inc), cosd(inc)


def fol2vec_rhr(strike, dip):
    return fol2vec_dd(strike + 90, dip)


def geo2vec_planar(*args):
    return {"dd": fol2vec_dd, "rhr": fol2vec_rhr}[apsg_conf["notation"]](*args)


##############################


def lin2vec_dd(azi, inc):
    return cosd(azi) * cosd(inc), sind(azi) * cosd(inc), sind(inc)


def geo2vec_linear(*args):
    return {"dd": lin2vec_dd, "rhr": lin2vec_dd}[apsg_conf["notation"]](*args)


##############################


def vec2fol_dd(v):
    n = v.uv()
    if n.z < 0:
        n = -n
    return (atan2d(n.y, n.x) + 180) % 360, 90 - asind(n.z)


def vec2fol_dd_signed(v):
    n = v.uv()
    return (atan2d(n.y, n.x) + 180) % 360, 90 - asind(n.z)


def vec2fol_rhr(v):
    n = v.uv()
    if n.z < 0:
        n = -n
    return (atan2d(n.y, n.x) + 90) % 360, 90 - asind(n.z)


def vec2fol_rhr_signed(v):
    n = v.uv()
    return (atan2d(n.y, n.x) + 90) % 360, 90 - asind(n.z)


def vec2geo_planar_signed(arg):
    return {"dd": vec2fol_dd_signed, "rhr": vec2fol_rhr_signed}[apsg_conf["notation"]](
        arg
    )


def vec2geo_planar(arg):
    return {"dd": vec2fol_dd, "rhr": vec2fol_rhr}[apsg_conf["notation"]](arg)


##############################


def vec2lin_dd(v):
    n = v.uv()
    if n.z < 0:
        n = -n
    return atan2d(n.y, n.x) % 360, asind(n.z)


def vec2lin_dd_signed(v):
    n = v.uv()
    return atan2d(n.y, n.x) % 360, asind(n.z)


def vec2geo_linear_signed(arg):
    return {"dd": vec2lin_dd_signed, "rhr": vec2lin_dd_signed}[apsg_conf["notation"]](
        arg
    )


def vec2geo_linear(arg):
    return {"dd": vec2lin_dd, "rhr": vec2lin_dd}[apsg_conf["notation"]](arg)
