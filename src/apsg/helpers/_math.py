import math

sqrt2 = math.sqrt(2.0)

### DEGREES TRIGINOMETRY ###


def sind(x):
    return math.sin(math.radians(x))


def cosd(x):
    return math.cos(math.radians(x))


def tand(x):
    return math.tan(math.radians(x))


def asind(x):
    return math.degrees(math.asin(x))


def acosd(x):
    return math.degrees(math.acos(x))


def atand(x):
    return math.degrees(math.atan(x))


def atan2d(y, x):
    return math.degrees(math.atan2(y, x))


def is_like_vec3(arg):
    if hasattr(arg, "__len__"):
        if len(arg) == 3:
            return True
    return False


def is_like_matrix3(arg):
    if hasattr(arg, "__len__"):
        if len(arg) == 3:
            if all([hasattr(row, "__len__") for row in arg]):
                if all([len(row) == 3 for row in arg]):
                    return True
    return False


def eformat(f, prec):
    s = "{:e}".format(f)
    m, e = s.split("e")
    return "{:.{:d}f}E{:0d}".format(float(m), prec, int(e))
