import math


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
