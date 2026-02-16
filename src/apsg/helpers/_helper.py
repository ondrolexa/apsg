# some utils
import json


def eformat(f, prec):
    s = "{:e}".format(f)
    m, e = s.split("e")
    return "{:.{:d}f}E{:0d}".format(float(m), prec, int(e))


def is_jsonable(x):
    try:
        json.dumps(x)
        return True
    except (TypeError, OverflowError):
        return False
