# some utils
import json


def eformat(f, prec):
    """Format float in scientific notation with given precision."""
    s = "{:e}".format(f)
    m, e = s.split("e")
    return "{:.{:d}f}E{:0d}".format(float(m), prec, int(e))


def is_jsonable(x):
    """Check if value is JSON-serializable."""
    try:
        json.dumps(x)
        return True
    except (TypeError, OverflowError):
        return False
