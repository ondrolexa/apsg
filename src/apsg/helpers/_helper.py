# some utils


def eformat(f, prec):
    s = "{:e}".format(f)
    m, e = s.split("e")
    return "{:.{:d}f}E{:0d}".format(float(m), prec, int(e))
