#!/usr/bin/python

import pkg_resources
import code
try:
    import readline
except ImportError:
    pass
from pylab import *
from apsg import *


def main():
    banner = '+----------------------------------------------------------+\n'
    banner += '    APSG toolbox '
    banner += pkg_resources.require('apsg')[0].version
    banner += ' - http://ondrolexa.github.io/apsg\n'
    banner += '+----------------------------------------------------------+'
    vars = globals().copy()
    vars.update(locals())
    shell = code.InteractiveConsole(vars)
    shell.interact(banner=banner)


if __name__ == "__main__":
    main()
