#!/usr/bin/python

import readline
import code
from pylab import *
from apsg import *

try:
    from apsg import __version__ as APSG_VERSION
except:
    APSG_VERSION = ''

def main():
    banner = '+----------------------------------------------------------+\n'
    banner += '    APSG toolbox '
    banner += APSG_VERSION
    banner += ' - http://ondrolexa.github.io/apsg\n'
    banner += '+----------------------------------------------------------+'
    vars = globals().copy()
    vars.update(locals())
    shell = code.InteractiveConsole(vars)
    shell.interact(banner=banner)

if __name__ == "__main__":
    main()
