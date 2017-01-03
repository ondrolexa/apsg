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
    banner = '+-----------------------------------------------------------+\n'
    banner += ' APSG '
    banner += APSG_VERSION
    banner += ' [interactive shell] - http://ondrolexa.github.io/apsg\n'
    banner += '+-----------------------------------------------------------+\n'
    banner += '\n'
    exitmsg = '\n... [Exiting the APSG interactive shell] ...\n'
    vars = globals().copy()
    vars.update(locals())
    shell = code.InteractiveConsole(vars)
    shell.interact(banner=banner, exitmsg=exitmsg)

if __name__ == "__main__":
    main()
