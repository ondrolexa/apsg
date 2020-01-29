#!/usr/bin/env python
# -*- coding: utf-8 -*-


"""
Run the interactive shell.
"""


import pkg_resources
import code

try:
    import readline  # NOQA
except ImportError:
    pass

from pylab import *  # NOQA
from apsg import *  # NOQA


def main():
    banner = "+----------------------------------------------------------+\n"
    banner += "    APSG toolbox "
    banner += pkg_resources.require("apsg")[0].version
    banner += " - http://ondrolexa.github.io/apsg\n"
    banner += "+----------------------------------------------------------+"
    vars = globals().copy()
    vars.update(locals())
    shell = code.InteractiveConsole(vars)
    shell.interact(banner=banner)


if __name__ == "__main__":
    main()
