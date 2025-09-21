#!/usr/bin/env python
# -*- coding: utf-8 -*-


"""
Run the interactive shell.
"""

import code

try:
    import readline  # NOQA
except ImportError:
    pass

import numpy as np
import matplotlib.pyplot as plt
import apsg
from apsg import *  # NOQA


def main():
    banner = "+----------------------------------------------------------+\n"
    banner += "    APSG toolbox "
    banner += apsg.__version__
    banner += " - http://ondrolexa.github.io/apsg\n"
    banner += "+----------------------------------------------------------+"
    vars = globals().copy()
    vars.update(locals())
    shell = code.InteractiveConsole(vars)
    shell.interact(banner=banner)


if __name__ == "__main__":
    main()
