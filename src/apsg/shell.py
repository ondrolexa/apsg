#!/usr/bin/env python
# -*- coding: utf-8 -*-


"""
Run the interactive shell.
"""

import readline  # noqa
import code
import numpy as np  # noqa
import matplotlib.pyplot as plt  # noqa
import apsg
from apsg import *  # noqa


def main():
    vars = globals().copy()
    vars.update(locals())
    banner = [
        "----------------------------------------------------------",
        f"  APSG toolbox {apsg.__version__} - https://github.com/ondrolexa/apsg",
        "----------------------------------------------------------",
    ]
    shell = code.InteractiveConsole(vars)
    shell.interact(banner="\n".join(banner), exitmsg="")


if __name__ == "__main__":
    main()
