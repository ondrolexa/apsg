#!/usr/bin/env python
# -*- coding: utf-8 -*-


"""
Run the package as application.
"""


import code
import pkg_resources
from argparse import ArgumentParser
try:
    import readline  # NOQA
except ImportError:
    pass

from pylab import *     # NOQA

# For `help(apsg)` working in console.
import apsg

from apsg import *      # NOQA
from apsg.math import * # NOQA


def run_console_mode():
    """
    Run the interactive console mode.
    """
    version = pkg_resources.require("apsg")[0].version

    banner =  "+------------------------------------------------------------------------------+\n"
    banner += "|                           APSG Toolbox | " + version + (31 * " ") +         "|\n"
    banner += "|                                                                              |\n"
    banner += "|                       http://ondrolexa.github.io/apsg                        |\n"
    banner += "|                                                                              |\n"
    banner += "|------------------------------------------------------------------------------|\n"
    banner += "| You can abort process using CTRL-Z + ENTER                                   |\n"
    banner += "+------------------------------------------------------------------------------+"

    vars = globals().copy()
    vars.update(locals())

    shell = code.InteractiveConsole(vars)
    shell.interact(banner=banner)


def run_jupyter_mode():
    pass


def main(args=None):
    """
    Main entry point for your project.

    Arguments:
        args : list
            A of arguments as if they were input in the command line. Leave it
            None to use sys.argv.
    """
    parser = ArgumentParser(description="APSG toolbox")

    parser.add_argument("-i", "--interactive", action='store_true', default=False, help=run_console_mode.__doc__)
    # todo parser.add_argument("-V", "--version", action='version')

    parser.set_defaults(func=lambda: parser.print_usage())

    result = parser.parse_args(args)

    if result.interactive:
        run_console_mode()
    else:
        result.func()

if __name__ == "__main__":
    main()
