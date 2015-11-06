#!/usr/bin/python

from pylab import *
from apsg import *

from subprocess import call
import platform
# import webbrowser
import sys

ion()

try:
    from apsg import __version__ as APSG_VERSION
except:
    APSG_VERSION = ''


# Command to clear the shell screen
def shellclear():
    if platform.system() == "Windows":
        return
    call("clear")


def magic_clear(self, arg):
    shellclear()


# def magic_docs(self, arg):
#     webbrowser.open('https://apsg.readthedocs.org/')

"""
If you run APSG directly, it will launch an IPython shell
"""


def setup_shell():

    banner = '+-----------------------------------------------------------+\n'
    banner += ' APSG '
    banner += APSG_VERSION
    banner += ' [interactive shell] - http://ondrolexa.github.io/apsg\n'
    banner += '+-----------------------------------------------------------+\n'
    banner += '\n'
    banner += 'Commands: \n'
    banner += '\t"exit()" or press "Ctrl+ D" to exit the shell\n'
    banner += '\t"clear" to clear the shell screen\n'
    banner += '\n'
    banner += 'Documentation:\n'
    banner += '\thelp(Fol), ?Fol, Fol?, or Fol()? all do the same\n'
    banner += '\t"docs" will launch webbrowser showing documentation'
    banner += '\n'
    exit_msg = '\n... [Exiting the APSG interactive shell] ...\n'

    try:
        from traitlets.config import Config
        from IPython.terminal.embed import InteractiveShellEmbed

        cfg = Config()
        cfg.PromptManager.in_template = "APSG:\\#> "
        cfg.PromptManager.out_template = "APSG:\\#: "
        apsgShell = InteractiveShellEmbed(config=cfg, banner1=banner,
                                          exit_msg = exit_msg)
        apsgShell.define_magic("clear", magic_clear)
        #apsgShell.define_magic("docs", magic_docs)

    except ImportError:
        try:
            from IPython.Shell import IPShellEmbed
            argsv = ['-pi1','APSG:\\#>','-pi2','   .\\D.:','-po','APSG:\\#>','-nosep']

            apsgShell = IPShellEmbed(argsv)
            apsgShell.set_banner(banner)
            apsgShell.set_exit_msg(exit_msg)
            apsgShell.IP.api.expose_magic("clear", magic_clear)
        except ImportError:
            raise("ERROR: IPython shell failed to load")

    return apsgShell

def self_update():
    URL = "https://github.com/ondrolexa/apsg/zipball/master"
    command = "pip install -U %s" % URL

    if os.getuid() == 0:
        command = "sudo " + command

    returncode = call(command, shell=True)
    sys.exit()


def main(*args):

    if len(sys.argv) > 1 and len(sys.argv[1]) > 1:
      flag = sys.argv[1]
      if flag == 'update':
        print("Updating APSG.....")
        self_update()
        

      if flag in ["--headless", "headless"]:
        # set SDL to use the dummy NULL video driver,
        #   so it doesn't need a windowing system.
        os.environ["SDL_VIDEODRIVER"] = "dummy"

    shellclear()

    apsgShell = setup_shell()
    #Note that all loaded libraries are inherited in the embedded ipython shell
    sys.exit(apsgShell())
