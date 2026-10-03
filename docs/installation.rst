============
Installation
============

Requirements
------------

APSG requires Python 3.12 or later. It depends on NumPy, SciPy, Matplotlib,
SQLAlchemy, pandas and pygeomag. The installers below take care of these.

Install APSG into a separate environment. uv is the default and the fastest
option; pip and conda/mamba are alternatives.

With uv (recommended)
---------------------

Install `uv <https://docs.astral.sh/uv/>`_, then create a virtual environment
and install APSG into it::

    uv venv
    uv pip install apsg

Activate the environment on Linux and macOS::

    source .venv/bin/activate

On Windows (Command Prompt or PowerShell)::

    .venv\Scripts\activate

To include JupyterLab and openpyxl (needed to read Excel files), use the ``lab`` extra::

    uv pip install "apsg[lab]"

In an existing project with a ``pyproject.toml``, add APSG as a dependency::

    uv add apsg

Command-line tool
~~~~~~~~~~~~~~~~~

To use APSG from the command line, install it as a tool. This puts an isolated
copy of APSG on your ``PATH``, without activating an environment::

    uv tool install apsg

Then start the interactive APSG shell, which runs ``from apsg import *`` for you::

    iapsg

To try the shell without installing, run::

    uvx --from apsg iapsg

Add the ``lab`` extra with ``uv tool install "apsg[lab]"`` to include JupyterLab.

Development install
~~~~~~~~~~~~~~~~~~~

To work on APSG itself, clone the repository and install it in editable mode
with all extras and development tools::

    git clone https://github.com/ondrolexa/apsg.git
    cd apsg
    uv sync --all-extras --dev

Run the test suite with ``uv run pytest``.

With pip
--------

Create and activate a virtual environment. On Linux and macOS::

    python -m venv .venv
    source .venv/bin/activate

On Windows (Command Prompt or PowerShell)::

    python -m venv .venv
    .venv\Scripts\activate

.. note::
   On Windows, PowerShell may refuse to run the activation script. To allow it
   for the current user, run::

       Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser

Install the latest stable release from PyPI::

    pip install apsg

To include JupyterLab and openpyxl (needed to read Excel files), use the ``lab`` extra::

    pip install "apsg[lab]"

To upgrade an existing installation without touching its dependencies::

    pip install --upgrade --no-deps apsg

To install the master branch from GitHub::

    pip install git+https://github.com/ondrolexa/apsg.git

Debian and Ubuntu system-wide installation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Recent Debian-based systems do not allow installing non-Debian packages
system-wide. Install the requirements with apt first, then install APSG with
pip::

    sudo apt install python3-numpy python3-matplotlib python3-scipy python3-sqlalchemy python3-pandas
    pip install --break-system-packages apsg

With conda or mamba
-------------------

If you already use conda or mamba, add the ``conda-forge`` channel and create
an environment with APSG::

    conda config --add channels conda-forge
    conda create -n apsg python apsg jupyterlab

or with mamba::

    mamba create -n apsg python apsg jupyterlab

To install APSG into an existing environment, run ``conda install apsg`` (or
``mamba install apsg``).
