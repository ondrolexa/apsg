============
Installation
============

Requirements
------------

You need Python 3.12 or later to run APSG. The package requires NumPy, SciPy,
Matplotlib, SQLAlchemy, pandas and pygeomag.

Create a virtual environment
----------------------------

It is strongly suggested to install APSG into a separate environment.

For Linux and macOS::

    python -m venv .venv
    source .venv/bin/activate

For Windows (Command Prompt or PowerShell)::

    python -m venv .venv
    .venv\Scripts\activate

.. note::
   On Microsoft Windows, it may be required to set the execution policy in
   PowerShell. You can do this by issuing the following PowerShell command::

       Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser

Install with pip
----------------

Install the latest stable version from PyPI within your environment::

    pip install apsg

To include JupyterLab and PyQt6 in the installation, use the ``extra`` option::

    pip install apsg[extra]

Upgrading via pip
~~~~~~~~~~~~~~~~~

To upgrade an existing version of APSG from PyPI::

    pip install apsg --upgrade --no-deps

Without the ``--no-deps`` flag, the dependencies (Matplotlib, NumPy, SciPy,
etc.) will also be upgraded if newer versions are available; use the
``--no-deps`` flag if you do not want this.

Install the development version
-------------------------------

The latest development version from the GitHub repository can be installed
with::

    pip install git+https://github.com/ondrolexa/apsg.git

Alternatively, clone the repository and do a local install with
`uv <https://docs.astral.sh/uv/>`_ (recommended for development)::

    git clone https://github.com/ondrolexa/apsg.git
    cd apsg
    uv sync --all-extras --dev

Comments on system-wide installations on Debian systems
-------------------------------------------------------

Latest Debian-based systems do not allow installing non-Debian packages
system-wide. However, installing all requirements allows installing APSG
system-wide without troubles.

Install requirements using apt::

    sudo apt install python3-numpy python3-matplotlib python3-scipy python3-sqlalchemy python3-pandas

and then install APSG using pip::

    pip install --break-system-packages apsg

Install with conda or mamba
---------------------------

If you already have conda or mamba installed, you can create an environment
with APSG by running::

    conda config --add channels conda-forge
    conda create -n apsg python apsg jupyterlab pyqt6

or using mamba::

    mamba create -n apsg python apsg jupyterlab pyqt6

To install APSG into an existing environment::

    conda install apsg

.. note::
   The ``conda-forge`` channel must be configured if you have not added it
   already (``conda config --add channels conda-forge``).
