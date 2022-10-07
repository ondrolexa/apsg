============
Installation
============

---------
Using pip
---------

To install APSG from PyPI, just execute::

    pip install apsg

To upgrade an existing version of APSG from PyPI, execute

    pip install apsg --upgrade --no-deps

Please note that the dependencies (Matplotlib, NumPy and SciPy) will also be upgraded if you omit the ``--no-deps`` flag;
use the ``--no-deps`` ("no dependencies") flag if you don't want this.

--------------
Master version
--------------

The APSG version on PyPI may always one step behind; you can install the latest development version from the GitHub repository by executing::

    pip install git+https://github.com/ondrolexa/apsg.git

-----------
Using Conda
-----------

The APSG package is also available through ``conda-forge``. To install APSG using conda, use the following command::

    conda install apsg --channel conda-forge

or simply

    conda install apsg

if you added ``conda-forge`` to your channels (``conda config --add channels conda-forge``).
