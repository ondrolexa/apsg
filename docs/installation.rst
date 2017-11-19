============
Installation
============

-------------
Conda package
-------------

For Anaconda distribution you can install directly from my personal channel::

    conda install -c ondrolexa apsg

---------
Using pip
---------

APSG requires dependencies which could be installed using conda::

    conda install numpy matplotlib scipy

or by any other mechanism (see `Installing Scientific Packages <https://packaging.python.org/science/>`_).

You can install APSG directly from github using pip::

    pip install https://github.com/ondrolexa/apsg/archive/master.zip

To safely upgrade installed APSG package use::

    pip install --upgrade --upgrade-strategy only-if-needed \
      https://github.com/ondrolexa/apsg/archive/master.zip


--------------------
Developement version
--------------------

To install most recent (and likely less stable) development version use::

    pip install https://github.com/ondrolexa/apsg/archive/develop.zip

To upgrade to latest development version use::

    pip install --upgrade --upgrade-strategy only-if-needed \
      https://github.com/ondrolexa/apsg/archive/develop.zip

