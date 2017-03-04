============
Installation
============

For Anaconda distribution (for now only Linux64 and Win64 supported) you can install from personal channel::

    conda install -c https://conda.anaconda.org/ondrolexa apsg

For other platforms install dependencies using conda::

    conda install numpy matplotlib scipy pyqt

or by any other mechanism (see `Installing Scientific Packages <https://packaging.python.org/science/>`_).

Than install apsg directly from github using pip::

    pip install https://github.com/ondrolexa/apsg/archive/master.zip

For upgrade use::

    pip install --upgrade --upgrade-strategy only-if-needed \
      https://github.com/ondrolexa/apsg/archive/master.zip
          

To install most recent (and likely less stable) development version use::

    pip install https://github.com/ondrolexa/apsg/archive/develop.zip


For upgrade to latest development version use::

    pip install --upgrade --upgrade-strategy only-if-needed \
      https://github.com/ondrolexa/apsg/archive/develop.zip

