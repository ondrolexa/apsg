# APSG - python package for structural geologists

[![GitHub version](https://badge.fury.io/gh/ondrolexa%2Fapsg.svg)](https://badge.fury.io/gh/ondrolexa%2Fapsg)
[![Build Status](https://travis-ci.org/ondrolexa/apsg.svg?branch=master)](https://travis-ci.org/ondrolexa/apsg)
[![Documentation Status](https://readthedocs.org/projects/apsg/badge/?version=stable)](https://apsg.readthedocs.io/en/stable/?badge=stable)
[![Anaconda-Server Badge](https://anaconda.org/ondrolexa/apsg/badges/version.svg)](https://anaconda.org/ondrolexa/apsg)
[![DOI](https://zenodo.org/badge/24879346.svg)](https://zenodo.org/badge/latestdoi/24879346)

APSG defines several new python classes to easily manage, analyze and
visualize orientational structural geology data.

## Installation

### PyPI

To install APSG, just execute
```
pip install apsg
```
Alternatively, you download the package manually from the Python Package Index [https://pypi.org/project/apsg](https://pypi.org/project/apsg), unzip it, navigate into the package, and use the command:
```
python setup.py install
```
#### Upgrading via pip

To upgrade an existing version of APSG from PyPI, execute
```
pip install apsg --upgrade --no-deps
```
Please note that the dependencies (Matplotlib, NumPy and SciPy) will also be upgraded if you omit the `--no-deps` flag; use the `--no-deps` ("no dependencies") flag if you don't want this.

#### Installing APSG from the source distribution

In rare cases, users reported problems on certain systems with the default pip installation command, which installs APSG from the binary distribution ("wheels") on PyPI. If you should encounter similar problems, you could try to install APSG from the source distribution instead via
```
pip install --no-binary :all: apsg
```
Also, I would appreciate it if you could report any issues that occur when using `pip install apsg` in hope that we can fix these in future releases.

### Conda

The APSG package is also available through `conda-forge`.

To install APSG using conda, use the following command:
```
conda install apsg --channel conda-forge
```
or simply
```
conda install apsg
```
if you added `conda-forge` to your channels (`conda config --add channels conda-forge`).

### Master version

The APSG version on PyPI may always one step behind; you can install the latest development version from the GitHub repository by executing
```
pip install git+git://github.com/ondrolexa/apsg.git
```
Or, you can fork the GitHub repository from [https://github.com/ondrolexa/apsg](https://github.com/ondrolexa/apsg) and install APSG from your local drive via
```
python setup.py install
```

## Getting started

You can see APSG in action in accompanied Jupyter notebook [http://nbviewer.jupyter.org/github/ondrolexa/apsg/blob/master/examples/apsg_tutorial.ipynb](http://nbviewer.jupyter.org/github/ondrolexa/apsg/blob/master/examples/apsg_tutorial.ipynb)

And for fun check how simply you can animate stereonets
[http://nbviewer.jupyter.org/github/ondrolexa/apsg/blob/master/examples/animation_example.ipynb](http://nbviewer.jupyter.org/github/ondrolexa/apsg/blob/master/examples/animation_example.ipynb)

## Documentation

Explore the full features of APSG. You can find detailed documentation [here](https://apsg.readthedocs.org).

## Contributing

Most discussion happens on [Github](https://github.com/ondrolexa/apsg). Feel free to open [an issue](https://github.com/ondrolexa/apsg/issues/new) or comment on any open issue or pull request. Check ``CONTRIBUTING.md`` for more details.

## Donate

APSG is an open-source project, available for you for free. It took a lot of time and resources to build this software. If you find this software useful and want to support its future development please consider donating me.

[![Donate via PayPal](https://www.paypalobjects.com/en_US/i/btn/btn_donateCC_LG.gif)](https://www.paypal.com/cgi-bin/webscr?cmd=_donations&business=QTYZWVUNDUAH8&item_name=APSG+development+donation&currency_code=EUR&source=url)

## License

APSG is free software: you can redistribute it and/or modify it under the terms of the MIT License. A copy of this license is provided in ``LICENSE`` file.
