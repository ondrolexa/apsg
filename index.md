<img src="https://ondrolexa.github.io/apsg/apsg_banner.svg" alt="APSG logo" width="640px"/>

[![PyPI version](https://badge.fury.io/py/apsg.png)](http://badge.fury.io/py/apsg)
[![GitHub version](https://badge.fury.io/gh/ondrolexa%2Fapsg.svg)](https://badge.fury.io/gh/ondrolexa%2Fapsg)
[![Build Status](https://travis-ci.org/ondrolexa/apsg.svg?branch=master)](https://travis-ci.org/ondrolexa/apsg)
[![Documentation Status](https://readthedocs.org/projects/apsg/badge/?version=stable)](https://apsg.readthedocs.io/en/stable/?badge=stable)
[![DOI](https://zenodo.org/badge/24879346.svg)](https://zenodo.org/badge/latestdoi/24879346)

APSG is package for structural geologists. It defines several new python classes to easily manage, analyze and visualize orientational structural geology data.

## Requirements

You need Python 3.6 or later to run APSG. The package requires [NumPy](https://numpy.org/) and [SciPy](https://www.scipy.org/), and the plotting routines require [Matplotlib](https://matplotlib.org/).

## Quick start

APSG can be installed using pip:
```
pip install apsg
```
If you want tu run the latest version of code, you can install it from git:
```
pip install git+git://github.com/ondrolexa/apsg.git
```

#### Upgrading via pip

To upgrade an existing version of APSG from PyPI, execute
```
pip install apsg --upgrade --no-deps
```
Please note that the dependencies (Matplotlib, NumPy and SciPy) will also be upgraded if you omit the `--no-deps` flag; use the `--no-deps` ("no dependencies") flag if you don't want this.

#### Conda

The APSG package is also available through `conda-forge`. Installing `apsg` from the `conda-forge` channel can be achieved by adding `conda-forge` to your channels with:

```
conda config --add channels conda-forge
```

Once the `conda-forge` channel has been enabled, `apsg` can be installed with:

```
conda install apsg
```

It is possible to list all of the versions of `apsg` available on your platform with:

```
conda search apsg --channel conda-forge
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
