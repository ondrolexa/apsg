<img src="https://ondrolexa.github.io/apsg/apsg_banner.svg" alt="APSG logo" width="300px"/>

[![PyPI - Version](https://img.shields.io/pypi/v/apsg)](https://pypi.org/project/apsg)
[![Conda](https://img.shields.io/conda/v/conda-forge/apsg)](https://anaconda.org/conda-forge/apsg)
[![Documentation Status](https://readthedocs.org/projects/apsg/badge/?version=stable)](https://apsg.readthedocs.io/en/stable/?badge=stable)
[![DOI](https://zenodo.org/badge/24879346.svg)](https://zenodo.org/badge/latestdoi/24879346)

## What is APSG?

APSG is the package for structural geologists. It defines several new python classes to easily manage, analyze and visualize orientational structural geology data.

## Major changes in class names and API from version 1.0.0

APSG has been significantly refactored from version 1.0 and several changes are
breaking backward compatibility. The main APSG namespace provides often-used
classes in lowercase names as aliases to `PascalCase` convention used in
modules to provide a simplified interface for users. The `PascalCase` names of
classes use longer and plain English names instead acronyms for better
readability.

Check [documentation](https://apsg.readthedocs.org) for more details.


## Requirements

You need Python 3.8 or later to run APSG. The package requires [NumPy](https://numpy.org/) and [SciPy](https://www.scipy.org/), and [Matplotlib](https://matplotlib.org/).

## Quick start

APSG can be installed using pip:
```
pip install apsg
```
If you want to run the latest version of the code, you can install it from git:
```
pip install git+https://github.com/ondrolexa/apsg.git
```
Alternatively, you can download the package manually from the GitHub repository [https://github.com/ondrolexa/apsg](https://github.com/ondrolexa/apsg), unzip it, navigate into the package, and use the command:
```
python setup.py install
```
or
```
pip install .
```

#### Upgrading via pip

To upgrade an existing version of APSG from PyPI, execute
```
pip install apsg --upgrade --no-deps
```
Please note that the dependencies (Matplotlib, NumPy and SciPy) will also be upgraded if you omit the `--no-deps` flag; use the `--no-deps` ("no dependencies") flag if you don't want this.

### Conda

The APSG package is also available on `conda-forge` channel.

#### Current release info

| Name | Downloads | Version | Platforms |
| --- | --- | --- | --- |
| [![Conda Recipe](https://img.shields.io/badge/recipe-apsg-green.svg)](https://anaconda.org/conda-forge/apsg) | [![Conda Downloads](https://img.shields.io/conda/dn/conda-forge/apsg.svg)](https://anaconda.org/conda-forge/apsg) | [![Conda Version](https://img.shields.io/conda/vn/conda-forge/apsg.svg)](https://anaconda.org/conda-forge/apsg) | [![Conda Platforms](https://img.shields.io/conda/pn/conda-forge/apsg.svg)](https://anaconda.org/conda-forge/apsg) |

#### Installing apsg

Installing `apsg` from the `conda-forge` channel can be achieved by adding `conda-forge` to your channels with:

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

## Documentation

Explore the full features of APSG. You can find detailed documentation [here](https://apsg.readthedocs.org).

## Contributing

Most discussion happens on [Github](https://github.com/ondrolexa/apsg). Feel free to open [an issue](https://github.com/ondrolexa/apsg/issues/new) or comment on any open issue or pull request. Check ``CONTRIBUTING.md`` for more details.

## Donate

APSG is an open-source project, available for you for free. It took a lot of time and resources to build this software. If you find this software useful and want to support its future development please consider donating me.

[![Donate via PayPal](https://www.paypalobjects.com/en_US/i/btn/btn_donateCC_LG.gif)](https://www.paypal.com/cgi-bin/webscr?cmd=_donations&business=QTYZWVUNDUAH8&item_name=APSG+development+donation&currency_code=EUR&source=url)

## License

APSG is free software: you can redistribute it and/or modify it under the terms of the MIT License. A copy of this license is provided in ``LICENSE`` file.
