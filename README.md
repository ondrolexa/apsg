<img src="https://ondrolexa.github.io/apsg/apsg_banner.svg" alt="APSG logo" width="300px"/>

[![PyPI - Version](https://img.shields.io/pypi/v/apsg)](https://pypi.org/project/apsg)
[![Conda](https://img.shields.io/conda/v/conda-forge/apsg)](https://anaconda.org/conda-forge/apsg)
[![Documentation Status](https://readthedocs.org/projects/apsg/badge/?version=stable)](https://apsg.readthedocs.io/en/stable/?badge=stable)
[![DOI](https://zenodo.org/badge/24879346.svg)](https://zenodo.org/badge/latestdoi/24879346)

## :thinking: What is APSG?

APSG is the package for structural geologists. It defines several new python classes to easily manage, analyze and visualize orientational structural geology data.

> [!IMPORTANT]
> APSG has been significantly refactored from version 1.0 and several changes are
> breaking backward compatibility. The main APSG namespace provides often-used
> classes in lowercase names as aliases to `PascalCase` convention used in
> modules to provide a simplified interface for users. The `PascalCase` names of
> classes use longer and plain English names instead acronyms for better readability.
>
> Check [documentation](https://apsg.readthedocs.org) for more details.

## :hammer_and_wrench: Requirements

You need Python 3.9 or later to run APSG. The package requires [NumPy](https://numpy.org/) and [SciPy](https://www.scipy.org/),
[Matplotlib](https://matplotlib.org/), [SciPy](https://scipy.org/), [SQLAlchemy](https://www.sqlalchemy.org/)
and [pandas](https://pandas.pydata.org/).

## :rocket: How to install

It is strongly suggested to install **apsg** into separate environment. You can create
Python virtual environment. For Linux and macOS use:

    python -m venv .venv
    source .venv/bin/activate

for Windows use Command Prompt or PowerShell:

    python -m venv .venv
    .venv\Scripts\activate

> [!NOTE]
> On Microsoft Windows, it may be required to set the execution policy in PowerShell for the user.
> You can do this by issuing the following PowerShell command:
> ```
> Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
> ```

and install latest stable version of **apsg** using pip within the environment:

    pip install apsg

To include jupyterlab and pyqt5 in installation, use `extra` option:

## I'm using conda or mamba to manage environments

    pip install apsg[extra]

or install **master** with:

    pip install git+https://github.com/ondrolexa/apsg.git

Alternatively, you can clone the repository and do a local install (recommended for dev):
    git clone https://github.com/ondrolexa/apsg.git
    cd apsg
    pip install -e .[dev]

#### Upgrading via pip

To upgrade an existing version of APSG from PyPI, execute:

    pip install apsg --upgrade --no-deps

#### Comments on system-wide instalations on Debian systems

Latest Debian-based systems does not allow to install non-debian packages system-wide.
However, installing all requirements allows to force install APSG system-wide without troubles.

Install requirements using apt:

    sudo apt install python3-numpy python3-matplotlib python3-scipy python3-sqlalchemy python3-pandas

and then install apsg using pip:

    pip install --break-system-packages apsg

### I'm using conda or mamba to manage environments

If you have already have conda or mamba installed, you can create environment with:

    conda config --add channels conda-forge
    conda create -n apsg python apsg jupyterlab pyqt

or using mamba

    mamba create -n apsg python apsg jupyterlab pyqt

#### Current release info

| Name | Downloads | Version | Platforms |
| --- | --- | --- | --- |
| [![Conda Recipe](https://img.shields.io/badge/recipe-apsg-green.svg)](https://anaconda.org/conda-forge/apsg) | [![Conda Downloads](https://img.shields.io/conda/dn/conda-forge/apsg.svg)](https://anaconda.org/conda-forge/apsg) | [![Conda Version](https://img.shields.io/conda/vn/conda-forge/apsg.svg)](https://anaconda.org/conda-forge/apsg) | [![Conda Platforms](https://img.shields.io/conda/pn/conda-forge/apsg.svg)](https://anaconda.org/conda-forge/apsg) |

## :blue_book: Documentation

Explore all the features of APSG. You can find detailed documentation [here](https://apsg.readthedocs.org).

## :computer: Contributing

Most discussion happens on [Github](https://github.com/ondrolexa/apsg). Feel free to open [an issue](https://github.com/ondrolexa/apsg/issues/new) or comment on any open issue or pull request. Check ``CONTRIBUTING.md`` for more details.

## :coin: Donate

APSG is an open-source project, available for you for free. It took a lot of time and resources to build this software. If you find this software useful and want to support its future development please consider donating me.

[![Donate via PayPal](https://www.paypalobjects.com/en_US/i/btn/btn_donateCC_LG.gif)](https://www.paypal.com/cgi-bin/webscr?cmd=_donations&business=QTYZWVUNDUAH8&item_name=APSG+development+donation&currency_code=EUR&source=url)

## License

APSG is free software: you can redistribute it and/or modify it under the terms of the MIT License. A copy of this license is provided in ``LICENSE`` file.
