<picture>
  <source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/ondrolexa/apsg/master/docs/images/apsg_banner_dark.svg">
  <img alt="APSG logo" src="https://raw.githubusercontent.com/ondrolexa/apsg/master/docs/images/apsg_banner_light.svg">
</picture>

[![PyPI - Version](https://img.shields.io/pypi/v/apsg)](https://pypi.org/project/apsg)
[![Conda](https://img.shields.io/conda/v/conda-forge/apsg)](https://anaconda.org/conda-forge/apsg)
[![Documentation Status](https://readthedocs.org/projects/apsg/badge/?version=stable)](https://apsg.readthedocs.io/en/stable/?badge=stable)
[![codecov](https://codecov.io/gh/ondrolexa/apsg/graph/badge.svg?token=YKXWmJJHw3)](https://codecov.io/gh/ondrolexa/apsg)
[![DOI](https://img.shields.io/badge/DOI-10.5281%2Fzenodo.593586-blue)](https://doi.org/10.5281/zenodo.593586)

APSG is the package for structural geologists. It defines several new Python classes to easily manage, analyze and visualize orientation structural geology data.

Check [CHANGELOG.md](https://github.com/ondrolexa/apsg/blob/master/CHANGELOG.md) for recent updates.

## Quick example

```python
from apsg import *

f = folset.random_fisher(position=fol(130, 60))
s = StereoNet()
s.great_circle(f)
s.point(f)
s.contour(f)
s.show()
```

`StereoNet` supports both equal-area (Schmidt, default) and equal-angle (Wulff) projections,
lower and upper hemisphere, and rotating the whole net independently of the plotted data.
See the stereonet tutorial in the [documentation](https://apsg.readthedocs.org) for examples.

## Requirements

APSG requires Python 3.12 or later. It depends on [NumPy](https://numpy.org/),
[SciPy](https://scipy.org/), [Matplotlib](https://matplotlib.org/),
[SQLAlchemy](https://www.sqlalchemy.org/), [pandas](https://pandas.pydata.org/),
[pygeomag](https://github.com/boxpet/pygeomag). The installers below take care of these.

## Installation

Install APSG into a separate environment. Pick one of the options below. uv is the default
and the fastest; pip and conda/mamba are alternatives.

### With uv (recommended)

Install [uv](https://docs.astral.sh/uv/), then create a virtual environment and install APSG
into it:

```sh
uv venv
uv pip install apsg
```

Activate the environment on Linux and macOS:

```sh
source .venv/bin/activate
```

On Windows (Command Prompt or PowerShell):

```powershell
.venv\Scripts\activate
```

To include JupyterLab and [openpyxl](https://openpyxl.readthedocs.io/) (needed to read Excel files), use the `lab` extra:

```sh
uv pip install "apsg[lab]"
```

In an existing project with a `pyproject.toml`, add APSG as a dependency instead:

```sh
uv add apsg
```

To use APSG from the command line, install it as a tool. This puts an isolated copy of APSG
on your `PATH`, without activating an environment:

```sh
uv tool install apsg
```

Then start the interactive APSG shell, which runs `from apsg import *` for you:

```sh
iapsg
```

To try the shell without installing, run `uvx --from apsg iapsg`. Add the `lab` extra with
`uv tool install "apsg[lab]"` to include JupyterLab.

### Development install

To work on APSG itself, clone the repository and install it in editable mode with all extras
and development tools:

```sh
git clone https://github.com/ondrolexa/apsg.git
cd apsg
uv sync --all-extras --dev
```

Run the test suite with `uv run pytest`.

### With pip

Create and activate a virtual environment. On Linux and macOS:

```sh
python -m venv .venv
source .venv/bin/activate
```

On Windows (Command Prompt or PowerShell):

```powershell
python -m venv .venv
.venv\Scripts\activate
```

> **Note:** On Windows, PowerShell may refuse to run the activation script. To allow it for
> the current user, run:
>
> ```powershell
> Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
> ```

Install the latest stable release from PyPI:

```sh
pip install apsg
```

To include JupyterLab and [openpyxl](https://openpyxl.readthedocs.io/) (needed to read Excel files), use the `lab` extra:

```sh
pip install "apsg[lab]"
```

To upgrade an existing installation without touching its dependencies:

```sh
pip install --upgrade --no-deps apsg
```

To install the master branch from GitHub:

```sh
pip install git+https://github.com/ondrolexa/apsg.git
```

### With conda or mamba

If you already use conda or mamba, add the `conda-forge` channel and create an environment
with APSG:

```sh
conda config --add channels conda-forge
conda create -n apsg python apsg jupyterlab
```

or with mamba:

```sh
mamba create -n apsg python apsg jupyterlab
```

To install APSG into an existing environment, run `conda install apsg` (or
`mamba install apsg`).

### Current release

| Name | Downloads | Version | Platforms |
| --- | --- | --- | --- |
| [![Conda Recipe](https://img.shields.io/badge/recipe-apsg-green.svg)](https://anaconda.org/conda-forge/apsg) | [![Conda Downloads](https://img.shields.io/conda/dn/conda-forge/apsg.svg)](https://anaconda.org/conda-forge/apsg) | [![Conda Version](https://img.shields.io/conda/vn/conda-forge/apsg.svg)](https://anaconda.org/conda-forge/apsg) | [![Conda Platforms](https://img.shields.io/conda/pn/conda-forge/apsg.svg)](https://anaconda.org/conda-forge/apsg) |

## Documentation

Explore all features of APSG in the [documentation](https://apsg.readthedocs.org).

## Contributing

Most discussion happens on [GitHub](https://github.com/ondrolexa/apsg). Feel free to open [an issue](https://github.com/ondrolexa/apsg/issues/new) or comment on any open issue or pull request. See [CONTRIBUTING.md](https://github.com/ondrolexa/apsg/blob/master/CONTRIBUTING.md) for more details.

## Donate

APSG is an open-source project, available for free. It took a lot of time and resources to build this software. If you find it useful and want to support its future development, please consider donating.

[![Donate via PayPal](https://www.paypalobjects.com/en_US/i/btn/btn_donateCC_LG.gif)](https://www.paypal.com/cgi-bin/webscr?cmd=_donations&business=QTYZWVUNDUAH8&item_name=APSG+development+donation&currency_code=EUR&source=url)

## License

APSG is free software: you can redistribute it and/or modify it under the terms of the MIT License. A copy of this license is provided in the [LICENSE](https://github.com/ondrolexa/apsg/blob/master/LICENSE) file.
