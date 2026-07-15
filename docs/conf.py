"""Sphinx configuration for apsg."""

from __future__ import annotations

import importlib.metadata

# -- Project information -----------------------------------------------------

project = "apsg"
copyright = "2015-2026, Ondrej Lexa"
author = "Ondrej Lexa"
release = importlib.metadata.version("apsg")

# -- General configuration ---------------------------------------------------

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.intersphinx",
    "sphinx.ext.viewcode",
    "sphinx_copybutton",
    "sphinx_design",
    "nbsphinx",
    "autodocsumm",
]

exclude_patterns = ["_build", "**.ipynb_checkpoints"]
source_suffix = ".rst"
master_doc = "index"

# -- autodoc -----------------------------------------------------------------

autodoc_member_order = "bysource"
autodoc_classwise = True
autodoc_typehints = "description"
napoleon_google_docstring = True
napoleon_numpy_docstring = False

# -- nbsphinx ----------------------------------------------------------------

nbsphinx_execute = "always"

# -- intersphinx -------------------------------------------------------------

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "pandas": ("https://pandas.pydata.org/pandas-docs/stable", None),
    "matplotlib": ("https://matplotlib.org/stable", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "scipy": ("https://docs.scipy.org/doc/scipy/", None),
    "sqlalchemy": ("https://docs.sqlalchemy.org/en/20/", None),
}

# -- HTML output -------------------------------------------------------------

html_theme = "shibuya"
html_title = "APSG"
html_css_variables = {
    "light": {
        "sy-color-primary": "#5c6bc0",
        "sy-color-primary-hover": "#7986cb",
    },
    "dark": {
        "sy-color-primary": "#7986cb",
        "sy-color-primary-hover": "#9fa8da",
    },
}
html_theme_options = {
    "globaltoc_collapse": False,
    "github_url": "https://github.com/ondrolexa/apsg",
    "nav_socials": ["github"],
}

# For light mode (e.g., github-light or poimandres)
pygments_style = "github-light-default"

# For dark mode (e.g., github-dark or tokyo-night)
pygments_dark_style = "github-dark-default"
