#!/usr/bin/env python
# -*- coding: utf-8 -*-

from os import path
from setuptools import setup, find_packages


CURRENT_PATH = path.abspath(path.dirname(__file__))


with open(path.join(CURRENT_PATH, "README.md")) as file:
    readme = file.read()


with open(path.join(CURRENT_PATH, "HISTORY.md")) as file:
    history = file.read()

setup(
    name="apsg",
    version="1.2.3",
    description="APSG - The package for structural geologists",
    long_description=readme + "\n\n" + history,
    long_description_content_type="text/markdown",
    author="Ondrej Lexa",
    author_email="lexa.ondrej@gmail.com",
    url="https://github.com/ondrolexa/apsg",
    license="MIT",
    keywords="apsg",
    python_requires=">=3.8",
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
    ],
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=["numpy", "matplotlib", "scipy", "sqlalchemy", "pandas"],
    extras_require={
        "docs": ["sphinx", "ipykernel", "nbsphinx"],
        "test": ["pytest", "black"],
        "extra": ["jupyterlab"],
    },
    project_urls={
        "Documentation": "https://apsg.readthedocs.io/",
        "Source Code": "https://github.com/ondrolexa/apsg/",
        "Bug Tracker": "https://github.com/ondrolexa/apsg/issues/",
    },
    entry_points="""
    [console_scripts]
    iapsg=apsg.shell:main
    """,
)
