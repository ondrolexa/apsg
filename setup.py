#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
from os import path
from setuptools import setup, find_packages


CURRENT_PATH = path.abspath(path.dirname(__file__))


with open(path.join(CURRENT_PATH, "README.md")) as file:
    readme = file.read()


with open(path.join(CURRENT_PATH, "HISTORY.md")) as file:
    history = file.read()


# Recipe from https://pypi.org/project/pytest-runner/
needs_pytest = {"pytest", "test", "ptr"}.intersection(sys.argv)
pytest_runner = ["pytest-runner"] if needs_pytest else []


setup(
    name="apsg",
    version="0.6.3",
    description="APSG - The package for structural geologists",
    long_description=readme + "\n\n" + history,
    long_description_content_type="text/markdown",
    author="Ondrej Lexa",
    author_email="lexa.ondrej@gmail.com",
    url="https://github.com/ondrolexa/apsg",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=["matplotlib", "numpy>=1.14", "scipy>=1.0",],
    extras_require={
        "testing": ["pytest", "radon",],
        "linting": ["black", "pylint",],
        "jupyter": ["jupyterlab",],
    },
    entry_points="""
    [console_scripts]
    iapsg=apsg.shell:main
    """,
    license="MIT",
    zip_safe=False,
    keywords="apsg",
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Natural Language :: English",
        "Programming Language :: Python :: 2.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.6",
    ],
    test_suite="tests",
    setup_requires=pytest_runner,
    tests_require=["pytest"],
)
