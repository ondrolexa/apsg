#!/usr/bin/env python
# -*- coding: utf-8 -*-


import io
import os
import re
import sys

from setuptools import setup, find_packages


HERE = os.path.abspath(os.path.dirname(__file__))


# Recipe from https://pypi.org/project/pytest-runner/
needs_pytest = {"pytest", "test", "ptr"}.intersection(sys.argv)
pytest_runner = ["pytest-runner"] if needs_pytest else []


def read(*names, **kwargs):
    """
    Read the files with a given encoding.
    """
    return io.open(os.path.join(
        os.path.dirname(__file__), *names), encoding=kwargs.get("encoding", "utf8")
    ).read()


# Extract an author, email and version.
package = {}
with io.open(os.path.join(HERE, "apsg", "__init__.py"), "rb") as f:
    file_content = f.read().decode('utf-8')
    package["author"] = re.search(r"^__author__ = ['\"]([^'\"]*)['\"]", file_content, re.M).group(1)
    package["version"] = re.search(r"^__version__ = ['\"]([^'\"]*)['\"]", file_content, re.M).group(1)
    package["email"] = re.search(r"^__email__ = ['\"]([^'\"]*)['\"]", file_content, re.M).group(1)


setup(
    name="apsg",
    version=package["version"],
    description="Structural geology package for Python",
    long_description=read("README.md") + "\n\n" + read("HISTORY.md"),
    long_description_content_type="text/markdown",
    author=package["author"],
    author_email=package["email"],
    url="https://github.com/ondrolexa/apsg",
    packages=find_packages(),
    install_requires=[
        "matplotlib",
        "numpy>=1.14",
        "scipy>=1.0",
        "typing",
    ],
    extras_require={
        "testing": [
            # "black",
            "pytest",
            "radon",
        ],
        "ipython": [
            "jupyter",
        ]
    },
    setup_requires=pytest_runner,
    entry_points={
        "console_scripts": [
            "apsg = apsg.__main__:main"
        ]
    },
    license="MIT",
    zip_safe=False,
    keywords="apsg",
    classifiers=[
        "Development Status :: 4 - Beta",
        'Intended Audience :: Education',
        'Intended Audience :: Developers',
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Natural Language :: English",
        "Programming Language :: Python :: 2.7",
        "Programming Language :: Python :: 3.4",
        "Programming Language :: Python :: 3.5",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7"
    ],
    test_suite="tests",
)
