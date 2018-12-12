#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
from os import path
from setuptools import setup, find_packages

this_directory = path.abspath(path.dirname(__file__))

with open(path.join(this_directory, 'README.md')) as f:
    readme = f.read()

with open(path.join(this_directory, 'HISTORY.md')) as f:
    history = f.read()

requirements = [
    'numpy',
    'matplotlib',
    'scipy'
]

# Recipe from https://pypi.org/project/pytest-runner/
needs_pytest = {'pytest', 'test', 'ptr'}.intersection(sys.argv)
pytest_runner = ['pytest-runner'] if needs_pytest else []

setup(
    name='apsg',
    version='0.6.1',
    description='APSG - structural geology module for Python',
    long_description=readme + '\n\n' + history,
    long_description_content_type='text/markdown',
    author='Ondrej Lexa',
    author_email='lexa.ondrej@gmail.com',
    url='https://github.com/ondrolexa/apsg',
    packages=find_packages(),
    install_requires=requirements,
    entry_points="""
    [console_scripts]
    iapsg=apsg.shell:main
    """,
    license="MIT",
    zip_safe=False,
    keywords='apsg',
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: MIT License',
        'Natural Language :: English',
        'Programming Language :: Python :: 2.7',
        'Programming Language :: Python :: 3',
    ],
    test_suite='tests',
    setup_requires=pytest_runner,
    tests_require=['pytest']
)
