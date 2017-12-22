#!/usr/bin/env python
# -*- coding: utf-8 -*-


from setuptools import setup, find_packages

from apsg import __version__, __author__, __email__


with open('README.rst') as readme_file:
    readme = readme_file.read()

with open('HISTORY.rst') as history_file:
    history = history_file.read().replace('.. :changelog:', '')

requirements = [
    'numpy',
    'matplotlib',
    'scipy'
]

test_requirements = [
    # TODO: put package test requirements here
]

setup(
    name='apsg',
    version=__version__,
    description='APSG - structural geology module for Python',
    long_description=readme + '\n\n' + history,
    author=__author__,
    author_email=__email__,
    url='https://github.com/ondrolexa/apsg',
    packages=find_packages(),
    install_requires=requirements,
    entry_points="""
    [console_scripts]
    iapsg=apsg.shell:main
    """,
    license="BSD",
    zip_safe=False,
    keywords='apsg',
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: BSD License',
        'Natural Language :: English',
        'Programming Language :: Python :: 2.7',
        'Programming Language :: Python :: 3',
    ],
    test_suite='tests',
    tests_require=test_requirements
)
