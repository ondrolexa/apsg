#!/usr/bin/env python
# -*- coding: utf-8 -*-


from setuptools import setup, find_packages

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
    version='0.5.4',
    description='APSG - structural geology module for Python',
    long_description=readme + '\n\n' + history,
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
    tests_require=test_requirements
)
