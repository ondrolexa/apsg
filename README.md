% Structural geology module for Python
% Ondrej Lexa <lexa.ondrej@gmail.com>
% 2014

First steps with APSG module
============================

APSG defines several new python classes to easily manage, analyze
and visualize orientational structural geology data. Base class `Vec3`
is derived from `numpy.array` class and affers several new method
which will be explained on following examples.

Download and install APSG module
--------------------------------

APSG is distributed as a single file with no traditional python install
implemented yet. For now, download `apsg.py` file and save it to
working directory or to any folder on `PYTHONPATH`.

Import APSG module
------------------

APSG module could be imported either into own namespace or into
active one for easier interactive work:

~~~~{.python}
>>> from apsg import *
~~~~~~~~~~~~~

Basic operation with vectors
----------------------------

Vector object of `Vec3` class could be created from any iterable
object as list, tuple or array:

~~~~{.python}
>>> u = Vec3([1, -2, 3])
>>> v = Vec3([-2, 1, 1])
~~~~~~~~~~~~~
