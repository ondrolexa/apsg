[![PyPI version](https://badge.fury.io/py/apsg.png)](http://badge.fury.io/py/apsg)

APSG is python module which defines several new classes to easily manage, analyze and visualize orientational structural geology data.

### Basic usage
APSG module could be imported either into own namespace or into active one for easier interactive work:

    from apsg import *

Base class `Vec3` is derived from `numpy.array` class and offers several new methods for simple vector manipulation. To work with orientational data in structural geology, APSG provide two classes derived from `Vec3` class. There is `Fol` class to represent planar features by planes and `Lin` class to represent linear feature by lines. Both classes provide all `Vec3` methods, but they differ in way how instance is created and how some calculations are evaluated, as structural geology data are commonly axial in nature.