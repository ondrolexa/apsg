================================
Welcome to APSG's documentation!
================================

.. image:: https://ondrolexa.github.io/apsg/apsg_banner.svg
   :width: 300
   :alt: APSG logo

APSG defines several new python classes to easily manage, analyze and
visualize orientational structural geology data. It is under active
developement, so until documenation will be finished, you can go trough
tutorial to see what APSG can do for you.

Usage
-----

To use APSG in a project::

    import apsg

To use APSG interactively it is easier to import into current namespace::

    from apsg import *


Changes in classnames and API
-----------------------------

.. note::
   APSG has been significantly refactored from version 1.0 and several changes are
   breaking backward compatibility. The main APSG namespace provides often-used
   classes in lowercase names as aliases to `PascalCase` convention used in
   modules to provide a simplified interface for users. The `PascalCase` names of
   classes use longer and plain English names instead acronyms for better
   readability.


If you already used older versions of APSG, check following table for new
names and aliases of most commonly used classes.

.. list-table::
   :widths: 15 20 50
   :header-rows: 1

   * - alias
     - class name
     - Description
   * - vec2
     - Vector2
     - A class to represent a 2D vector
   * - vec
     - Vector3
     - A class to represent a 3D vector
   * - lin
     - Lineation
     - A class to represent non-oriented (axial) linear feature
   * - fol
     - Foliation
     - A class to represent non-oriented planar feature
   * - pair
     - Pair
     - The class to store pair of planar and linear features
   * - fault
     - Fault
     - The class to store pair of planar and linear features
       together with sense of movement
   * - cone
     - Cone
     - The class to store cone with given axis, secant line and
       revolution angle in degrees
   * - vec2set
     - Vector2Set
     - Class to store set of ``Vector2`` features
   * - vecset
     - Vector3Set
     - Class to store set of ``Vector3`` features
   * - linset
     - LineationSet
     - Class to store set of ``Lineation`` features
   * - folset
     - FoliationSet
     - Class to store set of ``Foliation`` features
   * - pairset
     - PairSet
     - Class to store set of ``Pair`` features
   * - faultset
     - FaultSet
     - Class to store set of ``Fault`` features
   * - coneset
     - ConeSet
     - Class to store set of ``Cone`` features
   * - defgrad2
     - DeformationGradient2
     - The class to represent 2D deformation gradient tensor
   * - defgrad
     - DeformationGradient3
     - The class to represent 3D deformation gradient tensor
   * - velgrad2
     - VelocityGradient2
     - The class to represent 2D velocity gradient tensor
   * - velgrad
     - VelocityGradient3
     - The class to represent 3D velocity gradient tensor
   * - stress2
     - Stress2
     - The class to represent 2D stress tensor
   * - stress
     - Stress3
     - The class to represent 3D stress tensor
   * - ellipse
     - Ellipse
     - The class to represent 2D ellipse
   * - ellipsoid
     - Ellipsoid
     - The class to represent 3D ellipsoid
   * - ortensor2
     - OrientationTensor2
     - Represents an 2D orientation tensor
   * - ortensor
     - OrientationTensor3
     - Represents an 3D orientation tensor
   * - ellipseset
     - EllipseSet
     - Class to store set of ``Ellipse`` features
   * - ortensor2set
     - OrientationTensor2Set
     - Class to store set of ``OrientationTensor2`` features
   * - ellipsoidset
     - EllipsoidSet
     - Class to store set of ``Ellipsoid`` features
   * - ortensorset
     - OrientationTensor3Set
     - Class to store set of ``OrientationTensor3`` features


Check tutorials and module API for more details.

Contents
--------

.. toctree::
   :maxdepth: 2

   installation
   tutorials
   automodules
   contributing
   authors
