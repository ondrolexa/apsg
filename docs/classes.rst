============
APSG classes
============

The main APSG namespace provide most of the classes in lowercase names as
aliases to PascalCase convention used in modules to provides simplified
interface for users.

.. list-table:: Aliases of most commonly used classes
   :widths: 15 20 50
   :header-rows: 1

   * - alias
     - class name
     - Description
   * - vec2
     - Vector2
     - A class to represent a 2D vector
   * - vec3
     - Vector3
     - A class to represent a 3D vector
   * - lin
     - Lineation
     - A class to represent axial (non-oriented) linear feature (lineation)
   * - fol
     - Foliation
     - A class to represent non-oriented planar feature (foliation)
   * - pair
     - Pair
     - The class to store pair of planar and linear feature
   * - fault
     - Fault
     - The class to store ``Pair`` with associated sense of movement
   * - cone
     - Cone
     - The class to store cone with given axis, secant line and revolution angle in degrees
   * - vec2set
     - Vector2Set
     - Class to store set of ``Vector2`` features
   * - vec3set
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
   * - ellipsoidset
     - EllipsoidSet
     - Class to store set of ``Ellipsoid`` features
   * - ortensorset
     - OrientationTensor3Set
     - Class to store set of ``OrientationTensor3`` features

