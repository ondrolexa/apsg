.. :changelog:

History
-------

0.1.0 (01 Nov 2014)
-------------------

* First release of APSG

0.2 (09 Dec 2014)
-----------------

* new StereoNet class for Schmidt projection
* Quick plot when data are passed to StereoNet class instantiation
* mplstereonet dependency depreceated

* new Pair and Fault classes to manipulate paired data (full support in future)
* new uniform_lin and uniform_fol Group methods
* abs for Group implemented to calculate euclidean norms
* new Group method normalized
* new Group properties and methods to calculate spherical statistics

0.2 Patch 1 (09 Dec 2014)
-------------------------

* Quick plotting of groups fixed.

0.2 Patch 2 (17 Apr 2015)
-------------------------

* FaultSet class added. Fault and Hoeppner methods of StereoNet implemented
* VelGrad and DefGrad classes used for transformations added
* G class to quickly create groups from strings added.

0.2 Patch 3 (21 Oct 2015)
-------------------------

* New Docstrings format
* StereoNet.getfols method bug fixed.
* Shell scripts to run interactive session improved.

0.3 (09 Nov 2015)
-----------------

* Group fancy indexing implemented. Group could be indexed by sequences
  of indexes like list, tuple or array as well as sliced.
* Cluster class with hierarchical clustering implemented
* Group to_file and from_file methods implemented to store data in file
* Group copy method for shallow copy implemented
* StereoNet now accept Vec3 and Fault object as well for instant plotting.
* Ortensor updated with new properties E1,E2,E3 and Vollmer(1989) indexes
  P,G,R and C. Bug in Woodcocks's shape and strength values fixed.
* uniform_lin and uniform_fol improved.
* asvec3 method implemented for Fol and Lin
* fol_plot property of StereoNet allows choose poles or great circles for
  immediate plotting
* bootstrap method of Group provide generator of random resampling with
  replacements.
* Group examples method provide few well-known datasets.
* Matplotlib deprecation warnings are ignored by default

0.3 Patch 1 (20 Nov 2015)
-------------------------

* SDB class improved. Support basic filtering including tags
* StereoNet has close method to close figure and new method
  to re-initialize figure when closed in interactive mode
* iapsg shell script added to invoke apsg ipython shell

0.3 Patch 2 (22 Feb 2016)
-------------------------

* FabricPlot - triangular fabric plot added
* .asvec3 property has .V alias
* Resultant of Fol and Lin is calculated as vectorial in centered position
* dv property of Fol added to return dip slip vector

0.3 Patch 3 (04 Jun 2016)
-------------------------

* Added E1,E2,E3 properties and polar decomposition method to DefGrad object
* StereoNet has vector method to mimics lower and upper hemisphere plotting
  of Lin and Vec3 objects as used in paleomagnetic plots
* StereoNet could be initialized with subplots
* rake method of Fol added to return vector defined by rake
* Density could be initialized without data for user-defined calculations
  New method apply_func could be used to calculate density
* Contour(f) methods accept Density object as argument
* Added Group class methods to generate Spherical Fibonacci and Golden Section
  based uniform distributions of Vec3, Lin and Fol

0.3 Patch 4 (20 Jun 2016)
-------------------------

* RTD fix

0.3 Patch 5 (12 Nov 2016)
-------------------------

* Simple settings interface implemented in in apsg.core.seetings dictionary.
  To change: `from apsg.core import settings`
             `set setting['name']=value`
* `notation` seeting with values `dd` or `rhr` control how azimuth argument of
  Fol is represented.
* `vec2dd` setting with values `True` or `False` control how `Vec3` is
  represented.
* Vec3 could be instantiated by one arument (vector like), 2 arguments
  (azimuth, inclination) or 3 arguments (azimuth, inclination, magnitude).
* Group and FaultSet can return array or list of user-defined attributes of
  all elements

0.3 Patch 6 (03 Jan 2017)
-------------------------

* shell script iapsg opens interactive console
