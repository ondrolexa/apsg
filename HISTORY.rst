.. :changelog:

History
-------

0.1.0 (01 Nov 2014)
---------------------

* First release of APSG

0.2.0 (09 Dec 2014)
---------------------

* new StereoNet class for Schmidt projection
* Quick plot when data are passed to StereoNet class instantiation
* mplstereonet dependency depreceated

* new Pair and Fault classes to manipulate paired data (full support in future)
* new uniform_lin and uniform_fol Group methods
* abs for Group implemented to calculate euclidean norms
* new Group method normalized
* new Group properties and methods to calculate spherical statistics

0.2.1 (09 Dec 2014)
---------------------

* Quick plotting of groups fixed.

0.2.2 (17 Apr 2015)
---------------------

* FaultSet class added. Fault and Hoeppner methods of StereoNet implemented
* VelGrad and DefGrad classes used for transformations added
* G class to quickly create groups from strings added.

0.2.3 (21 Oct 2015)
---------------------

* New Docstrings format
* StereoNet.getfols method bug fixed.
* Shell scripts to run interactive session improved.

0.3.0 (09 Nov 2015)
---------------------

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

0.3.1 (20 Nov 2015)
---------------------

* SDB class improved. Support basic filtering including tags
* StereoNet has close method to close figure and new method
  to re-initialize figure when closed in interactive mode
* iapsg shell script added to invoke apsg ipython shell