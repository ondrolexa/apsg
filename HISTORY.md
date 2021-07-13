# Changes

## 0.7.1 (master)
 * paleomag rs3 input/output improved
 * Simple SQLAlchemy API to sdb database implemented
 * StereoNet arc method fixed
 * StereoNet polygon method added

## 0.7.0 (3 Feb 2021)

* Python 2 support dropped
* RosePlot added
* Ortensor has from_pairs method for Lisle tensor for orthogonal data
* StereoNet scatter method has labels kwarg to show hover annotations

### 0.6.3 (6 Dec 2019)

* Python 2/3 compatibility fix

### 0.6.2 (6 Dec 2019)

* few minor bugs fixed
* Stereogrid apply_func passes Vec3 instead numpy array
* Pair H method to get mutual rotation implemented
* velgrad method of DefGrad accepts steps kwarg
  to generate list of DefGrad tensors
* Added Tensor class to work with deformation tensors

### 0.6.1 (12 Dec 2018)

* Stereogrid always use Euclidean norms as weights
* DefGrad properties e1, e2, e3 (natural principal strains) added
* DefGrad properties eoct, goct (octahedral strains) added
* DefGrad from_ratios class method added
* DefGrad properties k, d, K, D (strain symmetries and intesities) added
* New class Ellipsoid added to work with ellipsoids
* FabricPLot renamed to VollmerPlot for consistency
* RamsayPlot, FlinnPlot and HsuPlot implemented
* All fabric plots have new path method accepting list of tensors

## 0.6.0 (07 Nov 2018)

* Stress always gives eigenvalues sorted
* Stress I1, I2, I3 properties for invariants implemented
* Stress mean_stress property implemented
* Stress hydrostatic and deviatoric properties implemented
* precision added to settings to control numerical comparisms
* figsize added to settings to control figure size across APSG
* Animation examples fixed
* rand class method implemented for Fol, Lin, Vec3 and Pair to
  generate random instance
* Group to_csv and from_csv improved
* SDB tags method works properly for multiple tags
* SDB can modify database metadata
* QGIS 3 plugin ReadSDB compatibility

### 0.5.4 (19 Oct 2018)

* StereoNet has cbpad keyword for colorbar padding
* FabricPlot bug introduced in 0.5.2 fixed.

### 0.5.3 (10 Oct 2018)

* Bugfix release

### 0.5.2 (10 Oct 2018)

* Fischer distribution sampling added
* transform method has norm kwarg to normalize tranformed vectors
* axisangle property to calculate axis and angle from rotation matrix
* StereoNet arc method added
* Vec3 and Group upper and flip properties implemented
* DefGrad, VelGrad and Stress rotate method accepts also rotation matrix
* velgrad method added to DefGrad to calculate matrix logarithm
* StereoGrid has new methods max, min, max_at, min_at

### 0.5.1 (05 Dec 2017)

* Kent distribution sampling added
* Automatic kernel density estimate for contouring
* UserWarnings fix

## 0.5.0 (19 Nov 2017)

* bux fix minor release

### 0.4.4 (25 Mar 2017)

* Group method centered improved
* Group method halfspace added to reorient all vectors towards resultant
  halfspace

### 0.4.3 (25 Mar 2017)

* Stress tensor with few basic methods implemented
* StereoGrid keyword argument 'weighted' to control weighting
* StereoNet kwargs are passed to underlying methods for immediate plots
* StereoNet tensor method implemented (draw eigenlins or fols based on
  fol_plot settings)
* Group totvar property and dot and proj methods implemented
* Fol and Lin dot method returns absolute value of dot product
* Vec3 H method to get mutual rotation implemented
* StereoNet.contourf method draw contour lines as well by default. Option
  clines controls it.
* centered bug fixed
* StereoNet allows simple animations. Add `animate=True` kwarg to plotting
  method and finally call StereoNet animate method.

### 0.4.1-2 (04 Mar 2017)

* bugfix

## 0.4.0 (04 Mar 2017)

* Density class renamed to StereoGrid
* Fault sense under rotation fixed
* FaultSet example provided
* Angelier-Mechler dihedra method implemented for FaultSet
* StereoNet accepts StereoGrid and Ortensor as quick plot arguments
* StereoNet instance has axtitle method to put text below stereonet

### 0.3.7 (05 Jan 2017)

* conda build for all platforms
* numpy, matplotlib and other helpres imported by default
* ortensor is normed by default
* ortensor MADp, MADo, MAD and kind properties added

### 0.3.6 (03 Jan 2017)

* shell script iapsg opens interactive console

### 0.3.5 (12 Nov 2016)

* Simple settings interface implemented in in apsg.core.seetings dictionary.
  To change settings use:
  ```
  from apsg.core import settings
  setting['name']=value
  ```
* `notation` setting with values `dd` or `rhr` control how azimuth argument of
  Fol is represented.
* `vec2dd` setting with values `True` or `False` control how `Vec3` is
  represented.
* Vec3 could be instantiated by one arument (vector like), 2 arguments
  (azimuth, inclination) or 3 arguments (azimuth, inclination, magnitude).
* Group and FaultSet can return array or list of user-defined attributes of
  all elements

### 0.3.4 (20 Jun 2016)

* RTD fix

### 0.3.3 (04 Jun 2016)

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

### 0.3.2 (22 Feb 2016)

* FabricPlot - triangular fabric plot added
* .asvec3 property has .V alias
* Resultant of Fol and Lin is calculated as vectorial in centered position
* dv property of Fol added to return dip slip vector

### 0.3.1 (20 Nov 2015)

* SDB class improved. Support basic filtering including tags
* StereoNet has close method to close figure and new method
  to re-initialize figure when closed in interactive mode
* iapsg shell script added to invoke apsg ipython shell

## 0.3.0 (09 Nov 2015)

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

### 0.2.3 (21 Oct 2015)

* New Docstrings format
* StereoNet.getfols method bug fixed.
* Shell scripts to run interactive session improved.

### 0.2.2 (17 Apr 2015)

* FaultSet class added. Fault and Hoeppner methods of StereoNet implemented
* VelGrad and DefGrad classes used for transformations added
* G class to quickly create groups from strings added.

### 0.2.1 (09 Dec 2014)

* Quick plotting of groups fixed.

## 0.2.0 (09 Dec 2014)

* new StereoNet class for Schmidt projection
* Quick plot when data are passed as argument `StereoNet` class instantiation
* mplstereonet dependency depreceated
* new `Pair` and `Fault` classes to manipulate paired data (full support in future)
* new `uniform_lin` and `uniform_fol` `Group` methods
* abs for `Group` implemented to calculate euclidean norms
* new `Group` method normalized
* new `Group` properties and methods to calculate spherical statistics

## 0.1.0 (01 Nov 2014)

* First release of APSG
