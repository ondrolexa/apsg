# Changes

### 1.2.3 (Nov 18 2024)
 * ClusterSet accepts PairSet and FaultSet
 * quicknet label option added
 * vector pow bug fix

### 1.2.2 (Oct 21 2024)
 * Fault sense could be defined by str, one of 's', 'd', 'n' and 'r'

### 1.2.1 (Sep 23 2024)
 * Fault sense could be defined by str, one of 's', 'd', 'n' and 'r'

### 1.2.0 (May 24 2024)
 * sqlalchemy and pandas added to requirements
 * quicknet fault bug fixed

### 1.1.5 (May 15 2024)
 * paleomag Core .dd bug fixed
 * fix round-off domain math error for acosd and asind

### 1.1.4 (Dec 13 2023)
 * Ellipsoid repr bugfix

### 1.1.3 (Oct 23 2023)
Bugfix release
 * slip and dilatation tendency methods added to stress
 * proj alias of project for FeatureSet added

### 1.1.2 (Oct 09 2023)
 * added title_kws argument for plotting routines

### 1.1.1 (Oct 06 2023)
 * sigma estimate contour fix

## 1.1.0 (Oct 04 2023)
APSG offers convenient pandas integration via pandas accessors.

See documentation and Pandas interface tutorial for further details.

 * StereoNet tensor method added
 * Cluster class renamed to ClusterSet

### 1.0.3 (Apr 30 2023)
 * lambda properties of tensors renamed to S
 * cursor coordinates in stereonet show lin and fol

### 1.0.1 (Nov 22 2022)
 * density_lookup method implemented for StereoNet.grid
 * Stress tensor sigma* properties using inverted order of eigenvalues
 * render2fig method of StereoNet implemented
 * vector-like objects are not iterable, so properly render in pandas 

## 1.0.0 (Oct 7 2022)
New major release

APSG has been significantly refactored from version 1.0 and several changes are
breaking backward compatibility. The main APSG namespace provides often-used
classes in lowercase names as aliases to `PascalCase` convention used in
modules to provide a simplified interface for users. The `PascalCase` names of
classes use longer and plain English names instead acronyms for better
readability.

See documentation for further details.

### 0.7.3 (Oct 6 2022)
 * figure window title removed from StereoNet
 * for future only bugfixes planned, foo further development see versions >=1.0

### 0.7.2 (Oct 6 2022)
 * bugfix release

### 0.7.1 (Jul 13 2021)
 * paleomag rs3 input/output improved
 * Simple SQLAlchemy API to sdb database implemented
 * StereoNet arc method fixed
 * StereoNet polygon method added

## 0.7.0 (Feb 3 2021)

* Python 2 support dropped
* RosePlot added
* Ortensor has from_pairs method for Lisle tensor for orthogonal data
* StereoNet scatter method has labels kwarg to show hover annotations

### 0.6.3 (Dec 6 2019)

* Python 2/3 compatibility fix

### 0.6.2 (Dec 6 2019)

* few minor bugs fixed
* Stereogrid apply_func passes Vec3 instead numpy array
* Pair H method to get mutual rotation implemented
* velgrad method of DefGrad accepts steps kwarg
  to generate list of DefGrad tensors
* Added Tensor class to work with deformation tensors

### 0.6.1 (Dec 12 2018)

* Stereogrid always use Euclidean norms as weights
* DefGrad properties e1, e2, e3 (natural principal strains) added
* DefGrad properties eoct, goct (octahedral strains) added
* DefGrad from_ratios class method added
* DefGrad properties k, d, K, D (strain symmetries and intesities) added
* New class Ellipsoid added to work with ellipsoids
* FabricPLot renamed to VollmerPlot for consistency
* RamsayPlot, FlinnPlot and HsuPlot implemented
* All fabric plots have new path method accepting list of tensors

## 0.6.0 (Nov 7 2018)

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

### 0.5.4 (Oct 19 2018)

* StereoNet has cbpad keyword for colorbar padding
* FabricPlot bug introduced in 0.5.2 fixed.

### 0.5.3 (Oct 10 2018)

* Bugfix release

### 0.5.2 (Oct 10 2018)

* Fischer distribution sampling added
* transform method has norm kwarg to normalize tranformed vectors
* axisangle property to calculate axis and angle from rotation matrix
* StereoNet arc method added
* Vec3 and Group upper and flip properties implemented
* DefGrad, VelGrad and Stress rotate method accepts also rotation matrix
* velgrad method added to DefGrad to calculate matrix logarithm
* StereoGrid has new methods max, min, max_at, min_at

### 0.5.1 (Dec 5 2017)

* Kent distribution sampling added
* Automatic kernel density estimate for contouring
* UserWarnings fix

## 0.5.0 (Nov 19 2017)

* bux fix minor release

### 0.4.4 (Mar 25 2017)

* Group method centered improved
* Group method halfspace added to reorient all vectors towards resultant
  halfspace

### 0.4.3 (Mar 25 2017)

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

### 0.4.1-2 (Mar 4 2017)

* bugfix

## 0.4.0 (Mar 4 2017)

* Density class renamed to StereoGrid
* Fault sense under rotation fixed
* FaultSet example provided
* Angelier-Mechler dihedra method implemented for FaultSet
* StereoNet accepts StereoGrid and Ortensor as quick plot arguments
* StereoNet instance has axtitle method to put text below stereonet

### 0.3.7 (Jan 5 2017)

* conda build for all platforms
* numpy, matplotlib and other helpres imported by default
* ortensor is normed by default
* ortensor MADp, MADo, MAD and kind properties added

### 0.3.6 (Jan 3 2017)

* shell script iapsg opens interactive console

### 0.3.5 (Nov 12 2016)

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

### 0.3.4 (Jun 20 2016)

* RTD fix

### 0.3.3 (Jun 4 2016)

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

### 0.3.2 (Feb 22 2016)

* FabricPlot - triangular fabric plot added
* .asvec3 property has .V alias
* Resultant of Fol and Lin is calculated as vectorial in centered position
* dv property of Fol added to return dip slip vector

### 0.3.1 (Nov 20 2015)

* SDB class improved. Support basic filtering including tags
* StereoNet has close method to close figure and new method
  to re-initialize figure when closed in interactive mode
* iapsg shell script added to invoke apsg ipython shell

## 0.3.0 (Nov 9 2015)

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

### 0.2.3 (Oct 21 2015)

* New Docstrings format
* StereoNet.getfols method bug fixed.
* Shell scripts to run interactive session improved.

### 0.2.2 (Apr 17 2015)

* FaultSet class added. Fault and Hoeppner methods of StereoNet implemented
* VelGrad and DefGrad classes used for transformations added
* G class to quickly create groups from strings added.

### 0.2.1 (Dec 9 2014)

* Quick plotting of groups fixed.

## 0.2.0 (Dec 9 2014)

* new StereoNet class for Schmidt projection
* Quick plot when data are passed as argument `StereoNet` class instantiation
* mplstereonet dependency depreceated
* new `Pair` and `Fault` classes to manipulate paired data (full support in future)
* new `uniform_lin` and `uniform_fol` `Group` methods
* abs for `Group` implemented to calculate euclidean norms
* new `Group` method normalized
* new `Group` properties and methods to calculate spherical statistics

## 0.1.0 (Nov 1 2014)

* First release of APSG
