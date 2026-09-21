# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).
Note: apsg's version numbers do not strictly follow [Semantic Versioning](https://semver.org/) --
a patch-level release can include breaking changes, marked below as **BREAKING**.

## [2.0.1] - master

### Added
- `StereoNet.dihedra()` fills the extensional dihedra (the quadrants containing the T axis) of faults; a `FaultSet` shows where its dihedra overlap.
- `StereoNet.beachball()` plots the beach ball of a stress tensor, filling the compressive quadrants (those containing the P axis); a `Stress3Set` shows where its beach balls overlap.
- `Arc`/`ArcSet` (and `StereoNet.arc()`) for plotting a curved path between two vectors, with an optional bow away from the plain great-circle connection. Arcs can be drawn as a line, as points, or as the filled polygon they bound (or the rest of the net, with `region="outside"`).

## [2.0.0] - 2026-09-21

### Added
- Multiple contour layers can now be overlaid on the same stereonet in one plot.
- `StereoNet.set_rotation()`/`.rotation` and `apsg.rotation_from_axis_angle()` to rotate a whole net (grid and data) around any axis.
- `StereoGrid` can now be saved to and loaded from a file.
- Config options extended.
- `kinematic_vorticity`, `vorticity_vector`, `vorticity_scalar` for velocity gradients.
- `flow_apophyses` to find the directions a progressive deformation neither stretches nor shrinks.
- `from_ellipsoid`/`from_ellipse` to recover the deformation that produced a measured strain ellipsoid/ellipse.
- `Stress2Set`, plus convenience shortcuts (rake, cone angles, shape/anisotropy parameters, etc.) across feature sets.
- `P_j` and `T` (Jelínek 1981 anisotropy degree and shape parameter) on `Ellipsoid`.
- `WebSDBSession` for connecting to a remote websdb database over the web.
- `Arc`/`ArcSet` (and `StereoNet.arc()`) for plotting a curved path between two vectors, with an optional bow away from the plain great-circle connection. Arcs can be drawn as a line, as points, or as the filled polygon they bound.
- `quicknet()` now also accepts arcs, tensors, stress tensors (and their sets), and `StereoGrid` contours.
- `ArcSet.from_vectors()` to build a set of arcs connecting consecutive vectors.
- `Rotation3.axisangle_from_vectors_axis()` returning the signed (axis, angle) pair behind `from_vectors_axis()`.
- `mean_tensor()` on `EllipsoidSet` and `Stress3Set` returning the mean tensor and confidence ellipses of its principal axes (linear perturbation method of Jelínek 1978, after Hext 1963), with optional `normalize` and `anisoft` (extra (n-1)/n factor as in jelinekstat/TomoFab) settings.
- `confidence(method="jelinek")` on `StereoNet` to plot the confidence ellipses of the principal axes of the mean tensor of an `EllipsoidSet` or `Stress3Set`: all three by default, or a single one selected with `which`.
- `tensor()` now accepts `mec` (marker edge color) when plotting principal directions.

### Changed
- **BREAKING:** the stereonet drawing engine was rebuilt on a real matplotlib map projection.
- **BREAKING:** contour density values from the "kamb" and "sph" methods are now on one common scale ("standard deviations above random").
- **BREAKING:** `contour()`: `clines` renamed to `filled`; default method changed from "sph" to "kamb"; densities above random are now shown by default (`clip=True`).
- **BREAKING:** `StereoNet` setup options renamed: `overlay_position` to `rotation`, `overlay` to `grid`, `overlay_step` to `grid_step`.
- **BREAKING:** rotating a net now also rotates the plotted data by default.

### Removed
- **BREAKING:** `contour()` options `sigmanorm`, `trimzero`, `show_data`, `data_kws` (plot data points separately with `point()` instead).
- **BREAKING:** `StereoGrid.contourf()`/`.contour()`/`.plotcountgrid()` (use `StereoNet.contour()` instead).

### Fixed
- `contour()`'s `clip` option now actually works (it silently did nothing before).
- Lines, poles, planes and contour plots on an upper-hemisphere stereonet now correctly appear at the antipodal position.
- `cone()`, `arc()`, `confidence()`, `fault()` and `pair()` plots now correctly respond to upper/lower hemisphere choice.
- The coordinate readout shown when hovering over a stereonet now always reports the correct orientation.
- `vector()`'s open (antipodal) marker now always matches its filled marker's color and no longer shifts later plots' colors.
- Hoeppner plot arrows are no longer hidden behind an oversized marker.
- Paleomagnetic `stereo_plot()` silently plotting nothing.
- `quicknet()` no longer silently ignores a `ConeSet`.
- `quicknet(..., fol_as_pole=True)` no longer crashes.
- `stress()` silently ignored its `mec` (marker edge color) option.
- `StereoNet.plot()` with a style now prints an error for invalid arguments, like the direct plotting methods, instead of crashing.
- `tensor()` and `stress()` error messages for invalid arguments no longer say "arrow".

## [1.5.1] - 2026-08-04

### Added
- `section` method on `Ellipsoid` to get a planar section as an `Ellipse`.
- `confidence` method on `StereoNet` to draw a Fisher, Bingham, Watson or bootstrap confidence cone/ellipse around orientation data.
- `csd` and `uniformity_test` methods on 2D and 3D feature sets; axial data (lines, poles) now uses Watson's test instead of a plain directional test.
- `random` method on `Vector3Set` for uniformly random orientations.

### Changed
- **BREAKING:** `Stress2`/`Stress3` now follow the geosciences/rock-mechanics sign convention (compression positive, tension negative).

### Removed
- `fisher_cone`/`fisher_cone_csd` (replaced by the new `confidence` method).

## [1.5.0] - 2026-07-15

### Added
- `StereoNet.point` method as a unified replacement for `line`/`pole`.
- `StereoNet.plot` method for plotting with reusable styles.
- Rotation classes derived from the deformation gradient class.
- `align` method on feature sets to find the best-fit rotation between two sets.
- Watson statistics.
- `method` keyword for choosing the contouring method.
- Similarity test for vector sets.
- Stress inversion method on fault sets.
- `angular_misfit` method on `Fault`/`FaultSet`.
- `strike` method on planar features.
- `from_declination` to correct for magnetic declination.
- `from_euler`, `from_quat`, `euler`, `quat` methods on deformation gradients.

### Changed
- Notation handling refactored, including quadrant notation.
- Pandas integration refactored.
- Spherical harmonics are now the default contouring method.
- Fisher statistics improved.
- Documentation updated.

### Fixed
- Rose diagram weighting.
- `scaled_eigenvectors`.

## [1.4.0] - 2026-02-20

### Added
- Custom attributes can now be attached to features.
- `add_vecs`, `add_fols`, `add_lins`, `add_faults` in the Pandas integration.
- `df` and `structdata` methods for reading SDB databases.

### Changed
- Global settings reorganized as a proper configuration object.

## [1.3.9] - 2026-02-05

### Fixed
- SDB database read/write bug.

## [1.3.8] - 2026-02-05

### Added
- Pair and fault retrieval for SDB databases.

### Changed
- SDB database read/write improved.

## [1.3.7] - 2025-12-08

### Added
- `transform` method on ellipse/ellipsoid sets.

### Fixed
- Text representation bug.

## [1.3.6] - 2025-11-13

### Changed
- Rose diagrams are now axial for directions and vectorial for vectors, as appropriate.

### Fixed
- Clustering bug.

## [1.3.5] - 2025-11-13

### Added
- `Direction` and `Direction2Set` classes.
- `from_ratio` method on stress tensors.
- Readable string representation for fault sense.

### Changed
- Clustering now uses different distance metrics for axial vs. vector data.
- Fault sense string representation is now used by default.

## [1.3.4] - 2025-09-24

### Changed
- `quicknet` now passes all keyword arguments through to the underlying `StereoNet` methods.

### Fixed
- Rounding bug.

## [1.3.3] - 2025-09-21

### Fixed
- Interactive shell (`iapsg`).
- SDB database tag reading.

## [1.3.2] - 2025-03-02

### Changed
- Build system switched to setuptools.

## [1.3.1] - 2025-02-28

### Changed
- SDB database interface updated.
- matplotlib 3.9 is now the minimum supported version.

### Fixed
- A plotting bug affecting collections.

## [1.3.0] - 2024-12-14

### Added
- `eigenlins`/`eigenfols` methods on 3D tensors.

### Changed
- Python 3.10 is now the minimum supported version.
- Pandas `.G` accessor now returns a proper apsg feature set.

## [1.2.3] - 2024-11-18

### Added
- `label` option for `quicknet`.

### Changed
- Clustering now accepts pair and fault sets.

### Fixed
- Vector exponentiation bug.

## [1.2.2] - 2024-10-21

### Changed
- Fault sense can now be given as a string (`'s'`, `'d'`, `'n'` or `'r'`).

## [1.2.1] - 2024-09-23

### Changed
- Fault sense can now be given as a string (`'s'`, `'d'`, `'n'` or `'r'`).

## [1.2.0] - 2024-05-24

### Changed
- SQLAlchemy and Pandas added as dependencies.

### Fixed
- A `quicknet` bug affecting faults.

## [1.1.5] - 2024-05-15

### Fixed
- Paleomagnetic core-orientation bug.
- Round-off error in angle calculations.

## [1.1.4] - 2023-12-13

### Fixed
- `Ellipsoid` text representation bug.

## [1.1.3] - 2023-10-23
Bugfix release.

### Added
- Slip and dilatation tendency methods on stress tensors.
- `proj` as a shorter alias for `project` on feature sets.

## [1.1.2] - 2023-10-09

### Added
- `title_kws` argument on plotting methods for custom title styling.

## [1.1.1] - 2023-10-06

### Fixed
- Contouring bug affecting the sigma estimate.

## [1.1.0] - 2023-10-04
APSG offers convenient Pandas integration via Pandas accessors. See documentation and the
Pandas interface tutorial for further details.

### Added
- `tensor` method on `StereoNet`.

### Changed
- `Cluster` class renamed to `ClusterSet`.

## [1.0.3] - 2023-04-30

### Added
- Cursor coordinates on a stereonet now show both line and plane orientation.

### Changed
- Lambda properties of tensors renamed to `S`.

## [1.0.1] - 2022-11-22

### Added
- `density_lookup` method on `StereoNet`'s grid.
- `render2fig` method on `StereoNet`.

### Changed
- Stress tensor sigma properties now use the standard (inverted) eigenvalue order.

### Fixed
- Vector-like objects are no longer iterable, so they now display correctly in Pandas.

## [1.0.0] - 2022-10-07
New major release. See documentation for further details.

### Changed
- **BREAKING:** significantly refactored from the 0.x series; the main namespace now provides
  short lowercase aliases (e.g. `lin`, `fol`) for the plain-English `PascalCase` class names
  used internally, for a simpler day-to-day interface.

## [0.7.3] - 2022-10-06

### Removed
- Figure window title from `StereoNet`.

## [0.7.2] - 2022-10-06

### Fixed
- General bug fixes.

## [0.7.1] - 2021-07-13

### Added
- Simple SQLAlchemy-based interface to SDB databases.
- `StereoNet.polygon` method.

### Changed
- Paleomagnetic RS3 file input/output improved.

### Fixed
- `StereoNet.arc` method.

## [0.7.0] - 2021-02-03

### Added
- `RosePlot` (rose diagram).
- `from_pairs` method on orientation tensors (Lisle tensor for orthogonal data).
- `labels` option on `StereoNet.scatter` for hover annotations.

### Removed
- Python 2 support.

## [0.6.3] - 2019-12-06

### Fixed
- Python 2/3 compatibility.

## [0.6.2] - 2019-12-06

### Added
- `Pair.H` method to get the mutual rotation between two pairs.
- `steps` option on `velgrad` to generate a series of deformation tensors.
- `Tensor` class for working with generic deformation tensors.

### Fixed
- Several minor bugs.
- `StereoGrid.apply_func` now passes a proper vector object instead of a raw array.

## [0.6.1] - 2018-12-12

### Added
- Natural principal strains (`e1`, `e2`, `e3`) on deformation gradients.
- Octahedral strains (`eoct`, `goct`) on deformation gradients.
- `from_ratios` class method on deformation gradients.
- Strain symmetry/intensity properties (`k`, `d`, `K`, `D`) on deformation gradients.
- `Ellipsoid` class for working with strain ellipsoids.
- `RamsayPlot`, `FlinnPlot` and `HsuPlot` fabric plots.
- `path` method on all fabric plots, accepting a list of tensors.

### Changed
- `StereoGrid` now always uses Euclidean norms as weights.
- `FabricPlot` renamed to `VollmerPlot`.

## [0.6.0] - 2018-11-07

### Added
- Stress invariants (`I1`, `I2`, `I3`).
- `mean_stress` property on stress tensors.
- `hydrostatic`/`deviatoric` properties on stress tensors.
- `precision` setting to control numerical comparisons.
- `figsize` setting to control figure size.
- `rand` class method to generate a random line/plane/vector/pair.
- SDB database metadata can now be modified.
- QGIS 3 plugin (ReadSDB) compatibility.

### Changed
- Stress eigenvalues are now always returned sorted.
- Group `to_csv`/`from_csv` improved.

### Fixed
- Animation examples.
- SDB `tags` method for multiple tags.

## [0.5.4] - 2018-10-19

### Added
- `cbpad` keyword on `StereoNet` for colorbar padding.

### Fixed
- `FabricPlot` bug introduced in 0.5.2.

## [0.5.3] - 2018-10-10

### Fixed
- General bug fixes.

## [0.5.2] - 2018-10-10

### Added
- Fisher distribution sampling.
- `norm` option on `transform` to normalize transformed vectors.
- `axisangle` property to compute axis and angle from a rotation matrix.
- `StereoNet.arc` method.
- `upper`/`flip` properties.
- `velgrad` method to compute the matrix logarithm of a deformation gradient.
- `max`, `min`, `max_at`, `min_at` methods on `StereoGrid`.

### Changed
- Rotate methods now also accept a rotation matrix directly.

## [0.5.1] - 2017-12-05

### Added
- Kent distribution sampling.
- Automatic kernel density estimate for contouring.

### Fixed
- Warnings cleanup.

## [0.5.0] - 2017-11-19

### Fixed
- Minor bugfix release.

## [0.4.4] - 2017-03-25

### Added
- `halfspace` method to reorient all vectors towards a common resultant halfspace.

### Changed
- `centered` method improved.

## [0.4.3] - 2017-03-25

### Added
- Stress tensor with basic methods.
- `weighted` keyword on `StereoGrid`.
- `StereoNet.tensor` method (draws eigen-lines or eigen-planes).
- `totvar`, `dot` and `proj` methods on feature groups.
- `Vec3.H` method to get the mutual rotation between two vectors.
- Simple animation support on `StereoNet` (`animate=True`).

### Changed
- `StereoNet` keyword arguments are now passed through for immediate plotting.
- `dot` method on planes/lines now returns the absolute value of the dot product.
- `StereoNet.contourf` now draws contour lines by default too (`clines` option).

### Fixed
- `centered` bug.

## [0.4.1-2] - 2017-03-04

### Fixed
- General bug fixes.

## [0.4.0] - 2017-03-04

### Added
- Angelier-Mechler dihedra method for fault sets.
- `StereoNet` accepts a `StereoGrid` or orientation tensor as a quick-plot argument.
- `StereoNet.axtitle` method to caption a plot.
- Fault set example.

### Changed
- `Density` class renamed to `StereoGrid`.

### Fixed
- Fault sense under rotation.

## [0.3.7] - 2017-01-05

### Added
- `MADp`, `MADo`, `MAD` and `kind` properties on orientation tensors.

### Changed
- Conda build available for all platforms.
- numpy, matplotlib and other common helpers imported by default.
- Orientation tensor is now normalized by default.

## [0.3.6] - 2017-01-03

### Added
- `iapsg` shell script to open an interactive console.

## [0.3.5] - 2016-11-12

### Added
- Simple settings interface.
- `notation` setting (`dd` or `rhr`) to control how azimuth is interpreted.
- `vec2dd` setting to control how vectors are displayed.
- Vectors can now be created from 1 (vector-like), 2 (azimuth, inclination) or 3 (azimuth, inclination, magnitude) arguments.
- Feature groups can now return an array or list of any user-defined attribute across all their elements.

## [0.3.4] - 2016-06-20

### Fixed
- Documentation build.

## [0.3.3] - 2016-06-04

### Added
- Principal strain properties and polar decomposition on deformation gradients.
- `StereoNet.vector` method to mimic lower/upper hemisphere plotting as used in paleomagnetic plots.
- Support for initializing `StereoNet` with subplots.
- `rake` method on planes.
- `apply_func` to compute density from a user-defined function, without needing input data upfront.
- Contour methods accept a density grid object directly.
- Uniform (Spherical Fibonacci / Golden Section) sampling for vectors, lines and planes.

## [0.3.2] - 2016-02-22

### Added
- Triangular fabric plot.
- `.V` as a shorter alias for `.asvec3`.
- `dv` property on planes to return the dip-slip vector.

### Changed
- Resultant of planes/lines is now calculated vectorially in the centered position.

## [0.3.1] - 2015-11-20

### Added
- Basic filtering by tags for SDB database support.
- `close` method on `StereoNet`, with the ability to re-initialize after closing in interactive mode.
- `iapsg` shell script to launch an apsg-aware IPython shell.

## [0.3.0] - 2015-11-09

### Added
- Fancy indexing -- a group can be indexed by a list/tuple/array of indices, or sliced.
- Hierarchical clustering.
- Groups can be saved to and loaded from file.
- Shallow-copy method for groups.
- `StereoNet` now also accepts vectors and faults for instant plotting.
- `E1`/`E2`/`E3` and Vollmer (1989) `P`/`G`/`R`/`C` indices on orientation tensors.
- `asvec3` method on planes and lines.
- `StereoNet` can plot planes as poles or great circles by default (`fol_plot`).
- `bootstrap` method for resampling with replacement.
- Built-in example datasets.

### Changed
- Uniform sampling of lines/planes improved.
- matplotlib deprecation warnings ignored by default.

### Fixed
- A bug in Woodcock's shape/strength values.

## [0.2.3] - 2015-10-21

### Changed
- Docstrings reformatted.
- Interactive-session shell scripts improved.

### Fixed
- A `StereoNet` data-retrieval bug.

## [0.2.2] - 2015-04-17

### Added
- `FaultSet` class; `Fault` and `Hoeppner` plotting methods on `StereoNet`.
- `VelGrad` and `DefGrad` classes for transformations.
- Quick group-creation helper.

## [0.2.1] - 2014-12-09

### Fixed
- A quick-plotting bug for groups.

## [0.2.0] - 2014-12-09

### Added
- `StereoNet` class for Schmidt projection plots.
- Quick plotting when data is passed directly to `StereoNet`.
- `Pair` and `Fault` classes for paired orientation data.
- `uniform_lin`/`uniform_fol` sampling methods.
- Absolute-value method to compute Euclidean norms for a group.
- Normalization method for groups.
- Spherical-statistics properties and methods for groups.

### Removed
- `mplstereonet` dependency.

## [0.1.0] - 2014-11-01

### Added
- First release of APSG.
