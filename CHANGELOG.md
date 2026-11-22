# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).
Note: apsg's version numbers do not strictly follow [Semantic Versioning](https://semver.org/) --
a patch-level release can include breaking changes, marked below as **BREAKING**.

## [2.1.0] - 2026-11-22

apsg 2.1 is a big step for orientation statistics. Fit, sample and compare more than a dozen
distributions, from Fisher and Kent to Bingham and Lisle, in one consistent way. Split axes into
fracture sets with Bingham mixtures, correct the orientation bias of boreholes and outcrops, and
export the parameters of every set for discrete fracture network models. Ask your data real
questions with a full set of hypothesis tests, each traced to its original source. Two new
subpackages bring paleomagnetism (`apsg.paleomag`: readers for six file formats, component analysis
including an automatic search, fold and reversal tests) and anisotropy of magnetic susceptibility
(`apsg.ams`). Statistics of lineations and foliations no longer depend on how the axes were stored.
Plots lay themselves out, legends go wherever you want, and bad input is caught at once instead of
crashing later. Large data sets are much faster, and the documentation has a fresh new home.

### Added

**Orientation distributions**
- One interface for fitting, sampling, density and model selection: uniform, von Mises, wrapped normal, wrapped Cauchy, cardioid, Jones-Pewsey, Fisher, Kent, ESAG, Watson, Bingham, ACG, Lisle (pairs) and `MatrixFisher` (rotations, with `RotationSet`). Use `linset.fit("kent")` or the aliases `dist_fisher`, `dist_kent`, ...
- Feature sets sample every distribution: `random(model=...)` and `random_<name>()` (e.g. `linset.random_watson()`), all with an `rng` keyword; `uniform_random()`, `uniform_sfs()` and `uniform_gss()` give uniform sets.
- `StereoNet.density()` and `RosePlot.density()` draw a fitted distribution; `StereoNet.confidence()` gains the methods `"kent"`, `"esag"`, `"lisle"` and `"bootstrap_ellipse"` (Constable & Tauxe 1990) and plots any `ConfidenceRegion`, e.g. `Kent.mean_confidence()`.
- `kappa_confidence()` and more fitting methods for concentrations (`"precision"`, `"corrected"`); `PairSet.lisle_statistics()` for the mean pair (Lisle 1989); `ACG.fit()` warns when the maximum-likelihood estimate does not exist.

**Fracture sets and sampling bias**
- `weights=` for `fit()` (Fisher, Watson, Bingham), `ortensor()` and the Fisher, Watson, Bingham and bootstrap statistics; `fit_info.n_eff` is the effective sample size.
- `terzaghi_weights()` corrects the orientation bias of boreholes, scanlines and outcrops.
- `BinghamMixture` and `Vector3Set.fit_mixture()` split axes into sets (number of components by BIC); `Vector3Set.cluster_bingham()` returns `BinghamClusters` with soft memberships, optional unassigned data, and `table()`, `to_dataframe()`, `to_csv()`, `to_json()` exporting the parameters of every set.

**Hypothesis tests** (all return a `TestResult` with statistic, p-value and `reject()`)
- Uniformity on the circle (Kuiper, Watson U², Hodges-Ajne, Rao) and of pairs (Arnold & Jupp), Fisher goodness of fit, Fisher against Kent, Watson against Bingham, likelihood ratio of nested distributions, common mean direction or axis, common concentration. `FisherQQPlot` shows Q-Q plots against the Fisher distribution.
- Docstrings of the statistics cite the primary sources with equation numbers.

**Circular and tensor statistics**
- `Vector2Set.median()`, `moment()`, `skewness()`, `kurtosis()`, `circular_dispersion()`, `angular_deviation()` and `bootstrap_statistics()`; `weighted=` for magnitude-weighted means; `RosePlot` accepts single `Vector2`/`Direction` features, `RosePlot.muci(method="bootstrap")` and `RosePlot(angles="math")`.
- `mean_tensor()` gives confidence intervals of the eigenvalues, Jelínek's (1978) anisotropy test and shape parameters (`jelinek_shape()`).

**Paleomagnetism (`apsg.paleomag`, import it explicitly)**
- `Specimen` and `Sample` read and write demagnetization data and fit components by PCA of lines and planes (MAD, DANG). Samples give Fisher means or combined line and great-circle means (McFadden & McElhinny 1988, Kirschvink, Bailey & Halls), bootstrap and McFadden fold tests, reversal tests (Watson F or bootstrap) and virtual geomagnetic poles.
- Formats: PMD (standard and Remasoft dialects, found automatically), Remasoft RS3, SQUID, AGICO JR6, Caltech CIT and 2G Enterprises (also continuous long cores, ARM and IRM steps). `Sample.from_file()`, `from_files()` and `from_directory(errors="warn")` read files with several specimens and whole folders; the format comes from the extension or the content.
- AGICO orientation codes P1-P4 of 3, 6, 9 or 12 (P2 0 or 90) are understood in RS3, JR6 and SUFAR files.
- Remasoft interpretations round-trip: `from_rs3(components=True)` imports the component and great-circle rows, `to_rs3()` writes them back, including the four fit options (free, `pca(origin=True)`, the new `pca(anchor="last")` and `pca(anchor=True)`).
- **Automatic components:** `find_component()`, `find_components()` and `interpret_auto()` pick the best linear segments of a Zijderveld path, `window_fits()` fits every window of steps, `ComponentCriteria` with `Sample.grade()` filters by MAD, DANG and number of steps, and `Sample.stack()` and `Sample.interpret()` help with whole sites. The search was tuned on 800 Remasoft picks of 439 specimens and is a heuristic to check on the Zijderveld diagram (`ZijderveldPlot(components=True)`, `ComponentSearchPlot`).
- Inclination-only means (McFadden & Reid, Arason & Levi), relative paleointensity (`Sample.rpi()`), `susceptibility_jump()`, `align_core_sections()` and `Sample.by_site()`.
- Plots share one interface (`PaleomagPlot`, configured by `apsg_conf.paleomag`): `ZijderveldPlot`, `DemagPlot`, `SpecimenStereoPlot`, `SpecimenPlot`, `KappaTPlot`, `FoldTestPlot`, `TiltComparisonPlot`, `ReversalTestPlot`, `InclinationOnlyPlot`, `LinesPlanesPlot`.

**Anisotropy of magnetic susceptibility (`apsg.ams`)**
- `AMSSpecimen` and `AMSSite` with core orientation, bedding, geographic and tectonic coordinates, Jelinek parameters, mean tensors with confidence regions and CSV input and output.
- Readers for AGICO instrument files: SUFAR, SAFYR and SUSAR reports (`read_asc()`, formerly `read_sufar()`), Kappabridge `.RAN` files (`read_ran()`), Anisoft `.AEX` and text tables (`read_aex()`, `read_anisoft_table()`); `read_ams()` picks the reader.
- Bootstrap of mean tensors and of the measurement error of a specimen, and a test of equal mean tensors of two samples (Hotelling or permutation).
- Plots `PjTPlot`, `LFPlot`, `PTPlot`, `AMSStereoPlot` and `BootstrapPlot` (`apsg_conf.ams`).
- Flow models: `AMSModel`, `GrainPopulation`, `jeffery_evolution()`, `orientation_tensor_from_ams()`, `ModelPlot`, and the grain models behind them (`VelocityGradient3.pseudo()`, `ACG.from_displacement()`, `ams_from_orientation()`, `ams_from_grains()`).

**Data and documentation**
- `apsg.data` provides example datasets without paths: `data.load("faults")`, `data.path("structures")`, `data.catalog()`. It holds the tutorial data, short published sets with citations (Fisher 1953, Fisher, Lewis & Embleton, Tauxe, Fisher 1993) and seeded synthetic samples, plus AMS data (`melechov`, `trebic`, `ams_two_sites`, ...) and `rs3_components` (six specimens with Remasoft picks).
- New tutorials on orientation distributions, paleomagnetism (artificial data) and AMS; the tutorials are renumbered, the various tutorial is now the last one.

### Changed
- **Plots lay themselves out:** `StereoNet`, `RosePlot` and the fabric plots use constrained layout and size the figure to the content (new `figsize` keyword; the figure grows by legend and title). `legend_position` (`"right"`, `"left"`, `"top"`, `"bottom"`) puts the legend on any side, its columns follow the plot size, and the `contour()` colorbar is a small inset (`colorbar_bounds`, `colorbar_fontsize`).
- **Much faster:** `Bingham` and `Lisle` fits are 10 to 100 times faster (`Lisle.fit` of 300 pairs: 5 ms instead of 0.7 s); pairwise `angle()`, `cross()` and clustering are vectorized (2000 items: 0.1 s instead of about 50 s); `random_kent()` is faster on concentrated data.
- Invalid arguments raise `TypeError`/`ValueError` or warn instead of failing with `AssertionError` or printing; files with unreadable rows (PMD, RS3, Anisoft tables) warn instead of dropping them silently.
- Statistics of lineation and foliation sets no longer depend on the stored signs of the axes, and `StereoNet.confidence()` draws the Watson cone for them by default.
- `RosePlot.muci()` draws the von Mises interval of `fisher_statistics()` by default, takes the level as a fraction (`level=0.95`), weights mean and interval by magnitude like the bars, and warns instead of failing for too few data.
- `from_csv()`/`to_csv()` honour `delimiter`; `SDBSession.getpairs()` raises `ValueError` for a wrong structure type; the documentation moved to MkDocs with the Material theme.
- **BREAKING:** `tight_layout` is removed from `StereoNet`, `RosePlot` and the fabric plots, which no longer use `apsg_conf.figsize`.
- **BREAKING:** `random()` of feature sets samples the uniform distribution models, so a seed gives other samples than before; `PairSet.random()` and `FaultSet.random()` create 100 items by default. `random_kent()` uses `beta = kappa / 4`.
- **BREAKING:** `fisher_statistics()["k"]` of 3D sets is Fisher's precision `(N - 1) / (N - R)`; the maximum-likelihood estimate is now `["kappa"]`. `csd()` uses the precision.
- **BREAKING:** `uniformity_test()` returns a `TestResult` (the old dict keys still work with a `FutureWarning`), and for lineations and foliations it is the Bingham test; the former test was not valid.
- **BREAKING:** The paleomagnetic `Core` class is removed from `apsg.feature` in favour of `Specimen` and `Sample`; its `pca()` returned wrong directions and MAD. The functions `zijderveld_plot`, `demag_plot` and `stereo_plot` are replaced by the plot classes.
- **BREAKING:** `SDBSession.getfaults()` is removed (the SDB format stores no sense of movement); use `getpairs()` and build the faults, e.g. `FaultSet([Fault(p, 1) for p in pairs])`.
- **BREAKING:** `Cone.apical_angle` is a property, `PairSet.from_csv()` no longer accepts `sense_str` and `scol`.

### Deprecated
- `fisher_statistics()`, `var()`, `delta()`, `rdegree()` and `csd()` of lineation and foliation sets; use `watson_statistics()`, `bingham_statistics()` or the distributions.
- `random_normal()` (use `random_fisher()` or `random_kent()`), `sfs()` and `gss()` (use `uniform_sfs()` and `uniform_gss()`), the `position` keyword of `random_fisher()`/`random_vonmises()` and `p` of `random_kent()` (use `mean`), and `confidence_level` of `RosePlot.muci()` (use `level`).

### Fixed
- **BREAKING:** wrong results corrected: the Bingham confidence ellipse (too large, wrongly oriented), `FaultSet.stress_inversion()`, circular statistics of `Vector2Set` and `Direction2Set`, the confidence cone of `watson_statistics()` (now Fisher, Lewis & Embleton 1987, with nominal coverage), `OrientationTensor3.strength` (Woodcock's `ln(S1/S3)`; was half of it), and `watson_statistics()` now recognizes girdles. The `Lisle` normalizer was inaccurate for `kappa1` near 0 with `kappa2` of 1000 or more, so such fits change; Lisle pairs with `kappa2=0` were not perpendicular.
- **BREAKING:** input is validated instead of stored: features need finite real components and angles (plunge in [-90, 90], dip in [0, 180], cone angle in [0, 180]), lineations, foliations and directions cannot be zero vectors, pairs and faults need non-parallel vectors, and an improper `Rotation2`/`Rotation3` (a reflection) raises `ValueError`.
- **BREAKING:** configuration values are validated when set (also by `update()` and `apsg_conf_context()`), unknown keys can't be set, and plot parameters fail at the call with `ValueError` instead of at render. Fitting a distribution or `fisher`/`watson`/`bingham` statistics of degenerate data (fewer than 2 items, identical, or on one great circle) raises `ValueError` instead of returning an infinite concentration.
- Sampling no longer hangs with an infinite concentration; `rvs(0)` returns an empty set. Statistics, plots and densities of empty sets raise a clear `ValueError` or draw nothing instead of failing obscurely. `from_csv()` and `feature_from_json()` report malformed input with `ValueError`/`TypeError` (naming file and line for CSV), and `from_array()` and pairwise operations of sets of different lengths raise instead of dropping items.
- Printing never fails (degenerate 2D tensors, huge matrix entries); Kamb contouring of 1,000,000 points needs about 0.6 GB instead of 22 GiB.
- `RosePlot`: `pdf()` of `Vector2Set` is weighted like the bars, `bar()`/`pdf()` honour `color`, and the `muci()` level for axial data and saving diagrams work.
- Mean direction of identical vectors, comparison of pairs and faults with reversed normals, shape parameters of tensors with a zero eigenvalue, `from apsg.helpers import *`, and several smaller bugs in `G()`, `Fault()`, `FaultSet.from_csv()` and `SDBSession`.

## [2.0.6] - 2026-10-16

### Changed
- **BREAKING:** `Stress3.shape_ratio` renamed to `stress_ratio`, now R = (σ2 - σ3) / (σ1 - σ3).

## [2.0.5] - 2026-10-07

### Changed
- `openpyxl` is now installed with the `lab` extra.

### Fixed
- Printed features no longer show `-0` or `360`.

## [2.0.4] - 2026-10-03

### Changed
- `defgrad()` of ellipses/ellipsoids returns the coaxial (symmetric) stretch by default.

### Fixed
- Product of a rotation with a non-orthogonal matrix.
- Fabric plots created one legend entry per member of an `EllipsoidSet`.
- `StereoNet.tensor()` accepts sets of tensors.

## [2.0.3] - 2026-10-03

### Added
- `apsg_conf.stress_convention` to select geological or mechanical sign convention of stress.
- `Ellipse.from_ratio()` and `Ellipsoid.from_ratios()`.
- `defgrad()` method on ellipses/ellipsoids.
- `form` option for `from_ellipse()`/`from_ellipsoid()`.
- `EllipseSet.rfphi()` for Rf/phi strain analysis.
- `robin()` mean strain of ellipse/ellipsoid sets (Robin 1977).

### Changed
- **BREAKING:** `DeformationGradient3.from_ratios()` requires two ratios, `DeformationGradient2.from_ratio()` requires `R >= 1`.
- **BREAKING:** Stress tensors are stored in the mechanical sign convention.
- **BREAKING:** `cauchy()` returns the physical (tension-positive) traction.

### Fixed
- `Stress2.signed_shear_stress()`.

## [2.0.2] - 2026-09-30

### Added
- `StereoNet.hoeppner()` accepts pairs.
- `name` option for `from_csv()`.

### Changed
- Default `cmap` of unclipped contours is `"RdBu_r"`.

### Fixed
- `grid=False` was ignored when a grid style option was set.

## [2.0.1] - 2026-09-25

### Added
- `StereoNet.dihedra()` for faults.
- `StereoNet.beachball()` for stress tensors.
- Filled arcs.

### Fixed
- `savefig()` left the figure open.

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
