# API reference

APSG is organized in layered subpackages. Most user code imports everything from the
top-level namespace with `from apsg import *`, which re-exports the public classes
listed on the module pages below under their own names and under short aliases.

| Module | Contents |
|----|----|
| [`apsg.feature`](feature.md) | geological features, feature sets, tensors, orientation distributions |
| [`apsg.plotting`](plotting.md) | `StereoNet`, `StereoGrid`, `RosePlot` and fabric plots |
| [`apsg.pandas`](pandas.md) | pandas extension types and accessors |
| [`apsg.database`](database.md) | PySDB `.sdb` files and websdb projects |
| [`apsg.math`](math.md) | vector and matrix base classes |
| [`apsg.config`](config.md) | global configuration `apsg_conf` |
| [`apsg.helpers`](helpers.md) | degree-based trigonometry and notation conversion |

## Aliases

Short aliases available after `from apsg import *`:

### Vectors and features

| Alias | Class |
|----|----|
| `arc` | [`Arc`][apsg.feature.Arc] |
| `cone` | [`Cone`][apsg.feature.Cone] |
| `dir2` | [`Direction`][apsg.feature.Direction] |
| `fault` | [`Fault`][apsg.feature.Fault] |
| `fol` | [`Foliation`][apsg.feature.Foliation] |
| `lin` | [`Lineation`][apsg.feature.Lineation] |
| `pair` | [`Pair`][apsg.feature.Pair] |
| `vec2` | [`Vector2`][apsg.math.Vector2] |
| `vec` | [`Vector3`][apsg.math.Vector3] |

### Feature sets

| Alias | Class |
|----|----|
| `arcset` | [`ArcSet`][apsg.feature.ArcSet] |
| `cluster` | [`ClusterSet`][apsg.feature.ClusterSet] |
| `coneset` | [`ConeSet`][apsg.feature.ConeSet] |
| `dir2set` | [`Direction2Set`][apsg.feature.Direction2Set] |
| `ellipseset` | [`EllipseSet`][apsg.feature.EllipseSet] |
| `ellipsoidset` | [`EllipsoidSet`][apsg.feature.EllipsoidSet] |
| `faultset` | [`FaultSet`][apsg.feature.FaultSet] |
| `folset` | [`FoliationSet`][apsg.feature.FoliationSet] |
| `linset` | [`LineationSet`][apsg.feature.LineationSet] |
| `ortensor2set` | [`OrientationTensor2Set`][apsg.feature.OrientationTensor2Set] |
| `ortensorset` | [`OrientationTensor3Set`][apsg.feature.OrientationTensor3Set] |
| `pairset` | [`PairSet`][apsg.feature.PairSet] |
| `stress2set` | [`Stress2Set`][apsg.feature.Stress2Set] |
| `stressset` | [`Stress3Set`][apsg.feature.Stress3Set] |
| `vec2set` | [`Vector2Set`][apsg.feature.Vector2Set] |
| `vecset` | [`Vector3Set`][apsg.feature.Vector3Set] |

### Tensors

| Alias | Class |
|----|----|
| `defgrad2` | [`DeformationGradient2`][apsg.feature.DeformationGradient2] |
| `defgrad` | [`DeformationGradient3`][apsg.feature.DeformationGradient3] |
| `ellipse` | [`Ellipse`][apsg.feature.Ellipse] |
| `ellipsoid` | [`Ellipsoid`][apsg.feature.Ellipsoid] |
| `ortensor2` | [`OrientationTensor2`][apsg.feature.OrientationTensor2] |
| `ortensor` | [`OrientationTensor3`][apsg.feature.OrientationTensor3] |
| `rotation2` | [`Rotation2`][apsg.feature.Rotation2] |
| `rotation` | [`Rotation3`][apsg.feature.Rotation3] |
| `stress2` | [`Stress2`][apsg.feature.Stress2] |
| `stress` | [`Stress3`][apsg.feature.Stress3] |
| `velgrad2` | [`VelocityGradient2`][apsg.feature.VelocityGradient2] |
| `velgrad` | [`VelocityGradient3`][apsg.feature.VelocityGradient3] |
| `matrix2` | [`Matrix2`][apsg.math.Matrix2] |
| `matrix` | [`Matrix3`][apsg.math.Matrix3] |

### Distributions

| Alias | Class |
|----|----|
| `dist_acg` | [`ACG`][apsg.feature.ACG] |
| `dist_esag` | [`ESAG`][apsg.feature.ESAG] |
| `dist_axial_vonmises` | [`AxialVonMises`][apsg.feature.AxialVonMises] |
| `dist_bingham` | [`Bingham`][apsg.feature.Bingham] |
| `dist_kent` | [`Kent`][apsg.feature.Kent] |
| `dist_lisle` | [`Lisle`][apsg.feature.Lisle] |
| `dist_vonmises` | [`VonMises`][apsg.feature.VonMises] |
| `dist_fisher` | [`VonMisesFisher`][apsg.feature.VonMisesFisher] |
| `dist_watson` | [`Watson`][apsg.feature.Watson] |
| `dist_wrapped_normal` | [`WrappedNormal`][apsg.feature.WrappedNormal] |

### Plotting styles

| Alias | Class |
|----|----|
| `fabricplot_styles` | [`FabricPlotStyleFactory`][apsg.plotting.FabricPlotStyleFactory] |
| `roseplot_styles` | [`RosePlotStyleFactory`][apsg.plotting.RosePlotStyleFactory] |
| `stereonet_styles` | [`StereoNetStyleFactory`][apsg.plotting.StereoNetStyleFactory] |
