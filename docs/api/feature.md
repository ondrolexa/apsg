# apsg.feature

The `apsg.feature` module provides classes to store and analyze structural geology data. It includes
representations of planar and linear features (`Foliation`, `Lineation`, `Fault`, `Cone`, `Arc`, `Pair`),
their corresponding sets (`FoliationSet`, `LineationSet`, etc.), tensor classes (`Stress3`,
`Ellipsoid`, `OrientationTensor3`), and higher-level analysis tools like `ClusterSet` for
clustering of orientation data.

The main APSG namespace provides lowercase aliases for commonly used classes (e.g. `fol` for
`Foliation`, `lin` for `Lineation`, `vec` for `Vector3`).
See the [alias table](index.md#aliases) for the full list.

## Usage

Create individual features:

```pycon
>>> from apsg import fol, lin, fault, pair, cone, arc, vec
>>> f = fol(120, 30)                 # dip direction=120, dip=30
>>> l = lin(210, 45)                 # trend=210, plunge=45
>>> f2 = fault(120, 30, 80, 20, 1)   # dip direction, dip, trend, plunge, sense
>>> p = pair(300, 20, 200, 60)       # dip direction, dip, trend, plunge
>>> c = cone(vec(45, 30), 20)        # axis + half-angle
>>> a = arc(lin(0, 0), lin(90, 0))   # curved path between two vectors
```

Create sets of features for batch analysis:

```pycon
>>> from apsg import folset, linset, vecset
>>> fols = folset.random_fisher(position=f, kappa=50, n=30)
>>> lins = linset.random_fisher(position=l, kappa=100, n=30)
>>> vecs = vecset.gss(n=50)
```

Analyze orientation data:

```pycon
>>> ot = fols.ortensor()
>>> ot.eigenvalues()
>>> ot.eigenvectors()
>>> fols.watson_statistics()           # mean axis, concentration, confidence cone
>>> vecs.fisher_statistics()           # for vectors; deprecated for axes
```

Fit, compare and sample probability distributions of orientations (Fisher, Kent,
ESAG, Watson, Bingham, ACG, von Mises, ...), see [Orientation distributions](../distributions.md):

```pycon
>>> from apsg import dist_kent
>>> d = dist_kent.fit(lins)            # or lins.fit("kent")
>>> d.bic(lins), d.mean_confidence()
>>> d.rvs(100, rng=1)                  # random sample, a Vector3Set
>>> fols.fit("watson")                 # bipolar (kappa > 0) or girdle (kappa < 0)
```

Cluster analysis:

```pycon
>>> from apsg import cluster
>>> cs = cluster(fols)
>>> cs.cluster(k=3)
```

Tensor analysis:

```pycon
>>> from apsg import ellipsoid, stress, ortensor
>>> e = ellipsoid.from_stretch(2, 1, 0.5)
>>> s = stress.from_ratio(r=0.5, mag=1)
>>> ot = ortensor.from_features(fols)
```

Stress ratio: `Stress3.stress_ratio` (and `Stress3Set.stress_ratio`) uses the structural
geology definition R = (σ2 - σ3) / (σ1 - σ3), so R = 0 for σ2 = σ3 and R = 1 for σ2 = σ1. The
seismological definition (σ1 - σ2) / (σ1 - σ3) equals `1 - S.stress_ratio`.
`stress.from_ratio(r, mag)` builds a tensor with `stress_ratio == r`.

Sign convention of stress tensors: `apsg_conf.stress_convention` selects `"geological"`
(default, compression positive) or `"mechanical"` (tension positive). The constructors and
`from_*` methods take input in the active convention, and reported values (principal
stresses, invariants, `normal_stress()`, eigenvalues, `repr`) follow it. The internal
matrix, returned by `np.asarray()`, is always mechanical, so `np.asarray(S)` is the negative
of the geological matrix. `cauchy()` and `stress_comp()` return the physical traction
`σ·n` in both conventions, and `shear_stress()`, `slip_tendency()` and `fault()` do not
depend on the convention:

```pycon
>>> from apsg import fol
>>> from apsg.config import apsg_conf_context
>>> S = stress.from_comp(xx=1, yy=1, zz=10)   # vertical compression 10
>>> S.normal_stress(fol(90, 60))              # +3.25, compression positive
>>> S.cauchy(fol(90, 60))                     # physical traction, tension positive
>>> with apsg_conf_context(stress_convention="mechanical"):
...     M = stress.from_comp(xx=-1, yy=-1, zz=-10)  # same state, tension positive
...     M.normal_stress(fol(90, 60))                # -3.25
```

Mean tensor of a group of tensors (`EllipsoidSet` or `Stress3Set`) with confidence ellipses
of its principal axes, following the linear perturbation method of Jelínek (1978):

```pycon
>>> from apsg import ellipsoidset
>>> es = ellipsoidset([ellipsoid.from_stretch(2 + 0.1 * i, 1, 0.5) for i in range(10)])
>>> res = es.mean_tensor()                     # normalize=True divides tensors by trace/3
>>> res["mean"], res["eigenvalues"]
>>> res["ellipses"][0]["mu"]                   # principal axis of the mean tensor
>>> res["ellipses"][0]["gamma"]                # semi-angles of its confidence ellipse (deg)
>>> es.mean_tensor(anisoft=True)               # extra (n-1)/n factor as in jelinekstat/TomoFab
>>> res["intervals"], res["anisotropy_test"]   # eigenvalue intervals, Jelinek's T² test
>>> res["shape"]                               # Jelinek (1981) P, L, F, Pj, T, U
```

Hypothesis tests return a `TestResult` (see [Orientation distributions](../distributions.md#hypothesis-tests)):

```pycon
>>> fols.uniformity_test().reject()           # Bingham test for axes
>>> vecs.uniformity_test().p_value            # Rayleigh test for vectors
>>> lins.common_mean_test(other_lins)         # common principal axis
```

Build a strain ellipse/ellipsoid directly from axial ratios, and recover the deformation
gradient that produced it (the inverse of `from_defgrad()`):

```pycon
>>> E = ellipsoid.from_ratios(Rxy=2, Ryz=3)
>>> F = E.defgrad()                            # F @ F.T == E (up to a rotational ambiguity)
```

Strain analysis of a population of deformed markers -- the Rf/phi vector-mean method (2D only)
and Robin's (1977) log-matrix mean (2D and 3D):

```pycon
>>> from apsg import ellipseset
>>> es = ellipseset([ellipse.from_ratio(R) for R in [1.8, 2.0, 2.2]])
>>> es.rfphi()                                 # strain ratio and orientation
>>> es.robin()                                 # log-Euclidean mean ellipse
>>> ellipsoidset([ellipsoid.from_ratios(Rxy=2, Ryz=r) for r in [2.8, 3.0, 3.2]]).robin()
```

Serialization:

```pycon
>>> from apsg.feature import feature_from_json
>>> data = f.to_json()
>>> f2 = feature_from_json(data)
>>> fols.to_csv('foliations.csv')
```

## Reference

### Features

Single geological features. Vector and axial base classes are in [`apsg.math`](math.md).

::: apsg.feature.Direction

::: apsg.feature.Lineation

::: apsg.feature.Foliation

::: apsg.feature.Pair

::: apsg.feature.Fault

::: apsg.feature.Cone

::: apsg.feature.Arc

### Feature sets

::: apsg.feature.FeatureSet

::: apsg.feature.Vector2Set

::: apsg.feature.Direction2Set

::: apsg.feature.Vector3Set

::: apsg.feature.LineationSet

::: apsg.feature.FoliationSet

::: apsg.feature.PairSet

::: apsg.feature.FaultSet

::: apsg.feature.ConeSet

::: apsg.feature.ArcSet

::: apsg.feature.EllipseSet

::: apsg.feature.OrientationTensor2Set

::: apsg.feature.Stress2Set

::: apsg.feature.EllipsoidSet

::: apsg.feature.OrientationTensor3Set

::: apsg.feature.Stress3Set

::: apsg.feature.RotationSet

::: apsg.feature.ClusterSet

::: apsg.feature.G

### 2D tensors

::: apsg.feature.DeformationGradient2

::: apsg.feature.Rotation2

::: apsg.feature.VelocityGradient2

::: apsg.feature.Stress2

::: apsg.feature.Ellipse

::: apsg.feature.OrientationTensor2

### 3D tensors

::: apsg.feature.DeformationGradient3

::: apsg.feature.Rotation3

::: apsg.feature.VelocityGradient3

::: apsg.feature.Stress3

::: apsg.feature.Ellipsoid

::: apsg.feature.OrientationTensor3

### Orientation distributions

See [Orientation distributions](../distributions.md) for an overview.

::: apsg.feature.OrientationDistribution

::: apsg.feature.Uniform2

::: apsg.feature.VonMises

::: apsg.feature.WrappedNormal

::: apsg.feature.AxialUniform2

::: apsg.feature.AxialVonMises

::: apsg.feature.Uniform3

::: apsg.feature.VonMisesFisher

::: apsg.feature.Kent

::: apsg.feature.ESAG

::: apsg.feature.AxialUniform3

::: apsg.feature.Watson

::: apsg.feature.Bingham

::: apsg.feature.ACG

::: apsg.feature.WrappedCauchy

::: apsg.feature.Cardioid

::: apsg.feature.JonesPewsey

::: apsg.feature.Lisle

::: apsg.feature.MatrixFisher

### Hypothesis tests

::: apsg.feature.TestResult

::: apsg.feature.rayleigh_test

::: apsg.feature.bingham_test

::: apsg.feature.kuiper_test

::: apsg.feature.watson_u2_test

::: apsg.feature.hodges_ajne_test

::: apsg.feature.rao_spacing_test

::: apsg.feature.frame_uniformity_test

::: apsg.feature.fisher_goodness_of_fit

::: apsg.feature.fisher_kent_test

::: apsg.feature.watson_symmetry_test

::: apsg.feature.likelihood_ratio_test

::: apsg.feature.common_mean_test

::: apsg.feature.axial_common_mean_test

::: apsg.feature.equal_kappa_test

::: apsg.feature.anisotropy_test

::: apsg.feature.jelinek_shape

### Paleomagnetism

::: apsg.feature.Core
