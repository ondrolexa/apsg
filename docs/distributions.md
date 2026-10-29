# Orientation distributions

APSG provides probability distributions for orientation data on the circle (2D
directions) and on the sphere (lines, poles and vectors). All of them share one
interface: they evaluate their density, generate random samples, are fitted to
data, give the confidence region of the mean and can be compared by information
criteria. They live in `apsg.feature` and have aliases with the prefix
`dist_` in the main `apsg` namespace (`dist_kent` for `Kent`, ...). See the
[distributions tutorial](notebooks/04_apsg_distributions.ipynb) for a worked example.

## Overview

The distributions are separated by the dimension of the data and by their
symmetry. *Vectorial* (directional) data distinguish a vector from its antipode,
e.g. paleomagnetic vectors or slip directions with sense. *Axial* data do not:
a lineation plunging 30° to 120° is the same line as one plunging -30° to 300°,
and a foliation is represented by its pole, which has no sense either.

|  | vectorial | axial |
|----|----|----|
| circle (`Vector2`, `Direction`) | `Uniform2`, `VonMises` (`vonmises`), `WrappedNormal` (`wrapped_normal`), `WrappedCauchy` (`wrapped_cauchy`), `Cardioid` (`cardioid`), `JonesPewsey` (`jones_pewsey`) | `AxialUniform2`, `AxialVonMises` (`axial_vonmises`) |
| sphere (`Vector3`, `Lineation`, `Foliation`) | `Uniform3`, `VonMisesFisher` (`fisher`), `Kent` (`kent`), `ESAG` (`esag`) | `AxialUniform3`, `Watson` (`watson`), `Bingham` (`bingham`), `ACG` (`acg`) |
| pairs (`Pair`, `Fault`) | | `Lisle` (`lisle`) |
| rotations (`Rotation3`) | `MatrixFisher` (`matrix_fisher`) | |

In parentheses are the short names accepted by `FeatureSet.fit()`, e.g.
`lineations.fit("kent")`; the `apsg` aliases add the prefix `dist_`.

The sources of the models (abbreviations as in the [references](#references)):

| model | density | estimation | sampling |
|----|----|----|----|
| `VonMises` | MJ (3.5.17) | MLE MJ (5.3.2); Best & Fisher approximation F93 (4.40); small-sample correction F93 (4.41) | Best & Fisher (1979), via scipy |
| `WrappedNormal` | MJ (3.5.63)-(3.5.66) | moments, numerical MLE | wrapped normal deviates |
| `WrappedCauchy` | MJ (3.5.68) | MLE by Kent & Tyler (1988), MJ (5.4.1)-(5.4.6) | wrapped Cauchy deviates |
| `Cardioid` | MJ (3.5.47) | moments, numerical MLE | rejection |
| `JonesPewsey` | Jones & Pewsey (2005); L&V (2.5) | numerical MLE | rejection |
| `AxialVonMises` | MJ (3.6.4) | as `VonMises`, doubled angles (F93 (2.36)) | as `VonMises` |
| `VonMisesFisher` | Fisher (1953); MJ (9.3.4) | MLE MJ (10.3.4); precision FLE (5.25); corrected FLE (5.26) | Wood (1994), via scipy |
| `Kent` | Kent (1982) (1.1) | moments Kent (1982) sec. 4, (4.9); numerical MLE | rejection |
| `ESAG` | Paine et al. (2018) (7) | numerical MLE | normal deviates |
| `Watson` | MJ (9.4.1) | MLE MJ (10.3.31)-(10.3.33) | as `Bingham` |
| `Bingham` | Bingham (1974); MJ (9.4.3) | MLE Bingham (1974) (6.3) | ACG envelope, Kent et al. (2013) (3.4)-(3.6) |
| `ACG` | Tyler (1987) (1) | fixed point Tyler (1987) (3) | normal deviates |
| `Lisle` | frame Bingham, Arnold & Jupp (2013) (7), with Lisle's (1989) tensor | numerical MLE | rejection |
| `MatrixFisher` | Kent et al. (2013) (4.2) | MLE, MJ (13.2.16) for the mode | Bingham on quaternions, Kent et al. (2013) |

## Choosing a model

On the sphere:

- `VonMisesFisher` (Fisher distribution): directions clustered with circular
  contours about the mean, one concentration `kappa`.
- `Kent`: directions clustered with elliptical contours, e.g. lineations
  spread along a foliation. `beta` controls the ellipticity (`beta = 0` is
  the Fisher distribution); it is unimodal for `2 beta < kappa`.
- `ESAG` (elliptically symmetric angular Gaussian): an alternative to Kent
  with a closed-form density, the direction of a normally distributed vector.
- `Watson`: axes with circular contours, either clustered about the mean axis
  (`kappa > 0`, bipolar) or spread in a girdle about the mean axis as its pole
  (`kappa < 0`).
- `Bingham`: axes with elliptical contours, covering clusters, girdles and
  everything between. `kappa_minor <= kappa_major <= 0` are the
  concentrations along the minor and major axis; a girdle has `kappa_major`
  near 0, with the minor axis as its pole.
- `ACG` (angular central Gaussian): the axial counterpart of ESAG, the axis of
  a normally distributed vector with zero mean.

On the circle:

- `VonMises`: directions (azimuths with sense).
- `AxialVonMises`: axes such as strikes or trends of lines, the von Mises
  distribution of the doubled angles; its `kappa` refers to the doubled
  angles.
- `WrappedNormal`: the normal distribution wrapped around the circle, an
  alternative to the von Mises distribution.

How the models relate: Fisher is Kent with `beta = 0` and the isotropic case of
ESAG, Watson is Bingham with equal concentrations along the major and minor axis,
and the axial von Mises distribution is the Bingham distribution on the circle.

## Common interface

Create a distribution from its parameters, sample it, and fit it back:

```pycon
>>> from apsg import *
>>> true = dist_kent(lin(120, 40), lin(30, 0), kappa=40, beta=15)
>>> l = true.rvs(150, rng=7, feature=lin)      # LineationSet
>>> d = dist_kent.fit(l)                       # or l.fit("kent")
>>> d.params
{'mean': ..., 'major': ..., 'kappa': 35.75..., 'beta': 12.66...}
```

- `pdf(x)`, `logpdf(x)` and `loglik(data)` accept a feature, a feature set
  or an array of vectors. Densities are taken with respect to arc length on the
  circle and area on the sphere, over the whole circle or sphere also for axial
  models, so the uniform density is `1/(2 pi)` or `1/(4 pi)`.
- `rvs(n, rng=None, feature=None)` returns a feature set. `rng` is an
  integer seed or a `numpy.random.Generator`; without it, the seed is drawn
  from numpy's global state, so `np.random.seed()` makes samples
  reproducible. `feature` selects the feature class, e.g. `feature=fol` for
  a `FoliationSet`. Axial samples get a random sign.
- `fit(data, method=...)` is a class method returning the fitted
  distribution. Its `fit_info` holds the sample size, the log-likelihood,
  whether the solver converged and the statistics used. The methods are
  maximum likelihood (`"mle"`, the default) and, where available, moment
  estimates (`"moments"` for `Kent` and `WrappedNormal`) or the
  approximation used by `fisher_statistics` (`"approx"` for the von Mises
  distributions). `FeatureSet.fit(name)` does the same, finding the model by
  its class, class name or short name.
- `mean`, `params`: the mean direction or axis and all parameters, which
  recreate the distribution as `type(d)(**d.params)`.
- `mean_confidence(level=0.95)` of a fitted distribution returns the
  confidence region of the mean, a cone (`alpha`) or an ellipse (`axes` and
  `gamma`); `as_dict()` gives the layout of the `*_statistics` methods.
  `kappa_confidence(level=0.95)` gives the interval of the concentration
  (`VonMisesFisher`, FLE (5.37); `VonMises`, F93 (4.48)-(4.55); bipolar
  `Watson`, FLE (6.26)).
- `aic(data)` and `bic(data)` for model selection, and `to_json()` for
  serialization through `feature_from_json()`.

Fitting a model of vectorial data to axial data, e.g. `fisher` or `kent` to a
`LineationSet`, warns and fits the data flipped into one hemisphere
(`halfspace()`), as `fisher_statistics()` does. This is only adequate for
clustered data; `watson`, `bingham` and `acg` are the models of axial data.

## Generating samples

Feature sets sample any model directly. `random(n, model=..., **params)` takes
the model as an instance, a class or a name, and each set also has a
`random_<name>()` method with the model's parameters:

```pycon
>>> linset.random(50)                                  # uniform axes
>>> linset.random(50, model="watson", mean=lin(120, 30), kappa=10)
>>> linset.random_watson(50, mean=lin(120, 30), kappa=10)   # the same
>>> folset.random_bingham(50, mean=lin(120, 30), major=lin(30, 0),
...                       kappa_major=-2, kappa_minor=-20)
```

| set | `random_<name>()` |
|---|---|
| `vec2set` | `vonmises`, `wrapped_normal`, `wrapped_cauchy`, `cardioid`, `jones_pewsey` |
| `dir2set` | the above and `axial_vonmises` |
| `vecset` | `fisher`, `kent`, `esag` |
| `linset`, `folset` | the above and `watson`, `bingham`, `acg` |
| `pairset`, `faultset` | `lisle` (faults get a random sense) |
| `rotationset` | `matrix_fisher` |

Without a model, `random()` (or `uniform_random()`) samples the uniform
distribution of the set's features. `uniform_sfs(n)` and `uniform_gss(n)` give
evenly spread, not random, vectors on the sphere (Fibonacci and golden section
spirals). All generators take `rng` like `rvs()`, but without it they draw from
numpy's global random state, so `np.random.seed()` reproduces their samples.

## Model selection

Compare the fits by an information criterion, lower is better:

```pycon
>>> fits = [l.fit(m) for m in ("fisher", "kent", "esag")]
>>> for d in fits:
...     print(type(d).__name__, d.bic(l))
VonMisesFisher -53.56...
Kent -124.89...
ESAG -125.57...
>>> best = min(fits, key=lambda d: d.bic(l))
```

The elliptical models describe these data much better than the Fisher
distribution. Compare only models of the same symmetry: an axial density spreads
its probability over both antipodes, so its log-likelihood of the same data is
lower by `N ln 2` and an axial model can't be compared with a vectorial one
this way. For axial data compare `watson`, `bingham` and `acg`.

A girdle, e.g. poles of folded foliations, is a Watson distribution with negative
`kappa`:

```pycon
>>> g = dist_watson(lin(0, 90), -20).rvs(100, rng=2, feature=fol)
>>> g.fit("watson")
Watson(mean=L:135/89, kappa=-18.7...)
```

## Plotting

`StereoNet.density()` contours the density of a distribution and
`StereoNet.confidence()` with `method="kent"` or `"esag"` draws the
confidence ellipse of the mean of the fitted model:

```pycon
>>> s = StereoNet()
>>> s.point(l)
>>> s.density(dist_kent.fit(l), levels=5)
>>> s.confidence(l, method="kent", color="r")
>>> s.show()
```

The density is shown in multiples of uniform distribution, for the axis a point
of the net stands for, i.e. a direction and its antipode combined. For a
distribution of axes it is `4 pi f(x)`. `StereoGrid.apply_distribution()`
computes the same values on a grid, which, unlike `apply_func()`, can be
saved. `RosePlot.density()` plots circular distributions in the same units as
`RosePlot.pdf()`:

```pycon
>>> d = dir2set.from_directions([12, 18, 195, 23, 7, 190, 15])
>>> r = RosePlot()
>>> r.pdf(d)
>>> r.density(d.fit("axial_vonmises"), fc="none", ec="r")
>>> r.show()
```

## Statistics methods of feature sets

The statistics methods of the feature sets are built on the distributions:

- `Vector2Set.fisher_statistics()`: `VonMises` (`AxialVonMises` for a
  `Direction2Set`) fitted with `method="approx"`, with the confidence
  interval of Upton (1986), as given by Zar. `VonMises.mean_confidence()`
  also offers the intervals of F93: `method="vonmises"` (4.42)-(4.43) for
  small samples and `method="dispersion"` (4.21)-(4.22), without a model, for
  `N >= 25`.
- `Vector3Set.fisher_statistics()`: Fisher's precision `k = (N - 1) / (N -
  R)` (Fisher, 1953; FLE (5.25)), the maximum-likelihood `kappa` and the
  confidence cone `alpha` of Fisher (1953) (alpha95); deprecated for axial
  data, see below. Before apsg 2.1 `k` was the maximum-likelihood estimate.
- `watson_statistics()`: `Watson`, with the confidence cone of FLE
  (6.11)-(6.13).
- `bingham_statistics()`: the large-sample confidence ellipse of an
  eigenvector of the orientation tensor (the moment version of Bingham's
  (1974) (7.24) ellipse) and the concentrations `k` of the fitted `Bingham`
  distribution.
- `bootstrap_statistics()`: a bootstrap cone or, with `method="ellipse"`,
  the elliptical region of Constable & Tauxe (1990) (13)-(15)
  (`StereoNet.confidence(method="bootstrap_ellipse")`).
- `random_vonmises()`, `random_fisher()` and `random_kent()` sample
  `VonMises`, `VonMisesFisher` and `Kent`.

## Circular descriptive statistics

`Vector2Set` has the descriptive statistics of MJ and F93; for a `Direction2Set`
they are computed from the doubled angles, with angles halved:

| method | statistic | source |
|----|----|----|
| `R(mean=True)`, `var()` | mean resultant, circular variance `1 - Rbar` | MJ (2.2.4), (2.3.3) |
| `csd()` | circular standard deviation `sqrt(-2 ln Rbar)` | MJ (2.3.11) |
| `angular_deviation()` | `sqrt(2 (1 - Rbar))` | MJ (2.3.12) |
| `circular_dispersion()` | `(1 - rho2) / (2 Rbar^2)` | MJ sec. 2.3.4 |
| `moment(p)`, `skewness()`, `kurtosis()` | trigonometric moments | MJ (2.4.2)-(2.4.13) |
| `median()` | circular median | F93 sec. 2.3.2 |

## Hypothesis tests

The tests return a `TestResult` with `statistic`, `p_value`, `df`,
`critical`, `reject(level)` and `extra`. They are functions of `apsg.feature`
working on arrays, and methods of the feature sets and distributions:

| question | method | test |
|----|----|----|
| uniform directions? | `Vector2Set`/`Vector3Set.uniformity_test()` | Rayleigh, MJ (10.4.4)-(10.4.6) |
| uniform on the circle, any alternative? | `uniformity_test(method=...)` | `"kuiper"` (F93 Table A5), `"watson"` (U², MJ sec. 6.3.5), `"hodges_ajne"` (MJ sec. 6.3.4), `"rao"` (MJ (6.3.45)) |
| uniform axes? | `LineationSet`/`FoliationSet.uniformity_test()` | Bingham, MJ (10.7.1)-(10.7.3) = Lisle (1985) `n I` |
| uniform pairs? | `PairSet.uniformity_test()` | Arnold & Jupp (2013) (5) |
| Fisher model adequate? | `VonMisesFisher.goodness_of_fit(data)` | colatitude `M_E`, longitude `M_U`, FLE (5.15)-(5.16) |
| circular or elliptical contours? | `Kent.score_test(data)` | Kent (1982) (5.5) |
| Watson or Bingham? | `Watson.symmetry_test(data)` | MJ (10.7.8)-(10.7.18) |
| nested models? | `likelihood_ratio_test(d0, d1, data)` | Wilks' theorem |
| common mean direction? | `common_mean_test(*others)` | sphere: Watson (1956) F, McFadden & Lowes (1981), Watson & Debiche (1992); circle: Watson-Williams, uniform scores, two-sample U² (MJ ch. 7-8) |
| common principal axis? | `LineationSet.common_mean_test(*others)` | FLE (7.72)-(7.76), corrected |
| common concentration? | `equal_kappa_test(*others)` | F, MJ (10.5.3); Bartlett, MJ (10.6.21) |
| isotropic mean tensor? | `EllipsoidSet.mean_tensor()["anisotropy_test"]` | Jelínek (1978) (44)-(46) |

```pycon
>>> lins.uniformity_test().reject()
>>> fols.common_mean_test(other_fols).p_value
>>> vecs.equal_kappa_test(other_vecs)
```

Some printed formulas the tests follow deviate from a source, where the
source is in error: FLE's `M_E`, `M_U` constants are the original ones
(MJ misquotes them), the Watson & Debiche statistic has its factor 2, and the
axial common-mean test drops the spurious factor 2 Watson (1983) introduced
into FLE's `g`.

## Resultant statistics of axial data

`R()` of a `LineationSet` or `FoliationSet` is the principal axis of the
orientation tensor, with the length of the resultant of the axes oriented to its
side (`halfspace()`), so it does not depend on the signs the axes are stored
with. The statistics built on the resultant of vectors -- `fisher_statistics()`,
`var()`, `delta()`, `rdegree()` and `csd()` -- describe clustered axial
data only, and for axial sets they are deprecated: they warn in apsg 2.1 and will
raise `TypeError` in apsg 2.2. Use instead:

- `watson_statistics()`, `bingham_statistics()` or a fitted `watson` or
  `bingham` distribution for the mean axis, concentration and confidence
  region;
- the orientation tensor for the spread and shape of the fabric, e.g.
  `1 - l.ortensor().eigenvalues(0)`, `l.ortensor().strength`,
  `.shape` or `.Intensity`;
- or, for clustered data, the vectors of the axes oriented to one side,
  `l.halfspace().to_vec().fisher_statistics()`.

`StereoNet.confidence()` draws the Watson cone by default for lineation and
foliation sets (around the pole of the girdle for girdle data) and the Fisher cone
for vectors.

## Pairs and faults

A `Pair` (foliation and lineation) is a frame of three orthogonal axes, the
lineation `l`, `n x l` and the foliation normal `n`. Reversing the lineation
or the normal gives the same pair. Lisle (1989) averages pairs by the tensor
`mean(l l' - n n')`: its first and last eigenvectors are the mean lineation and
the mean foliation normal, which are orthogonal, unlike the means of the
lineations and the poles taken separately. `PairSet.lisle_statistics()` returns
the mean pair, the eigenvalues and large-sample confidence ellipses of the mean
lineation and normal, and `StereoNet.confidence()` draws them for pair and fault
sets:

```pycon
>>> s = pairs.lisle_statistics()
>>> s["mean"], s["lin"]["gamma"], s["fol"]["gamma"]
>>> net = StereoNet()
>>> net.pair(pairs)
>>> net.pair(s["mean"], color="r")
>>> net.confidence(pairs)                    # method "lisle"
```

The `Lisle` distribution (`dist_lisle`) is the model behind this mean: its density
with respect to the uniform distribution of pairs is
`exp(l' L l - n' L n) / c` with `L = diag(kappa1 + kappa2, kappa2, 0)` in the
frame of the mean pair, its maximum-likelihood mean is Lisle's mean, and it does
not depend on the signs of `l` and `n`. `kappa1` measures the concentration
of the lineation within the foliation (rotation about the normal) and `kappa2`
that of the foliation about the lineation, e.g. folding about it:

```pycon
>>> d = pairs.fit("lisle")                    # or dist_lisle.fit(pairs)
>>> d.kappa1, d.kappa2, d.mean_confidence(which="fol")
>>> d.rvs(100, rng=1)                         # PairSet
```

The sense of faults is not used by either; `OrientationTensor3.from_pairs()` is
Lisle's tensor (shifted by default, see its documentation). `Lisle` is apsg's own
model: the frame Bingham distribution of Arnold & Jupp (2013) whose sufficient
statistic is Lisle's tensor. `PairSet.uniformity_test()` is the test of
Arnold & Jupp (2013).

## Rotations

A rotation, e.g. a directed frame of a lineation with sense and a foliation normal
with a defined side, is a `Rotation3`; a `RotationSet` holds several of them,
with `mean()` the rotation closest to their mean matrix (MJ (13.2.16)). The
`MatrixFisher` distribution has the density `exp(tr(F' R))` with respect to the
uniform distribution of rotations; the proper singular value decomposition `F =
U diag(psi) V'` gives its mode `U V'` and concentrations `psi`:

```pycon
>>> d = dist_matrix_fisher.from_mode(rotation.from_axisangle(lin(120, 30), 40), 20)
>>> r = d.rvs(50, rng=1)                      # RotationSet
>>> dist_matrix_fisher.fit(r).mean, r.mean()
```

A `Pair` is an axial frame, four rotations at once, so pairs are described by
Lisle's tensor and the `Lisle` distribution, not by the matrix Fisher one.

## Abbreviations

MJ: Mardia & Jupp (2000); FLE: Fisher, Lewis & Embleton (1987); F93: Fisher (1993);
L&V: Ley & Verdebout (2017).

## References

- Arnold, R., Jupp, P.E. (2013). Statistics of orthogonal axial frames.
  Biometrika 100, 571-586.
- Bingham, C. (1974). An antipodally symmetric distribution on the sphere. The
  Annals of Statistics 2, 1201-1225.
- Constable, C., Tauxe, L. (1990). The bootstrap for magnetic susceptibility
  tensors. Journal of Geophysical Research 95, 8383-8395.
- Fisher, R.A. (1953). Dispersion on a sphere. Proceedings of the Royal Society
  of London A 217, 295-305.
- Fisher, N.I. (1993). Statistical Analysis of Circular Data. Cambridge
  University Press.
- Fisher, N.I., Lewis, T., Embleton, B.J.J. (1987). Statistical Analysis of
  Spherical Data. Cambridge University Press.
- Jelínek, V. (1978). Statistical processing of anisotropy of magnetic
  susceptibility measured on groups of specimens. Studia Geophysica et
  Geodaetica 22, 50-62.
- Jelínek, V. (1981). Characterization of the magnetic fabric of rocks.
  Tectonophysics 79, T63-T67.
- Jones, M.C., Pewsey, A. (2005). A family of symmetric distributions on the
  circle. Journal of the American Statistical Association 100, 1422-1428.
- Kent, J.T. (1982). The Fisher-Bingham distribution on the sphere. Journal of
  the Royal Statistical Society B 44, 71-80.
- Kent, J.T., Ganeiber, A.M., Mardia, K.V. (2013). A new method to simulate
  the Bingham and related distributions in directional data analysis with
  applications. Preprint, arXiv:1310.8110.
- Ley, C., Verdebout, T. (2017). Modern Directional Statistics. CRC Press.
- Lisle, R.J. (1985). The use of the orientation tensor for the description and
  statistical testing of fabrics. Journal of Structural Geology 7, 115-117.
- Lisle, R.J. (1989). The statistical analysis of orthogonal orientation data.
  Journal of Geology 97, 360-364.
- López-Custodio, P.C. (2025). A cheat sheet for probability distributions of
  orientational data. Mechanism and Machine Theory 210, 105995.
- Mardia, K.V., Jupp, P.E. (2000). Directional Statistics. Wiley.
- McFadden, P.L., Lowes, F.J. (1981). The discrimination of mean directions
  drawn from Fisher distributions. Geophysical Journal of the Royal
  Astronomical Society 67, 19-33.
- Paine, P.J., Preston, S.P., Tsagris, M., Wood, A.T.A. (2018). An elliptically
  symmetric angular Gaussian distribution. Statistics and Computing 28,
  689-697.
- Tyler, D.E. (1987). Statistical analysis for the angular central Gaussian
  distribution on the sphere. Biometrika 74, 579-589.
- Zar, J.H. Biostatistical Analysis. Prentice Hall. Eq. 26.24-26.25, the
  approximation of Upton (1986).
