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
| circle (`Vector2`, `Direction`) | `Uniform2`, `VonMises` (`vonmises`), `WrappedNormal` (`wrapped_normal`) | `AxialUniform2`, `AxialVonMises` (`axial_vonmises`) |
| sphere (`Vector3`, `Lineation`, `Foliation`) | `Uniform3`, `VonMisesFisher` (`fisher`), `Kent` (`kent`), `ESAG` (`esag`) | `AxialUniform3`, `Watson` (`watson`), `Bingham` (`bingham`), `ACG` (`acg`) |

In parentheses are the short names accepted by `FeatureSet.fit()`, e.g.
`lineations.fit("kent")`; the `apsg` aliases add the prefix `dist_`.

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
- `aic(data)` and `bic(data)` for model selection, and `to_json()` for
  serialization through `feature_from_json()`.

Fitting a model of vectorial data to axial data, e.g. `fisher` or `kent` to a
`LineationSet`, warns and fits the data flipped into one hemisphere
(`halfspace()`), as `fisher_statistics()` does. This is only adequate for
clustered data; `watson`, `bingham` and `acg` are the models of axial data.

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

The statistics methods of the feature sets are built on the distributions and
return the same dict as before:

- `Vector2Set.fisher_statistics()`: `VonMises` (`AxialVonMises` for a
  `Direction2Set`) fitted with `method="approx"`, with the confidence
  interval of Upton (1986), as given by Zar.
- `Vector3Set.fisher_statistics()`: `VonMisesFisher`, with the confidence
  cone of Fisher (1953); deprecated for axial data, see below.
- `watson_statistics()`: `Watson`.
- `bingham_statistics()`: the large-sample confidence ellipse of an
  eigenvector of the orientation tensor and the concentrations `k` of the
  fitted `Bingham` distribution.
- `random_vonmises()`, `random_fisher()` and `random_kent()` sample
  `VonMises`, `VonMisesFisher` and `Kent`.

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
Lisle's tensor (shifted by default, see its documentation).

## References

- Fisher, R.A. (1953). Dispersion on a sphere. Proceedings of the Royal Society
  of London A 217, 295-305.
- Fisher, N.I. (1993). Statistical Analysis of Circular Data. Cambridge
  University Press.
- Fisher, N.I., Lewis, T., Embleton, B.J.J. (1987). Statistical Analysis of
  Spherical Data. Cambridge University Press.
- Kent, J.T. (1982). The Fisher-Bingham distribution on the sphere. Journal of
  the Royal Statistical Society B 44, 71-80.
- Kent, J.T., Ganeiber, A.M., Mardia, K.V. (2013). A new method to simulate
  the Bingham and related distributions in directional data analysis with
  applications. Preprint, arXiv:1310.8110.
- Lisle, R.J. (1989). The statistical analysis of orthogonal orientation data.
  Journal of Geology 97, 360-364.
- López-Custodio, P.C. (2025). A cheat sheet for probability distributions of
  orientational data. Mechanism and Machine Theory 210, 105995.
- Paine, P.J., Preston, S.P., Tsagris, M., Wood, A.T.A. (2018). An elliptically
  symmetric angular Gaussian distribution. Statistics and Computing 28,
  689-697.
- Tyler, D.E. (1987). Statistical analysis for the angular central Gaussian
  distribution on the sphere. Biometrika 74, 579-589.
- Zar, J.H. Biostatistical Analysis. Prentice Hall. Eq. 26.24-26.25, the
  approximation of Upton (1986).
