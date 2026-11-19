# apsg.ams

Anisotropy of magnetic susceptibility (AMS). The module is not imported by
`from apsg import *`; import it explicitly:

```pycon
>>> from apsg import velgrad
>>> from apsg.ams import *
>>> L = velgrad.from_comp(xy=1)                    # simple shear
>>> K = grain_K("prolate", 3)                      # grains of axial ratio 3
>>> D = L.pseudo(K).defgrad(time=3)                # pseudo-displacement of the grains
>>> fabric = ACG.from_displacement(D)              # March / Willis distribution of the axes
>>> k = ams_from_orientation(fabric, K=0.9, delta=0.3)
>>> k.eigenlins()
```

### Specimens and sites

An [`AMSSpecimen`][apsg.ams.AMSSpecimen] holds the susceptibility tensor of one specimen with its
bulk susceptibility, core orientation and bedding, and rotates it to specimen, geographic or
tectonic coordinates. An [`AMSSite`][apsg.ams.AMSSite] collects the specimens of a site: normalised
tensors as `EllipsoidSet`, the mean tensor with confidence ellipses (Jelinek, 1978), the
anisotropy test and a table of the parameters of Jelinek (1981).

::: apsg.ams.AMSSpecimen

::: apsg.ams.AMSSite

### Statistics

`AMSSite.mean()` is the linear perturbation method of Jelinek (1978). `AMSSite.bootstrap()`
resamples the normalised tensors (Constable and Tauxe, 1990) and gives bootstrapped eigenvalues,
eigenvectors and a region containing the stated fraction of the bootstrapped axes, which
`StereoNet.confidence(method="bootstrap_tensor")` draws; `AMSSpecimen.bootstrap()` simulates the
measurement error of one specimen. `AMSSite.compare()` tests whether two sites have the same
mean tensor (Hotelling's T² of the deviatoric components, or a permutation test).

::: apsg.ams.compare_tensor_means

::: apsg.ams.tensor_bootstrap

::: apsg.ams.tensor_error_bootstrap

::: apsg.ams.TensorBootstrap

### Plots

All plots share the interface of the other figure plots (`show()`, `savefig()`, `render2fig()`)
and are configured by `apsg_conf.ams`.

::: apsg.ams.PjTPlot

::: apsg.ams.LFPlot

::: apsg.ams.PTPlot

::: apsg.ams.AMSStereoPlot

::: apsg.ams.BootstrapPlot

::: apsg.ams.ModelPlot

### Flow models

[`AMSModel`][apsg.ams.AMSModel] gives the AMS of populations of uniaxial grains
([`GrainPopulation`][apsg.ams.GrainPopulation]) carried by a flow; `evolve()` returns an
[`AMSEvolution`][apsg.ams.AMSEvolution] with a table of the parameters, the orientation of the axes
and the switch of the lineation. [`jeffery_evolution`][apsg.ams.jeffery_evolution] integrates
the rotation of rigid triaxial ellipsoids (Jeffery, 1922) for a sample of grains, and
[`orientation_tensor_from_ams`][apsg.ams.orientation_tensor_from_ams] inverts the AMS ellipsoid
of a rock with one kind of uniaxial grains to the orientation tensor of the grains.

::: apsg.ams.GrainPopulation

::: apsg.ams.AMSModel

::: apsg.ams.AMSEvolution

::: apsg.ams.jeffery_evolution

::: apsg.ams.jeffery_rotations

::: apsg.ams.ams_from_rotations

::: apsg.ams.orientation_tensor_from_ams

### Tables

Reports (`.asc`) and text tables exported by Anisoft or SUFAR are read by `read_sufar`,
`read_anisoft_table` and `read_ams` (which recognises the layout); binary `.ams` and `.ran` files
must be exported to text first. Tables with the six tensor components (`k11 k22 k33 k12 k23 k13`, the order of AGICO files),
optional bulk susceptibility, core orientation and bedding are read from CSV files or data frames
and written back by `AMSSite.to_csv()`.

::: apsg.ams.read_ams

::: apsg.ams.read_sufar

::: apsg.ams.read_anisoft_table

::: apsg.ams.read_csv

::: apsg.ams.from_dataframe

::: apsg.ams.to_dataframe

### Example data

`data.load("ams_two_sites")` gives two artificial sites made with the grain models below, see
[`apsg.data`](data.md).

### Grain models

A rigid grain that does not interact with its neighbours turns in a homogeneous flow like a
material line carried by the reduced flow `K E + W` (Willis, 1977): `K` is the kinematic
coefficient of the grain ([`grain_K`][apsg.ams.grain_K], Jeffery's
`(r² - 1) / (r² + 1)` for spheroids) and
[`VelocityGradient3.pseudo`][apsg.feature.VelocityGradient3.pseudo] builds the flow. Axes that
start uniformly distributed are then angular central Gaussian distributed
([`ACG.from_displacement`][apsg.feature.ACG.from_displacement]), and the susceptibility tensor
of the rock follows from their orientation tensor.

::: apsg.ams.grain_K

::: apsg.ams.kinematic_K

::: apsg.ams.osborn_factors

::: apsg.ams.grain_susceptibilities

::: apsg.ams.grain_anisotropy

::: apsg.ams.ams_from_orientation

::: apsg.ams.ams_from_grains
