# apsg.paleomag

Paleomagnetic demagnetization data. The module is not imported by `from apsg import *`;
import it explicitly:

```pycon
>>> from apsg.paleomag import *
>>> s = Specimen.from_file("specimen.rs3")
>>> s.interpret("ChRM", slice(3, None))
>>> ZijderveldPlot(s, coords="tilt").show()
```

A [`Specimen`][apsg.paleomag.Specimen] reads PMD, Remasoft RS3 (thermal or AF is guessed from the levels, as the file has no unit), SQUID, AGICO JR6, Caltech CIT and 2G Enterprises files (several specimens per file through [`Sample.from_file`][apsg.paleomag.Sample.from_file]) and fits
components by principal component analysis, by hand or by the automatic search of
[`Specimen.find_components`][apsg.paleomag.Specimen.find_components] (windows of steps, dynamic-programming
segments, site stacking with [`Sample.stack`][apsg.paleomag.Sample.stack]); a [`Sample`][apsg.paleomag.Sample] collects
specimens for mean directions, fold and reversal tests and virtual geomagnetic poles.

### Specimens and samples

::: apsg.paleomag.Specimen

::: apsg.paleomag.Sample

::: apsg.paleomag.DemagStep

::: apsg.paleomag.Component

::: apsg.paleomag.ComponentCriteria

### Functions

::: apsg.paleomag.lines_planes_mean

::: apsg.paleomag.kirschvink_mean

::: apsg.paleomag.bailey_halls_mean

::: apsg.paleomag.mcfadden_fold_test

::: apsg.paleomag.inclination_only_mean

::: apsg.paleomag.relative_paleointensity

::: apsg.paleomag.RPIEstimate

::: apsg.paleomag.align_core_sections

::: apsg.paleomag.vgp

::: apsg.paleomag.core_rotation

::: apsg.paleomag.bedding_vector

::: apsg.paleomag.untilt_rotation

### Plots

All plots are `PaleomagPlot` subclasses: keyword arguments override `apsg_conf.paleomag`,
`show()`, `savefig()` and `render2fig()` work as for `RosePlot` and the fabric plots.

::: apsg.paleomag.PaleomagPlot

::: apsg.paleomag.ZijderveldPlot

::: apsg.paleomag.DemagPlot

::: apsg.paleomag.KappaTPlot

::: apsg.paleomag.ComponentSearchPlot

::: apsg.paleomag.SpecimenStereoPlot

::: apsg.paleomag.SpecimenPlot

::: apsg.paleomag.FoldTestPlot

::: apsg.paleomag.TiltComparisonPlot

::: apsg.paleomag.ReversalTestPlot

::: apsg.paleomag.LinesPlanesPlot

::: apsg.paleomag.InclinationOnlyPlot
