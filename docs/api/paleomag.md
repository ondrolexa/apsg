# apsg.paleomag

Paleomagnetic demagnetization data. The module is not imported by `from apsg import *`;
import it explicitly:

```pycon
>>> from apsg.paleomag import *
>>> s = Specimen.from_file("specimen.rs3")
>>> s.interpret("ChRM", slice(3, None))
>>> zijderveld_plot(s, coords="tilt")
```

A [`Specimen`][apsg.paleomag.Specimen] reads PMD, Remasoft RS3 and SQUID files and fits
components by principal component analysis; a [`Sample`][apsg.paleomag.Sample] collects
specimens for mean directions, fold and reversal tests and virtual geomagnetic poles.

### Specimens and samples

::: apsg.paleomag.Specimen

::: apsg.paleomag.Sample

::: apsg.paleomag.DemagStep

::: apsg.paleomag.Component

### Functions

::: apsg.paleomag.lines_planes_mean

::: apsg.paleomag.kirschvink_mean

::: apsg.paleomag.bailey_halls_mean

::: apsg.paleomag.vgp

::: apsg.paleomag.core_rotation

::: apsg.paleomag.bedding_vector

::: apsg.paleomag.untilt_rotation

### Plots

::: apsg.paleomag.zijderveld_plot

::: apsg.paleomag.demag_plot

::: apsg.paleomag.stereo_plot
