# apsg.data

Datasets bundled with APSG for tutorials, examples and reproducible tests. They live in the
package (one small file per group, the 71 paleomagnetic specimens in a single zip), so they are
available after `pip install apsg` without paths or downloads.

```pycon
>>> from apsg import data
>>> data.catalog()                           # all datasets with description and source
>>> data.info("fisher_1953")                 # citation and remarks
>>> f = data.load("mele")                    # FaultSet
>>> smp = data.load("pmdata", site="PM04")   # paleomagnetic Sample
>>> df = pd.read_csv(data.path("structures"))
>>> data.tauxe_table_4_2()                   # dict: {"normal": ..., "reversed": ...}
```

Three groups of data exist: the **tutorial data** of the notebooks, **published** short tables
cited from textbooks and papers (Fisher 1953, Fisher, Lewis & Embleton 1987, Fisher 1993, Mardia &
Jupp 2000, Tauxe 2003, Lisle 1989, ...), and **synthetic** data with known properties generated with
fixed seeds (Fisher, Kent, Watson and uniform samples, a folded
bedding set, AMS and stress tensors, faults). `pmdata` is artificial: names, locations and
numbers are invented.

## Available data

<!-- catalog:start -->
| Name | Kind | n | Description | Source |
|----|----|----|----|----|
| `mele` | faults | 94 | Fault-slip data: fault plane, slickenline and sense of movement. | tutorial data of APSG |
| `structures` | table | 253 | Lineation and foliation measurements of several sites, one row per measurement (columns site, structure, azi, inc). | tutorial data of APSG |
| `structures_wide` | table | 132 | The same kind of data in wide format: one row per measurement with the columns of S1, L1, S2 and L2 (empty where not measured). | tutorial data of APSG |
| `pmdata` | paleomag | 71 | 71 thermally demagnetized drill-core specimens of 10 sites (dict of ``Sample`` by site, or one ``Sample`` with ``site=``). | artificial, seeded random generator: normal and reversed ChRM acquired before tilting, soft overprint, some great-circle and poorly behaved specimens |
| `pmdata_sites` | table | 10 | Site table of ``pmdata``: location of the hypothetical study, bedding and number of specimens. | artificial, seeded random generator |
| `fisher_1953` | vectors | 9 | Remanent magnetization of nine specimens of the 1947-48 Icelandic lava flow (declination, inclination). | Fisher, R.A. (1953) Dispersion on a sphere. Proc. R. Soc. London A 217, 295-305, Table 1; also Mardia & Jupp (2000) Table 9.1. |
| `fle_b2` | vectors | 26 | Palaeomagnetic sample of 26 directions (declination, inclination) from Fisher, Lewis & Embleton, Appendix B2; the Fisher goodness-of-fit example. | Fisher, Lewis & Embleton (1987) Statistical Analysis of Spherical Data, Appendix B2 (Example 5.17). |
| `mj_two_samples` | groups | 20 | Two samples of ten directions (azimuth, inclination) for two-sample tests (groups I and II). | Mardia & Jupp (2000) Directional Statistics, Example 10.7 (Table 10.3, Vistelius 1966). |
| `tauxe_table_4_2` | groups | 49 | Normal and reversed polarity directions (declination, inclination) for the reversal test; 25 normal and 24 reversed. | Tauxe, L. (2003) Paleomagnetic Principles and Practice, Table 4.2 (Example 4.9). |
| `fisher_1993_b3` | circular | 76 | Circular sample B3, n = 76 directions (degrees). | Fisher, N.I. (1993) Statistical Analysis of Circular Data, Appendix B3. |
| `fisher_1993_b4` | circular | 50 | Circular sample B4, n = 50 directions (degrees). | Fisher, N.I. (1993) Statistical Analysis of Circular Data, Appendix B4. |
| `fisher_1993_b5` | circular | 60 | Orientations of 60 feldspar laths in basalt, axial data (degrees). | Fisher, N.I. (1993) Statistical Analysis of Circular Data, Appendix B5. |
| `fisher_1993_b7` | circular | 100 | Circular sample B7, n = 100 directions (degrees). | Fisher, N.I. (1993) Statistical Analysis of Circular Data, Appendix B7. |
| `fisher_1993_b6` | groups | 100 | Circular sample B6: three sets of directions (degrees). | Fisher, N.I. (1993) Statistical Analysis of Circular Data, Appendix B6. |
| `fisher_1993_b8` | grouped | 100 | Grouped axial counts of Appendix B8, nine classes of 20 degrees (0-180). | Fisher, N.I. (1993) Statistical Analysis of Circular Data, Appendix B8. |
| `fisher_1993_b9` | grouped | 279 | Grouped directions of Appendix B9, 36 classes of 10 degrees. | Fisher, N.I. (1993) Statistical Analysis of Circular Data, Appendix B9. |
| `fisher_1993_b11` | circular | 22 | Circular sample B11, n = 22 directions (degrees). | Fisher, N.I. (1993) Statistical Analysis of Circular Data, Appendix B11. |
| `fisher_1993_b12` | circular | 15 | Homing directions of 15 pigeons (degrees), Schmidt-Koenig (1963). | Fisher, N.I. (1993) Statistical Analysis of Circular Data, Appendix B12; Mardia & Jupp (2000) Ex. 7.2. |
| `gorleston_wind` | groups | 49 | Wind directions (degrees) at Gorleston in four seasons. | Mardia, K.V. (1972) Statistics of Directional Data, Table 6.3 (= Fisher 1993 B14 = Mardia & Jupp 2000 Table 7.2). |
| `mallard` | grouped | 714 | Departure directions of 714 mallards: counts in 18 classes of 20 degrees. | Mardia (1972) Table 1.1; Mardia & Jupp (2000) Example 2.4. |
| `leukaemia_cases` | grouped | 506 | Monthly counts of leukaemia cases: one class of 30 degrees per month. | Mardia & Jupp (2000) Example 6.3. |
| `johnson_wehrly` | table | 19 | Wind direction (degrees) and ozone concentration, 19 pairs, for circular-linear association. | Johnson & Wehrly (1977); Mardia & Jupp (2000) Table 11.1 (= Fisher 1993 B18). |
| `lisle_1989` | groups | 8 | Rhoscolyn fold example: four bedding/cleavage intersection lineations (trend, plunge) and the bedding (dip direction, dip) of the same four localities, for orthogonal mean axes. | Lisle, R.J. (1989) The statistical analysis of orthogonal orientation data. J. Geol. 97, 359-364, Table 2 (p. 363). |
| `fisher_k50` | vectors | 100 | 100 directions from a Fisher distribution, mean 120/35, kappa 50. | synthetic, vecset.random_fisher(100, kappa=50), seed 101 |
| `fisher_weak` | vectors | 100 | 100 directions from a weakly concentrated Fisher distribution, mean 40/60, kappa 5. | synthetic, vecset.random_fisher(100, kappa=5), seed 102 |
| `kent` | vectors | 150 | 150 directions from an elongated Kent distribution, mean 80/25, kappa 30, beta 10. | synthetic, vecset.random_kent(150, kappa=30, beta=10), seed 103 |
| `watson_bipolar` | lines | 100 | 100 axes from a bipolar Watson distribution, axis 250/20, kappa 15. | synthetic, linset.random_watson(100, kappa=15), seed 104 |
| `watson_girdle` | lines | 100 | 100 axes from a Watson girdle distribution around the pole 30/50, kappa -8. | synthetic, linset.random_watson(100, kappa=-8), seed 105 |
| `uniform_sphere` | lines | 100 | 100 axes uniformly distributed on the sphere. | synthetic, linset.random(100), seed 106 |
| `two_clusters` | groups | 100 | Two Watson clusters of axes (60 around 40/30 and 40 around 250/15), to test clustering. | synthetic, linset.random_watson, seeds 107 and 108 |
| `vonmises_mixture` | circular | 120 | 120 directions (degrees), a mixture of two von Mises distributions (40, kappa 8, n 70 and 220, kappa 4, n 50). | synthetic, vec2set.random_vonmises, seeds 109 and 110 |
| `fold_data` | groups | 100 | Poles of 60 bedding planes of a cylindrical fold (axis 20/10) and 40 hinge-parallel lineations. | synthetic, numpy, seed 111 |
| `ams_sites` | tensors | 20 | Anisotropy of magnetic susceptibility tensors of 20 sites with a foliated, weakly lineated fabric. | synthetic, numpy, seed 112 |
| `stress_tensors` | stress | 15 | 15 stress tensors (compression positive, sigma1 about 210/70, ratio 0.57). | synthetic, numpy, seed 113 |
| `synthetic_faults` | faults | 60 | 60 normal faults in two conjugate sets (dip directions 60 and 240, dip 60) with dip-slip slickenlines. | synthetic, numpy, seed 114 |
| `ams_two_sites` | ams | 26 | AMS of two sites, 14 and 12 specimens with core orientation and bedding: oblate grains in flattening (AM1), prolate grains in transpression (AM2). | synthetic, apsg.ams grain models, numpy, seed 115 |
<!-- catalog:end -->

## Functions

::: apsg.data.catalog

::: apsg.data.info

::: apsg.data.load

::: apsg.data.path

::: apsg.data.pmd_file

::: apsg.data.names

::: apsg.data.Dataset
