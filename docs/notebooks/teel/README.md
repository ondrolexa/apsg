# Teel basalts data for the paleomagnetism tutorial

Demagnetization data of a subset of the Teel Formation basalt flows (Late Ordovician,
Zavkhan terrane, Mongolia), used by `09_apsg_paleomag.ipynb`.

**Source.** Kilian, T.M., Swanson-Hysell, N.L., Bold, U., Crowley, J. and Macdonald, F.A.
(2016). Paleomagnetism of the Teel basalts from the Zavkhan terrane: Implications for
Paleozoic paleogeography in Mongolia and the growth of continental crust. *Lithosphere* 8(6),
699-715, [doi:10.1130/L552.1](https://doi.org/10.1130/L552.1), with the GSA Data Repository
Item 2016284. The data are published in the MagIC database
(<http://earthref.org/MAGIC/doi/10.1130/L552.1>, contribution 18692) and in
<https://github.com/Swanson-Hysell-Group/2016_Teel_Basalts>. The repository states no licence,
so only a small subset is redistributed here for teaching; please cite the paper when you use
it.

**Files**

| File | Content |
|----|----|
| `Z31.1a.pmd` ... `Z51.8a.pmd` | 67 specimens of 8 sites (Z31, Z34, Z36, Z38, Z40, Z47, Z49, Z51), PMD format read by `Specimen.from_pmd` |
| `teel_components.csv` | the authors' line fits (PmagPy `demag_gui`; geographic coordinates, step range in degrees C) |
| `teel_sites.csv` | the authors' site means of the magnetite component in both coordinate systems, with VGP and paleolatitude |

**Conversion.** The files were made by a one-off script (not kept) that followed the steps below. The steps (NRM and thermal steps up to 582 C) come from `magic_measurements.txt` of
the GitHub repository (specimen coordinates, moment in Am2). The specimen volume is taken as the
nominal 1 cm3 of the authors' CIT files, so the files hold the magnetization in A/m. Steps the
authors flagged as bad are left out, and so is the liquid nitrogen step (77 K). The core azimuth
and hade are the MagIC sample azimuth and minus the dip, the bedding is the MagIC bedding dip
direction and dip. The site means come from the tables of the Data Repository PDF.

**Checks.** apsg's principal component analysis of the written files over the authors' step ranges
reproduces their MAG and HEM line fits (median deviation 0.05 degrees, 95 fits). The geographic and
tilt-corrected directions apsg computes for each step agree with the CIT files to 0.2 and 0.7
degrees (median); site Z36 differs by up to 2.4 degrees, because its CIT geographic declinations are
about 2 degrees larger than the MagIC orientation gives.

**Note on MagIC 3.0.** In the MagIC 3.0 contribution file the tilt-corrected directions of the
specimens and sites tables are not a rotation of the geographic ones (the same bedding gives
different rotations within a flow); they are not used here. The tables of the Data Repository,
which are used instead, are consistent with the bedding.
