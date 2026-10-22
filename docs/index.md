---
hide:
  - navigation
---

# Welcome to APSG

![APSG logo](images/apsg_banner_light.svg#only-light){ width="300" }
![APSG logo](images/apsg_banner_dark.svg#only-dark){ width="300" }

**APSG** brings the essential analytical toolkit of structural geology
directly into Python. From processing raw field measurements and
performing directional statistics to simplified stress and strain
tensor analysis and creating publication-ready stereonets and fabric
diagrams, APSG handles the underlying mathematics while keeping every
step transparent, documented, and reproducible. It provides an intuitive
pythonic bridge between field observations, quantitative analysis, and
tectonic interpretation — all seamlessly integrated into your Jupyter notebook.
{ .lead }

<div class="grid cards" markdown>

-   :material-compass-outline:{ .lg .middle } __Structural features__

    ---

    Lineations, foliations, faults, pairs, cones and arcs behave like
    proper geological objects, with degree-based angles and RHR/dip-direction
    notation built in.

-   :material-layers-triple-outline:{ .lg .middle } __Feature sets__

    ---

    Group any feature into a set for vectorized statistics, filtering,
    rotation and resampling — no manual loops over measurements.

-   :material-chart-scatter-plot:{ .lg .middle } __Tensors & fabric analysis__

    ---

    Deformation gradients, orientation and stress tensors, with
    Vollmer, Ramsay, Flinn and Hsu fabric plots ready to go.

-   :material-circle-slice-8:{ .lg .middle } __Stereonets & rose diagrams__

    ---

    Publication-ready equal-area/equal-angle stereonets, contouring and
    rose diagrams, styled through a single configuration object.

-   :material-table:{ .lg .middle } __Pandas integration__

    ---

    APSG features live directly inside a `DataFrame` column via a
    pandas extension type, so your existing pandas workflow just works.

-   :material-database-outline:{ .lg .middle } __Database support__

    ---

    Read and write field data from legacy PySDB `.sdb` files or a
    remote websdb project with the same feature-based API.

</div>

## Usage

To use APSG in a project:

```python
import apsg
```

To use APSG interactively it is easier to import into current namespace:

```python
from apsg import *
```

A quick stereonet of random Fisher-distributed foliations:

```python
from apsg import *

f = folset.random_fisher(position=fol(130, 60))
s = StereoNet()
s.great_circle(f)
s.point(f)
s.contour(f)
s.show()
```

[Install APSG](installation.md){ .md-button .md-button--primary }
[Tutorials](tutorials/index.md){ .md-button }
[API reference](api/index.md){ .md-button }
