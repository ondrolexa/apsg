================================
Welcome to APSG's documentation!
================================

.. image:: /images/apsg_banner_light.svg
   :class: light-only
   :width: 300
   :alt: APSG logo

.. image:: /images/apsg_banner_dark.svg
   :class: dark-only
   :width: 300
   :alt: APSG logo

.. rst-class:: lead

   **APSG** brings the essential analytical toolkit of structural geology
   directly into Python. From processing raw field measurements and
   performing directional statistics to simplified stress and strain
   tensor analysis and creating publication-ready stereonets and fabric
   diagrams, APSG handles the underlying mathematics while keeping every
   step transparent, documented, and reproducible. It provides an intuitive
   pythonic bridge between field observations, quantitative analysis, and
   tectonic interpretation — all seamlessly integrated into your Jupyter notebook.

|

.. grid:: 1 1 2 3
    :gutter: 2
    :padding: 0
    :class-row: surface

    .. grid-item-card:: :octicon:`north-star` Structural features

        Lineations, foliations, faults, pairs, cones and arcs behave like
        proper geological objects, with degree-based angles and RHR/dip-direction
        notation built in.

    .. grid-item-card:: :octicon:`stack` Feature sets

        Group any feature into a set for vectorized statistics, filtering,
        rotation and resampling — no manual loops over measurements.

    .. grid-item-card:: :octicon:`graph` Tensors & fabric analysis

        Deformation gradients, orientation and stress tensors, with
        Vollmer, Ramsay, Flinn and Hsu fabric plots ready to go.

    .. grid-item-card:: :octicon:`telescope` Stereonets & rose diagrams

        Publication-ready equal-area/equal-angle stereonets, contouring and
        rose diagrams, styled through a single configuration object.

    .. grid-item-card:: :octicon:`table` Pandas integration

        APSG features live directly inside a ``DataFrame`` column via a
        pandas extension type, so your existing pandas workflow just works.

    .. grid-item-card:: :octicon:`database` Database support

        Read and write field data from legacy PySDB ``.sdb`` files or a
        remote websdb project with the same feature-based API.

Usage
-----

To use APSG in a project::

    import apsg

To use APSG interactively it is easier to import into current namespace::

    from apsg import *


Check tutorials and module API for more details.

Contents
--------

.. toctree::
   :maxdepth: 2

   installation
   tutorials
   automodules
   contributing
   authors
