==============
config module
==============

The :mod:`apsg.config` module provides a hierarchical configuration system based on dataclasses with a ``Mapping`` mixin. The global instance ``apsg_conf`` is used throughout APSG to control defaults for notation, rounding, figure properties, and plotting defaults.

Usage
-----

Access configuration values using dot notation::

    >>> from apsg.config import apsg_conf
    >>> apsg_conf.ndigits
    3
    >>> apsg_conf.stereonet.kind
    'equal-area'
    >>> apsg_conf.stereonet_point.mfc
    >>> apsg_conf.roseplot.bins
    36

Modify values in-place::

    >>> apsg_conf.ndigits = 5
    >>> apsg_conf.stereonet.kind = 'equal-angle'
    >>> apsg_conf.roseplot_bars.ec = 'gray'

The ``Mapping`` interface enables dict-like access::

    >>> apsg_conf['figsize']
    (8, 6)
    >>> dict(apsg_conf.stereonet)
    {...}

Use :meth:`~BaseConfig.update` for bulk updates from a dictionary. Nested configs are updated recursively::

    >>> apsg_conf.update({'ndigits': 4, 'stereonet': {'kind': 'equal-area'}})

To obtain a plain dictionary (e.g. to pass as ``**kwargs``), call :meth:`~BaseConfig.copy`::

    >>> kwargs = apsg_conf.stereonet_point.copy()
    >>> kwargs['mfc'] = 'red'

Classes
-------

.. automodule:: apsg.config
    :autosummary:
    :members:
    :show-inheritance:
    :autosummary-no-nesting:

Default values
--------------

AppConfig
^^^^^^^^^

.. list-table::
    :header-rows: 1
    :widths: 25 25 50

    * - Field
      - Default
      - Description
    * - ``notation``
      - ``"dd"``
      - Notation for geological measurements (``"dd"``, ``"rhr"`` or ``"quadrant"``)
    * - ``vec2geo``
      - ``False``
      - Represent ``Vector3`` using geological notation
    * - ``ndigits``
      - ``3``
      - Rounding precision in ``__repr__``
    * - ``figsize``
      - ``(8, 6)``
      - Default figure size (width, height)
    * - ``dpi``
      - ``100``
      - Default figure DPI
    * - ``facecolor``
      - ``"white"``
      - Default figure facecolor
    * - ``stereonet``
      - ``StereonetConfig()``
      - Stereonet projection parameters
    * - ``stereonet_point``
      - ``StereonetPointConfig()``
      - Default kwargs for point markers
    * - ``stereonet_pole``
      - ``StereonetPoleConfig()``
      - Default kwargs for pole markers
    * - ``stereonet_vector``
      - ``StereonetVectorConfig()``
      - Default kwargs for vector markers
    * - ``stereonet_great_circle``
      - ``StereonetGreatCircleConfig()``
      - Default kwargs for great circles
    * - ``stereonet_arc``
      - ``StereonetArcConfig()``
      - Default kwargs for arcs
    * - ``stereonet_scatter``
      - ``StereonetScatterConfig()``
      - Default kwargs for scatter plots
    * - ``stereonet_cone``
      - ``StereonetConeConfig()``
      - Default kwargs for cones
    * - ``stereonet_pair``
      - ``StereonetPairConfig()``
      - Default kwargs for pairs
    * - ``stereonet_fault``
      - ``StereonetFaultConfig()``
      - Default kwargs for faults
    * - ``stereonet_hoeppner``
      - ``StereonetHoeppnerConfig()``
      - Default kwargs for Hoeppner plots
    * - ``stereonet_arrow``
      - ``StereonetArrowConfig()``
      - Default kwargs for arrows
    * - ``stereonet_tensor``
      - ``StereonetTensorConfig()``
      - Default kwargs for tensor plots
    * - ``stereonet_stress``
      - ``StereonetStressConfig()``
      - Default kwargs for stress plots
    * - ``stereonet_contour``
      - ``StereonetContourConfig()``
      - Default kwargs for contour plots
    * - ``roseplot``
      - ``RoseplotConfig()``
      - Roseplot global parameters
    * - ``roseplot_bar``
      - ``RoseplotBarConfig()``
      - Default kwargs for roseplot bars
    * - ``roseplot_pdf``
      - ``RoseplotPdfConfig()``
      - Default kwargs for roseplot PDF
    * - ``roseplot_muci``
      - ``RoseplotMuciConfig()``
      - Default kwargs for roseplot confidence interval
    * - ``fabricplot``
      - ``FabricplotConfig()``
      - Fabricplot global parameters
    * - ``fabricplot_point``
      - ``FabricplotPointConfig()``
      - Default kwargs for fabric plot points
    * - ``fabricplot_path``
      - ``FabricplotPathConfig()``
      - Default kwargs for fabric plot paths

StereonetConfig
^^^^^^^^^^^^^^^

.. list-table::
    :header-rows: 1
    :widths: 25 25 50

    * - Field
      - Default
      - Description
    * - ``kind``
      - ``"equal-area"``
      - Projection type (``"equal-area"`` / ``"equal-angle"``)
    * - ``overlay_position``
      - ``(0, 0, 0, 0)``
      - Overlay position as ``(x, y, z, sense)``
    * - ``rotate_data``
      - ``False``
      - Rotate data together with overlay
    * - ``minor_ticks``
      - ``None``
      - Minor tick spacing (``None`` to disable)
    * - ``major_ticks``
      - ``None``
      - Major tick spacing (``None`` to disable)
    * - ``overlay``
      - ``True``
      - Show grid overlay
    * - ``overlay_step``
      - ``15``
      - Grid step in degrees
    * - ``overlay_resolution``
      - ``181``
      - Grid resolution
    * - ``clip_pole``
      - ``15``
      - Clipped cone around poles (degrees)
    * - ``hemisphere``
      - ``"lower"``
      - Hemisphere (``"lower"`` or ``"upper"``)
    * - ``grid_type``
      - ``"gss"``
      - Contouring grid type (``"gss"`` / ``"sfs"``)
    * - ``grid_n``
      - ``3000``
      - Number of counting points in grid
    * - ``tight_layout``
      - ``False``
      - Matplotlib tight layout
    * - ``title_kws``
      - ``{}``
      - Keyword arguments for suptitle

Marker sub-configs
^^^^^^^^^^^^^^^^^^

These classes control the appearance of point-like markers on stereonet plots.
They all inherit from ``StereonetMarkerConfig``.

.. list-table::
    :header-rows: 1
    :widths: 20 14 14 14 14 14 14

    * - Field
      - ``StereonetPointConfig``
      - ``StereonetPoleConfig``
      - ``StereonetVectorConfig``
      - ``StereonetHoeppnerConfig``
      - ``FabricplotPointConfig``
      - ``FabricplotPathConfig``
    * - ``alpha``
      - ``None``
      - ``None``
      - ``None``
      - ``None``
      - ``None``
      - ``None``
    * - ``color``
      - ``None``
      - ``None``
      - ``None``
      - ``None``
      - ``None``
      - ``None``
    * - ``mec``
      - ``None``
      - ``None``
      - ``None``
      - ``None``
      - ``None``
      - ``None``
    * - ``mfc``
      - ``None``
      - ``None``
      - ``None``
      - ``None``
      - ``None``
      - ``None``
    * - ``ls``
      - ``"none"``
      - ``"none"``
      - ``"none"``
      - ``"none"``
      - ``"none"``
      - ``"-"``
    * - ``marker``
      - ``"o"``
      - ``"o"``
      - ``"o"``
      - ``"o"``
      - ``"o"``
      - ``None``
    * - ``mew``
      - ``1``
      - ``1``
      - ``2``
      - ``1``
      - ``1``
      - ``1``
    * - ``ms``
      - ``6``
      - ``6``
      - ``6``
      - ``5``
      - ``8``
      - ``6``

Line sub-configs
^^^^^^^^^^^^^^^^

These classes control the appearance of lines on stereonet plots.

.. list-table::
    :header-rows: 1
    :widths: 15 14 14 14 14 14

    * - Field
      - ``StereonetGreatCircleConfig``
      - ``StereonetArcConfig``
      - ``StereonetConeConfig``
      - ``StereonetFaultConfig``
      - ``StereonetPairConfig``
    * - ``alpha``
      - ``None``
      - ``None``
      - ``None``
      - ``None``
      - ``None``
    * - ``color``
      - ``None``
      - ``None``
      - ``None``
      - ``None``
      - ``None``
    * - ``ls``
      - ``"-"``
      - ``"-"``
      - ``"-"``
      - ``"-"``
      - ``"-"``
    * - ``lw``
      - ``1.5``
      - ``1.5``
      - ``1.5``
      - ``1.5``
      - ``1.5``
    * - ``line_marker``
      - |nbsp|
      - |nbsp|
      - |nbsp|
      - |nbsp|
      - ``"o"``

StereonetScatterConfig
^^^^^^^^^^^^^^^^^^^^^^

.. list-table::
    :header-rows: 1
    :widths: 25 25 50

    * - Field
      - Default
      - Description
    * - ``alpha``
      - ``None``
      - Transparency
    * - ``s``
      - ``None``
      - Marker size
    * - ``c``
      - ``None``
      - Marker color
    * - ``linewidths``
      - ``1.5``
      - Edge line width
    * - ``marker``
      - ``"o"``
      - Marker style
    * - ``cmap``
      - ``None``
      - Colormap
    * - ``legend``
      - ``False``
      - Show legend
    * - ``num``
      - ``"auto"``
      - Number of features shown

StereonetArrowConfig
^^^^^^^^^^^^^^^^^^^^

.. list-table::
    :header-rows: 1
    :widths: 25 25 50

    * - Field
      - Default
      - Description
    * - ``color``
      - ``None``
      - Arrow color
    * - ``width``
      - ``2``
      - Arrow width (dots)
    * - ``headwidth``
      - ``5``
      - Arrow head width
    * - ``pivot``
      - ``"mid"``
      - Pivot point (``"mid"``, ``"tail"``, etc.)
    * - ``units``
      - ``"dots"``
      - Arrow units

StereonetTensorConfig
^^^^^^^^^^^^^^^^^^^^^

.. list-table::
    :header-rows: 1
    :widths: 25 25 50

    * - Field
      - Default
      - Description
    * - ``planes``
      - ``True``
      - Plot planes
    * - ``alpha``
      - ``None``
      - Transparency
    * - ``color``
      - ``None``
      - Color
    * - ``ls``
      - ``"-"``
      - Line style
    * - ``lw``
      - ``1.5``
      - Line width
    * - ``marker``
      - ``"o"``
      - Marker style
    * - ``mew``
      - ``1``
      - Marker edge width
    * - ``ms``
      - ``9``
      - Marker size

StereonetStressConfig
^^^^^^^^^^^^^^^^^^^^^

.. list-table::
    :header-rows: 1
    :widths: 25 25 50

    * - Field
      - Default
      - Description
    * - ``alpha``
      - ``None``
      - Transparency
    * - ``color``
      - ``None``
      - Color
    * - ``ls``
      - ``"none"``
      - Line style
    * - ``marker``
      - ``"*"``
      - Marker style (star)
    * - ``mew``
      - ``1``
      - Marker edge width
    * - ``ms``
      - ``12``
      - Marker size

StereonetContourConfig
^^^^^^^^^^^^^^^^^^^^^^

.. list-table::
    :header-rows: 1
    :widths: 25 25 50

    * - Field
      - Default
      - Description
    * - ``alpha``
      - ``None``
      - Transparency
    * - ``antialiased``
      - ``True``
      - Antialiasing
    * - ``method``
      - ``"sph"``
      - Contouring method (``"sph"``, ``"kamb"``, ``"schmidt"``)
    * - ``n_max``
      - ``6``
      - Max contour level
    * - ``cmap``
      - ``"Greys"``
      - Colormap
    * - ``levels``
      - ``6``
      - Number of contour levels
    * - ``clines``
      - ``True``
      - Show contour lines
    * - ``linewidths``
      - ``1``
      - Contour line width
    * - ``linestyles``
      - ``None``
      - Contour line styles
    * - ``colorbar``
      - ``False``
      - Show colorbar
    * - ``trimzero``
      - ``True``
      - Trim zero contours
    * - ``sigma``
      - ``None``
      - Sigma value for Kamb method
    * - ``sigmanorm``
      - ``True``
      - Sigma normalization
    * - ``show_data``
      - ``False``
      - Show data points
    * - ``data_kws``
      - ``{}``
      - Keyword arguments for data points

RoseplotConfig
^^^^^^^^^^^^^^

.. list-table::
    :header-rows: 1
    :widths: 25 25 50

    * - Field
      - Default
      - Description
    * - ``bins``
      - ``36``
      - Number of bins
    * - ``density``
      - ``True``
      - Use density instead of counts
    * - ``arrowness``
      - ``0.95``
      - Arrow shape factor
    * - ``rwidth``
      - ``1``
      - Bar relative width
    * - ``scaled``
      - ``False``
      - Bins scaled by area
    * - ``kappa``
      - ``250``
      - Von Mises shape parameter
    * - ``pdf_res``
      - ``901``
      - PDF resolution
    * - ``title``
      - ``None``
      - Plot title
    * - ``grid``
      - ``True``
      - Show grid lines
    * - ``grid_kws``
      - ``{}``
      - Keyword arguments for ``Axes.grid``
    * - ``tight_layout``
      - ``False``
      - Matplotlib tight layout
    * - ``title_kws``
      - ``{}``
      - Keyword arguments for suptitle

RoseplotBarConfig / RoseplotPdfConfig
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. list-table::
    :header-rows: 1
    :widths: 25 25 50

    * - Field
      - Default
      - Description
    * - ``alpha``
      - ``None``
      - Transparency
    * - ``color``
      - ``None``
      - Color
    * - ``ec``
      - ``None``
      - Edge color
    * - ``fc``
      - ``None``
      - Face color
    * - ``ls``
      - ``"-"``
      - Line style
    * - ``lw``
      - ``1.5``
      - Line width
    * - ``legend``
      - ``False``
      - Show legend

RoseplotMuciConfig
^^^^^^^^^^^^^^^^^^

.. list-table::
    :header-rows: 1
    :widths: 25 25 50

    * - Field
      - Default
      - Description
    * - ``confidence_level``
      - ``95``
      - Confidence level (percent)
    * - ``alpha``
      - ``None``
      - Transparency
    * - ``color``
      - ``"r"``
      - Color (red)
    * - ``ls``
      - ``"-"``
      - Line style
    * - ``lw``
      - ``1.5``
      - Line width
    * - ``n_resamples``
      - ``9999``
      - Number of resamples

FabricplotConfig
^^^^^^^^^^^^^^^^

.. list-table::
    :header-rows: 1
    :widths: 25 25 50

    * - Field
      - Default
      - Description
    * - ``ticks``
      - ``True``
      - Show ticks
    * - ``n_ticks``
      - ``10``
      - Number of ticks per axis
    * - ``tick_size``
      - ``0.2``
      - Tick size
    * - ``margin``
      - ``0.05``
      - Plot margin
    * - ``grid``
      - ``True``
      - Show grid
    * - ``grid_color``
      - ``"k"``
      - Grid line color
    * - ``grid_style``
      - ``":"``
      - Grid line style (dotted)
    * - ``title``
      - ``None``
      - Plot title
    * - ``tight_layout``
      - ``False``
      - Matplotlib tight layout
    * - ``title_kws``
      - ``{}``
      - Keyword arguments for suptitle

.. |nbsp| unicode:: 0xA0
   :trim:
