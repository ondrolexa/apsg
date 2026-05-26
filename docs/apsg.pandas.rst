=============
pandas module
=============

The :mod:`apsg.pandas` module bridges APSG feature sets with `pandas <https://pandas.pydata.org/>`_.
It provides array wrappers (``FolArray``, ``LinArray``, ``Vec3Array``, ``FaultArray``) that store
APSG features as pandas extension arrays, enabling seamless integration with DataFrames.

Usage
-----

Start by importing ``pd`` from the :mod:`apsg.pandas` submodule and feature aliases from :mod:`apsg`::

    >>> from apsg import fol, lin
    >>> from apsg.pandas import pd

Create a DataFrame with numerical columns for azimuth and inclination::

    >>> df = pd.DataFrame({
    ...     "azi": [145, 156, 173, 142, 153],
    ...     "inc": [38, 42, 36, 54, 41],
    ... })

Convert numerical columns into an APSG feature column using the ``apsg`` accessor::

    >>> df = df.apsg.create_fols(columns=["azi", "inc"])

Now use column-specific accessors for analysis and plotting::

    >>> df.fol.G            # FeatureSet from the fol column
    >>> df.fol.fisher_k()   # Fisher precision parameter
    >>> df.fol.ortensor()   # Orientation tensor

The same approach works for linear features using ``create_lins``::

    >>> df = df.apsg.create_lins(columns=["azi", "inc"], name="lins")
    >>> df.lin.G
    >>> df.lin.fisher_k()

.. automodule:: apsg.pandas
    :autosummary:
    :members:
    :show-inheritance:
    :autosummary-no-nesting:
