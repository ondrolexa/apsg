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

Now use the ``G`` Series accessor to retrieve the underlying APSG feature set::

    >>> df.fols.G()              # FeatureSet from the fols column
    >>> df.fols.G().fisher_statistics()["k"]   # Fisher precision parameter
    >>> df.fols.G().ortensor()   # Orientation tensor

The same approach works for linear features using ``create_lins``::

    >>> df = df.apsg.create_lins(columns=["azi", "inc"], name="lins")
    >>> df.lins.G()
    >>> df.lins.G().fisher_statistics()["k"]

Column names that are not valid Python identifiers require bracket access::

    >>> df = df.apsg.create_lins(columns=["azi", "inc"], name="my lins")
    >>> df["my lins"].G()

.. automodule:: apsg.pandas
    :autosummary:
    :members:
    :show-inheritance:
    :autosummary-no-nesting:
