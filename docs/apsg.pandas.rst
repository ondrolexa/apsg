=============
pandas module
=============

The :mod:`apsg.pandas` module bridges APSG feature sets with `pandas <https://pandas.pydata.org/>`_.
It provides array wrappers (``FolArray``, ``LinArray``, ``Vec3Array``, ``FaultArray``) that store
APSG features as pandas extension arrays, enabling seamless integration with DataFrames.

Usage
-----

Start by importing ``pd`` from the :mod:`apsg.pandas` submodule::

    >>> from apsg.pandas import pd

Create a DataFrame with numerical columns for azimuth and inclination::

    >>> df = pd.DataFrame({
    ...     "azi": [145, 156, 173, 142, 153],
    ...     "inc": [38, 42, 36, 54, 41],
    ... })

Now use the ``fol`` accessor to retrieve the underlying APSG feature set::

    >>> df.fol()              # FeatureSet from the fols column
    >>> df.fol().fisher_statistics()["k"]   # Fisher precision parameter
    >>> df.fol().ortensor()   # Orientation tensor

The same approach works for linear features using ``lin`` accessor::

    >>> df.lin()
    >>> df.lin().fisher_statistics()

If your columns have different names as default `azi` and `inc`, use `set_columns` method::

    >>> df.lin.set_columns(azi="trend", inc="plunge")()
    >>> df.lin.set_columns(azi="trend", inc="plunge").plot()

.. automodule:: apsg.pandas
    :autosummary:
    :members:
    :show-inheritance:
    :autosummary-no-nesting:
