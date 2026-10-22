# apsg.pandas

The `apsg.pandas` module bridges APSG feature sets with [pandas](https://pandas.pydata.org/).
It provides array wrappers (`FolArray`, `LinArray`, `Vec3Array`, `FaultArray`) that store
APSG features as pandas extension arrays, enabling seamless integration with DataFrames.

## Usage

Start by importing `pd` from the `apsg.pandas` submodule:

```pycon
>>> from apsg.pandas import pd
```

Create a DataFrame with numerical columns for azimuth and inclination:

```pycon
>>> df = pd.DataFrame({
...     "azi": [145, 156, 173, 142, 153],
...     "inc": [38, 42, 36, 54, 41],
... })
```

Now use the `fol` accessor to retrieve the underlying APSG feature set:

```pycon
>>> df.fol()              # FeatureSet from the fols column
>>> df.fol().watson_statistics()["k"]   # Watson concentration parameter
>>> df.fol().ortensor()   # Orientation tensor
```

The same approach works for linear features using `lin` accessor:

```pycon
>>> df.lin()
>>> df.lin().fit("bingham")
```

If your columns have different names as default `azi` and `inc`, use `set_columns` method:

```pycon
>>> df.lin.set_columns(azi="trend", inc="plunge")()
>>> df.lin.set_columns(azi="trend", inc="plunge").plot()
```

## Reference

### Accessors

::: apsg.pandas.VecAccessor

::: apsg.pandas.Vec2Accessor

::: apsg.pandas.DirAccessor

::: apsg.pandas.LinAccessor

::: apsg.pandas.FolAccessor

::: apsg.pandas.FaultAccessor

### Extension arrays

::: apsg.pandas.Vec3Array

::: apsg.pandas.Vec2Array

::: apsg.pandas.DirArray

::: apsg.pandas.LinArray

::: apsg.pandas.FolArray

::: apsg.pandas.FaultArray

### Groupby functions

Available as `apsg.pandas.gbf`.

::: apsg.pandas._gbfunctions
    options:
      show_root_heading: false
