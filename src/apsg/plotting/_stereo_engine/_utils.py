"""Small helpers shared by ``_axes.py`` and ``_stereogrid.py``.

Kept in their own module (rather than living in ``_axes.py``) so
``_stereogrid.py`` can use them without an import cycle (``_axes.py``
imports ``StereoGrid`` from ``_stereogrid.py``).
"""

import numpy as np


def _as_vectors(xyz):
    """Normalize array-like input (shape (3,) or (N, 3)) to a unit-vector
    (N, 3) array. Anything numpy can coerce with ``asarray`` works, which
    includes apsg's ``Vector3``/``Vector3Set`` (they implement
    ``__array__``) without this module depending on apsg."""
    v = np.atleast_2d(np.asarray(xyz, dtype=float))
    norm = np.linalg.norm(v, axis=-1, keepdims=True)
    return v / norm
