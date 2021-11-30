import numpy as np

# ############################################################################
# Following counting routines are from Joe Kington's mplstereonet
# https://github.com/joferkington/mplstereonet
# ############################################################################


def _kamb_radius(n, sigma):
    """Radius of kernel for Kamb-style smoothing."""
    a = sigma ** 2 / (float(n) + sigma ** 2)
    return 1 - a


def _kamb_units(n, radius):
    """Normalization function for Kamb-style counting."""
    return np.sqrt(n * radius * (1 - radius))


# ############################################################################
# All of the following kernel functions return an _unsummed_ distribution and
# a normalization factor.
# ############################################################################


def _exponential_kamb(cos_dist, sigma=3):
    """Kernel function from Vollmer for exponential smoothing."""
    n = float(cos_dist.size)
    f = 2 * (1.0 + n / sigma ** 2)
    count = np.exp(f * (cos_dist - 1))
    units = np.sqrt(n * (f / 2.0 - 1) / f ** 2)
    return count, units


def _linear_inverse_kamb(cos_dist, sigma=3):
    """Kernel function from Vollmer for linear smoothing."""
    n = float(cos_dist.size)
    radius = _kamb_radius(n, sigma)
    f = 2 / (1 - radius)
    # cos_dist = cos_dist[cos_dist >= radius]
    count = f * (cos_dist - radius)
    count[cos_dist < radius] = 0
    return count, _kamb_units(n, radius)


def _square_inverse_kamb(cos_dist, sigma=3):
    """Kernel function from Vollemer for inverse square smoothing."""
    n = float(cos_dist.size)
    radius = _kamb_radius(n, sigma)
    f = 3 / (1 - radius) ** 2
    # cos_dist = cos_dist[cos_dist >= radius]
    count = f * (cos_dist - radius) ** 2
    count[cos_dist < radius] = 0
    return count, _kamb_units(n, radius)


def _kamb_count(cos_dist, sigma=3):
    """Original Kamb kernel function (raw count within radius)."""
    n = float(cos_dist.size)
    dist = _kamb_radius(n, sigma)
    # count = (cos_dist >= dist)
    count = np.array(cos_dist >= dist, dtype=float)
    return count, _kamb_units(n, dist)


def _schmidt_count(cos_dist, sigma=None):
    """Schmidt (a.k.a. 1%) counting kernel function."""
    radius = 0.01
    count = (1 - cos_dist) <= radius
    # To offset the count.sum() - 0.5 required for the kamb methods...
    count = 0.5 / count.size + count
    return count, cos_dist.size * radius
