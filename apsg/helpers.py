# -*- coding: utf-8 -*-

from __future__ import division
import numpy as np


def sind(x):
    return np.sin(np.deg2rad(x))


def cosd(x):
    return np.cos(np.deg2rad(x))


def tand(x):
    return np.tan(np.deg2rad(x))


def asind(x):
    return np.rad2deg(np.arcsin(x))


def acosd(x):
    return np.rad2deg(np.arccos(x))


def atand(x):
    return np.rad2deg(np.arctan(x))


def atan2d(x1, x2):
    return np.rad2deg(np.arctan2(x1, x2))


def getldd(x, y):
    return (atan2d(x, y) % 360,
            90 - 2 * asind(np.sqrt((x * x + y * y) / 2)))


def getfdd(x, y):
    return (atan2d(-x, -y) % 360,
            2 * asind(np.sqrt((x * x + y * y) / 2)))


def l2v(azi, inc):
    return np.array([np.atleast_1d(cosd(azi) * cosd(inc)),
                     np.atleast_1d(sind(azi) * cosd(inc)),
                     np.atleast_1d(sind(inc))])


def p2v(azi, inc):
    return np.array([np.atleast_1d(-cosd(azi) * sind(inc)),
                     np.atleast_1d(-sind(azi) * sind(inc)),
                     np.atleast_1d(cosd(inc))])


def v2l(u):
    n = u / np.sqrt(np.sum(u * u, axis=0))
    ix = n[2] < 0
    n.T[ix] = -n.T[ix]
    azi = atan2d(n[1], n[0]) % 360
    inc = asind(n[2])
    return azi, inc


def v2p(u):
    n = u / np.sqrt(np.sum(u * u, axis=0))
    ix = n[2] < 0
    n.T[ix] = -n.T[ix]
    azi = (atan2d(n[1], n[0]) + 180) % 360
    inc = 90 - asind(n[2])
    return azi, inc


def l2xy(azi, inc):
    r = np.sqrt(2) * sind(45 - inc / 2)
    return r * sind(azi), r * cosd(azi)


def rodrigues(k, v, theta):
    return v * cosd(theta) + np.cross(k.T, v.T).T * sind(theta) + \
        k * np.dot(k.T, v) * (1 - cosd(theta))


def angle_metric(u, v):
    return np.degrees(np.arccos(np.abs(np.dot(u, v))))

# ----------------------------------------------------------------
# Following counting routines are from Joe Kington's mplstereonet
# https://github.com/joferkington/mplstereonet


def _kamb_radius(n, sigma):
    """Radius of kernel for Kamb-style smoothing."""
    a = sigma**2 / (float(n) + sigma**2)
    return (1 - a)


def _kamb_units(n, radius):
    """Normalization function for Kamb-style counting."""
    return np.sqrt(n * radius * (1 - radius))


# All of the following kernel functions return an _unsummed_ distribution and
# a normalization factor
def _exponential_kamb(cos_dist, sigma=3):
    """Kernel function from Vollmer for exponential smoothing."""
    n = float(cos_dist.size)
    f = 2 * (1.0 + n / sigma**2)
    count = np.exp(f * (cos_dist - 1))
    units = np.sqrt(n * (f / 2.0 - 1) / f**2)
    return count, units


def _linear_inverse_kamb(cos_dist, sigma=3):
    """Kernel function from Vollmer for linear smoothing."""
    n = float(cos_dist.size)
    radius = _kamb_radius(n, sigma)
    f = 2 / (1 - radius)
    # cos_dist = cos_dist[cos_dist >= radius]
    count = (f * (cos_dist - radius))
    count[cos_dist < radius] = 0
    return count, _kamb_units(n, radius)


def _square_inverse_kamb(cos_dist, sigma=3):
    """Kernel function from Vollemer for inverse square smoothing."""
    n = float(cos_dist.size)
    radius = _kamb_radius(n, sigma)
    f = 3 / (1 - radius)**2
    # cos_dist = cos_dist[cos_dist >= radius]
    count = (f * (cos_dist - radius)**2)
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
    count = ((1 - cos_dist) <= radius)
    # To offset the count.sum() - 0.5 required for the kamb methods...
    count = 0.5 / count.size + count
    return count, cos_dist.size * radius
# ------------------------------------------------------------------
