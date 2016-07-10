# -*- coding: utf-8 -*-

from __future__ import division
import numpy as np

# lambdas
sind = lambda x: np.sin(np.deg2rad(x))
cosd = lambda x: np.cos(np.deg2rad(x))
tand = lambda x: np.tan(np.deg2rad(x))
asind = lambda x: np.rad2deg(np.arcsin(x))
acosd = lambda x: np.rad2deg(np.arccos(x))
atand = lambda x: np.rad2deg(np.arctan(x))
atan2d = lambda x1, x2: np.rad2deg(np.arctan2(x1, x2))

getldd = lambda x, y: (atan2d(x, y) % 360,
                       90 - 2 * asind(np.sqrt((x * x + y * y) / 2)))
getfdd = lambda x, y: (atan2d(-x, -y) % 360,
                       2 * asind(np.sqrt((x * x + y * y) / 2)))

l2v = lambda azi, inc: np.array([np.atleast_1d(cosd(azi) * cosd(inc)),
                                 np.atleast_1d(sind(azi) * cosd(inc)),
                                 np.atleast_1d(sind(inc))])

p2v = lambda azi, inc: np.array([np.atleast_1d(-cosd(azi) * sind(inc)),
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
