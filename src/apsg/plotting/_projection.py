import numpy as np

from apsg.helpers import sind, cosd, tand, asind, acosd, atand, atan2d, sqrt2
from apsg.base_classes import Vec3, Axial, Matrix3
from apsg.geodata import Lin, Fol, Pair
from apsg.tensors import DefGrad


class Projection:
    def __init__(self, **kwargs):
        self.rotate_data = kwargs.get("rotate_data", False)
        self.grid_position = kwargs.get("grid_position", Pair())
        self.polehole = kwargs.get("polehole", 20)
        self.hemisphere = kwargs.get("hemisphere", "lower")
        self.gridstep = kwargs.get("gridstep", 15)  # grid step
        self.resolution = kwargs.get("resolution", 361)  # number of grid lines points
        self.R = np.array(DefGrad.from_pair(self.grid_position))
        self.Ri = np.linalg.inv(self.R)

    def project_grid(self, x, y, z, clip_polehole=False):
        if clip_polehole:
            polehole = [
                Lin(0, 0).angle(Vec3(xx, yy, zz)) < self.polehole
                for xx, yy, zz in zip(x, y, z)
            ]
        else:
            polehole = []
        x, y, z = self.R.dot((x, y, z))
        X, Y = self.project(x, y, z)
        inside = X * X + Y * Y < 1.0
        X[polehole] = np.nan
        Y[polehole] = np.nan
        return X[inside], Y[inside]

    def project_data(self, x, y, z):
        if self.rotate_data:
            x, y, z = self.R.dot((x, y, z))
        X, Y = self.project(x, y, z)
        inside = X * X + Y * Y < 1.0
        return X[inside], Y[inside]

    def inverse_data(self, X, Y):
        if X * X + Y * Y > 1.0:
            return None
        x, y, z = self.inverse(X, Y)
        if self.rotate_data:
            x, y, z = self.Ri.dot((x, y, z))
        return x, y, z

    def get_grid_overlay(self):
        angles_gc = np.linspace(-90 + 1e-7, 90 - 1e-7, int(self.resolution / 2))
        # angles_lon = np.linspace(-180 + 1e-7, 180 - 1e-7, self.resolution)
        angles_sc = np.linspace(-180 + 1e-7, 180 - 1e-7, self.resolution)
        # lats
        lat_e, lat_w = {}, {}
        for dip in range(self.gridstep, 90, self.gridstep):
            f = Fol(90, dip)
            if f.transform(self.R).angle(Fol(0, 0)) > 1e-6:
                fdv = f.transform(self.R).dipvec().transform(self.Ri)
                X, Y = self.project_grid(
                    *np.array([fdv.rotate(f, a) for a in angles_gc]).T,
                    clip_polehole=True,
                )
                lat_e[dip] = dict(x=X.tolist(), y=Y.tolist())
            f = Fol(270, dip)
            if f.transform(self.R).angle(Fol(0, 0)) > 1e-6:
                fdv = f.transform(self.R).dipvec().transform(self.Ri)
                X, Y = self.project_grid(
                    *np.array([fdv.rotate(f, a) for a in angles_gc]).T,
                    clip_polehole=True,
                )
                lat_w[dip] = dict(x=X.tolist(), y=Y.tolist())

        # lons
        lon_n, lon_s = {}, {}
        for dip in range(self.gridstep, 90, self.gridstep):
            if dip >= self.polehole:
                l = Lin(0, dip)
                X, Y = self.project_grid(
                    *np.array([l.rotate(Lin(0, 0), a) for a in angles_sc]).T,
                    clip_polehole=True,
                )
                lon_n[dip] = dict(x=X.tolist(), y=Y.tolist())
                l = Lin(180, dip)
                X, Y = self.project_grid(
                    *np.array([l.rotate(Lin(180, 0), a) for a in angles_sc]).T,
                    clip_polehole=True,
                )
                lon_s[dip] = dict(x=X.tolist(), y=Y.tolist())

        # pole holes rims
        if self.polehole > 0:
            l = Lin(0, self.polehole)
            X, Y = self.project_grid(
                *np.array([l.rotate(Lin(0, 0), a) for a in angles_sc]).T
            )
            polehole_n = dict(x=X.tolist(), y=Y.tolist())
            l = Lin(180, self.polehole)
            X, Y = self.project_grid(
                *np.array([l.rotate(Lin(180, 0), a) for a in angles_sc]).T
            )
            polehole_s = dict(x=X.tolist(), y=Y.tolist())
        else:
            polehole_n, polehole_s = {}, {}

        # Principal cross N-S
        f = Fol(90, 90)
        if f.transform(self.R).angle(Fol(0, 0)) > 1e-6:
            fdv = f.transform(self.R).dipvec().transform(self.Ri)
            X, Y = self.project_grid(*np.array([fdv.rotate(f, a) for a in angles_gc]).T)
            main_ns = dict(x=X.tolist(), y=Y.tolist())
        else:
            main_ns = {}
        # Principal cross E-W
        f = Fol(0, 90)
        if f.transform(self.R).angle(Fol(0, 0)) > 1e-6:
            fdv = f.transform(self.R).dipvec().transform(self.Ri)
            X, Y = self.project_grid(*np.array([fdv.rotate(f, a) for a in angles_gc]).T)
            main_ew = dict(x=X.tolist(), y=Y.tolist())
        else:
            main_ew = {}
        # Principal horizontal
        f = Fol(0, 0)
        if f.transform(self.R).angle(Fol(0, 0)) > 1e-6:
            fdv = f.transform(self.R).dipvec().transform(self.Ri)
            X, Y = self.project_grid(*np.array([fdv.rotate(f, a) for a in angles_gc]).T)
            main_h = dict(x=X.tolist(), y=Y.tolist())
        else:
            main_h = {}

        return dict(
            lat_e=lat_e,
            lat_w=lat_w,
            lon_n=lon_n,
            lon_s=lon_s,
            polehole_n=polehole_n,
            polehole_s=polehole_s,
            main_ns=main_ns,
            main_ew=main_ew,
            main_h=main_h,
        )


class EqualAreaProj(Projection):
    name = "Equal-area"
    netname = "Schmidt net"

    def project(self, x, y, z):
        z[np.isclose(1 + z, np.zeros_like(z))] = 1e-7 - 1
        sqz = np.sqrt(1 / (1 + z))
        return y * sqz, x * sqz

    def inverse(self, X, Y):
        X, Y = X * sqrt2, Y * sqrt2
        x = np.sqrt(1 - (X * X + Y * Y) / 4.0) * Y
        y = np.sqrt(1 - (X * X + Y * Y) / 4.0) * X
        z = 1.0 - (X * X + Y * Y) / 2
        return x, y, z


class EqualAngleProj(Projection):
    name = "Equal-angle"
    netname = "Wulff net"

    def project(self, x, y, z):
        z[np.isclose(1 + z, np.zeros_like(z))] = 1e-7 - 1
        return y / (1 + z), x / (1 + z)

    def inverse(self, X, Y):
        x = 2.0 * Y / (1.0 + X * X + Y * Y)
        y = 2.0 * X / (1.0 + X * X + Y * Y)
        z = (1.0 - X * X + Y * Y) / (1.0 + X * X + Y * Y)
        return x, y, z


##############################################
#   THIS IS DEPRECATED AND WILL BE REMOVED   #
##############################################

### STEREONET TRANSFORMATIONS ###


def getldd(x, y):
    return (atan2d(x, y) % 360, 90 - 2 * asind(math.sqrt((x * x + y * y) / 2)))


def getfdd(x, y):
    return (atan2d(-x, -y) % 360, 2 * asind(math.sqrt((x * x + y * y) / 2)))


def l2v(azi, inc):
    return math.array(
        [
            math.atleast_1d(cosd(azi) * cosd(inc)),
            math.atleast_1d(sind(azi) * cosd(inc)),
            math.atleast_1d(sind(inc)),
        ]
    )


def p2v(azi, inc):
    return math.array(
        [
            math.atleast_1d(-cosd(azi) * sind(inc)),
            math.atleast_1d(-sind(azi) * sind(inc)),
            math.atleast_1d(cosd(inc)),
        ]
    )


def v2l(u):
    n = u / math.sqrt(math.sum(u * u, axis=0))
    ix = n[2] < 0
    n.T[ix] = -n.T[ix]
    azi = atan2d(n[1], n[0]) % 360
    inc = asind(n[2])
    return azi, inc


def v2p(u):
    n = u / math.sqrt(math.sum(u * u, axis=0))
    ix = n[2] < 0
    n.T[ix] = -n.T[ix]
    azi = (atan2d(n[1], n[0]) + 180) % 360
    inc = 90 - asind(n[2])
    return azi, inc


def l2xy(azi, inc):
    r = math.sqrt(2) * sind(45 - inc / 2)
    return r * sind(azi), r * cosd(azi)


def rodrigues(k, v, theta):
    return (
        v * cosd(theta)
        + math.cross(k.T, v.T).T * sind(theta)
        + k * math.dot(k.T, v) * (1 - cosd(theta))
    )


def angle_metric(u, v):
    return math.degrees(math.arccos(math.abs(math.dot(u, v))))
