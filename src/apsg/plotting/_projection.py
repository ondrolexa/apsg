import numpy as np

from apsg.helpers._math import sqrt2
from apsg.math._vector import Vector3
from apsg.feature._geodata import Lineation, Foliation, Pair
from apsg.feature._tensor3 import DeformationGradient3


class Projection:
    def __init__(self, **kwargs):
        self.rotate_data = kwargs.get("rotate_data", False)
        self.overlay_position = Pair(kwargs.get("overlay_position", (0, 0, 0, 0)))
        self.clip_pole = kwargs.get("clip_pole", 15)
        self.hemisphere = kwargs.get("hemisphere", "lower")
        self.overlay_step = kwargs.get("overlay_step", 15)  # grid step
        self.overlay_resolution = kwargs.get(
            "overlay_resolution", 361
        )  # number of grid lines points
        self.overlay_cross_size = kwargs.get("overlay_cross_size", 3)
        self.R = np.array(DeformationGradient3.from_pair(self.overlay_position))
        self.Ri = np.linalg.inv(self.R)

    def project_data(self, x, y, z, clip_inside=True):
        if self.rotate_data:
            x, y, z = self.R.dot((x, y, z))
        if self.hemisphere == "upper":
            X, Y = self._project(-x, -y, -z)
            if clip_inside:
                outside = X * X + Y * Y > 1.0
                X[outside] = np.nan
                Y[outside] = np.nan
            return -X, -Y
        else:
            X, Y = self._project(x, y, z)
            if clip_inside:
                outside = X * X + Y * Y > 1.0
                X[outside] = np.nan
                Y[outside] = np.nan
            return X, Y

    def project_data_antipodal(self, x, y, z, clip_inside=True):
        if self.rotate_data:
            x, y, z = self.R.dot((x, y, z))
        X1, Y1 = self._project(x, y, z)
        X2, Y2 = self._project(-x, -y, -z)
        if clip_inside:
            outside1 = X1 * X1 + Y1 * Y1 > 1.0
            outside2 = X2 * X2 + Y2 * Y2 > 1.0
            X1[outside1] = np.nan
            Y1[outside1] = np.nan
            X2[outside2] = np.nan
            Y2[outside2] = np.nan
        return X1, Y1, -X2, -Y2

    def inverse_data(self, X, Y):
        if X * X + Y * Y > 1.0:
            return None
        x, y, z = self._inverse(X, Y)
        if self.rotate_data:
            x, y, z = self.Ri.dot((x, y, z))
        return x, y, z

    def project_overlay(self, x, y, z):
        x, y, z = self.R.dot((x, y, z))
        X, Y = self._project(x, y, z)
        outside = X * X + Y * Y >= 1.0
        X[outside] = np.nan
        Y[outside] = np.nan
        return X, Y

    def get_grid_overlay(self):
        angles_gc = np.linspace(-90 + 1e-7, 90 - 1e-7, int(self.overlay_resolution / 2))
        angles_gc_clipped = np.linspace(
            -90 + self.clip_pole + 1e-7,
            90 - self.clip_pole - 1e-7,
            int(self.overlay_resolution / 2),
        )
        angles_sc = np.linspace(-180 + 1e-7, 180 - 1e-7, self.overlay_resolution)
        angles_cross = np.linspace(
            -self.overlay_cross_size,
            self.overlay_cross_size,
            2 * self.overlay_cross_size,
        )

        # latitude grid
        lat_e, lat_w = {}, {}
        for dip in range(self.overlay_step, 90, self.overlay_step):
            f = Foliation(90, dip)
            if f.transform(self.R).angle(Foliation(0, 0)) > 0.1:
                fdv = f.dipvec()
                X, Y = self.project_overlay(
                    *np.array(
                        [np.asarray(fdv.rotate(f, a)) for a in angles_gc_clipped]
                    ).T
                )
                lat_e[dip] = dict(x=X, y=Y)
                X, Y = self.project_overlay(
                    *np.array(
                        [-np.asarray(fdv.rotate(f, a)) for a in angles_gc_clipped]
                    ).T
                )
                lat_e[-dip] = dict(x=X, y=Y)
            f = Foliation(270, dip)
            if f.transform(self.R).angle(Foliation(0, 0)) > 0.1:
                fdv = f.dipvec()
                X, Y = self.project_overlay(
                    *np.array(
                        [np.asarray(fdv.rotate(f, a)) for a in angles_gc_clipped]
                    ).T
                )
                lat_w[dip] = dict(x=X, y=Y)
                X, Y = self.project_overlay(
                    *np.array(
                        [-np.asarray(fdv.rotate(f, a)) for a in angles_gc_clipped]
                    ).T
                )
                lat_w[-dip] = dict(x=X, y=Y)

        # longitude grid
        lon_n, lon_s = {}, {}
        for dip in range(self.overlay_step, 90, self.overlay_step):
            if dip >= self.clip_pole:
                lon = Vector3(0, dip)
                X, Y = self.project_overlay(
                    *np.array(
                        [np.asarray(lon.rotate(Lineation(0, 0), a)) for a in angles_sc]
                    ).T
                )
                lon_n[dip] = dict(x=X, y=Y)
                lon = Vector3(180, dip)
                X, Y = self.project_overlay(
                    *np.array(
                        [
                            np.asarray(lon.rotate(Lineation(180, 0), a))
                            for a in angles_sc
                        ]
                    ).T
                )
                lon_s[dip] = dict(x=X, y=Y)

        # pole holes rims
        if self.clip_pole > 0:
            lon = Vector3(0, self.clip_pole)
            X, Y = self.project_overlay(
                *np.array(
                    [np.asarray(lon.rotate(Vector3(0, 0), a)) for a in angles_sc]
                ).T
            )
            polehole_n = dict(x=X, y=Y)
            lon = Vector3(180, self.clip_pole)
            X, Y = self.project_overlay(
                *np.array(
                    [np.asarray(lon.rotate(Vector3(180, 0), a)) for a in angles_sc]
                ).T
            )
            polehole_s = dict(x=X, y=Y)
        else:
            polehole_n, polehole_s = {}, {}

        # Principal axis X
        X = Vector3(1, 0, 0)
        X1, Y1 = self.project_overlay(
            *np.array(
                [np.asarray(X.rotate(Vector3(0, 1, 0), a)) for a in angles_cross]
            ).T
        )
        X2, Y2 = self.project_overlay(
            *np.array(
                [np.asarray(X.rotate(Vector3(0, 0, 1), a)) for a in angles_cross]
            ).T
        )
        X3, Y3 = self.project_overlay(
            *np.array(
                [-np.asarray(X.rotate(Vector3(0, 1, 0), a)) for a in angles_cross]
            ).T
        )
        X4, Y4 = self.project_overlay(
            *np.array(
                [-np.asarray(X.rotate(Vector3(0, 0, 1), a)) for a in angles_cross]
            ).T
        )
        main_x = dict(
            x=np.hstack((X1, np.nan, X2, np.nan, X3, np.nan, X4)),
            y=np.hstack((Y1, np.nan, Y2, np.nan, Y3, np.nan, Y4)),
        )
        # Principal axis Y
        Y = Vector3(0, 1, 0).transform(self.R).lower().transform(self.Ri)
        X1, Y1 = self.project_overlay(
            *np.array(
                [np.asarray(Y.rotate(Vector3(1, 0, 0), a)) for a in angles_cross]
            ).T
        )
        X2, Y2 = self.project_overlay(
            *np.array(
                [np.asarray(Y.rotate(Vector3(0, 0, 1), a)) for a in angles_cross]
            ).T
        )
        X3, Y3 = self.project_overlay(
            *np.array(
                [-np.asarray(Y.rotate(Vector3(1, 0, 0), a)) for a in angles_cross]
            ).T
        )
        X4, Y4 = self.project_overlay(
            *np.array(
                [-np.asarray(Y.rotate(Vector3(0, 0, 1), a)) for a in angles_cross]
            ).T
        )
        main_y = dict(
            x=np.hstack((X1, np.nan, X2, np.nan, X3, np.nan, X4)),
            y=np.hstack((Y1, np.nan, Y2, np.nan, Y3, np.nan, Y4)),
        )
        # Principal axis Z
        Z = Vector3(0, 0, 1).transform(self.R).lower().transform(self.Ri)
        X1, Y1 = self.project_overlay(
            *np.array(
                [np.asarray(Z.rotate(Vector3(1, 0, 0), a)) for a in angles_cross]
            ).T
        )
        X2, Y2 = self.project_overlay(
            *np.array(
                [np.asarray(Z.rotate(Vector3(0, 1, 0), a)) for a in angles_cross]
            ).T
        )
        X3, Y3 = self.project_overlay(
            *np.array(
                [-np.asarray(Z.rotate(Vector3(1, 0, 0), a)) for a in angles_cross]
            ).T
        )
        X4, Y4 = self.project_overlay(
            *np.array(
                [-np.asarray(Z.rotate(Vector3(0, 1, 0), a)) for a in angles_cross]
            ).T
        )
        main_z = dict(
            x=np.hstack((X1, np.nan, X2, np.nan, X3, np.nan, X4)),
            y=np.hstack((Y1, np.nan, Y2, np.nan, Y3, np.nan, Y4)),
        )

        # Principal plane XZ
        f = Foliation(90, 90)
        if f.transform(self.R).angle(Foliation(0, 0)) > 0.1:
            fdv = f.transform(self.R).dipvec().transform(self.Ri)
            X, Y = self.project_overlay(
                *np.array([np.asarray(fdv.rotate(f, a)) for a in angles_gc]).T
            )
            main_xz = dict(x=X, y=Y)
        else:
            main_xz = {}
        # Principal plane YZ
        f = Foliation(0, 90)
        if f.transform(self.R).angle(Foliation(0, 0)) > 0.1:
            fdv = f.transform(self.R).dipvec().transform(self.Ri)
            X, Y = self.project_overlay(
                *np.array([np.asarray(fdv.rotate(f, a)) for a in angles_gc]).T
            )
            main_yz = dict(x=X, y=Y)
        else:
            main_yz = {}
        # Principal plane XY
        f = Foliation(0, 0)
        if f.transform(self.R).angle(Foliation(0, 0)) > 0.1:
            fdv = f.transform(self.R).dipvec().transform(self.Ri)
            X, Y = self.project_overlay(
                *np.array([np.asarray(fdv.rotate(f, a)) for a in angles_gc]).T
            )
            main_xy = dict(x=X, y=Y)
        else:
            main_xy = {}

        return dict(
            lat_e=lat_e,
            lat_w=lat_w,
            lon_n=lon_n,
            lon_s=lon_s,
            polehole_n=polehole_n,
            polehole_s=polehole_s,
            main_xz=main_xz,
            main_yz=main_yz,
            main_xy=main_xy,
            main_x=main_x,
            main_y=main_y,
            main_z=main_z,
        )


class EqualAreaProj(Projection):
    name = "Equal-area"
    netname = "Schmidt net"

    def _project(self, x, y, z):
        # normalize
        d = np.sqrt(x * x + y * y + z * z)
        if any(d == 0):
            return np.nan, np.nan
        else:
            x, y, z = x / d, y / d, z / d
            z[np.isclose(1 + z, np.zeros_like(z))] = 1e-6 - 1
            sqz = np.sqrt(1 / (1 + z))
            return y * sqz, x * sqz

    def _inverse(self, X, Y):
        X, Y = X * sqrt2, Y * sqrt2
        x = np.sqrt(1 - (X * X + Y * Y) / 4.0) * Y
        y = np.sqrt(1 - (X * X + Y * Y) / 4.0) * X
        z = 1.0 - (X * X + Y * Y) / 2
        return x, y, z


class EqualAngleProj(Projection):
    name = "Equal-angle"
    netname = "Wulff net"

    def _project(self, x, y, z):
        # normalize
        d = np.sqrt(x * x + y * y + z * z)
        if any(d == 0):
            return np.nan, np.nan
        else:
            z[np.isclose(1 + z, np.zeros_like(z))] = 1e-7 - 1
            return y / (1 + z), x / (1 + z)

    def _inverse(self, X, Y):
        x = 2.0 * Y / (1.0 + X * X + Y * Y)
        y = 2.0 * X / (1.0 + X * X + Y * Y)
        z = (1.0 - X * X + Y * Y) / (1.0 + X * X + Y * Y)
        return x, y, z
