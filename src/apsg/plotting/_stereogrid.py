import numpy as np
import matplotlib.tri as tri
from matplotlib.patches import Circle
import matplotlib.pyplot as plt

from apsg.config import apsg_conf
from apsg.feature._geodata import Lineation
from apsg.feature._container import Vector3Set
from apsg.plotting._projection import EqualAreaProj, EqualAngleProj


class StereoGrid:
    """
    The class to store values with associated uniformly positions.

    ``StereoGrid`` is used to calculate continous functions on sphere e.g. density
    distribution.

    Keyword Args:
        kind (str): Equal area ("equal-area", "schmidt" or "earea") or equal angle
            ("equal-angle", "wulff" or "eangle") projection. Default is "equal-area"
        hemisphere (str): "lower" or "upper". Default is "lower"
        overlay_position (tuple or Pair): Position of overlay X, Y, Z given by Pair.
            X is direction of linear element, Z is normal to planar.
            Default is (0, 0, 0, 0)
        rotate_data (bool): Whether data should be rotated together with overlay.
            Default False
        minor_ticks (None or float): Default None
        major_ticks (None or float): Default None
        overlay (bool): Whether to show overlay. Default is True
        overlay_step (float): Grid step of overlay. Default 15
        overlay_resolution (float): Resolution of overlay. Default 181
        clip_pole (float): Clipped cone around poles. Default 15
        grid_type (str): Type of contouring grid "gss" or "sfs". Default "gss"
        grid_n (int): Number of counting points in grid. Default 3000

    Note: Euclidean norms are used as weights. Normalize data if you dont want to use
    weigths.

    """

    def __init__(self, **kwargs):
        # parse options
        self.grid_n = kwargs.get("grid_n", 2000)
        # grid type
        if kwargs.get("grid_type", "gss") == "gss":
            self.grid = Vector3Set.uniform_gss(n=self.grid_n)
        else:
            self.grid = Vector3Set.uniform_sfs(n=self.grid_n)
        # projection
        kind = str(kwargs.get("kind", "equal-area")).lower()
        if kind in ["equal-area", "schmidt", "earea"]:
            self.proj = EqualAreaProj(**kwargs)
        elif kind in ["equal-angle", "wulff", "eangle"]:
            self.proj = EqualAngleProj(**kwargs)
        else:
            raise TypeError("Only 'Equal-area' and 'Equal-angle' implemented")
        # initial values
        self.values = np.zeros(self.grid_n, dtype=float)
        self.calculated = False
        self.density_params = None
        self.features = None

    def __repr__(self):
        if self.calculated:
            info = (
                f"\nMaximum: {self.max():.4f} at {self.max_at()}"
                + f"\nMinimum: {self.min():.4f} at {self.min_at()}"
            )
        else:
            info = ""
        return f"StereoGrid {self.proj.__class__.__name__} {self.grid_n} points." + info

    def min(self):
        """Returns minimum value of the grid"""
        return self.values.min()

    def max(self):
        """Returns maximum value of the grid"""
        return self.values.max()

    def min_at(self):
        """Returns position of minimum value of the grid as ``Lineation``"""
        return Lineation(self.grid[self.values.argmin()])

    def max_at(self):
        """Returns position of maximum value of the grid as ``Lineation``"""
        return Lineation(self.grid[self.values.argmax()])

    def calculate_density(self, features, **kwargs):
        """Calculate density distribution of vectors from ``FeatureSet`` object.

        The modified Kamb contouring technique with exponential smoothing is used.

        Args:
            sigma (float): if none sigma is calculated automatically.
              Default None
            sigmanorm (bool): If True counting is normalized to sigma
              multiples. Default True
            trimzero: if True, zero contour is not drawn. Default True

        """
        # parse options
        sigma = kwargs.get("sigma", None)
        self.features = np.atleast_2d(features)
        n = len(self.features)
        if sigma is None:
            # k = estimate_k(features)
            # sigma = np.sqrt(2 * n / (k - 2))
            # Totally empirical as estimate_k is problematic
            if n < 10:
                sigma = 3
            else:
                sigma = np.sqrt(2 * n / (np.log(n) - 2)) / 3
        k = 2 * (1.0 + n / sigma**2)
        # method = kwargs.get("method", "exp_kamb")
        trim = kwargs.get("trimzero", True)
        sigmanorm = kwargs.get("sigmanorm", True)
        # do calc
        scale = np.sqrt(n * (k / 2.0 - 1) / k**2)
        cnt = np.exp(k * (np.abs(np.dot(self.grid, self.features.T)) - 1))
        self.values = cnt.sum(axis=1) / scale
        if sigmanorm:
            self.values /= sigma
        self.values[self.values < 0] = 0
        if trim:
            self.values[self.values == 0] = np.finfo(float).tiny
        self.calculated = True
        self.density_params = k, scale, sigma, sigmanorm

    def density_lookup(self, v):
        """
        Calculate density distribution value at position given by vector

        Note: you need to calculate density before using this method

        Args:
            v: Vector3 like object

        Keyword Args:
            p (int): power. Default 2
        """
        if self.density_params is not None:
            k, scale, sigma, sigmanorm = self.density_params
            cnt = np.exp(k * (np.abs(np.dot(v.normalized(), self.features.T)) - 1))
            val = cnt.sum() / scale
            if sigmanorm:
                val /= sigma
            return val
        else:
            raise ValueError(
                "No density distribution calculated. Use calculate_density method."
            )

    def apply_func(self, func, *args, **kwargs):
        """Calculate values of user-defined function on sphere.

        Function must accept Vector3 like (or 3 elements array)
        as first argument and return scalar value.

        Args:
            func (function): function used to calculate values
            *args: passed to function func as args
            **kwargs: passed to function func as kwargs

        """
        for i in range(self.grid_n):
            self.values[i] = func(self.grid[i], *args, **kwargs)
        self.calculated = True

    def contourf(self, *args, **kwargs):
        """
        Draw filled contours of values using tricontourf.

        Keyword Args:
            levels (int or list): number or values of contours. Default 6
            cmap: matplotlib colormap used for filled contours. Default "Greys"
            colorbar (bool): Show colorbar. Default False
            alpha (float): transparency. Default None
            antialiased (bool): Default True
        """
        colorbar = kwargs.get("colorbar", False)
        parsed = {}
        parsed["alpha"] = kwargs.get("alpha", 1)
        parsed["antialiased"] = kwargs.get("antialiased", True)
        parsed["cmap"] = kwargs.get("cmap", "Greys")
        parsed["levels"] = kwargs.get("levels", 6)

        fig, ax = plt.subplots(figsize=apsg_conf["figsize"])
        ax.set_aspect(1)
        ax.set_axis_off()

        # Projection circle frame
        theta = np.linspace(0, 2 * np.pi, 200)
        ax.plot(np.cos(theta), np.sin(theta), "k", lw=2)
        # add clipping circle
        primitive = Circle(
            (0, 0),
            radius=1,
            edgecolor="black",
            fill=False,
            clip_box="None",
            label="_nolegend_",
        )
        ax.add_patch(primitive)
        dcgrid = np.asarray(self.grid).T
        X, Y = self.proj.project_data(*dcgrid, clip_inside=False)
        cf = ax.tricontourf(X, Y, self.values, **parsed)
        for collection in cf.collections:
            collection.set_clip_path(primitive)
        ax.set_xlim(-1.05, 1.05)
        ax.set_ylim(-1.05, 1.05)
        if colorbar:
            fig.colorbar(cf, ax=ax, shrink=0.6)
        plt.show()

    def contour(self, *args, **kwargs):
        """
        Draw contour lines of values using tricontour.

        Keyword Args:
            levels (int or list): number or values of contours. Default 6
            cmap: matplotlib colormap used for filled contours. Default "Greys"
            colorbar (bool): Show colorbar. Default False
            alpha (float): transparency. Default None
            antialiased (bool): Default True
            linewidths (float): contour lines width
            linestyles (str): contour lines style
        """
        colorbar = kwargs.get("colorbar", False)
        parsed = {}
        parsed["alpha"] = kwargs.get("alpha", 1)
        parsed["antialiased"] = kwargs.get("antialiased", True)
        parsed["cmap"] = kwargs.get("cmap", "Greys")
        parsed["linewidths"] = kwargs.get("linewidths", 1)
        parsed["linestyles"] = kwargs.get("linestyles", "-")
        parsed["levels"] = kwargs.get("levels", 6)

        fig, ax = plt.subplots(figsize=apsg_conf["figsize"])
        ax.set_aspect(1)
        ax.set_axis_off()

        # Projection circle frame
        theta = np.linspace(0, 2 * np.pi, 200)
        ax.plot(np.cos(theta), np.sin(theta), "k", lw=2)
        # add clipping circle
        primitive = Circle(
            (0, 0),
            radius=1,
            edgecolor="black",
            fill=False,
            clip_box="None",
            label="_nolegend_",
        )
        ax.add_patch(primitive)
        dcgrid = np.asarray(self.grid).T
        X, Y = self.proj.project_data(*dcgrid, clip_inside=False)
        cf = ax.tricontour(X, Y, self.values, **parsed)
        for collection in cf.collections:
            collection.set_clip_path(primitive)
        if colorbar:
            fig.colorbar(cf, ax=ax, shrink=0.6)
        plt.show()

    def plotcountgrid(self, **kwargs):
        """Show counting grid."""

        proj = EqualAreaProj(**kwargs)

        fig, ax = plt.subplots(figsize=apsg_conf["figsize"])
        ax.set_aspect(1)
        ax.set_axis_off()

        # Projection circle
        # Projection circle frame
        theta = np.linspace(0, 2 * np.pi, 200)
        ax.plot(np.cos(theta), np.sin(theta), "k", lw=2)
        # add clipping circle
        primitive = Circle(
            (0, 0),
            radius=1,
            edgecolor="black",
            fill=False,
            clip_box="None",
            label="_nolegend_",
        )
        ax.add_patch(primitive)
        dcgrid = np.asarray(self.grid).T
        dcgrid = dcgrid[:, dcgrid[2] >= -0.5]
        X, Y = proj.project_data(*dcgrid, clip_inside=False)
        triang = tri.Triangulation(X, Y)
        tp = ax.triplot(triang, "bo-")
        for h in tp:
            h.set_clip_path(primitive)
        fig.tight_layout()
        plt.show()

    def angmech(self, faults, **kwargs):
        """Implementation of Angelier-Mechler dihedra method

        Args:
            faults: ``FaultSet`` of data

        Kwargs:
            method: 'probability' or 'classic'. Classic method assigns +/-1
            to individual positions, while 'probability' returns maximum
            likelihood estimate.
            Other kwargs are passed to contourf

        """

        def angmech(dc, fs):
            val = 0
            for f in fs:
                val += 2 * float(np.sign(dc.dot(f.fvec)) == np.sign(dc.dot(f.lvec))) - 1
            return val

        def angmech2(dc, fs):
            val = 0
            d = Lineation(dc)
            for f in fs:
                s = 2 * float(np.sign(dc.dot(f.fvec)) == np.sign(dc.dot(f.lvec))) - 1
                lprob = 1 - abs(45 - f.lin.angle(d)) / 45
                fprob = 1 - abs(45 - f.fol.angle(d)) / 45
                val += s * lprob * fprob
            return val

        method = kwargs.pop("method", "classic")
        if method == "probability":
            self.apply_func(angmech2, faults)
        else:
            self.apply_func(angmech, faults)
