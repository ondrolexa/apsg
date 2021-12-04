import numpy as np
import matplotlib.tri as tri
from matplotlib.patches import Circle
import matplotlib.pyplot as plt

from apsg.feature._statistics import estimate_k
from apsg.feature._container import Vector3Set


class StereoGrid:
    """
    The class to store regular grid of values to be contoured on ``StereoNet``.

    ``StereoGrid`` object could be calculated from ``Group`` object or by user-
    defined function, which accept unit vector as argument.

    Args:
        n: number of grid points Default 2000
        grid_type: type of grid 'gss' or 'sfs'. Default 'gss'
  
    Note: Euclidean norms are used as weights. Normalize data if you dont want to use weigths.

    """

    def __init__(self, **kwargs):
        # parse options
        self.n = kwargs.get("n", 2000)
        if kwargs.get("grid_type", "gss") == "gss":
            self.grid = Vector3Set.uniform_gss(n=self.n)
        else:
            self.grid = Vector3Set.uniform_sfs(n=self.n)
        self.values = np.zeros(self.n, dtype=float)

    def __repr__(self):
        maxazi, maxinc = self.max_at()
        minazi, mininc = self.min_at()
        if self.max() > self.min():
            info = (
                f"\nMaximum: {self.max():.4f} at V:{maxazi:.0f}/{maxinc:.0f}"
                + f"\nMinimum: {self.min():.4f} at V:{minazi:.0f}/{mininc:.0f}"
            )
        else:
            info = ""
        return f"StereoGrid with {self.n} points." + info

    def min(self):
        return self.values.min()

    def max(self):
        return self.values.max()

    def min_at(self):
        return self.grid[self.values.argmin()].geo

    def max_at(self):
        return self.grid[self.values.argmax()].geo

    def calculate_density(self, features, **kwargs):
        """Calculate density of elements from ``FeatureSet`` object.

        Args:
            sigma: if none k is calculated automatically. Default None
            trim: if True, values < 0 are clipped to zero. . Default True
        
        """
        # parse options
        sigma = kwargs.get("sigma", None)
        n = len(features)
        if sigma is None:
            k = estimate_k(features)
            sigma = np.sqrt(2 * n / (k - 2))
        else:
            k = 2 * (1.0 + n / sigma ** 2)
        # method = kwargs.get("method", "exp_kamb")
        trim = kwargs.get("trim", True)
        # do calc
        scale = np.sqrt(n * (k / 2.0 - 1) / k ** 2)
        self.values = (
            np.exp(k * (np.abs(np.dot(self.grid, np.asarray(features).T)) - 1)).sum(
                axis=1
            )
            / scale
            / sigma
        )
        if trim:
            self.values[self.values < 0] = np.finfo(float).tiny

    def apply_func(self, func, *args, **kwargs):
        """Calculate values using function passed as argument.
        Function must accept Vector3 like (or 3 elements array)
        as argument and return scalar value.

        """
        for i in range(self.n):
            self.values[i] = func(self.grid[i], *args, **kwargs)

    def contourf(self, *args, **kwargs):
        """ Show filled contours of values."""

        n = kwargs.get("n", 7)
        kind = str(kwargs.get("kind", "Equal-area")).lower()
        if kind in ["equal-area", "schmidt", "earea"]:
            from apsg.plotting._projection import EqualAreaProj

            proj = EqualAreaProj(**kwargs)
        elif kind in ["equal-angle", "wulff", "eangle"]:
            from apsg.plotting._projection import EqualAngleProj

            proj = EqualAngleProj(**kwargs)
        else:
            raise TypeError("Only 'Equal-area' and 'Equal-angle' implemented")
        contour_defaults = {"cmap": "Greys", "linestyles": "-", "antialiased": True}

        fig, ax = plt.subplots()
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
        X, Y = proj.project_data(*dcgrid, clip_inside=False)
        intervals = np.linspace(0, self.max(), n)
        cf = ax.tricontourf(X, Y, self.values, levels=intervals, **contour_defaults)
        for collection in cf.collections:
            collection.set_clip_path(primitive)
        plt.colorbar(cf, format="%3.2f", spacing="proportional")
        plt.show()

    def contour(self, *args, **kwargs):
        """ Show contours of values."""
        fig, ax = plt.subplots(figsize=settings["figsize"])
        # Projection circle
        ax.text(0, 1.02, "N", ha="center", va="baseline", fontsize=16)
        ax.add_artist(plt.Circle((0, 0), 1, color="w", zorder=0))
        ax.add_artist(plt.Circle((0, 0), 1, color="None", ec="k", zorder=3))
        ax.set_aspect("equal")
        plt.tricontour(self.triang, self.values, *args, **kwargs)
        plt.colorbar()
        plt.axis("off")
        plt.show()

    def plotcountgrid(self, **kwargs):
        """ Show counting grid."""

        proj = EqualAreaProj(**kwargs)

        fig, ax = plt.subplots()
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
