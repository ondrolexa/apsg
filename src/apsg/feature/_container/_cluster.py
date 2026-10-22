"""Hierarchical clustering of vector and pair sets."""

import matplotlib.pyplot as plt
import numpy as np
from scipy.cluster.hierarchy import dendrogram, fcluster, linkage

from apsg.config import apsg_conf
from apsg.feature._container._pairsets import PairSet
from apsg.feature._container._vectorsets import Vector2Set, Vector3Set
from apsg.math._vector import Axial2, Axial3


class ClusterSet:
    """
    Provides a hierarchical clustering using `scipy.cluster` routines.
    The distance matrix is calculated as an angle between features, where ``Foliation``
    and ``Lineation`` use axial angles while ``Vector3`` uses direction angles.

    Args:
        d (FeatureSet): Data to cluster, e.g. ``Vector3Set``, ``Vector2Set`` or ``PairSet``.

    Keyword Args:
        maxclust (int): Desired number of clusters. Default 2.
        angle (float): Forms flat clusters so that the original observations in each
            cluster have no greater angle. Default is None to use maxclust criterion.
        method (str): Method for calculating the distance between the newly formed
            cluster and observations. Default is 'average' for UPGMA algorithm.

    """

    def __init__(self, d, **kwargs):
        if not isinstance(d, (Vector2Set, Vector3Set, PairSet)):
            raise TypeError("Only vec2set, vecset and pairset could be clustered")
        self.data = d.copy()
        self.maxclust = kwargs.get("maxclust", 2)
        self.angle = kwargs.get("angle", None)
        self.method = kwargs.get("method", "average")
        self.pdist = self.data.angle()
        self.linkage()
        self.cluster()

    def __repr__(self):
        info = f"Already {len(self.groups)} clusters created."
        if self.angle is not None:
            crit = f"Criterion: Angle\nSettings: distance={self.angle:.4g}\n"
        else:
            crit = f"Criterion: Maxclust\nSettings: maxclust={self.maxclust:.4g}\n"
        return (
            "ClusterSet\n"
            + f"Number of data: {len(self.data)}\n"
            + f"Linkage method: {self.method}\n"
            + crit
            + info
        )

    def cluster(self, **kwargs):
        """Do clustering of data.

        Result is stored as tuple of Groups in ``groups`` property.

        Keyword Args:
            maxclust (int): Desired number of clusters. Default 2.
            angle (float): Forms flat clusters so that the original observations in each
                cluster have no greater angle. Default is None to use maxclust criterion.

        Returns:
            None
        """

        self.maxclust = kwargs.get("maxclust", self.maxclust)
        self.angle = kwargs.get("angle", self.angle)

        if self.angle is not None:
            self.idx = fcluster(self.Z, self.angle, criterion="distance")
        else:
            self.idx = fcluster(self.Z, self.maxclust, criterion="maxclust")
        self.groups = tuple(
            self.data[np.flatnonzero(self.idx == c)] for c in np.unique(self.idx)
        )

    def linkage(self, **kwargs):
        """Do linkage of distance matrix.

        method (str): Method for calculating the distance between the newly formed
            cluster and observations. Default is 'average' for UPGMA algorithm.

        Returns:
            None
        """

        self.method = kwargs.get("method", self.method)
        if issubclass(self.data.__feature_class__, (Axial2, Axial3)):
            self.Z = linkage(self.pdist, method=self.method, metric=angle_metric_axial)
        else:
            self.Z = linkage(self.pdist, method=self.method, metric=angle_metric)

    def dendrogram(self, **kwargs):
        """Show dendrogram."""

        _fig, ax = plt.subplots(figsize=apsg_conf.figsize)
        dendrogram(self.Z, ax=ax, **kwargs)
        plt.show()

    def elbow(self, no_plot=False, n=None):
        """Plot within groups variance vs. number of clusters.

        The variance is the spherical variance for vectors and ``1 - tau1``,
        with ``tau1`` the largest eigenvalue of the orientation tensor, for axes
        (lineations, foliations).
        """

        if n is None:
            idx = fcluster(self.Z, len(self.data), criterion="maxclust")
            nclust = list(np.arange(1, np.sqrt(idx.max() / 2) + 1, dtype=int))
        else:
            nclust = list(np.arange(1, n + 1, dtype=int))
        within_grp_var = []
        mean_var = []
        for k in nclust:
            idx = fcluster(self.Z, k, criterion="maxclust")
            grp = [np.flatnonzero(idx == c) for c in np.unique(idx)]
            var = [100 * _dispersion(self.data[ix]) for ix in grp]
            within_grp_var.append(var)
            mean_var.append(np.mean(var))
        if not no_plot:
            _fig, ax = plt.subplots(figsize=apsg_conf.figsize)
            ax.boxplot(within_grp_var, positions=nclust)
            ax.plot(nclust, mean_var, "k")
            ax.set_xlabel("Number of clusters")
            ax.set_ylabel("Variance")
            ax.set_title("Within-groups variance vs. number of clusters")
            plt.show()
        else:
            return nclust, within_grp_var

    @property
    def R(self):
        """Return group of clusters resultants."""

        return type(self.data)([group.R() for group in self.groups])


def _dispersion(group):
    """Spherical variance of vectors, ``1 - tau1`` of axes (orientation tensor)."""
    if issubclass(group.__feature_class__, Axial3):
        return 1 - group.ortensor().eigenvalues(0)
    return group.var()


def angle_metric(u, v):
    return np.degrees(np.arccos(np.dot(u, v)))


def angle_metric_axial(u, v):
    return np.degrees(np.arccos(np.abs(np.dot(u, v))))
