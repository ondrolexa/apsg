"""Clustering of vector and pair sets: hierarchical and by mixture models."""

import json
from contextlib import nullcontext

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import dendrogram, fcluster, linkage

from apsg.config import apsg_conf, apsg_conf_context
from apsg.feature._container._pairsets import PairSet
from apsg.feature._container._vectorsets import Vector2Set, Vector3Set
from apsg.feature._geodata import Lineation
from apsg.helpers._helper import check_weights
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


class BinghamClusters:
    """Clusters of axes found by a mixture of Bingham distributions.

    Unlike the hierarchical ``ClusterSet``, every datum has a graded
    membership (``responsibilities``) of all clusters, so the overlap of
    fracture sets is kept; ``idx`` and ``groups`` assign each datum to its most
    probable cluster. Created by ``Vector3Set.cluster_bingham``.

    Args:
        data (Vector3Set): axes to cluster, e.g. ``LineationSet``.
        mixture (BinghamMixture): the mixture fitted to `data`.

    Keyword Args:
        threshold (float): minimum membership of the most probable cluster, in
            (0, 1], below which a datum is left unassigned (``idx`` is -1).
            Default None assigns every datum.
        weights: the observation weights used to fit `mixture` (normalized to
            sum 1), kept for the effective sample sizes of ``table``. Default
            None, all equal.

    Attributes:
        data: copy of the clustered set.
        mixture: the fitted ``BinghamMixture``.
        responsibilities: ``(N, K)`` array of memberships, rows sum to 1.
        idx: ``(N,)`` array with the cluster of every datum (0 to K-1), or -1.
        groups: tuple of K sets, the data of each cluster (possibly empty).
        unassigned: set of the data below `threshold`.
    """

    def __init__(self, data, mixture, *, threshold=None, weights=None):
        if threshold is not None and not 0 < threshold <= 1:
            raise ValueError(f"threshold must be in (0, 1], got {threshold!r}")
        self.data = data.copy()
        self.mixture = mixture
        self.threshold = threshold
        self.weights = (
            None if weights is None else check_weights(weights, len(self.data))
        )
        self.responsibilities = mixture.responsibilities(data)
        idx = np.argmax(self.responsibilities, axis=1)
        if threshold is not None:
            idx[self.responsibilities.max(axis=1) < threshold] = -1
        self.idx = idx
        self.groups = tuple(
            self.data[np.flatnonzero(idx == k)] for k in range(len(mixture))
        )
        self.unassigned = self.data[np.flatnonzero(idx == -1)]

    def __repr__(self):
        sizes = ", ".join(str(len(g)) for g in self.groups)
        extra = f"; {len(self.unassigned)} unassigned" if len(self.unassigned) else ""
        return f"BinghamClusters: {len(self)} clusters of sizes {sizes}{extra}"

    def __len__(self):
        return len(self.groups)

    def __getitem__(self, k):
        return self.groups[k]

    @property
    def R(self):
        """Return set of cluster resultants (principal axes of the groups)."""
        return type(self.data)([group.R() for group in self.groups if len(group)])

    def summary(self, level=0.95):
        """Parameters of every cluster, e.g. as input of a DFN generator.

        Args:
            level: confidence level of the ellipse. Default 0.95.

        Returns:
            list: one dict per cluster with the mixing ``weight``, the number
            ``n`` of assigned data, the mean axis ``mean``, the ``major`` axis
            and the concentrations ``kappa_major`` and ``kappa_minor`` of the
            component, and the confidence ``ellipse`` of the mean axis of the
            assigned data (``bingham_statistics``, None for fewer than 5
            data).
        """
        out = []
        for weight, comp, group in zip(
            self.mixture.weights, self.mixture.components, self.groups
        ):
            ellipse = None
            if len(group) >= 5:
                try:
                    ellipse = group.bingham_statistics(level)
                except ValueError:
                    ellipse = None
            out.append(
                {
                    "weight": float(weight),
                    "n": len(group),
                    "mean": comp.mean,
                    "major": comp.params["major"],
                    "kappa_major": comp.kappa_major,
                    "kappa_minor": comp.kappa_minor,
                    "ellipse": ellipse,
                }
            )
        return out

    def table(self, level=0.95, notation=None):
        """Parameters of every cluster as flat records, e.g. for a DFN generator.

        Args:
            level: confidence level of the cone and the ellipse. Default 0.95.
            notation: orientation notation of ``mean_azimuth`` and
                ``mean_inclination`` for foliations, "dd" (dip direction, dip)
                or "rhr" (strike, dip). Default None uses ``apsg_conf.notation``.

        Returns:
            list: one dict per cluster, ordered by mixing weight, with
            ``set``, ``weight`` (mixing weight), ``n`` (data assigned to the
            cluster), ``n_eff`` (effective size of the memberships, accounting
            for weights), ``mean_azimuth`` and ``mean_inclination`` (the mean
            of the cluster as a feature of the data, e.g. a foliation whose pole
            is the mean axis), ``pole_trend`` and ``pole_plunge`` (the mean
            axis as a lineation), ``kappa_major`` and ``kappa_minor`` (Bingham
            concentrations), ``fisher_kappa`` and ``fisher_k`` (Fisher
            concentration and precision of the assigned data folded to one
            hemisphere), ``cone`` (confidence cone of the mean axis),
            ``gamma_major`` and ``gamma_minor`` (semi-angles of the Bingham
            confidence ellipse) and ``level``. Values that need more data than
            a cluster has (2 for the Fisher values, 5 for the ellipse) are NaN.
        """
        nan = float("nan")
        feature_cls = self.data.__feature_class__
        w_all = self.weights if self.weights is not None else None
        rows = []
        with _notation_context(notation):
            for k, (weight, comp, group) in enumerate(
                zip(self.mixture.weights, self.mixture.components, self.groups)
            ):
                mean = np.asarray(comp.mean, dtype=float)
                member = self.responsibilities[:, k]
                if w_all is not None:
                    member = member * w_all
                total = member.sum()
                n_eff = float(total**2 / np.sum(member**2)) if total > 0 else nan
                wk = None if w_all is None else w_all[self.idx == k]
                if wk is not None and len(wk) and wk.sum() <= 0:
                    wk = None
                row = dict.fromkeys(
                    (
                        "fisher_kappa",
                        "fisher_k",
                        "cone",
                        "gamma_major",
                        "gamma_minor",
                    ),
                    nan,
                )
                if len(group) >= 2:
                    try:
                        fs = group._fisher_statistics(level, weights=wk)
                        row.update(
                            fisher_kappa=float(fs["kappa"]),
                            fisher_k=float(fs["k"]),
                            cone=float(fs["alpha"]),
                        )
                    except ValueError:
                        pass
                if len(group) >= 5:
                    try:
                        bs = group.bingham_statistics(level, weights=wk)
                        row.update(
                            gamma_major=float(bs["gamma"][0]),
                            gamma_minor=float(bs["gamma"][1]),
                        )
                    except ValueError:
                        pass
                azi, inc = feature_cls(mean).geo
                trend, plunge = Lineation(mean).geo
                rows.append(
                    {
                        "set": k,
                        "weight": float(weight),
                        "n": len(group),
                        "n_eff": n_eff,
                        "mean_azimuth": float(azi),
                        "mean_inclination": float(inc),
                        "pole_trend": float(trend),
                        "pole_plunge": float(plunge),
                        "kappa_major": float(comp.kappa_major),
                        "kappa_minor": float(comp.kappa_minor),
                        **{
                            key: row[key]
                            for key in (
                                "fisher_kappa",
                                "fisher_k",
                                "cone",
                                "gamma_major",
                                "gamma_minor",
                            )
                        },
                        "level": float(level),
                    }
                )
        return rows

    def to_dataframe(self, level=0.95, notation=None):
        """Return the ``table`` as a ``pandas.DataFrame``, one row per cluster."""
        return pd.DataFrame(self.table(level, notation))

    def to_csv(self, path, level=0.95, notation=None, **kwargs):
        """Write the ``table`` as a CSV file, one row per cluster.

        Args:
            path: file name or buffer.
            level: confidence level. Default 0.95.
            notation: notation of foliations, see ``table``.
            **kwargs: passed to ``pandas.DataFrame.to_csv``.
        """
        self.to_dataframe(level, notation).to_csv(path, index=False, **kwargs)

    def to_json(self, path=None, level=0.95, notation=None):
        """Return the ``table`` with the fitted mixture as a JSON-serializable dict.

        Args:
            path: optional file name to write the JSON to.
            level: confidence level. Default 0.95.
            notation: notation of foliations, see ``table``.

        Returns:
            dict: ``datatype``, ``notation``, ``level``, ``n`` and ``n_eff`` of
            the data, ``unassigned`` count, ``threshold``, ``sets`` (the rows
            of ``table``, NaN as None) and the ``mixture`` (``BinghamMixture``
            JSON, which ``feature_from_json`` reads back).
        """
        sets = [
            {
                k: (None if isinstance(v, float) and np.isnan(v) else v)
                for k, v in row.items()
            }
            for row in self.table(level, notation)
        ]
        n = len(self.data)
        w = self.weights
        out = {
            "datatype": type(self).__name__,
            "notation": notation if notation is not None else apsg_conf.notation,
            "level": float(level),
            "n": n,
            "n_eff": float(n if w is None else 1.0 / np.sum(w**2)),
            "unassigned": len(self.unassigned),
            "threshold": self.threshold,
            "sets": sets,
            "mixture": self.mixture.to_json(),
        }
        if path is not None:
            with open(path, "w") as f:
                json.dump(out, f, indent=2)
        return out


def _notation_context(notation):
    """Context temporarily setting the notation, a no-op for None."""
    return apsg_conf_context(notation=notation) if notation else nullcontext()


def _dispersion(group):
    """Spherical variance of vectors, ``1 - tau1`` of axes (orientation tensor)."""
    if issubclass(group.__feature_class__, Axial3):
        return 1 - group.ortensor().eigenvalues(0)
    return group.var()


def angle_metric(u, v):
    return np.degrees(np.arccos(np.dot(u, v)))


def angle_metric_axial(u, v):
    return np.degrees(np.arccos(np.abs(np.dot(u, v))))
